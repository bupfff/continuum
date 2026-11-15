//! M12 gate 3: glTF skins and morph targets, posed as Blender poses them.
//!
//! The user's order (2026-11-15): animated characters imported and played
//! before anti-aliasing. M12 gates 1–2 render geometry an application poses
//! (`RendererOptions::deforming`, `Renderer::set_vertices`). This gate
//! brings glTF's own deformation:
//! - **Read:** skins (joints, inverse bind matrices), each vertex's joints
//!   and weights (`JOINTS_0`, `WEIGHTS_0`), morph targets' position and
//!   normal deltas, and the mesh's default weights. Kept per triangle
//!   corner, in the order and with the degenerate triangles left out as
//!   the geometry built from that primitive.
//! - **Animation:** scale and morph-weight channels as well as translation
//!   and rotation.
//! - **Placed:** a node with a skin, or a mesh with morph targets, gets a
//!   geometry of its own (poses differ per node). A skinned mesh sits at the
//!   world origin with its skinned rest pose as its shape (glTF ignores the
//!   skinned node's own transform); a morphing mesh without a skin keeps
//!   its node's transform.
//! - **Posed:** `Animations::deform(animations, time)` gives each deforming
//!   geometry's triangles and normals at a time: the targets' deltas by the
//!   weights, then linear blend skinning by the joints' world matrices times
//!   their inverse bind matrices. `Imported::deforming` lists the geometries
//!   for `RendererOptions::deforming`.
//!
//! Test file: `experiments/room3d/cycles/make_skinned_test.py`, made in
//! Blender 5.2 (no download): a tube 2 m tall of 32 segments by 24 rings,
//! skinned to three bones with automatic weights, with a shape key; a
//! 48-frame animation (24 fps) bends the middle bone 60° and back, twists
//! the top bone, and drives the shape key 0 → 1 → 0.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Posed as Blender poses it:** at 0, 0.25, 0.5, 0.75, 1.0, 1.25,
//!    1.5 and 1.9 s (both animations), every triangle corner of Continuum's
//!    pose lies within 0.2 mm (1e-4 of the tube's height) of a vertex of
//!    Blender's own import of the same file evaluated at that time
//!    (`compare_skinned_pose.py`: its glTF importer, armature and shape
//!    keys), and the mean is below 0.05 mm.
//! 2. **Played:** the imported tube rendered through `Renderer` with the
//!    cache, posed every frame from `Animations::deform` over the
//!    animation (48 frames at 24 fps, three loops), settles to an image
//!    within 1% relative RMSE of a renderer built with the final pose as
//!    its rest pose (the lag caught up), and the pose shows (at least 5%
//!    from the rest pose's image).
//!
//! Reported: the import's and a pose's CPU time; the tube's corner count.
//!
//! Development before the run (disclosed):
//! - **Reference fixes in `compare_skinned_pose.py`:**
//!   - Blender's importer puts glTF time t at frame t × fps (its exporter
//!     wrote frame f at f / fps), not 1 + t × fps: the script was a frame
//!     off.
//!   - It leaves the shape keys' animation as a muted NLA track; the
//!     script makes it active with its action slot (Blender 4.4 on).
//! - **`Renderer::is_settled`** also waits while deforming geometry's
//!   lighting lags its pose or movers' pages re-bake: it read settled
//!   before the catch-up ended (10% from the built pose; 0.0000% after).
//! - **After the fixes, development** (0.6 s, one loop): 0.001 mm, and
//!   0.0000% played.
//!
//! `cargo run --release -p room3d --bin m12_gate3`
//!
//! `M12_DEV=1`: one comparison (0.6 s) and one loop played (development).

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec2, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, Light, MaskConeKernel, RectLight, Renderer, RendererOptions,
    wgpu,
};
use room3d::cycles::relative_rmse;

const W: u32 = 320;
const H: u32 = 240;

fn blender() -> String {
    std::env::var("BLENDER")
        .unwrap_or_else(|_| "/Applications/Blender.app/Contents/MacOS/Blender".into())
}

/// Blender's verdict on Continuum's pose `corners` (Z-up world) of `glb`
/// at `seconds`: the largest and mean distance to its own evaluated mesh.
fn compare(glb: &Path, seconds: f32, corners: &[Vec3], work: &Path) -> (f64, f64, String) {
    let file = work.join(format!("m12_gate3_corners_{seconds:.2}.f32"));
    let bytes: Vec<u8> = corners
        .iter()
        .flat_map(|v| v.to_array())
        .flat_map(f32::to_le_bytes)
        .collect();
    std::fs::write(&file, bytes).expect("corners");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("cycles/compare_skinned_pose.py");
    let out = Command::new(blender())
        .args(["-b", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(glb)
        .arg(format!("{seconds}"))
        .arg(&file)
        .arg(std::env::var("M12_BLENDER_ANIM").unwrap_or_default())
        .output()
        .expect("Blender");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let line = text
        .lines()
        .find(|l| l.starts_with("max "))
        .unwrap_or_else(|| panic!("no comparison from Blender: {text}"))
        .to_string();
    let words: Vec<&str> = line.split_whitespace().collect();
    (words[1].parse().unwrap(), words[3].parse().unwrap(), line)
}

fn target(r: &Renderer) -> wgpu::TextureView {
    r.kernel()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn main() {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let glb = out_dir.join("skinned/tube.glb");
    if !glb.exists() {
        println!(
            "Make the test file first: Blender -b --factory-startup --python experiments/room3d/cycles/make_skinned_test.py -- {}",
            glb.display()
        );
        return;
    }
    let t = Instant::now();
    let imported = continuum_gltf::load(&glb).expect("import");
    let import_ms = t.elapsed().as_secs_f64() * 1000.0;
    // (Development, `M12_ANIM=0,1`: the animations posed.)
    let animations: Vec<usize> = match std::env::var("M12_ANIM") {
        Ok(list) => list.split(',').filter_map(|a| a.parse().ok()).collect(),
        Err(_) => (0..imported.animations.list.len()).collect(),
    };
    let deforming = imported.deforming();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let mut pass = true;
    say(format!(
        "imported in {import_ms:.1} ms: {} deforming geometries, {} animations, {} channels skipped",
        deforming.len(),
        animations.len(),
        imported.stats.animation_channels_skipped
    ));
    // 1. Posed as Blender poses it.
    let mut worst: f64 = 0.0;
    let mut worst_mean: f64 = 0.0;
    let mut pose_ms = 0.0;
    let mut corners_n = 0;
    let dev = std::env::var_os("M12_DEV").is_some();
    let dev_time: f32 = std::env::var("M12_T")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0.6);
    let dev_times = [dev_time];
    let times: &[f32] = if dev {
        &dev_times
    } else {
        &[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.9]
    };
    for &seconds in times {
        let t = Instant::now();
        let poses = imported.animations.deform(&animations, seconds);
        pose_ms = t.elapsed().as_secs_f64() * 1000.0;
        let corners: Vec<Vec3> = poses
            .iter()
            .flat_map(|p| p.triangles.iter().flatten().copied())
            .collect();
        corners_n = corners.len();
        let (max, mean, line) = compare(&glb, seconds, &corners, &out_dir);
        worst = worst.max(max);
        worst_mean = worst_mean.max(mean);
        say(format!("   {seconds:.2} s: {line}"));
    }
    let c1 = worst <= 2e-4 && worst_mean < 5e-5 && corners_n > 0;
    pass &= c1;
    say(format!(
        "1. posed as Blender poses it: largest distance {:.4} mm (at most 0.2), largest mean {:.4} mm (below 0.05) over {corners_n} corners: {}; a pose took {pose_ms:.2} ms",
        1e3 * worst,
        1e3 * worst_mean,
        verdict(c1)
    ));
    // 2. Played through the renderer, with a ground and a light.
    let mut scene = imported.scene.clone();
    let ground = scene.geometries.len();
    let h = 4.0;
    scene.geometries.push(Surface::from_mesh(
        vec![
            [
                Vec3::new(-h, -h, 0.0),
                Vec3::new(h, -h, 0.0),
                Vec3::new(h, h, 0.0),
            ],
            [
                Vec3::new(-h, -h, 0.0),
                Vec3::new(h, h, 0.0),
                Vec3::new(-h, h, 0.0),
            ],
        ],
        None,
        Some(vec![
            [Vec2::ZERO, Vec2::X, Vec2::ONE],
            [Vec2::ZERO, Vec2::ONE, Vec2::Y],
        ]),
    ));
    for v in [
        &mut scene.geometry_textures,
        &mut scene.geometry_normal_maps,
        &mut scene.geometry_metallic_roughness,
        &mut scene.geometry_emissive_textures,
    ] {
        if !v.is_empty() {
            v.push(None);
        }
    }
    for v in [&mut scene.geometry_alpha_cutoff, &mut scene.geometry_blend] {
        if !v.is_empty() {
            v.push(None);
        }
    }
    if !scene.geometry_materials.is_empty() {
        scene.geometry_materials.push(None);
    }
    if !scene.geometry_emission.is_empty() {
        scene.geometry_emission.push(Vec3::ZERO);
    }
    if !scene.geometry_colors.is_empty() {
        scene.geometry_colors.push(None);
    }
    scene.instances.push(Instance {
        geometry: ground,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position: DVec3::ZERO,
        albedo: Vec3::splat(0.6),
    });
    let light = Light::white(
        RectLight {
            corner: Vec3::new(2.0, -3.0, 4.0),
            edge_u: Vec3::new(0.6, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.6, 0.0),
            one_sided: false,
        },
        40.0,
    );
    let camera = Camera {
        eye: Vec3::new(0.0, -5.0, 1.6),
        target: Vec3::new(0.0, 0.0, 1.0),
        vertical_fov_degrees: 45.0,
    };
    let options = || RendererOptions {
        cache_options: CacheOptions {
            frame_target_ms: 0.0,
            ..CacheOptions::default()
        },
        deforming: deforming.clone(),
        ..Default::default()
    };
    let make = |scene: &continuum_gpu::InstancedScene| {
        let kernel = MaskConeKernel::new().expect("GPU");
        Renderer::new(
            kernel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            (W, H),
            scene,
            DVec3::ZERO,
            &[light],
            options(),
        )
    };
    let settle = |r: &mut Renderer, view: &wgpu::TextureView| {
        for _ in 0..16 {
            r.render(camera, view);
        }
        let mut frames = 0;
        while frames < 3000 && !r.is_settled(8) {
            r.render(camera, view);
            frames += 1;
        }
        r.read_colors()
    };
    let mut r = make(&scene);
    let view = target(&r);
    let rest = settle(&mut r, &view);
    let duration = imported
        .animations
        .list
        .iter()
        .map(|a| a.duration)
        .fold(0.0f32, f32::max);
    let last = 2.0 * duration / 3.0;
    for k in 0..if dev { 48 } else { 144 } {
        let seconds = (k as f32 / 24.0) % duration;
        for p in imported.animations.deform(&animations, seconds) {
            r.set_vertices(p.geometry, p.triangles, p.normals);
        }
        r.render(camera, &view);
    }
    for p in imported.animations.deform(&animations, last) {
        r.set_vertices(p.geometry, p.triangles, p.normals);
    }
    let played = settle(&mut r, &view);
    let mut posed = scene.clone();
    for p in imported.animations.deform(&animations, last) {
        posed.geometries[p.geometry].triangles = p.triangles;
        posed.geometries[p.geometry].normals = p.normals;
    }
    let mut b = make(&posed);
    let view_b = target(&b);
    let built = settle(&mut b, &view_b);
    let (e, shown) = (relative_rmse(&played, &built), relative_rmse(&rest, &built));
    let c2 = e <= 0.01 && shown >= 0.05;
    pass &= c2;
    say(format!(
        "2. played (three loops, then held at {last:.2} s): {:.4}% from a renderer built with that pose (at most 1%); the rest pose {:.2}% from it (at least 5%): {}",
        100.0 * e,
        100.0 * shown,
        verdict(c2)
    ));
    let save = |name: &str, px: &[[f32; 3]]| {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let c = p.map(|v| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
            image.set(i % W as usize, i / W as usize, c);
        }
        image
            .save(&out_dir.join(format!("m12_gate3_{name}.png")))
            .expect("save");
    };
    save("rest", &rest);
    save("played", &played);
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(
        out_dir.join(if dev {
            "m12_gate3_dev.txt"
        } else {
            "m12_gate3.txt"
        }),
        report,
    )
    .expect("report");
}
