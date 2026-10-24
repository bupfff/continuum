//! M2 gate 5: moving objects with exact soft shadows, live on the GPU.
//!
//! A port of 3D gate 4 to the GPU lighting cache. Three boxes move through the
//! render scene. They have no texels and are shaded per pixel; static texels are
//! re-baked only in the 8×8 blocks whose cone volume (the hull of the block and
//! the light) a mover's swept bounds can reach. Everything else is reused.
//!
//! Pass criterion, fixed before the first run: (1) after every frame's update,
//! every texel equals a full re-bake with the movers at their new positions,
//! within 1e-5 of the peak texel irradiance; (2) the mean frame (mover upload,
//! hierarchy rebuild, block marking, re-bake, and the 1920×1080 cached render,
//! each submitted and waited on) takes at most 16.7 ms on the M4 Air, over 120
//! frames. The full re-bake time and the fraction of blocks re-baked are
//! reported.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum_gpu::{Camera, Lightmap, LiveRenderer, MaskConeKernel, RectLight, wgpu};
use flatland::img::Image;
use room3d::{movers, render_scene, to_rgb};

const W: u32 = 1920;
const H: u32 = 1080;
const FRAMES: u32 = 120;
const TEXEL: f32 = 0.005;
const TOLERANCE: f32 = 1e-5;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let mut scene = render_scene();
    let first = scene.add_movers(0.0);
    let l = &scene.room.light;
    let light = RectLight {
        corner: l.corner,
        edge_u: l.edge_u,
        edge_v: l.edge_v,
    };
    let camera = Camera {
        eye: scene.eye,
        target: scene.target,
        vertical_fov_degrees: scene.vertical_fov_degrees,
    };
    let mut lightmap = Lightmap::boxes(&scene.room.meshes, TEXEL, |m| m < first);
    let bake = kernel.bake(&mut lightmap, light, &scene.room.meshes);
    let mut renderer = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        light,
        &scene.room.meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
    );
    let target = kernel.device().create_texture(&wgpu::TextureDescriptor {
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
    });
    let view = target.create_view(&Default::default());

    let (mut updates, mut renders, mut full) = (Vec::new(), Vec::new(), Vec::new());
    let (mut rebaked, mut max_diff, mut failures) = (0u64, 0.0f32, 0usize);
    let mut texels = Vec::new();
    for f in 1..=FRAMES {
        let now = movers(f as f32);
        let changes: Vec<_> = now
            .iter()
            .enumerate()
            .map(|(k, mesh)| (first + k, mesh.clone()))
            .collect();
        let start = Instant::now();
        renderer.update_meshes(&changes);
        renderer.wait();
        let updated = Instant::now();
        renderer.frame(camera, camera.vertical_fov_degrees, &view);
        renderer.wait();
        updates.push(updated - start);
        renders.push(updated.elapsed());

        rebaked += renderer.rebaked_blocks() as u64;
        texels = renderer.read_texels();
        scene.room.meshes.truncate(first);
        scene.room.meshes.extend(now);
        let (reference, time) =
            kernel.irradiance_with(&lightmap.receivers, light, &scene.room.meshes, true, 1);
        full.push(time);
        let peak = reference.iter().copied().fold(0.0, f32::max);
        for (a, b) in texels.iter().zip(&reference) {
            let d = (a - b).abs() / peak;
            max_diff = max_diff.max(d);
            if d > TOLERANCE {
                failures += 1;
            }
        }
    }

    // The last frame, rendered from the incrementally updated cache.
    lightmap.irradiance = texels;
    let (pixels, _) = kernel.render_cached(
        camera,
        W,
        H,
        light,
        &scene.room.meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
        1,
    );
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(
            i % W as usize,
            i / W as usize,
            to_rgb(continuum::glam::Vec3::from_array(*p)),
        );
    }
    image.save(&out_dir.join("m2_gate5.png"))?;

    let ms = |d: &Duration| d.as_secs_f64() * 1000.0;
    let mean = |v: &[Duration]| v.iter().map(ms).sum::<f64>() / v.len() as f64;
    let frames: Vec<Duration> = updates.iter().zip(&renders).map(|(a, b)| *a + *b).collect();
    let worst = frames.iter().map(ms).fold(0.0, f64::max);
    let (update_ms, render_ms, frame_ms, full_ms) =
        (mean(&updates), mean(&renders), mean(&frames), mean(&full));
    let fraction = rebaked as f64 / (FRAMES as f64 * renderer.blocks() as f64);
    let pass = failures == 0 && frame_ms <= 16.7;
    let mut out = String::new();
    writeln!(out, "# M2 gate 5 results\n").unwrap();
    writeln!(
        out,
        "Render scene ({first} static meshes) plus 3 moving boxes, {FRAMES} frames, {} at {W}×{H}. \
         Lightmap: {} texels in {} blocks (initial bake {:.1} ms).\n",
        kernel.adapter_name,
        lightmap.texels(),
        renderer.blocks(),
        ms(&bake)
    )
    .unwrap();
    writeln!(
        out,
        "- Update (upload, hierarchy, marking, re-bake): {update_ms:.2} ms mean.\n\
         - Cached render at {W}×{H}: {render_ms:.2} ms mean.\n\
         - Frame (update + render): {frame_ms:.2} ms mean, {worst:.2} ms worst.\n\
         - Full re-bake for comparison (GPU time only): {full_ms:.2} ms mean.\n\
         - Blocks re-baked per frame: {:.1}%.\n\
         - Exactness: max difference {max_diff:.1e} of peak; texel-frames over {TOLERANCE:e}: {failures}.",
        100.0 * fraction
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nExact and at most 16.7 ms per frame → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m2_gate5_results.md"), &out)?;
    Ok(())
}
