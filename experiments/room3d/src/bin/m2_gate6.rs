//! M2 gate 6: one-bounce indirect light from cone G-buffers, live on the GPU.
//!
//! Probes sit about 0.04 apart on every static face. Each caches a hemicube
//! G-buffer (16×16 bins per full face, 768 per probe): the nearest static
//! surface along each bin's center direction. Geometry is static, so this is
//! built once. Each frame, every probe gathers bounce radiance (the hit
//! surface's albedo times its cached direct irradiance, which gate 5 keeps exact
//! as boxes move), with moving boxes composited by depth as black occluders. The
//! render adds bilinearly interpolated probe irradiance on static surfaces.
//!
//! Pass criterion, fixed before the first run:
//! 1. Exact incremental updates: on every one of 120 frames, each probe's
//!    indirect irradiance equals a from-scratch gather that traces every bin
//!    through the current scene (movers in place, black), within 1e-5 of the
//!    peak probe value.
//! 2. Accuracy: on the last frame, the probes' indirect irradiance is within 5%
//!    relative RMSE (sqrt(Σ|a − b|² / Σ|b|²) over RGB, over probes the
//!    reference marks valid) of a 64×64-per-face reference gather (16× the
//!    directions) with the same direct cache and movers.
//! 3. Speed: the mean frame (gate 5's update plus the gather, and the 1920×1080
//!    render with direct and indirect light, each submitted and waited on) takes
//!    at most 16.7 ms on the M4 Air.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{
    Camera, Lightmap, LiveRenderer, MaskConeKernel, PROBE_RESOLUTION, RectLight, wgpu,
};
use flatland::img::Image;
use room3d::{movers, render_scene, to_rgb};

const W: u32 = 1920;
const H: u32 = 1080;
const FRAMES: u32 = 120;
const TEXEL: f32 = 0.005;
const PROBES: f32 = 0.04;
const TOLERANCE: f32 = 1e-5;
const REFERENCE_RESOLUTION: u32 = 64;

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
        one_sided: false,
    };
    let camera = Camera {
        eye: scene.eye,
        target: scene.target,
        vertical_fov_degrees: scene.vertical_fov_degrees,
    };
    let mut lightmap =
        Lightmap::from_meshes(&scene.room.meshes, TEXEL, |m| m < first).with_probes(PROBES);
    kernel.bake(&mut lightmap, light, &scene.room.meshes);
    let start = Instant::now();
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
    renderer.wait();
    let setup = start.elapsed();
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

    let (mut updates, mut renders) = (Vec::new(), Vec::new());
    let (mut max_diff, mut failures) = (0.0f32, 0usize);
    let mut probes = Vec::new();
    for f in 1..=FRAMES {
        let changes: Vec<_> = movers(f as f32)
            .into_iter()
            .enumerate()
            .map(|(k, mesh)| (first + k, mesh))
            .collect();
        let start = Instant::now();
        renderer.update_meshes(&changes);
        renderer.wait();
        let updated = Instant::now();
        renderer.frame(camera, camera.vertical_fov_degrees, &view);
        renderer.wait();
        updates.push(updated - start);
        renders.push(updated.elapsed());

        probes = renderer.read_probes();
        let scratch = renderer.probes_from_scratch(&kernel, PROBE_RESOLUTION);
        let peak = scratch
            .iter()
            .flat_map(|p| &p[..3])
            .copied()
            .fold(0.0, f32::max);
        for (a, b) in probes.iter().zip(&scratch) {
            let d = (0..4).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max) / peak;
            max_diff = max_diff.max(d);
            if d > TOLERANCE {
                failures += 1;
            }
        }
        scene.room.meshes.truncate(first);
        scene
            .room
            .meshes
            .extend(changes.into_iter().map(|(_, mesh)| mesh));
    }

    // Accuracy on the last frame against 16× the directions.
    let reference = renderer.probes_from_scratch(&kernel, REFERENCE_RESOLUTION);
    let (mut err, mut norm, mut counted) = (0.0f64, 0.0f64, 0usize);
    for (a, b) in probes.iter().zip(&reference) {
        if b[3] > 0.0 {
            counted += 1;
            for c in 0..3 {
                err += ((a[c] - b[c]) as f64).powi(2);
                norm += (b[c] as f64).powi(2);
            }
        }
    }
    let rmse = (err / norm).sqrt();

    // The gather alone, for the breakdown (not part of the criterion).
    let mut gather = Duration::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        renderer.regather();
        renderer.wait();
        gather = gather.min(start.elapsed());
    }

    // The last frame from the incrementally updated caches.
    lightmap.irradiance = renderer.read_texels();
    lightmap.indirect = probes;
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
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(&out_dir.join("m2_gate6.png"))?;

    let ms = |d: &Duration| d.as_secs_f64() * 1000.0;
    let mean = |v: &[Duration]| v.iter().map(ms).sum::<f64>() / v.len() as f64;
    let frames: Vec<Duration> = updates.iter().zip(&renders).map(|(a, b)| *a + *b).collect();
    let worst = frames.iter().map(ms).fold(0.0, f64::max);
    let (update_ms, render_ms, frame_ms) = (mean(&updates), mean(&renders), mean(&frames));
    let pass = failures == 0 && rmse <= 0.05 && frame_ms <= 16.7;
    let bins = 3 * PROBE_RESOLUTION * PROBE_RESOLUTION;
    let mut out = String::new();
    writeln!(out, "# M2 gate 6 results\n").unwrap();
    writeln!(
        out,
        "Render scene ({first} static meshes) plus 3 moving boxes, {FRAMES} frames, {} at {W}×{H}. \
         {} probes about {PROBES} apart, {bins} bins each ({:.0} MB of G-buffers); setup \
         (upload, G-buffer build, first gather) {:.1} ms.\n",
        kernel.adapter_name,
        renderer.probes(),
        renderer.probes() as f64 * bins as f64 * 8.0 / 1e6,
        ms(&setup)
    )
    .unwrap();
    writeln!(
        out,
        "- Update (gate 5's update plus the gather): {update_ms:.2} ms mean; the gather alone: {:.2} ms.\n\
         - Render at {W}×{H} with direct and indirect light: {render_ms:.2} ms mean.\n\
         - Frame (update + render): {frame_ms:.2} ms mean, {worst:.2} ms worst.\n\
         - Exactness against a from-scratch trace: max difference {max_diff:.1e} of peak; probe-frames over {TOLERANCE:e}: {failures}.\n\
         - Accuracy against {REFERENCE_RESOLUTION}×{REFERENCE_RESOLUTION} bins per face: relative RMSE {:.2}% over {counted} valid probes.",
        ms(&gather),
        100.0 * rmse
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nExact, within 5% of the reference, and at most 16.7 ms per frame → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m2_gate6_results.md"), &out)?;
    Ok(())
}
