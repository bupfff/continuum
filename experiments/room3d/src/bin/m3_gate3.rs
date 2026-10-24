//! M3 gate 3: the showcase in real time, against Cycles.
//!
//! The showcase scene with its moving objects, direct light cached in texels
//! with exact incremental re-bakes, and one bounce of indirect light from probe
//! G-buffers, measured against an independent path tracer.
//!
//! Pass criterion, fixed before the first run:
//! 1. Exact incremental updates: on every frame of the camera path, lightmap
//!    texels equal a full re-bake and probes equal a from-scratch trace, within
//!    1e-5 of peak.
//! 2. Speed: the mean frame (movers, re-bake, gather, and the 1920×1080 render
//!    with direct and indirect light, each submitted and waited on) over the
//!    300-frame camera path takes at most 16.7 ms, warm, on the M4 Air. A
//!    sustained number (after 60 s of continuous rendering) is reported.
//! 3. Quality: the static showcase frame (movers removed) at 960×540 is within
//!    5% relative RMSE (linear RGB) of a Cycles render of the same scene:
//!    Lambertian albedos, the same one-sided rectangular emitter, direct light
//!    plus one diffuse bounce, pixel-center sampling, 4096 samples per pixel.
//!    The reference's own noise is reported.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{Hierarchy, Lightmap, LiveRenderer, MaskConeKernel, PROBE_RESOLUTION, wgpu};
use flatland::img::Image;
use room3d::cycles::{self, relative_rmse};
use room3d::showcase::{PATH_FRAMES, PROBES, Showcase, TEXEL};
use room3d::to_rgb;

const W: u32 = 1920;
const H: u32 = 1080;
const QW: u32 = 960;
const QH: u32 = 540;
const TOLERANCE: f32 = 1e-5;
const SAMPLES: u32 = 4096;

fn movers(scene: &Showcase) -> Vec<(usize, &continuum::mesh::Surface)> {
    (scene.first_mover..scene.surfaces.len())
        .map(|m| (m, &scene.surfaces[m]))
        .collect()
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;

    // 1 and 2: the camera path with moving objects, verified every frame.
    let mut scene = Showcase::new();
    let mut lightmap =
        Lightmap::new(&scene.surfaces, TEXEL, |m| m < scene.first_mover).with_probes(PROBES);
    let bake = kernel.bake(&mut lightmap, scene.light, &scene.meshes());
    let start = Instant::now();
    let mut renderer = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        scene.light,
        &scene.meshes(),
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
    let step = |renderer: &mut LiveRenderer, scene: &mut Showcase, f: u32| {
        scene.set_frame(f as f32);
        let start = Instant::now();
        renderer.update_surfaces(&movers(scene));
        renderer.wait();
        let updated = Instant::now();
        renderer.frame(
            scene.camera(((f - 1) % PATH_FRAMES + 1) as f32),
            scene.vertical_fov_degrees,
            &view,
        );
        renderer.wait();
        (updated - start, updated.elapsed())
    };
    let (mut updates, mut renders) = (Vec::new(), Vec::new());
    let (mut texel_diff, mut texel_failures) = (0.0f32, 0usize);
    let (mut probe_diff, mut probe_failures) = (0.0f32, 0usize);
    let mut rebaked = 0u64;
    for f in 1..=PATH_FRAMES {
        let (u, r) = step(&mut renderer, &mut scene, f);
        updates.push(u);
        renders.push(r);
        rebaked += renderer.rebaked_blocks() as u64;
        let texels = renderer.read_texels();
        let (full, _) = kernel.irradiance_with(
            &lightmap.receivers,
            scene.light,
            &scene.meshes(),
            Hierarchy::Triangles,
            1,
        );
        let peak = full.iter().copied().fold(0.0, f32::max);
        for (a, b) in texels.iter().zip(&full) {
            let d = (a - b).abs() / peak;
            texel_diff = texel_diff.max(d);
            texel_failures += (d > TOLERANCE) as usize;
        }
        let probes = renderer.read_probes();
        let scratch = renderer.probes_from_scratch(&kernel, PROBE_RESOLUTION);
        let peak = scratch
            .iter()
            .flat_map(|p| &p[..3])
            .copied()
            .fold(0.0, f32::max);
        for (a, b) in probes.iter().zip(&scratch) {
            let d = (0..4).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max) / peak;
            probe_diff = probe_diff.max(d);
            probe_failures += (d > TOLERANCE) as usize;
        }
    }
    // Sustained: a minute of continuous frames, then the path again.
    let start = Instant::now();
    let mut f = PATH_FRAMES;
    while start.elapsed() < Duration::from_secs(60) {
        f += 1;
        step(&mut renderer, &mut scene, f);
    }
    let sustained: Vec<Duration> = (1..=PATH_FRAMES)
        .map(|k| {
            let (u, r) = step(&mut renderer, &mut scene, f + k);
            u + r
        })
        .collect();

    // 3: the static showcase against Cycles.
    let still = Showcase::new().without_movers();
    let camera = still.camera(0.0);
    let mut still_map = Lightmap::new(&still.surfaces, TEXEL, |_| true).with_probes(PROBES);
    kernel.bake(&mut still_map, still.light, &still.meshes());
    let still_renderer = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        QW,
        QH,
        still.light,
        &still.meshes(),
        &still.albedo,
        still.radiance,
        &still_map,
    );
    still_renderer.wait();
    let render = |map: &Lightmap| {
        kernel
            .render_cached(
                camera,
                QW,
                QH,
                still.light,
                &still.meshes(),
                &still.albedo,
                still.radiance,
                map,
                1,
            )
            .0
    };
    let ours_direct = render(&still_map);
    still_map.indirect = still_renderer.read_probes();
    let ours = render(&still_map);
    let work = out_dir.join("cycles");
    let reference = |samples, bounces, seed| {
        cycles::render(
            &still.surfaces,
            &still.albedo,
            still.light,
            still.radiance,
            camera,
            (QW, QH),
            samples,
            bounces,
            seed,
            &work,
        )
    };
    let cycles_full = reference(SAMPLES, 1, 0)?;
    let cycles_direct = reference(SAMPLES, 0, 0)?;
    let (noise_a, noise_b) = (reference(SAMPLES / 4, 1, 1)?, reference(SAMPLES / 4, 1, 2)?);
    let rmse = relative_rmse(&ours, &cycles_full.pixels);
    let rmse_direct = relative_rmse(&ours_direct, &cycles_direct.pixels);
    let indirect = |full: &[[f32; 3]], direct: &[[f32; 3]]| -> Vec<[f32; 3]> {
        full.iter()
            .zip(direct)
            .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
            .collect()
    };
    let (ours_ind, cycles_ind) = (
        indirect(&ours, &ours_direct),
        indirect(&cycles_full.pixels, &cycles_direct.pixels),
    );
    let full_mean = cycles_full
        .pixels
        .iter()
        .flatten()
        .map(|&x| x as f64)
        .sum::<f64>()
        / (3 * cycles_full.pixels.len()) as f64;
    let ind_mean =
        cycles_ind.iter().flatten().map(|&x| x as f64).sum::<f64>() / (3 * cycles_ind.len()) as f64;
    let rmse_ind = relative_rmse(&ours_ind, &cycles_ind);
    let noise = relative_rmse(&noise_a.pixels, &noise_b.pixels) / 2f64.sqrt() / 2.0;

    let gap = 8;
    let (w, h) = (QW as usize, QH as usize);
    let mut sheet = Image::new(3 * w + 4 * gap, h + 2 * gap, [28; 3]);
    for i in 0..w * h {
        let (x, y) = (i % w, i / w);
        let (a, b) = (
            Vec3::from_array(ours[i]),
            Vec3::from_array(cycles_full.pixels[i]),
        );
        sheet.set(gap + x, gap + y, to_rgb(a));
        sheet.set(2 * gap + w + x, gap + y, to_rgb(b));
        let d = ((a - b).abs().max_element() / full_mean as f32 * 255.0 * 4.0).min(255.0) as u8;
        sheet.set(3 * gap + 2 * w + x, gap + y, [d, d, d]);
    }
    sheet.save(&out_dir.join("m3_gate3_vs_cycles.png"))?;
    let mut image = Image::new(w, h, [0; 3]);
    for i in 0..w * h {
        image.set(i % w, i / w, to_rgb(Vec3::from_array(ours[i])));
    }
    image.save(&out_dir.join("m3_gate3_ours.png"))?;

    let mean = |v: &[Duration]| v.iter().map(|&d| ms(d)).sum::<f64>() / v.len() as f64;
    let frames: Vec<Duration> = updates.iter().zip(&renders).map(|(a, b)| *a + *b).collect();
    let worst = frames.iter().map(|&d| ms(d)).fold(0.0, f64::max);
    let frame_ms = mean(&frames);
    let exact = texel_failures == 0 && probe_failures == 0;
    let pass = exact && frame_ms <= 16.7 && rmse <= 0.05;
    let mut out = String::new();
    writeln!(out, "# M3 gate 3 results\n").unwrap();
    writeln!(
        out,
        "Showcase with 2 moving objects on {}: {} meshes, {} texels, {} probes ({} on moving objects). \
         Bake {:.0} ms; caches (static coverage, G-buffers, first gather) {:.0} ms.\n",
        kernel.adapter_name,
        scene.surfaces.len(),
        lightmap.texels(),
        renderer.probes(),
        renderer.probes() as usize - lightmap.static_probes,
        ms(bake),
        ms(setup)
    )
    .unwrap();
    writeln!(
        out,
        "1. Exactness over {PATH_FRAMES} frames: texels max difference {texel_diff:.1e} of peak ({texel_failures} over {TOLERANCE:e}); \
         probes max difference {probe_diff:.1e} ({probe_failures} over).\n\
         2. Frame at {W}×{H} (update + render, each waited on): {frame_ms:.2} ms mean ({:.2} update, {:.2} render), {worst:.2} ms worst; \
         blocks re-baked per frame {:.1}%. Sustained after 60 s: {:.2} ms mean.\n\
         3. Static frame at {QW}×{QH} against Cycles ({} spp, 1 diffuse bounce, {} in {:.0} s): relative RMSE {:.2}%. \
         Direct light only (Cycles 0 bounces): {:.2}%. Indirect only: {:.2}% of the full image's mean ({:.1}% of the indirect mean). \
         Cycles noise at {} spp: {:.2}%.",
        mean(&updates),
        mean(&renders),
        100.0 * rebaked as f64 / (PATH_FRAMES as f64 * renderer.blocks() as f64),
        mean(&sustained),
        SAMPLES,
        cycles_full.device,
        cycles_full.seconds,
        100.0 * rmse,
        100.0 * rmse_direct,
        100.0 * rmse_ind * ind_mean / full_mean,
        100.0 * rmse_ind,
        SAMPLES,
        100.0 * noise,
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nExact, at most 16.7 ms per frame, and within 5% of Cycles → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m3_gate3_results.md"), &out)?;
    Ok(())
}
