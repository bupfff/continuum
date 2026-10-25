//! M5 gate 4: sky light through the view-dependent cache, at scale.
//!
//! M5 gate 3 matched Cycles on the million-boulder field with sky light, but
//! per-pixel sky cones cost about 100–200 µs per pixel (six cones, most of it
//! in finding coverage). This gate bakes the sky into the view-dependent
//! lighting cache instead, and makes sky cones cheaper:
//! - Cache pages hold two channels per texel: irradiance per unit sun radiance
//!   and per unit sky radiance; shading applies the radiances. Pixels with no
//!   page use unshadowed sun and an unoccluded upper-hemisphere sky.
//! - Sky cones prefilter 8 times as coarsely as the light's cones
//!   (`set_sky_prefilter(8.0)`): their bins are 11° wide. Measured before this
//!   gate on the field at 320×180: 7.7× cheaper, 0.97% from factor 1.
//! - Each face's sky light is its whole horizon-clipped window weight minus
//!   the weights of covered bins (identical to 8e-8).
//!
//! Scene: the million-boulder field (sun 3 km for criteria 2–4, as gates
//! 8–11; 1 km for criterion 1, as gate 3), sky from `Field::sky`, prefilter 1
//! bin with object-level prefiltering, cache defaults (frame target 14 ms),
//! hardware camera rays, gate 8's four viewpoints at 1280×720.
//!
//! Pass criterion, fixed before the first run:
//! 1. Accuracy at the new sky prefilter: per-pixel sky only, 4×4 per pixel,
//!    within 3% of Cycles (instanced, 1024 spp), as gate 3's criterion 1.
//! 2. The sun channel unchanged: warm cached images with no sky within 0.2
//!    points of gate 8's relative RMSE against per-pixel exact cones.
//! 3. Sky in the cache: at each viewpoint, the warm cached sky-only image is
//!    within 5% relative RMSE of the per-pixel sky-only render (1 sample per
//!    pixel, same prefiltering).
//! 4. Speed with sun and sky: warm, each viewpoint at most 16.7 ms per frame;
//!    gate 8's flight from an empty cache, three runs, each with mean frame at
//!    most 16.7 ms, 99th percentile at most 33.3 ms, fallback at most 10%.
//!
//! Reported: frames until warm (sun and sky), bake budget, samples at their
//! wanted mip, and the cost of a sky texel.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, RectLight, wgpu};
use room3d::cycles::{relative_rmse, render_instanced};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const SKY_PREFILTER: f32 = 8.0;

/// A cached renderer warmed at `camera`: frames until 16 quiet frames (read
/// back), then the wall time per frame over 60 frames, and the image.
struct Warm {
    frames: Option<usize>,
    frame_time: Duration,
    image: Vec<[f32; 3]>,
}

fn warm(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Warm {
    let mut frames = 0usize;
    let mut warm_from = None;
    while frames < 20_000 {
        r.frame(camera, target);
        frames += 1;
        let latest = r.cache_stats().unwrap().frame as usize;
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            warm_from = Some(latest - 16);
            break;
        }
    }
    r.finish();
    let start = Instant::now();
    for _ in 0..60 {
        r.frame(camera, target);
    }
    r.finish();
    let frame_time = start.elapsed() / 60;
    Warm {
        frames: warm_from,
        frame_time,
        image: r.read_colors(),
    }
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(SKY_PREFILTER);

    // 1. Accuracy at the new sky prefilter, against Cycles (gate 3's setup).
    let near = Field::new(1_000_000, DVec3::ZERO).with_sun_at(1000.0);
    let n_origin = near.origin_near_camera();
    let (n_camera, n_light) = near.relative(n_origin);
    let sky = near.sky();
    kernel.set_hardware_rays(false);
    kernel.set_sky(sky);
    let (big, big_t, _) = kernel.render_instanced(
        n_camera,
        640 * 4,
        360 * 4,
        n_light,
        &near.scene,
        n_origin,
        0.0,
        1,
    );
    let mut sky_ss = vec![[0.0f32; 3]; 640 * 360];
    for y in 0..360 * 4 {
        for x in 0..640 * 4 {
            let p = big[y * 640 * 4 + x];
            let o = &mut sky_ss[(y / 4) * 640 + x / 4];
            for c in 0..3 {
                o[c] += p[c] / 16.0;
            }
        }
    }
    drop(big);
    let reference = render_instanced(
        &near.scene,
        n_origin,
        n_light,
        0.0,
        sky,
        n_camera,
        (640, 360),
        1024,
        0,
        1,
        &out_dir.join("m5_gate4_cycles"),
    )?;
    let accuracy = relative_rmse(&sky_ss, &reference.pixels);
    println!(
        "1. sky only vs Cycles at sky prefilter ×{SKY_PREFILTER}: {:.2}% ({:.0} s for 4×4)",
        100.0 * accuracy,
        big_t.as_secs_f64()
    );
    drop(near);

    // 2–4 on gate 8's field and viewpoints (sun 3 km), hardware rays.
    kernel.set_hardware_rays(true);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let sky = field.sky();
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |ahead: f32, height: f32, down: f32| {
        let d = down.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(0.0, 1.7, 0.0), 0.1449),
        ("10 m up", view(0.0, 10.0, 15.0), 0.2587),
        ("40 m up", view(0.0, 40.0, 30.0), 0.3088),
        ("150 m up", view(0.0, 150.0, 45.0), 0.3880),
    ];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
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
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let cached = |radiance: f32, sky: Vec3, light: RectLight| {
        kernel.set_sky(sky);
        InstancedRenderer::with_cache(
            &kernel,
            format,
            W,
            H,
            1,
            light,
            &field.scene,
            origin,
            radiance,
            CacheOptions::default(),
        )
    };
    let per_pixel = |camera: Camera, radiance: f32, sky: Vec3, prefilter: f32| {
        kernel.set_sky(sky);
        kernel.set_prefilter(prefilter);
        let pixels = kernel
            .render_instanced(camera, W, H, light, &field.scene, origin, radiance, 1)
            .0;
        kernel.set_prefilter(1.0);
        pixels
    };

    let mut rows = Vec::new();
    for (name, camera, gate8) in views {
        // 2. Sun only.
        let mut r = cached(field.radiance, Vec3::ZERO, light);
        let sun = warm(&mut r, camera, &target);
        drop(r);
        let exact = per_pixel(camera, field.radiance, Vec3::ZERO, 0.0);
        let sun_rmse = relative_rmse(&sun.image, &exact);
        // 3. Sky only.
        let mut r = cached(0.0, sky, light);
        let sky_only = warm(&mut r, camera, &target);
        drop(r);
        let sky_pp = per_pixel(camera, 0.0, sky, 1.0);
        let sky_rmse = relative_rmse(&sky_only.image, &sky_pp);
        // 4. Sun and sky, warm.
        let mut r = cached(field.radiance, sky, light);
        let both = warm(&mut r, camera, &target);
        drop(r);
        println!(
            "{name}: sun {:.2}% (gate 8 {:.2}%), sky cached vs per-pixel {:.2}%, warm sun+sky {:.2} ms after {:?} frames (sky only after {:?})",
            100.0 * sun_rmse,
            100.0 * gate8,
            100.0 * sky_rmse,
            ms(both.frame_time),
            both.frames,
            sky_only.frames
        );
        rows.push((
            name,
            sun_rmse,
            gate8,
            sky_rmse,
            both,
            sky_only.frames,
            sun.frames,
        ));
    }

    // 4. Flights with sun and sky.
    let mut flights = Vec::new();
    for _ in 0..3 {
        let mut r = cached(field.radiance, sky, light);
        let mut times = Vec::new();
        let start = Instant::now();
        for f in 0..600 {
            let t = f as f32 / 599.0;
            let camera = view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t);
            let call = Instant::now();
            r.frame(camera, &target);
            times.push(call.elapsed());
        }
        r.finish();
        let total = start.elapsed();
        let log = r.cache_log().to_vec();
        drop(r);
        let mut sorted = times.clone();
        sorted.sort();
        let fallback = log
            .iter()
            .map(|s| s.fallback_pixels as f64 / (W * H) as f64)
            .sum::<f64>()
            / log.len() as f64;
        let samples: u64 = log.iter().map(|s| s.samples as u64).sum();
        let at_wanted: u64 = log.iter().map(|s| s.at_wanted as u64).sum();
        let budget = log.iter().map(|s| s.budget as f64).sum::<f64>() / log.len() as f64;
        let texels: u64 = log.iter().map(|s| s.texels_baked as u64).sum();
        let flight = (
            total / 600,
            sorted[(sorted.len() * 99) / 100 - 1],
            *sorted.last().unwrap(),
            fallback,
            at_wanted as f64 / samples.max(1) as f64,
            budget,
            texels,
        );
        println!(
            "flight: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, at wanted {:.1}%, budget {:.0}, texels {}",
            ms(flight.0),
            ms(flight.1),
            ms(flight.2),
            100.0 * flight.3,
            100.0 * flight.4,
            flight.5,
            flight.6
        );
        flights.push(flight);
    }

    let c1 = accuracy <= 0.03;
    let c2 = rows.iter().all(|r| (r.1 - r.2).abs() <= 0.002);
    let c3 = rows.iter().all(|r| r.3 <= 0.05);
    let c4 = rows
        .iter()
        .all(|r| r.4.frames.is_some() && ms(r.4.frame_time) <= 16.7)
        && flights
            .iter()
            .all(|f| ms(f.0) <= 16.7 && ms(f.1) <= 33.3 && f.3 <= 0.10);
    let pass = c1 && c2 && c3 && c4;

    let mut out = String::new();
    writeln!(out, "# M5 gate 4 results\n").unwrap();
    writeln!(
        out,
        "{}, 999,986 boulders, sky prefilter ×{SKY_PREFILTER}, prefilter 1 bin with object-level prefiltering, cache defaults (frame target 14 ms), hardware rays at 1280×720 for criteria 2–4.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "1. Sky only, 4×4 per pixel (640×360, sun 1 km), vs Cycles (1024 spp, {:.1} s): **{:.2}%** (ours {:.0} s).\n",
        reference.seconds,
        100.0 * accuracy,
        big_t.as_secs_f64()
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | 2. Sun only vs per-pixel exact (gate 8) | 3. Sky only, cached vs per-pixel | 4. Warm sun and sky | Frames until warm: sun / sky / both |\n|---|---|---|---|---|"
    )
    .unwrap();
    for (name, sun, gate8, sky_rmse, both, sky_frames, sun_frames) in &rows {
        let f = |x: &Option<usize>| x.map_or("not warm".to_string(), |v| v.to_string());
        writeln!(
            out,
            "| {name} | {:.2}% ({:.2}%) | {:.2}% | {:.2} ms | {} / {} / {} |",
            100.0 * sun,
            100.0 * gate8,
            100.0 * sky_rmse,
            ms(both.frame_time),
            f(sun_frames),
            f(sky_frames),
            f(&both.frames)
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nFlights with sun and sky (600 frames from an empty cache):\n\n| Run | Mean | 99th percentile | Worst | Fallback | Samples at wanted mip | Bake budget mean | Texels baked |\n|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for (i, f) in flights.iter().enumerate() {
        writeln!(
            out,
            "| {} | {:.2} ms | {:.2} ms | {:.2} ms | {:.2}% | {:.1}% | {:.0} | {} |",
            i + 1,
            ms(f.0),
            ms(f.1),
            ms(f.2),
            100.0 * f.3,
            100.0 * f.4,
            f.5,
            f.6
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n## Verdict\n\n1. Sky accuracy within 3% of Cycles: {}\n2. Sun channel unchanged (within 0.2 points of gate 8): {}\n3. Cached sky within 5% of per-pixel: {}\n4. Speed with sun and sky (warm and three flights): {}\n\n→ {}",
        if c1 { "pass" } else { "fail" },
        if c2 { "pass" } else { "fail" },
        if c3 { "pass" } else { "fail" },
        if c4 { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m5_gate4_results.md"), &out)?;
    Ok(())
}
