//! M5 gate 5: sky light through the view cache at speed.
//!
//! Gate 4 found sky texels about 100× costlier than sun texels and its timings
//! confounded by the machine. Since then (development measurements in
//! `docs/results/m5-sky-bake-diagnostics.md`):
//! - Sky samples are six bake jobs, one per cube face, face by face; sky cones
//!   skip geometry below the receiver's horizon (exact for the integral).
//! - Own-object sky masks: each geometry's coverage of its own sky, through
//!   cube faces in its frame, is baked once for all its instances
//!   (`CacheOptions::self_spacing`, 1 cm); sky texels walk only the other
//!   objects and combine per bin. It removed 85% of a sky texel's
//!   triangle-hierarchy work.
//! - The bake budget follows GPU time: bake and camera passes are timed with
//!   timestamp queries, bake time is fit as latency plus cost per unit, and
//!   the budget fits the frame target; a sky sample counts 10 sun texels.
//! - The cache's bakes walk alone; per-pixel cones walk in subgroup packets.
//!
//! Scene: the million-boulder field, sun 3 km (as gates 8–11), sky from
//! `Field::sky`, prefilter 1 bin with object-level prefiltering, sky
//! prefilter ×8, cache defaults (frame target 14 ms), hardware camera rays,
//! gate 8's four viewpoints at 1280×720.
//!
//! Control, before anything else: gate 11's sun-only warm frame at the
//! standing viewpoint must take at most 8.03 ms (gate 11's own number), or
//! the machine is not fit to measure and the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. Masks are faithful: at each viewpoint, the warm cached sky-only image
//!    with own-object masks is within 0.5% relative RMSE of the warm cache
//!    without them, its texels walking their own objects through faces in
//!    the object's frame (`self_spacing` 0, `set_sky_object_frame`).
//! 2. The sun channel is unchanged: warm cached sun-only images within 0.2
//!    points of gate 8's relative RMSE against per-pixel exact cones.
//! 3. Speed with sun and sky: warm, each viewpoint at most 16.7 ms per frame;
//!    gate 8's flight from an empty cache, three runs, each with mean frame
//!    at most 16.7 ms, 99th percentile at most 33.3 ms, fallback at most 10%.
//! 4. Warm-up with sun and sky: at each viewpoint, frames until warm at most
//!    10 times the sun-only cache's (measured in the same run).
//!
//! Reported, not gated: per-pixel sky only against Cycles (gate 4's criterion
//! 1: 640×360, 4×4 per pixel, sun 1 km, instanced Cycles at 1,024 spp) with
//! faces in world axes and in object frames. Development runs before this
//! gate found the sky's binning error depends on the cube's orientation
//! (1.9–4.2% from Cycles at the standing viewpoint), so gate 4's 3% bound
//! measured a lucky orientation; it is reported here instead. Also reported:
//! the masks' precompute time and memory, GPU bake and camera times, budget,
//! sky samples, and the cached sky against per-pixel sky.

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
    while frames < 40_000 {
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

/// 4×4 supersampled per-pixel render, box filtered to `w × h`.
fn supersampled(
    kernel: &MaskConeKernel,
    camera: Camera,
    (w, h): (u32, u32),
    light: RectLight,
    field: &Field,
    origin: DVec3,
) -> Vec<[f32; 3]> {
    let (big, _, _) =
        kernel.render_instanced(camera, w * 4, h * 4, light, &field.scene, origin, 0.0, 1);
    let mut px = vec![[0.0f32; 3]; (w * h) as usize];
    for y in 0..(h * 4) as usize {
        for x in 0..(w * 4) as usize {
            let p = big[y * (w * 4) as usize + x];
            let o = &mut px[(y / 4) * w as usize + x / 4];
            for c in 0..3 {
                o[c] += p[c] / 16.0;
            }
        }
    }
    px
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
    kernel.set_hardware_rays(true);

    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let sky = field.sky();
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
    let cached = |radiance: f32, sky: Vec3, options: CacheOptions| {
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
            options,
        )
    };

    // Control: gate 11's sun-only warm frame at the standing viewpoint.
    let control = {
        let mut r = cached(field.radiance, Vec3::ZERO, CacheOptions::default());
        warm(&mut r, ground, &target)
    };
    let fit = ms(control.frame_time) <= 8.03;
    println!(
        "control: sun-only warm frame at the standing viewpoint {:.2} ms (gate 11: 8.03 ms) after {:?} frames: {}",
        ms(control.frame_time),
        control.frames,
        if fit {
            "fit"
        } else {
            "NOT FIT: the run is void"
        }
    );
    if !fit {
        return Ok(());
    }

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

    // The masks' precompute, timed on its own.
    let precompute = {
        let start = Instant::now();
        let r = cached(0.0, sky, CacheOptions::default());
        let t = start.elapsed();
        let bytes = r.cache_bytes();
        drop(r);
        let start = Instant::now();
        let r = cached(
            0.0,
            sky,
            CacheOptions {
                self_spacing: 0.0,
                ..Default::default()
            },
        );
        let base = start.elapsed();
        let without = r.cache_bytes();
        drop(r);
        (t.saturating_sub(base), bytes - without)
    };
    println!(
        "own-object masks: precompute {:.0} ms, {:.1} MB",
        ms(precompute.0),
        precompute.1 as f64 / 1e6
    );

    struct Row {
        name: &'static str,
        sun_rmse: f64,
        gate8: f64,
        masks_vs_walk: f64,
        cached_vs_per_pixel: f64,
        both: Warm,
        frames: [Option<usize>; 4],
    }
    let mut rows = Vec::new();
    for (name, camera, gate8) in views {
        // 2. Sun only.
        let mut r = cached(field.radiance, Vec3::ZERO, CacheOptions::default());
        let sun = warm(&mut r, camera, &target);
        drop(r);
        kernel.set_sky(Vec3::ZERO);
        kernel.set_prefilter(0.0);
        let exact = kernel
            .render_instanced(camera, W, H, light, &field.scene, origin, field.radiance, 1)
            .0;
        kernel.set_prefilter(1.0);
        let sun_rmse = relative_rmse(&sun.image, &exact) as f64;
        // 1. Sky only, with masks and without (faces in object frames).
        let mut r = cached(0.0, sky, CacheOptions::default());
        let masked = warm(&mut r, camera, &target);
        drop(r);
        kernel.set_sky_object_frame(true);
        let mut r = cached(
            0.0,
            sky,
            CacheOptions {
                self_spacing: 0.0,
                ..Default::default()
            },
        );
        let walked = warm(&mut r, camera, &target);
        drop(r);
        // The per-pixel sky with faces in object frames, for reference.
        kernel.set_sky(sky);
        let per_pixel = kernel
            .render_instanced(camera, W, H, light, &field.scene, origin, 0.0, 1)
            .0;
        kernel.set_sky_object_frame(false);
        let masks_vs_walk = relative_rmse(&masked.image, &walked.image) as f64;
        let cached_vs_per_pixel = relative_rmse(&masked.image, &per_pixel) as f64;
        // 3–4. Sun and sky, warm.
        let mut r = cached(field.radiance, sky, CacheOptions::default());
        let both = warm(&mut r, camera, &target);
        drop(r);
        println!(
            "{name}: sun {:.2}% (gate 8 {:.2}%); sky masks vs walk {:.3}%; cached vs per-pixel sky {:.2}%; warm sun+sky {:.2} ms; frames until warm sun / sky (masks) / sky (walk) / both: {:?} / {:?} / {:?} / {:?}",
            100.0 * sun_rmse,
            100.0 * gate8,
            100.0 * masks_vs_walk,
            100.0 * cached_vs_per_pixel,
            ms(both.frame_time),
            sun.frames,
            masked.frames,
            walked.frames,
            both.frames
        );
        rows.push(Row {
            name,
            sun_rmse,
            gate8,
            masks_vs_walk,
            cached_vs_per_pixel,
            frames: [sun.frames, masked.frames, walked.frames, both.frames],
            both,
        });
    }

    // 3. Flights with sun and sky.
    struct Flight {
        mean: Duration,
        p99: Duration,
        worst: Duration,
        fallback: f64,
        at_wanted: f64,
        budget: f64,
        sky_samples: u64,
        gpu_bake: f64,
        gpu_camera: f64,
    }
    let mut flights = Vec::new();
    for _ in 0..3 {
        let mut r = cached(field.radiance, sky, CacheOptions::default());
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
        let n = log.len() as f64;
        let samples: u64 = log.iter().map(|s| s.samples as u64).sum();
        let at_wanted: u64 = log.iter().map(|s| s.at_wanted as u64).sum();
        let flight = Flight {
            mean: total / 600,
            p99: sorted[(sorted.len() * 99) / 100 - 1],
            worst: *sorted.last().unwrap(),
            fallback: log
                .iter()
                .map(|s| s.fallback_pixels as f64 / (W * H) as f64)
                .sum::<f64>()
                / n,
            at_wanted: at_wanted as f64 / samples.max(1) as f64,
            budget: log.iter().map(|s| s.budget as f64).sum::<f64>() / n,
            sky_samples: log.iter().map(|s| s.sky_samples as u64).sum(),
            gpu_bake: log.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n,
            gpu_camera: log.iter().map(|s| s.gpu_render_ms as f64).sum::<f64>() / n,
        };
        println!(
            "flight: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, at wanted {:.1}%, budget {:.0}, sky samples {}, GPU bake {:.2} ms, camera {:.2} ms",
            ms(flight.mean),
            ms(flight.p99),
            ms(flight.worst),
            100.0 * flight.fallback,
            100.0 * flight.at_wanted,
            flight.budget,
            flight.sky_samples,
            flight.gpu_bake,
            flight.gpu_camera
        );
        flights.push(flight);
    }

    // Reported: per-pixel sky only against Cycles, world axes and object
    // frames (gate 4's criterion 1 setup).
    let near = Field::new(1_000_000, DVec3::ZERO).with_sun_at(1000.0);
    let n_origin = near.origin_near_camera();
    let (n_camera, n_light) = near.relative(n_origin);
    let n_sky = near.sky();
    kernel.set_hardware_rays(false);
    kernel.set_sky(n_sky);
    let world = supersampled(&kernel, n_camera, (640, 360), n_light, &near, n_origin);
    kernel.set_sky_object_frame(true);
    let object = supersampled(&kernel, n_camera, (640, 360), n_light, &near, n_origin);
    kernel.set_sky_object_frame(false);
    let reference = render_instanced(
        &near.scene,
        n_origin,
        n_light,
        0.0,
        n_sky,
        n_camera,
        (640, 360),
        1024,
        0,
        1,
        &out_dir.join("m5_gate5_cycles"),
    )?;
    let vs_world = relative_rmse(&world, &reference.pixels) as f64;
    let vs_object = relative_rmse(&object, &reference.pixels) as f64;
    println!(
        "sky only vs Cycles: world axes {:.2}%, object frames {:.2}%",
        100.0 * vs_world,
        100.0 * vs_object
    );

    let c1 = rows.iter().all(|r| r.masks_vs_walk <= 0.005);
    let c2 = rows.iter().all(|r| (r.sun_rmse - r.gate8).abs() <= 0.002);
    let c3 = rows
        .iter()
        .all(|r| r.both.frames.is_some() && ms(r.both.frame_time) <= 16.7)
        && flights
            .iter()
            .all(|f| ms(f.mean) <= 16.7 && ms(f.p99) <= 33.3 && f.fallback <= 0.10);
    let c4 = rows.iter().all(|r| match (r.frames[0], r.frames[3]) {
        (Some(sun), Some(both)) => both <= 10 * sun.max(1),
        _ => false,
    });
    let pass = c1 && c2 && c3 && c4;

    let mut out = String::new();
    let f = |x: &Option<usize>| x.map_or("not warm".to_string(), |v| v.to_string());
    writeln!(out, "# M5 gate 5 results\n").unwrap();
    writeln!(
        out,
        "{}, 999,986 boulders, sky prefilter ×{SKY_PREFILTER}, prefilter 1 bin with object-level prefiltering, cache defaults (frame target 14 ms), hardware rays at {W}×{H}.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "Control: sun-only warm frame at the standing viewpoint {:.2} ms (gate 11: 8.03 ms).\n",
        ms(control.frame_time)
    )
    .unwrap();
    writeln!(
        out,
        "Own-object masks: precompute {:.0} ms, {:.1} MB.\n",
        ms(precompute.0),
        precompute.1 as f64 / 1e6
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | 1. Sky: masks vs walk | 2. Sun vs per-pixel exact (gate 8) | Cached vs per-pixel sky | 3. Warm sun and sky | 4. Frames until warm: sun / sky (masks) / sky (walk) / both |\n|---|---|---|---|---|---|"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            out,
            "| {} | {:.3}% | {:.2}% ({:.2}%) | {:.2}% | {:.2} ms | {} / {} / {} / {} |",
            r.name,
            100.0 * r.masks_vs_walk,
            100.0 * r.sun_rmse,
            100.0 * r.gate8,
            100.0 * r.cached_vs_per_pixel,
            ms(r.both.frame_time),
            f(&r.frames[0]),
            f(&r.frames[1]),
            f(&r.frames[2]),
            f(&r.frames[3])
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nFlights with sun and sky (600 frames from an empty cache):\n\n| Run | Mean | 99th percentile | Worst | Fallback | Samples at wanted mip | Bake budget mean | Sky samples | GPU bake | GPU camera |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for (i, fl) in flights.iter().enumerate() {
        writeln!(
            out,
            "| {} | {:.2} ms | {:.2} ms | {:.2} ms | {:.2}% | {:.1}% | {:.0} | {} | {:.2} ms | {:.2} ms |",
            i + 1,
            ms(fl.mean),
            ms(fl.p99),
            ms(fl.worst),
            100.0 * fl.fallback,
            100.0 * fl.at_wanted,
            fl.budget,
            fl.sky_samples,
            fl.gpu_bake,
            fl.gpu_camera
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nReported: sky only, 4×4 per pixel (640×360, sun 1 km) vs Cycles (1,024 spp, {:.1} s): world axes {:.2}%, object frames {:.2}%.",
        reference.seconds,
        100.0 * vs_world,
        100.0 * vs_object
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\n1. Masks faithful (within 0.5%): {}\n2. Sun channel unchanged (within 0.2 points of gate 8): {}\n3. Speed with sun and sky (warm and three flights): {}\n4. Warm-up with sun and sky at most 10× sun-only: {}\n\n→ {}",
        if c1 { "pass" } else { "fail" },
        if c2 { "pass" } else { "fail" },
        if c3 { "pass" } else { "fail" },
        if c4 { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m5_gate5_results.md"), &out)?;
    Ok(())
}
