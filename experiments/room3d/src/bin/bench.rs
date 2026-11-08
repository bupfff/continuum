//! Benchmark (not a gate): the gates' timing measurements in one headless
//! run, to compare GPUs (the NVIDIA benchmark, `docs/BENCHMARK.md`). Every
//! number here was first measured on the M4 Air by a gate; this binary
//! repeats the timing parts only (no Cycles references).
//!
//! Suite, 1280×720, release build:
//! 1. The million-boulder field, M5 gate 8's flight (600 frames from an
//!    empty cache, three times each): the sun; the sun and the field's sky;
//!    M5 gate 6's three colored lights. Mean, 99th percentile and worst
//!    frame (wall time per call, the GPU pipelined as in an application),
//!    fallback, GPU bake and camera time.
//! 2. Warm frames at M5 gate 5's four viewpoints with the sun and sky
//!    (median of 60 frames each waited for, after 16 quiet frames).
//! 3. The sky's warm-up at the standing viewpoint, fixed budget 11,000
//!    (M5 gate 9): frames until warm with the sky later, and without.
//! 4. M6's textured scene with M10 gate 10's materials and bounce light:
//!    warm frame (median of 30 frames each waited for).
//! 5. Silhouette rays (M7 gate 4): the field from 150 m up, sun only,
//!    warm frame with one ray, four quarters and two quarters.
//!
//! The report goes to stdout and `out/bench_<adapter>.txt`. `--quick` runs
//! one flight of each kind and skips part 3.
//!
//! `cargo run --release -p room3d --bin bench`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, CacheStats, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu,
};
use room3d::field::Field;
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;
const FLIGHT: usize = 600;

/// Renders until `quiet` consecutive frames requested and baked nothing;
/// the frames it took (None past 60,000).
fn settle(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    quiet: usize,
) -> Option<usize> {
    let start = r.cache_log().len();
    for k in 1..=60_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + quiet
            && r.cache_log()[latest - quiet..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return Some(k);
        }
    }
    None
}

/// The median of `n` frames, each waited for, in milliseconds.
fn warm_ms(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView, n: usize) -> f64 {
    let mut t: Vec<f64> = (0..n)
        .map(|_| {
            let s = Instant::now();
            r.frame(camera, target);
            r.wait();
            s.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[n / 2]
}

struct Flight {
    mean: f64,
    p99: f64,
    worst: f64,
    fallback: f64,
    bake: f64,
    camera: f64,
}

fn summarize(times: &[Duration], total: Duration, log: &[CacheStats]) -> Flight {
    let mut sorted: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    sorted.sort_by(f64::total_cmp);
    let n = log.len().max(1) as f64;
    Flight {
        mean: total.as_secs_f64() * 1000.0 / times.len() as f64,
        p99: sorted[(sorted.len() * 99) / 100 - 1],
        worst: *sorted.last().unwrap(),
        fallback: log.iter().map(|s| s.fallback_pixels as f64).sum::<f64>() / n / (W * H) as f64,
        bake: log.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n,
        camera: log.iter().map(|s| s.gpu_render_ms as f64).sum::<f64>() / n,
    }
}

fn main() -> std::io::Result<()> {
    let quick = std::env::args().any(|a| a == "--quick");
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let features = kernel.device().features();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    say(format!(
        "adapter: {}; OS: {}; ray queries: {}; timestamps: {}; subgroups: {}",
        kernel.adapter_name,
        std::env::consts::OS,
        features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
        features.contains(wgpu::Features::TIMESTAMP_QUERY),
        features.contains(wgpu::Features::SUBGROUP)
    ));
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
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
    let start = Instant::now();
    let field = Field::new(1_000_000, DVec3::ZERO);
    say(format!(
        "field built in {:.1} s (CPU)",
        start.elapsed().as_secs_f64()
    ));
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let sun = Light::white(sun, field.radiance);
    let three = field.colored_lights(origin, 1000.0).to_vec();
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
    let flight_camera = |f: usize| {
        let t = f as f32 / (FLIGHT - 1) as f32;
        view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t)
    };
    let field_renderer = |lights: &[Light], options: CacheOptions| {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            lights,
            &field.scene,
            origin,
            options,
        )
    };

    // 1. Flights.
    let flights = if quick { 1 } else { 3 };
    for (name, lights, sky_on) in [
        ("sun", vec![sun], false),
        ("sun and sky", vec![sun], true),
        ("three colored lights", three.clone(), false),
    ] {
        kernel.set_sky(if sky_on { sky } else { Vec3::ZERO });
        for k in 0..flights {
            let mut r = field_renderer(&lights, CacheOptions::default());
            let mut times = Vec::with_capacity(FLIGHT);
            let begin = Instant::now();
            for f in 0..FLIGHT {
                let call = Instant::now();
                r.frame(flight_camera(f), &target);
                times.push(call.elapsed());
            }
            r.finish();
            let s = summarize(&times, begin.elapsed(), r.cache_log());
            say(format!(
                "1. flight, {name}, run {}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, GPU bake {:.2} ms, camera {:.2} ms",
                k + 1,
                s.mean,
                s.p99,
                s.worst,
                100.0 * s.fallback,
                s.bake,
                s.camera
            ));
        }
    }

    // 2. Warm frames at the four viewpoints, sun and sky.
    kernel.set_sky(sky);
    for (name, camera) in [
        ("standing", view(0.0, 1.7, 0.0)),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ] {
        let mut r = field_renderer(&[sun], CacheOptions::default());
        let frames = settle(&mut r, camera, &target, 16);
        let ms = warm_ms(&mut r, camera, &target, 60);
        say(format!(
            "2. warm frame, sun and sky, {name}: {ms:.2} ms (warm after {} frames)",
            frames.map_or("never".into(), |f| f.to_string())
        ));
    }

    // 3. The sky's warm-up at a fixed budget.
    if !quick {
        for sky_later in [true, false] {
            let mut r = field_renderer(
                &[sun],
                CacheOptions {
                    bake_budget: 11_000,
                    frame_target_ms: 0.0,
                    sky_later,
                    ..Default::default()
                },
            );
            let begin = Instant::now();
            let frames = settle(&mut r, view(0.0, 1.7, 0.0), &target, 16);
            r.finish();
            say(format!(
                "3. sky warm-up, standing, budget 11,000, sky later {sky_later}: {} frames, {:.1} s",
                frames.map_or("never".into(), |f| f.to_string()),
                begin.elapsed().as_secs_f64()
            ));
        }
    }

    // 4. The textured scene with materials and bounce light.
    let t = Textured::new().with_materials();
    kernel.set_sky(t.sky);
    {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions {
                indirect: true,
                ..Default::default()
            },
        );
        let frames = settle(&mut r, t.camera, &target, 32);
        let ms = warm_ms(&mut r, t.camera, &target, 30);
        say(format!(
            "4. textured scene, materials and bounce light: warm frame {ms:.2} ms (settled in {} frames)",
            frames.map_or("never".into(), |f| f.to_string())
        ));
    }

    // 5. Silhouette rays from 150 m up, sun only.
    kernel.set_sky(Vec3::ZERO);
    let high = view(0.0, 150.0, 45.0);
    let mut one = 0.0;
    for (name, edge_rays, edge_samples) in [
        ("one ray", false, 4),
        ("four quarters", true, 4),
        ("two quarters", true, 2),
    ] {
        let mut r = field_renderer(
            &[sun],
            CacheOptions {
                edge_rays,
                edge_samples,
                ..Default::default()
            },
        );
        settle(&mut r, high, &target, 32);
        let ms = warm_ms(&mut r, high, &target, 30);
        if !edge_rays {
            one = ms;
        }
        r.finish();
        let edges = r.cache_stats().map_or(0, |s| s.edge_pixels);
        say(format!(
            "5. silhouette rays, field from 150 m, {name}: warm frame {ms:.2} ms ({:.2}× one ray; edge pixels {:.1}%)",
            ms / one,
            100.0 * edges as f64 / (W * H) as f64
        ));
    }

    let tag: String = kernel
        .adapter_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    std::fs::write(out_dir.join(format!("bench_{tag}.txt")), report)?;
    Ok(())
}
