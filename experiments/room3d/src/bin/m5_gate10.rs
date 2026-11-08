//! M5 gate 10: a guessed sky from coarser pages.
//!
//! With the sky later (M5 gate 9), a page shows the open sky over the
//! pixel's normal until its own sky is baked: exact shadows, but no sky
//! darkening between and under objects. In flights most pixels show it
//! (the M4 benchmark's sun-and-sky flight: 82.9% fallback, mostly guessed
//! skies; `docs/results/bench-m4.md`).
//!
//! Method (`CacheOptions::coarse_sky_guess`): a page whose sky is still a
//! guess shows the sky of the finest coarser resident page whose sky is
//! baked, if any (the open sky otherwise). Coarse pages are asked for first,
//! and their skies are queued first, so in motion a blurred but occluded sky
//! is often there. A page used this way counts as used, so eviction keeps
//! it. Pixels shown so still count as fallback: their sky is approximate.
//!
//! Scene: the million-boulder field, the sun and the field's sky, 1280×720,
//! hardware camera rays, cache defaults but a fixed bake budget of 11,000
//! units a frame (`frame_target_ms` 0), so frame counts do not depend on
//! the machine's heat. Configurations: the open-sky guess (the baseline)
//! and the coarser pages' sky.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Quality in flight:** M5 gate 8's flight (600 frames from an empty
//!    cache); at frames 100, 200, …, 600, relative RMSE against that frame's
//!    settled image (a fresh cache at that camera, settled: 32 quiet
//!    frames); the mean over the six at most 0.8× the baseline's.
//! 2. **Quality while warming** (M5 gate 9's standing view): relative RMSE
//!    against the settled image, every 20th frame until the baseline is
//!    warm, averaged: at most the baseline's (1.0×).
//! 3. **Cost:** the flight's mean frame (wall time per call, from flights
//!    without readbacks, one each, the coarse sky first) at most 1.05× the
//!    baseline's.
//!
//! Reported: fallback; frames until warm at the standing view.
//!
//! `cargo run --release -p room3d --bin m5_gate10`
//!
//! `M5_DEV=1`: a development version (640×360, 100,000 boulders, a
//! 200-frame flight, three quality frames), whose numbers are not the
//! gate's.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const FLIGHT: usize = 600;

/// Renders until 32 frames rendered since the call requested and baked
/// nothing; the frames it took.
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    let start = r.cache_log().len();
    for k in 1..=60_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return k;
        }
    }
    panic!("the view never settled");
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M5_DEV").is_some();
    let (w, h, flight, boulders) = if dev {
        (640, 360, 200, 100_000)
    } else {
        (1280, 720, FLIGHT, 1_000_000)
    };
    if dev {
        println!("DEVELOPMENT RUN ({w}×{h}, {boulders} boulders): not the gate's numbers");
    }
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let field = Field::new(boulders, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let sun = Light::white(sun, field.radiance);
    kernel.set_sky(field.sky());
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
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
        let t = (f * FLIGHT / flight) as f32 / (FLIGHT - 1) as f32;
        view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t)
    };
    let renderer = |coarse_sky_guess: bool| {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &[sun],
            &field.scene,
            origin,
            CacheOptions {
                bake_budget: 11_000,
                frame_target_ms: 0.0,
                coarse_sky_guess,
                ..Default::default()
            },
        )
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 3. Cost, first: one flight each, no readbacks.
    let mut means = [0.0f64; 2];
    let mut fallback = [0.0f64; 2];
    for (i, coarse) in [true, false].into_iter().enumerate() {
        let mut r = renderer(coarse);
        let begin = Instant::now();
        for f in 0..flight {
            r.frame(flight_camera(f), &target);
        }
        r.finish();
        means[i] = begin.elapsed().as_secs_f64() * 1000.0 / flight as f64;
        let log = r.cache_log();
        fallback[i] = log.iter().map(|s| s.fallback_pixels as f64).sum::<f64>()
            / log.len() as f64
            / (w * h) as f64;
    }
    let c3 = means[0] <= 1.05 * means[1];
    say(format!(
        "3. cost: flight mean {:.2} ms with coarser skies against {:.2} ms ({:.3}×, at most 1.05×): {}; reported: fallback {:.1}% and {:.1}%",
        means[0],
        means[1],
        means[0] / means[1],
        verdict(c3),
        100.0 * fallback[0],
        100.0 * fallback[1]
    ));

    // 1. Quality in flight.
    let samples: Vec<usize> = if dev {
        vec![50, 100, 200]
    } else {
        (1..=6).map(|k| 100 * k).collect()
    };
    let read_flight = |coarse: bool| {
        let mut r = renderer(coarse);
        let mut images = Vec::new();
        for f in 0..flight {
            r.frame(flight_camera(f), &target);
            if samples.contains(&(f + 1)) {
                r.finish();
                images.push(r.read_colors());
            }
        }
        images
    };
    let coarse_images = read_flight(true);
    let open_images = read_flight(false);
    let (mut e_coarse, mut e_open) = (0.0, 0.0);
    for (i, &frame) in samples.iter().enumerate() {
        let camera = flight_camera(frame - 1);
        let mut r = renderer(true);
        settle(&mut r, camera, &target);
        r.finish();
        let reference = r.read_colors();
        let (a, b) = (
            relative_rmse(&coarse_images[i], &reference),
            relative_rmse(&open_images[i], &reference),
        );
        e_coarse += a / samples.len() as f64;
        e_open += b / samples.len() as f64;
        say(format!(
            "frame {frame}: {:.2}% from settled with coarser skies, {:.2}% with the open sky",
            100.0 * a,
            100.0 * b
        ));
    }
    let c1 = e_coarse <= 0.8 * e_open;
    say(format!(
        "1. quality in flight: mean {:.2}% against {:.2}% ({:.2}×, at most 0.8×): {}",
        100.0 * e_coarse,
        100.0 * e_open,
        e_coarse / e_open,
        verdict(c1)
    ));

    // 2. Quality while warming at the standing view.
    let standing = view(0.0, 1.7, 0.0);
    let mut reference = renderer(true);
    settle(&mut reference, standing, &target);
    reference.finish();
    let reference = reference.read_colors();
    let warming = |coarse: bool| {
        let mut r = renderer(coarse);
        let mut errors = Vec::new();
        let start = r.cache_log().len();
        for k in 1..=60_000usize {
            r.frame(standing, &target);
            if k % 20 == 0 {
                r.finish();
                errors.push((k, relative_rmse(&r.read_colors(), &reference)));
            }
            let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
            if latest >= start + 32
                && r.cache_log()[latest - 32..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0)
            {
                return (k, errors);
            }
        }
        (60_000, errors)
    };
    let (open_frames, open_errors) = warming(false);
    let (coarse_frames, coarse_errors) = warming(true);
    let mean_until = |errors: &[(usize, f64)]| {
        let ks: Vec<usize> = (1..=open_frames / 20).map(|i| 20 * i).collect();
        let at = |k: usize| {
            errors
                .iter()
                .take_while(|e| e.0 <= k)
                .last()
                .map_or(0.0, |e| e.1)
        };
        ks.iter().map(|&k| at(k)).sum::<f64>() / ks.len().max(1) as f64
    };
    let (w_coarse, w_open) = (mean_until(&coarse_errors), mean_until(&open_errors));
    let c2 = w_coarse <= w_open;
    say(format!(
        "2. quality while warming: mean {:.2}% against {:.2}% ({:.2}×, at most 1.0×): {}; reported: warm in {coarse_frames} and {open_frames} frames",
        100.0 * w_coarse,
        100.0 * w_open,
        w_coarse / w_open,
        verdict(c2)
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c3),
        verdict(c1 && c2 && c3)
    ));
    if !dev {
        std::fs::write(out_dir.join("m5_gate10.txt"), report)?;
    }
    Ok(())
}
