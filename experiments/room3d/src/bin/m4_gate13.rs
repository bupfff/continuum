//! M4 gate 13: the error while flying, reported each frame (Landmark 2).
//!
//! M4 gate 12 measured the error while flying (20.8% from settled, relative
//! RMSE, on M9's flight at a fixed budget) against renderers settled at the
//! same cameras; nothing in the renderer reports it. The design (DESIGN
//! §3.3, "The error while flying, reported"): each frame `N` probe pixels,
//! on a low-discrepancy pattern that moves each frame, are shaded as shown
//! and again with each light's share and the sky's computed at the pixel's
//! own surface point instead of read from pages. Over the probes, `sqrt(3 · mean d) / mean s` estimates the
//! image's relative RMSE (`d` the squared difference summed over channels,
//! `s` the exact value summed over channels), and a one-sided 95% bound
//! adds 1.645 standard errors of `d`'s mean. `CacheStats` carries both.
//!
//! **Changed before the run, after development runs (`M4_DEV=1`,
//! disclosed):** the stub had each light's share from the bake's light cone
//! and the sky's from its 192 rays, one probe an invocation. That cost 6.6
//! ms of GPU time a frame (1.37× the flight): 512 serial cone walks and 192
//! serial rays each, with too few invocations to hide their latency. Now a
//! workgroup per probe traces hardware rays: 64 to an 8×8 grid on each
//! light (the visible share of its irradiance), and the sky's 192 bins,
//! three a lane. The estimates moved by under 0.1 points (27.58 → 27.56% at
//! the first check). The development run then gave 1.035× for criterion 1,
//! 10 of 10 for criterion 2 and 1.199× for criterion 3; its settled
//! references' own estimates were 10–23%: the settled cache's error
//! against lighting exact at each pixel, which the estimate includes. The
//! criteria are unchanged.
//!
//! Scene and flight are gate 12's: M9's flight (600 frames from standing
//! height to 150 m, pitching down 45°) over the million-boulder field with
//! sun and sky (sky through rays), 1280×720, 512 probes.
//!
//! Pass criteria, fixed before the run:
//! 1. **Cost** (run first): three pairs of the flight with default options,
//!    probes off then on, interleaved: the mean frame with probes at most
//!    1.05× without (over the three flights each).
//! 2. **Calibration:** the flight at gate 12's fixed budget (11,000 units a
//!    frame); every 60th frame, the reported bound at least the error
//!    measured against a renderer settled at that camera (gate 12's method)
//!    in at least 9 of the 10 checks.
//! 3. **Tightness:** the reported estimate's mean over the 10 checks within
//!    0.8–1.3× the measured mean.
//!
//! Reported: each check's estimate, bound and measured error; the estimate
//! at each settled reference (the cache's own floor against exact lighting);
//! the probe pass's GPU time.
//!
//! `cargo run --release -p room3d --bin m4_gate13`
//!
//! `M4_DEV=1`: a development check (100,000 boulders, a 200-frame flight);
//! not the gate's numbers.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const FLIGHT: usize = 600;
const FIXED: u32 = 11_000;
const EVERY: usize = 60;
const PROBES: u32 = 512;

/// Frames until 16 in a row request and bake nothing (at most 60,000).
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Option<usize> {
    for _ in 0..60_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return Some(latest - 16);
        }
    }
    None
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M4_DEV").is_some();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say("DEVELOPMENT RUN: 100,000 boulders, a 200-frame flight; not the gate's numbers".into());
    }
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let field = Field::new(if dev { 100_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
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
    let options = |probes: u32, fixed: bool| CacheOptions {
        error_probes: probes,
        frame_target_ms: if fixed { 0.0 } else { 14.0 },
        bake_budget: if fixed { FIXED } else { 20_000 },
        ..Default::default()
    };
    let renderer = |o: CacheOptions| {
        InstancedRenderer::with_cache(
            &kernel,
            format,
            W,
            H,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            o,
        )
    };
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let flight_camera = |f: usize| {
        let t = f as f32 / (FLIGHT - 1) as f32;
        let d = (45.0 * t).to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, 1.7 + 148.3 * t) + flat * (300.0 * t);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let frames = if dev { 200 } else { FLIGHT };

    // 1. Cost, first: flights with default options, probes off then on.
    let fly = |probes: u32| {
        let mut r = renderer(options(probes, false));
        let begin = Instant::now();
        for f in 0..frames {
            r.frame(flight_camera(f * FLIGHT / frames), &target);
        }
        r.finish();
        let mean = begin.elapsed() / frames as u32;
        let log = r.cache_log();
        let timed: Vec<f64> = log
            .iter()
            .filter(|s| s.timed && s.gpu_render_ms > 0.0)
            .map(|s| s.gpu_render_ms as f64)
            .collect();
        let camera_ms = timed.iter().sum::<f64>() / timed.len().max(1) as f64;
        (mean, camera_ms)
    };
    let pairs = std::env::var("M4_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if dev { 1 } else { 3 });
    let mut means = [0.0f64; 2];
    let mut camera = [0.0f64; 2];
    for _ in 0..pairs {
        for (k, probes) in [0u32, PROBES].into_iter().enumerate() {
            let (mean, camera_ms) = fly(probes);
            means[k] += ms(mean) / pairs as f64;
            camera[k] += camera_ms / pairs as f64;
            say(format!(
                "1. flight, {probes} probes: mean {:.2} ms, camera passes {camera_ms:.2} ms on the GPU",
                ms(mean)
            ));
        }
    }
    let c1 = means[1] <= 1.05 * means[0];
    say(format!(
        "1. cost: with probes {:.2} ms against {:.2} ms ({:.3}×; at most 1.05×): {}; camera passes {:.2} against {:.2} ms (probe pass about {:.2} ms)",
        means[1],
        means[0],
        means[1] / means[0],
        if c1 { "PASS" } else { "FAIL" },
        camera[1],
        camera[0],
        camera[1] - camera[0]
    ));

    // 2–3. The reported error against settled references.
    let mut r = renderer(options(PROBES, true));
    let every = EVERY * frames / FLIGHT;
    let mut checks = Vec::new();
    for f in 0..frames {
        let g = f * FLIGHT / frames;
        r.frame(flight_camera(g), &target);
        if (f + 1) % every == 0 {
            r.finish();
            let flying = r.read_colors();
            let stats = *r.cache_log().last().expect("a frame");
            let mut reference = renderer(options(PROBES, true));
            let settled = settle(&mut reference, flight_camera(g), &target);
            reference.finish();
            let floor = reference
                .cache_log()
                .last()
                .map_or(0.0, |s| s.error_estimate);
            let measured = relative_rmse(&flying, &reference.read_colors());
            say(format!(
                "   frame {}: measured {:.2}%, estimate {:.2}%, bound {:.2}% ({} probes on surfaces); settled reference's own estimate {:.2}% (settled in {})",
                f + 1,
                100.0 * measured,
                100.0 * stats.error_estimate,
                100.0 * stats.error_bound,
                stats.error_probes,
                100.0 * floor,
                settled.map_or("never".into(), |s| format!("{s} frames"))
            ));
            checks.push((
                measured,
                stats.error_estimate as f64,
                stats.error_bound as f64,
            ));
        }
    }
    let held = checks.iter().filter(|c| c.2 >= c.0).count();
    let c2 = held >= 9;
    say(format!(
        "2. calibration: the bound at least the measured error in {held} of {} checks (at least 9): {}",
        checks.len(),
        if c2 { "PASS" } else { "FAIL" }
    ));
    let n = checks.len().max(1) as f64;
    let measured = checks.iter().map(|c| c.0).sum::<f64>() / n;
    let estimate = checks.iter().map(|c| c.1).sum::<f64>() / n;
    let ratio = estimate / measured.max(1e-12);
    let c3 = (0.8..=1.3).contains(&ratio);
    say(format!(
        "3. tightness: mean estimate {:.2}% against measured {:.2}% ({ratio:.3}×; 0.8–1.3×): {}",
        100.0 * estimate,
        100.0 * measured,
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m4_gate13_dev.txt"
        } else {
            "m4_gate13.txt"
        }),
        report,
    )?;
    Ok(())
}
