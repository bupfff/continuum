//! M14 gate 11: the frame-period cap times the renderer's own frames.
//!
//! The cap on the bake budget (M14 gate 9, `CacheOptions::frame_budget_ms`)
//! measures the period between frames on the CPU. That period includes
//! whatever the caller does between frames: in M5 gate 13's development, a
//! warm-up that read the image back every 20th frame held the cap near its
//! floor (it shrinks in proportion to a long frame, and grows only while
//! frames take under 85% of the budget), and a view took up to 1,895 frames
//! to warm instead of 423.
//!
//! Now the cap measures the time spent inside the renderer's frame call
//! (which waits for the GPU when the GPU is behind), not the gap between
//! calls: what the caller does between frames, a readback or an engine's
//! own work or a wait for the display, no longer moves the bake budget.
//!
//! Pass criteria, fixed before the implementation:
//! 1. **The frame budget still holds** (run first): M14 gate 10's timed
//!    field and flights (36,864 unique rocks, camera and lighting levels on
//!    demand, a uniform sky and bounce light, default cache options, three
//!    1,800-frame flights at 1280×720 after the build and two minutes idle):
//!    each flight's mean and 99th percentile at most 16.7 ms, worst at most
//!    33.3 ms (gate 10's criterion 1).
//! 2. **The caller's stalls no longer slow warming:** at M5 gate 13's four
//!    viewpoints (the million-boulder field, sun and sky through rays,
//!    default options, 1280×720), a cache warmed from empty while the image
//!    is read back every 20th frame warms (16 quiet frames) in at most 1.25×
//!    the frames of one warmed undisturbed in the same run.
//!
//! **Disclosed (before the run):** with `sky_dev` on the full field the
//! views from 40 and 10 m up warmed in 412 and 364 frames reading back every
//! 20th frame against 421 and 374 undisturbed (before the change, the same
//! readbacks took up to 1,895 frames). The development mode passed: flights
//! 8.74–8.95 ms mean, p99 13.9–14.2 ms on 2,304 rocks; warm in 179 and 360
//! frames reading back against 165 and 354 on 100,000 boulders.
//!
//! `M14_DEV=1`: a development check (2,304 rocks with short flights,
//! 100,000 boulders at two viewpoints); not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m14_gate11`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, StreamOptions, wgpu};
use room3d::field::Field;
use room3d::rocks::Rocks;

/// The uniform sky's radiance (as M14 gate 10).
const SKY: f32 = 0.05;
const W: u32 = 1280;
const H: u32 = 720;
const FOV: f32 = 50.0;
const SPACING: f64 = 6.0;

#[derive(Clone, Copy)]
struct Size {
    side: usize,
    flight_half: f32,
    flight: usize,
    budget: u32,
    cache: u64,
    lit_slots: u32,
    lit_cache: u64,
    idle: u64,
    boulders: usize,
    views: usize,
}

const GATE: Size = Size {
    side: 192,
    flight_half: 192.0,
    flight: 1800,
    budget: 8 << 20,
    cache: 2 << 30,
    lit_slots: 491_520,
    lit_cache: 256 << 20,
    idle: 120,
    boulders: 1_000_000,
    views: 4,
};
const DEV: Size = Size {
    side: 48,
    flight_half: 60.0,
    flight: 300,
    budget: 3 << 20,
    cache: 256 << 20,
    lit_slots: 61_440,
    lit_cache: 32 << 20,
    idle: 0,
    boulders: 100_000,
    views: 2,
};

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    sorted[((sorted.len() * p) / 100).min(sorted.len() - 1)]
}

/// M14 gate 10's settle: the cache, streaming and lighting levels quiet
/// for 16 frames (at most `cap`).
fn settle_streamed(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    cap: usize,
) -> usize {
    let mut quiet = 0;
    for f in 1..=cap {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        let cache_quiet = latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0);
        let stream_quiet = r.stream_log().last().is_none_or(|s| {
            s.loads == 0 && s.evictions == 0 && s.waiting_production == 0 && s.deferred == 0
        });
        let (lit, waiting) = r.lighting_log();
        let lit_quiet = waiting == 0
            && lit
                .last()
                .is_none_or(|s| s.loads == 0 && s.waiting_production == 0);
        quiet = if stream_quiet && lit_quiet {
            quiet + 1
        } else {
            0
        };
        if cache_quiet && quiet >= 16 && f >= 16 {
            r.finish();
            return f;
        }
    }
    r.finish();
    cap
}

/// Frames until 16 in a row request and bake nothing (at most 60,000),
/// reading the image back every `every`th frame if given.
fn warm(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    every: Option<usize>,
) -> Option<usize> {
    for k in 1..=60_000usize {
        r.frame(camera, target);
        if every.is_some_and(|e| k % e == 0) {
            r.finish();
            std::hint::black_box(r.read_colors());
        }
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
    let dev = std::env::var_os("M14_DEV").is_some();
    let size = if dev { DEV } else { GATE };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say("DEVELOPMENT RUN: not the gate's numbers".into());
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
    kernel.set_sky(Vec3::splat(SKY));
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

    // 1. M14 gate 10's timed field and flights, first.
    let built = Instant::now();
    let rocks = Rocks::lighting_on_demand(size.side, SPACING);
    say(format!(
        "1. {} unique rocks, lighting levels on demand, built in {:.1} s; a uniform sky ({SKY}) and bounce light",
        rocks.positions.len(),
        built.elapsed().as_secs_f64()
    ));
    let mut r = InstancedRenderer::with_streaming(
        &kernel,
        format,
        W,
        H,
        1,
        &[Rocks::sun()],
        &rocks.scene,
        DVec3::ZERO,
        Some(CacheOptions {
            indirect: true,
            ..Default::default()
        }),
        StreamOptions {
            budget_triangles: size.budget,
            cpu_cache_bytes: size.cache,
            lighting_cache_bytes: size.lit_cache,
            lighting_triangles: size.lit_slots,
            ..Default::default()
        },
    );
    r.wait();
    if size.idle > 0 {
        std::thread::sleep(Duration::from_secs(size.idle));
    }
    let camera = |k: usize, f: usize| {
        room3d::rocks::flight(
            k,
            f as f32 / (size.flight - 1) as f32,
            size.flight_half,
            FOV,
        )
    };
    let mut c1 = true;
    for k in 0..3 {
        let warmed = settle_streamed(&mut r, camera(k, 0), &target, 1200);
        let cache_skip = r.cache_log().len();
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            r.frame(camera(k, f), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let mean = begin.elapsed() / size.flight as u32;
        let mut sorted = times.clone();
        sorted.sort();
        let (p99, worst) = (percentile(&sorted, 99), *sorted.last().unwrap());
        let ok = ms(mean) <= 16.7 && ms(p99) <= 16.7 && ms(worst) <= 33.3;
        c1 &= ok;
        let log = &r.cache_log()[cache_skip..];
        let n = log.len().max(1) as f64;
        say(format!(
            "1. flight {k}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms (at most 16.7, 16.7, 33.3): {} (warmed first for {warmed} frames); frames over 16.7 ms: {}; bake budget {:.0} on average (cap {:.0}), {:.1} pages baked a frame",
            ms(mean),
            ms(p99),
            ms(worst),
            if ok { "PASS" } else { "FAIL" },
            times.iter().filter(|t| ms(**t) > 16.7).count(),
            log.iter().map(|c| c.budget as f64).sum::<f64>() / n,
            log.iter().map(|c| c.budget_cap as f64).sum::<f64>() / n,
            log.iter().map(|c| c.pages_baked as f64).sum::<f64>() / n,
        ));
    }
    say(format!(
        "1. the frame budget: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));
    drop(r);
    drop(rocks);

    // 2. Warm-ups reading the image back, against undisturbed ones.
    let field = Field::new(size.boulders, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
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
        ("standing", view(0.0, 1.7, 0.0)),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ];
    let renderer = || {
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
            CacheOptions::default(),
        )
    };
    let mut c2 = true;
    for &(name, camera) in views.iter().take(size.views) {
        let plain = warm(&mut renderer(), camera, &target, None);
        let read = warm(&mut renderer(), camera, &target, Some(20));
        let ok = matches!((plain, read), (Some(p), Some(r)) if r as f64 <= 1.25 * p as f64);
        c2 &= ok;
        say(format!(
            "2. {name}: warm in {} frames reading back every 20th, {} undisturbed ({}; at most 1.25×): {}",
            read.map_or("never".into(), |f| f.to_string()),
            plain.map_or("never".into(), |f| f.to_string()),
            match (plain, read) {
                (Some(p), Some(r)) => format!("{:.2}×", r as f64 / p as f64),
                _ => "-".into(),
            },
            if ok { "PASS" } else { "FAIL" }
        ));
    }
    say(format!(
        "2. the caller's stalls: {}",
        if c2 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m14_gate11_dev.txt"
        } else {
            "m14_gate11.txt"
        }),
        report,
    )?;
    Ok(())
}
