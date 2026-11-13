//! M14 gate 14: several bounces with streamed lighting.
//!
//! M10 gate 25 carried light past its first bounce (`CacheOptions::bounces`;
//! the showcase room 6.54% from Cycles at 12 bounces with 8, against 45.36%
//! with one), with bounce rays through hardware ray queries, on scenes held
//! wholly in memory. M14 gate 10 flew 36,864 unique rocks with a sky and one
//! bounce within 60 fps, lighting levels streamed and direct light
//! certified. Untried: both together. With several bounces a bounce sample
//! also gathers the bounce light of the pages its rays hit, and pages whose
//! hits hold too few bounces ask for those pages with samples of their own:
//! more pages, on objects whose lighting levels may not be resident.
//!
//! Method: gate 10's binary with `bounces: 8` (M10 gate 25's setting); no
//! other change.
//!
//! Scenes: gate 10's (a) 9,216 rocks beside every lighting level resident,
//! and (b) 36,864 rocks; a uniform sky and bounce light, 8 bounces.
//!
//! Pass criterion, fixed before the run (gate 10's first three):
//! 1. **Frame budget** (b): each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//! 2. **Certified** (a): sun and sky pages shown, in flight and settled,
//!    identical (1e-5) to every lighting level resident; at least 100,000
//!    sun and 30,000 sky pages.
//! 3. **Bounce converges** (a): settled images within 1% of the
//!    all-resident renderer's (both with 8 bounces).
//!
//! Reported: frames to settle (gate 10 with one bounce: 8,000, 8,000 and
//! 6,026 here); the settled images against one bounce's.
//!
//! **Disclosed (before the run):** the development mode (2,304 and 576
//! rocks, 300-frame flights) passed: flights 7.7–9.5 ms on average (99th
//! percentile 12.9–14.1 ms), 240,108 sun and sky pages compared with none
//! differing, settled images 0.047–0.061% from all resident; one bounce's
//! settled images 0.10–0.13% from 8 bounces'. The streamed renderer settled
//! none of the last views within 8,000 frames (the reference in 1,055–1,365).
//!
//! `cargo run --release -p room3d --bin m14_gate14`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, CachePage, Camera, InstancedRenderer, InstancedScene, MaskConeKernel,
    StreamOptions, wgpu,
};

/// Bounces (M10 gate 25's setting).
const BOUNCES: u32 = 8;

/// The uniform sky's radiance (the sun's irradiance is about 0.9).
const SKY: f32 = 0.05;
use room3d::rocks::Rocks;

const W: u32 = 1280;
const H: u32 = 720;
const FOV: f32 = 50.0;
const SPACING: f64 = 6.0;

#[derive(Clone, Copy)]
struct Size {
    /// The certified field (a), its flights' half width; the timed field
    /// (b), its flights' half width.
    small: usize,
    small_half: f32,
    side: usize,
    flight_half: f32,
    flight: usize,
    budget: u32,
    cache: u64,
    lit_slots: u32,
    lit_cache: u64,
    every: usize,
    idle: u64,
    min_sun: usize,
    min_sky: usize,
}

const GATE: Size = Size {
    small: 96,
    small_half: 288.0,
    side: 192,
    flight_half: 192.0,
    flight: 1800,
    budget: 8 << 20,
    cache: 2 << 30,
    lit_slots: 491_520,
    lit_cache: 256 << 20,
    every: 60,
    idle: 120,
    min_sun: 100_000,
    min_sky: 30_000,
};
const DEV: Size = Size {
    small: 24,
    small_half: 72.0,
    side: 48,
    flight_half: 60.0,
    flight: 300,
    budget: 3 << 20,
    cache: 256 << 20,
    lit_slots: 61_440,
    lit_cache: 32 << 20,
    every: 30,
    idle: 0,
    min_sun: 1_000,
    min_sky: 300,
};

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    sorted[((sorted.len() * p) / 100).min(sorted.len() - 1)]
}

/// Renders at `camera` until the cache has baked and been asked for
/// nothing, streaming has loaded nothing, and no page waits for a lighting
/// level (nor a level for its production), for 16 frames each (at most
/// `cap` frames). Frames rendered.
fn settle(
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

/// Pages of `ours` (only those shown, if `shown_only`; with `sky`, only
/// those whose sky is baked, in both) also shown in `theirs`: how many, how
/// many differ beyond 1e-5 (relative, absolute below 1), and the largest
/// difference.
fn compare(
    ours: &[CachePage],
    theirs: &[CachePage],
    shown_only: bool,
    sky: bool,
) -> (usize, usize, f32) {
    let theirs: HashMap<(u32, u32, u32, [u32; 2]), &Vec<f32>> = theirs
        .iter()
        .filter(|p| p.shown && (!sky || p.sky_baked))
        .map(|p| ((p.object, p.chart, p.mip, p.page), &p.texels))
        .collect();
    let (mut common, mut differing, mut worst) = (0, 0, 0.0f32);
    for p in ours
        .iter()
        .filter(|p| (p.shown || !shown_only) && (!sky || p.sky_baked))
    {
        let Some(t) = theirs.get(&(p.object, p.chart, p.mip, p.page)) else {
            continue;
        };
        common += 1;
        let d = p
            .texels
            .iter()
            .zip(t.iter())
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        if d > 1e-5 {
            differing += 1;
        }
        worst = worst.max(d);
    }
    (common, differing, worst)
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
    let options = StreamOptions {
        budget_triangles: size.budget,
        cpu_cache_bytes: size.cache,
        lighting_cache_bytes: size.lit_cache,
        ..Default::default()
    };
    // The cache's frame target and frame budget at their defaults (14 ms of
    // GPU time, 11 ms frames); bounce rays through hardware ray queries (the
    // renderer's choice with hardware camera rays).
    let cache = CacheOptions {
        indirect: true,
        bounces: BOUNCES,
        ..Default::default()
    };
    let make_with = |scene: &InstancedScene, lighting_triangles: u32, cache: CacheOptions| {
        InstancedRenderer::with_streaming(
            &kernel,
            format,
            W,
            H,
            1,
            &[Rocks::sun()],
            scene,
            DVec3::ZERO,
            Some(cache),
            StreamOptions {
                lighting_triangles,
                ..options
            },
        )
    };
    let make = |scene: &InstancedScene, lighting_triangles: u32| {
        make_with(scene, lighting_triangles, cache)
    };

    // (b) The timed field, and criterion 4's large field at creation.
    let built = Instant::now();
    let rocks = Rocks::lighting_on_demand(size.side, SPACING);
    let count = rocks.positions.len();
    say(format!(
        "(b) {count} unique rocks, lighting levels on demand, built in {:.1} s; a uniform sky ({SKY}) and bounce light, {BOUNCES} bounces",
        built.elapsed().as_secs_f64()
    ));
    let level = rocks
        .scene
        .level_source
        .clone()
        .expect("a source")
        .lighting(1)
        .expect("a lighting level");
    let mut r = make(&rocks.scene, size.lit_slots);
    r.wait();

    // 1. The frame budget, first: after idling, each flight settled at its
    // first view, then flown as fast as it renders.
    if size.idle > 0 {
        std::thread::sleep(Duration::from_secs(size.idle));
    }
    let camera = |k: usize, f: usize, half: f32| {
        room3d::rocks::flight(k, f as f32 / (size.flight - 1) as f32, half, FOV)
    };
    let mut budget_pass = true;
    for k in 0..3 {
        let warm = settle(&mut r, camera(k, 0, size.flight_half), &target, 1200);
        let skip = r.lighting_log().0.len();
        let cache_skip = r.cache_log().len();
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            r.frame(camera(k, f, size.flight_half), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let mean = begin.elapsed() / size.flight as u32;
        let mut sorted = times.clone();
        sorted.sort();
        let (p99, worst) = (percentile(&sorted, 99), *sorted.last().unwrap());
        let ok = ms(mean) <= 16.7 && ms(p99) <= 16.7 && ms(worst) <= 33.3;
        budget_pass &= ok;
        let lit = &r.lighting_log().0[skip..];
        let cache_log = &r.cache_log()[cache_skip..];
        let n = cache_log.len().max(1) as f64;
        say(format!(
            "1. flight {k}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms (at most 16.7, 16.7, 33.3): {} (warmed first for \
             {warm} frames); frames over 16.7 ms: {}; bake budget {:.0} on average (cap {:.0}), {:.1} pages baked a \
             frame; lighting levels: {} loads, {} produced, at most {} pinned",
            ms(mean),
            ms(p99),
            ms(worst),
            if ok { "PASS" } else { "FAIL" },
            times.iter().filter(|t| ms(**t) > 16.7).count(),
            cache_log.iter().map(|c| c.budget as f64).sum::<f64>() / n,
            cache_log.iter().map(|c| c.budget_cap as f64).sum::<f64>() / n,
            cache_log.iter().map(|c| c.pages_baked as f64).sum::<f64>() / n,
            lit.iter().map(|s| s.loads).sum::<u32>(),
            lit.iter().map(|s| s.produced).sum::<u32>(),
            lit.iter().map(|s| s.pinned).max().unwrap_or(0),
        ));
    }
    drop(r);
    drop(rocks);

    // (a) Certified, and bounce converging: the smaller field streamed
    // beside the same scene with every lighting level resident.
    let streamed_rocks = Rocks::lighting_on_demand(size.small, SPACING);
    let full_rocks = Rocks::with_proxies(size.small, SPACING);
    let small_count = streamed_rocks.positions.len();
    let slots = (small_count as u64 * level.triangles.len() as u64 / 8) as u32;
    let mut ours = make(&streamed_rocks.scene, slots);
    let mut reference = make(&full_rocks.scene, 0);
    say(format!(
        "(a) {small_count} unique rocks, lighting slots {slots} triangles (an eighth), against every lighting level \
         resident"
    ));
    let mut one = make_with(
        &full_rocks.scene,
        0,
        CacheOptions {
            bounces: 1,
            ..cache
        },
    );
    let (mut sun_compared, mut sky_compared) = (0usize, 0usize);
    let mut certified = true;
    let mut bounce_pass = true;
    for k in 0..3 {
        let first = settle(&mut ours, camera(k, 0, size.small_half), &target, 3000);
        settle(&mut reference, camera(k, 0, size.small_half), &target, 3000);
        let (mut sun, mut sky) = ((0usize, 0usize, 0.0f32), (0usize, 0usize, 0.0f32));
        let mut hidden = 0;
        for f in 0..size.flight {
            ours.frame(camera(k, f, size.small_half), &target);
            reference.frame(camera(k, f, size.small_half), &target);
            if f % size.every == 0 {
                let a = ours.read_cache_pages(0);
                hidden += a.iter().filter(|p| !p.shown).count();
                let b = reference.read_cache_pages(0);
                let (c, d, w) = compare(&a, &b, true, false);
                sun = (sun.0 + c, sun.1 + d, sun.2.max(w));
                let a = ours.read_cache_pages(1);
                let b = reference.read_cache_pages(1);
                let (c, d, w) = compare(&a, &b, true, true);
                sky = (sky.0 + c, sky.1 + d, sky.2.max(w));
            }
        }
        let frames = settle(
            &mut ours,
            camera(k, size.flight - 1, size.small_half),
            &target,
            8000,
        );
        let their_frames = settle(
            &mut reference,
            camera(k, size.flight - 1, size.small_half),
            &target,
            8000,
        );
        let (sc, sd, sw) = compare(
            &ours.read_cache_pages(0),
            &reference.read_cache_pages(0),
            true,
            false,
        );
        let (kc, kd, kw) = compare(
            &ours.read_cache_pages(1),
            &reference.read_cache_pages(1),
            true,
            true,
        );
        sun_compared += sun.0 + sc;
        sky_compared += sky.0 + kc;
        let ok = sun.1 == 0 && sky.1 == 0 && sd == 0 && kd == 0;
        certified &= ok;
        say(format!(
            "2. flight {k}: in flight {} sun pages compared ({} differing, worst {:.2e}) and {} sky pages ({} \
             differing, worst {:.2e}), {hidden} hidden waiting; settled (here in {frames} frames, there in \
             {their_frames}; first view here in {first}) {sc} sun pages ({sd} differing, worst {sw:.2e}) and {kc} \
             sky pages ({kd} differing, worst {kw:.2e}): {}",
            sun.0,
            sun.1,
            sun.2,
            sky.0,
            sky.1,
            sky.2,
            if ok { "PASS" } else { "FAIL" }
        ));
        // 3. The settled images.
        let a = ours.read_colors();
        let b = reference.read_colors();
        let luminance = |p: &[f32; 3]| (p[0] + p[1] + p[2]) as f64 / 3.0;
        let mean = b.iter().map(luminance).sum::<f64>() / b.len() as f64;
        let mean_difference = |a: &[[f32; 3]], b: &[[f32; 3]]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (0..3).map(|c| (x[c] - y[c]).abs() as f64).sum::<f64>() / 3.0)
                .sum::<f64>()
                / b.len() as f64
        };
        let ratio = mean_difference(&a, &b) / mean.max(1e-9);
        let ok = ratio <= 0.01;
        bounce_pass &= ok;
        // Reported: one bounce's settled image (every level resident).
        let one_frames = settle(
            &mut one,
            camera(k, size.flight - 1, size.small_half),
            &target,
            8000,
        );
        let c = one.read_colors();
        let one_mean = c.iter().map(luminance).sum::<f64>() / c.len() as f64;
        say(format!(
            "3. flight {k}: settled image's mean difference {:.3}% of its mean (at most 1%): {}; one bounce's \
             settled image (in {one_frames} frames) {:.3}% from {BOUNCES} bounces', its mean {:.4} against {:.4}",
            100.0 * ratio,
            if ok { "PASS" } else { "FAIL" },
            100.0 * mean_difference(&c, &b) / mean.max(1e-9),
            one_mean,
            mean
        ));
    }
    let enough = sun_compared >= size.min_sun && sky_compared >= size.min_sky;
    certified &= enough;
    say(format!(
        "2. compared {sun_compared} sun pages (at least {}) and {sky_compared} sky pages (at least {}): {}",
        size.min_sun,
        size.min_sky,
        if enough { "PASS" } else { "FAIL" }
    ));

    drop(ours);
    drop(reference);
    drop(one);

    let pass = certified && bounce_pass && budget_pass;
    say(format!(
        "M14 gate 14: {} (frame budget {}, certified {}, bounce {})",
        if pass { "PASS" } else { "FAIL" },
        if budget_pass { "pass" } else { "fail" },
        if certified { "pass" } else { "fail" },
        if bounce_pass { "pass" } else { "fail" },
    ));
    let name = if dev {
        "m14_gate14_dev.txt"
    } else {
        "m14_gate14.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
