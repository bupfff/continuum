//! M14 gate 1: unique geometry streamed by need (camera levels).
//!
//! Until now every triangle of a scene was resident: the 10¹⁰ triangles of
//! M4 were a million copies of one boulder. Unique geometry at that density
//! does not fit, so it must be streamed: kept on the GPU only where the
//! view needs its detail, within a fixed budget, with something standing in
//! for the rest.
//!
//! Method (the first step of M14):
//! - Each geometry may carry finer *camera levels* (`Levels`), coarse to
//!   fine, sharing its charts, each with its largest distance from the
//!   finest. The geometry itself stays the *lighting level*: always
//!   resident, what mask cones, the cache's bakes and bounce light see, and
//!   the camera's stand-in.
//! - Camera rays (hardware ray queries) see each geometry at its finest
//!   resident level. After the visibility pass, a feedback pass turns each
//!   pixel's hit distance into the level its object needs (the coarsest
//!   whose distance from the finest stays under `pixel_error` pixels) and
//!   keeps the highest per geometry. The CPU reads that back two frames
//!   later, loads the most needed levels into a fixed pool (triangles,
//!   normals, chart coordinates; a bottom-level structure each), evicts the
//!   least recently needed, and points the geometry's top-level instances at
//!   the new level.
//! - Lighting is unchanged and certified as before, for the lighting
//!   level. Streaming the lighting level by cone footprint is a later gate.
//!
//! Scene: 256 unique procedural rocks (`room3d::rocks`, each its own
//! geometry: band-limited octaves, one more per level; four levels, the
//! lighting level always resident; 65,536 triangles at the finest) 6 m
//! apart on a 96 m square of a ground, lit by a sun; 1280×720, the view
//! cache on, hardware camera rays. Budget 2.5 × 2²⁰ camera-level triangles,
//! half a pixel of error, at most 2¹⁷ triangles loaded a frame. Three
//! flights cross the field turning half a circle (each ends looking back
//! across it). The fully resident renderer (every rock's finest level
//! uploaded at start) is the reference.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Scale:** the rocks' finest levels hold at least 5× the camera
//!    budget's triangles, and the resident camera-level triangles never
//!    exceed the budget (checked every frame of every run).
//! 2. **Frame rate:** on three flights, the streamed renderer's mean frame
//!    is at most 16.7 ms and its 99th percentile at most 33.3 ms, and its
//!    mean at most 1.10× the fully resident renderer's on the same flight
//!    (three interleaved rounds per flight, resident then streamed, each
//!    settled at the flight's first view; timed before everything else).
//! 3. **Coverage, settled:** at each flight's last view, once streaming and
//!    the cache are quiet, the streamed renderer's coverage error against
//!    the resident renderer is at most 0.25× one ray's coverage error.
//!    Coverage error: per pixel, half the sum over objects (a miss counted
//!    as one) of the absolute difference in the share of 4×4 sub-pixel
//!    samples each covers, averaged over pixels. One ray's: the resident
//!    renderer's pixel centre against its own 4×4 samples.
//! 4. **Coverage in flight:** replaying each flight frame by frame, every
//!    10th frame, the share of pixels whose visible object differs from the
//!    resident renderer's at the same frame is at most 1% on average.
//!
//! Reported: memory (the GPU device's allocations), loads and evictions
//! per flight, the cache's fallback in flight against the resident
//! renderer's, and the settled images' colour difference (4×4
//! supersampled).
//!
//! Changed before the first run, after development runs (`M14_DEV=1`, 36
//! rocks; and `stream_dev`, which sizes the working set; disclosed), the
//! criteria unchanged:
//! - the flights turn, so their last views look across the field (they
//!   ended looking away from it, which made criterion 3 empty);
//! - each flight is timed in three interleaved rounds (M10 gate 18's
//!   lesson), with both renderers alive throughout, instead of once,
//!   resident then streamed;
//! - the rocks' octaves fall as frequency^-1.8, not ^-1.2: at -1.2 a view
//!   of this field wanted up to 4.9 million camera-level triangles, more
//!   than any budget a fifth of the scene's; at -1.8, at most 1.9 million.
//!
//! `cargo run --release -p room3d --bin m14_gate1`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, CacheStats, Camera, InstancedRenderer, Light, MaskConeKernel, StreamOptions,
    StreamStats, wgpu,
};
use room3d::rocks::Rocks;

const W: u32 = 1280;
const H: u32 = 720;
const FOV: f32 = 50.0;

#[derive(Clone, Copy)]
struct Size {
    side: usize,
    spacing: f64,
    flight: usize,
    budget: u32,
}

const GATE: Size = Size {
    side: 16,
    spacing: 6.0,
    flight: 600,
    budget: 5 << 19,
};
const DEV: Size = Size {
    side: 6,
    spacing: 6.0,
    flight: 120,
    budget: 9 << 17,
};

/// Flight `k`'s camera at `t` in [0, 1] (`room3d::rocks::flight`).
fn flight_camera(k: usize, t: f32, half: f32) -> Camera {
    room3d::rocks::flight(k, t, half, FOV)
}

/// Bytes the GPU device has allocated (Metal's `currentAllocatedSize`; 0
/// elsewhere).
fn gpu_bytes(device: &wgpu::Device) -> u64 {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLDevice;
        if let Some(d) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            return d.raw_device().currentAllocatedSize() as u64;
        }
    }
    let _ = device;
    0
}

/// One timed flight: frame times, total, the cache's and streaming's logs,
/// and the frames its first view took to settle.
type Run = (
    Vec<Duration>,
    Duration,
    Vec<CacheStats>,
    Vec<StreamStats>,
    usize,
);

struct Flight {
    mean: Duration,
    p99: Duration,
    worst: Duration,
    fallback: f64,
    loads: u64,
    evictions: u64,
    loaded: u64,
    max_resident: u64,
    over_budget: usize,
    cpu_mean: f64,
    cpu_max: f64,
}

fn summarize(
    times: &[Duration],
    total: Duration,
    cache: &[CacheStats],
    stream: &[StreamStats],
    budget: u64,
) -> Flight {
    let mut sorted = times.to_vec();
    sorted.sort();
    let n = cache.len().max(1) as f64;
    let s = stream.len().max(1) as f64;
    Flight {
        mean: total / times.len() as u32,
        p99: sorted[(sorted.len() * 99) / 100 - 1],
        worst: *sorted.last().unwrap(),
        fallback: cache.iter().map(|c| c.fallback_pixels as f64).sum::<f64>() / n / (W * H) as f64,
        loads: stream.iter().map(|x| x.loads as u64).sum(),
        evictions: stream.iter().map(|x| x.evictions as u64).sum(),
        loaded: stream.iter().map(|x| x.loaded_triangles).sum(),
        max_resident: stream
            .iter()
            .map(|x| x.resident_triangles)
            .max()
            .unwrap_or(0),
        over_budget: stream
            .iter()
            .filter(|x| x.resident_triangles > budget)
            .count(),
        cpu_mean: stream.iter().map(|x| x.cpu_ms as f64).sum::<f64>() / s,
        cpu_max: stream.iter().map(|x| x.cpu_ms as f64).fold(0.0, f64::max),
    }
}

/// Renders at `camera` until the cache has baked and been asked for
/// nothing for 16 frames and streaming has loaded and evicted nothing for
/// 16 frames (at most `cap` frames). Frames rendered.
fn settle(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    cap: usize,
) -> usize {
    for f in 1..=cap {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        let cache_quiet = latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0);
        let log = r.stream_log();
        let stream_quiet = log.len() < 16
            || log[log.len() - 16..]
                .iter()
                .all(|s| s.loads == 0 && s.evictions == 0);
        if cache_quiet && stream_quiet && f >= 16 {
            r.finish();
            return f;
        }
    }
    r.finish();
    cap
}

/// The 4×4 sub-pixel offsets.
fn offsets() -> Vec<(f32, f32)> {
    let o = [-0.375, -0.125, 0.125, 0.375];
    o.iter()
        .flat_map(|&y| o.iter().map(move |&x| (x, y)))
        .collect()
}

/// Visible objects at each 4×4 offset and the summed colors, at `camera`.
fn samples(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<Vec<u32>>, Vec<[f32; 3]>) {
    let mut ids = Vec::new();
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    for (dx, dy) in offsets() {
        r.set_pixel_offset(dx, dy);
        r.frame(camera, target);
        r.finish();
        ids.push(r.read_visibility());
        for (c, s) in color.iter_mut().zip(r.read_colors()) {
            for k in 0..3 {
                c[k] += s[k] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    (ids, color)
}

/// Mean over pixels of half the sum over objects of the difference in the
/// share of samples each covers.
fn coverage_error(a: &[Vec<u32>], b: &[Vec<u32>]) -> f64 {
    let pixels = a[0].len();
    let (wa, wb) = (1.0 / a.len() as f64, 1.0 / b.len() as f64);
    let mut total = 0.0;
    let mut shares: HashMap<u32, f64> = HashMap::new();
    for p in 0..pixels {
        shares.clear();
        for s in a {
            *shares.entry(s[p]).or_default() += wa;
        }
        for s in b {
            *shares.entry(s[p]).or_default() -= wb;
        }
        total += 0.5 * shares.values().map(|c| c.abs()).sum::<f64>();
    }
    total / pixels as f64
}

fn relative_rms(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        for k in 0..3 {
            num += ((x[k] - y[k]) as f64).powi(2);
            den += (y[k] as f64).powi(2);
        }
    }
    (num / den.max(1e-30)).sqrt()
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M14_DEV").is_some();
    let mut size = if dev { DEV } else { GATE };
    if dev
        && let Some(b) = std::env::var("M14_BUDGET")
            .ok()
            .and_then(|b| b.parse().ok())
    {
        size.budget = b;
    }
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
    kernel.set_sky(Vec3::ZERO);
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

    let built = Instant::now();
    let rocks = Rocks::new(size.side, size.spacing);
    let half = rocks.half as f32;
    let finest = rocks.finest_triangles();
    let errors: Vec<[f32; 4]> = rocks
        .scene
        .geometry_levels
        .iter()
        .flatten()
        .map(|l| {
            [
                l.base_error,
                l.finer[0].error,
                l.finer[1].error,
                l.finer[2].error,
            ]
        })
        .collect();
    let mean_error = |k: usize| errors.iter().map(|e| e[k]).sum::<f32>() / errors.len() as f32;
    say(format!(
        "scene: {} unique rocks {} m apart ({:.0} m square), built in {:.1} s; finest levels {} triangles; \
         mean level errors {:.1} / {:.1} / {:.1} / {:.1} mm (lighting level, then finer)",
        size.side * size.side,
        size.spacing,
        2.0 * half,
        built.elapsed().as_secs_f64(),
        finest,
        1e3 * mean_error(0),
        1e3 * mean_error(1),
        1e3 * mean_error(2),
        1e3 * mean_error(3),
    ));
    let budget = size.budget as u64;
    let ratio = finest as f64 / budget as f64;
    say(format!(
        "1. scale: finest levels {} triangles against a budget of {} ({:.2}×, at least 5×)",
        finest, budget, ratio
    ));
    let lights: [Light; 1] = [rocks.sun];
    let streamed_options = StreamOptions {
        budget_triangles: size.budget,
        ..Default::default()
    };
    let resident_options = StreamOptions {
        resident: true,
        ..Default::default()
    };
    // Both renderers live through the gate (so the scene's levels can be
    // freed); each is settled at a flight's first view before it flies.
    let make = |options: StreamOptions| {
        InstancedRenderer::with_streaming(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            &rocks.scene,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            options,
        )
    };
    let device = kernel.device().clone();
    let before = gpu_bytes(&device);
    let mut streamed = make(streamed_options);
    streamed.wait();
    let streamed_bytes = gpu_bytes(&device).saturating_sub(before);
    let mid = gpu_bytes(&device);
    let mut resident = make(resident_options);
    resident.wait();
    let resident_bytes = gpu_bytes(&device).saturating_sub(mid);
    drop(rocks);
    let created = gpu_bytes(&device);

    // 2. Frame rate, first: per flight, three interleaved rounds (resident,
    // then streamed), each settled at the flight's first view, then timed.
    let mut speed_pass = true;
    for k in 0..3 {
        let camera = |f: usize| flight_camera(k, f as f32 / (size.flight - 1) as f32, half);
        let fly = |r: &mut InstancedRenderer| {
            let warmup = settle(r, camera(0), &target, 4000);
            let skip_cache = r.cache_log().len();
            let skip_stream = r.stream_log().len();
            let mut times = Vec::with_capacity(size.flight);
            let begin = Instant::now();
            for f in 0..size.flight {
                let call = Instant::now();
                r.frame(camera(f), &target);
                times.push(call.elapsed());
            }
            r.finish();
            let total = begin.elapsed();
            let cache = &r.cache_log()[skip_cache.min(r.cache_log().len())..];
            (
                times,
                total,
                cache.to_vec(),
                r.stream_log()[skip_stream..].to_vec(),
                warmup,
            )
        };
        let mut runs: [Vec<Run>; 2] = [Vec::new(), Vec::new()];
        for _ in 0..3 {
            runs[0].push(fly(&mut resident));
            runs[1].push(fly(&mut streamed));
        }
        let pooled = |runs: &[Run]| {
            let times: Vec<Duration> = runs.iter().flat_map(|r| r.0.clone()).collect();
            let total: Duration = runs.iter().map(|r| r.1).sum();
            let cache: Vec<CacheStats> = runs.iter().flat_map(|r| r.2.clone()).collect();
            let stream: Vec<StreamStats> = runs.iter().flat_map(|r| r.3.clone()).collect();
            let warm: Vec<usize> = runs.iter().map(|r| r.4).collect();
            (summarize(&times, total, &cache, &stream, budget), warm)
        };
        let (r, rw) = pooled(&runs[0]);
        let (s, sw) = pooled(&runs[1]);
        let rounds: Vec<String> = (0..3)
            .map(|i| {
                format!(
                    "{:.2}/{:.2}",
                    ms(runs[0][i].1) / size.flight as f64,
                    ms(runs[1][i].1) / size.flight as f64
                )
            })
            .collect();
        let ok = ms(s.mean) <= 16.7
            && ms(s.p99) <= 33.3
            && s.mean.as_secs_f64() <= 1.10 * r.mean.as_secs_f64();
        speed_pass &= ok;
        say(format!(
            "2. flight {k}: resident mean {:.2} ms, p99 {:.2}, worst {:.2}, fallback {:.2}%; \
             streamed mean {:.2} ms, p99 {:.2}, worst {:.2}, fallback {:.2}%; {:.3}× resident: {}",
            ms(r.mean),
            ms(r.p99),
            ms(r.worst),
            100.0 * r.fallback,
            ms(s.mean),
            ms(s.p99),
            ms(s.worst),
            100.0 * s.fallback,
            s.mean.as_secs_f64() / r.mean.as_secs_f64(),
            if ok { "PASS" } else { "FAIL" }
        ));
        say(format!(
            "   rounds (resident/streamed mean ms): {}; settled at the first view in {:?} / {:?} frames",
            rounds.join(", "),
            rw,
            sw
        ));
        say(format!(
            "   streaming over the rounds: {} loads ({} triangles), {} evictions, at most {} resident triangles, \
             {} frames over budget; CPU {:.2} ms mean, {:.2} ms worst",
            s.loads, s.loaded, s.evictions, s.max_resident, s.over_budget, s.cpu_mean, s.cpu_max
        ));
    }
    say(format!(
        "memory (GPU allocations): the streamed renderer {:.0} MB at creation (its pool, lighting levels and cache), \
         the fully resident one {:.0} MB; both together grew {:.0} MB over the flights (the streamed one's level \
         structures, both caches); the streamed one keeps its levels in CPU memory as the stand-in for a disk ({:.0} MB)",
        streamed_bytes as f64 / 1e6,
        resident_bytes as f64 / 1e6,
        gpu_bytes(&device).saturating_sub(created) as f64 / 1e6,
        streamed.stream_cpu_bytes() as f64 / 1e6,
    ));

    // 3 and 4. Coverage, with the same renderers: the resident one's
    // sampled frames and settled last view, then the streamed replay.
    let mut settled_pass = true;
    let mut flight_pass = true;
    for k in 0..3 {
        let camera = |f: usize| flight_camera(k, f as f32 / (size.flight - 1) as f32, half);
        let last = camera(size.flight - 1);
        let mut resident_ids = Vec::new();
        for f in (0..size.flight).step_by(10) {
            resident.frame(camera(f), &target);
            resident.finish();
            resident_ids.push(resident.read_visibility());
        }
        settle(&mut resident, last, &target, 4000);
        resident.frame(last, &target);
        resident.finish();
        let resident_centre = resident.read_visibility();
        let (resident_samples, resident_color) = samples(&mut resident, last, &target);
        settle(&mut streamed, camera(0), &target, 4000);
        let mut differing = Vec::new();
        for f in 0..size.flight {
            streamed.frame(camera(f), &target);
            if f % 10 == 0 {
                streamed.finish();
                let ids = streamed.read_visibility();
                let reference = &resident_ids[f / 10];
                let n = ids.iter().zip(reference).filter(|(a, b)| a != b).count();
                differing.push(n as f64 / ids.len() as f64);
            }
        }
        let frames = settle(&mut streamed, last, &target, 4000);
        let (streamed_samples, streamed_color) = samples(&mut streamed, last, &target);
        let ours = coverage_error(&streamed_samples, &resident_samples);
        let one_ray = coverage_error(&[resident_centre], &resident_samples);
        let ratio = ours / one_ray.max(1e-12);
        let ok3 = ratio <= 0.25;
        settled_pass &= ok3;
        let mean_diff = differing.iter().sum::<f64>() / differing.len() as f64;
        let worst_diff = differing.iter().copied().fold(0.0, f64::max);
        let ok4 = mean_diff <= 0.01;
        flight_pass &= ok4;
        say(format!(
            "3. flight {k} settled (streamed in {frames} frames): coverage error {:.4}% against one ray's {:.4}%: \
             {:.3}× (at most 0.25×): {}; colour difference {:.3}% (4×4 supersampled, reported)",
            100.0 * ours,
            100.0 * one_ray,
            ratio,
            if ok3 { "PASS" } else { "FAIL" },
            100.0 * relative_rms(&streamed_color, &resident_color),
        ));
        say(format!(
            "4. flight {k} in flight: pixels whose object differs {:.3}% on average (at most 1%), \
             worst sampled frame {:.3}%: {}",
            100.0 * mean_diff,
            100.0 * worst_diff,
            if ok4 { "PASS" } else { "FAIL" }
        ));
    }
    let over_budget = streamed
        .stream_log()
        .iter()
        .filter(|x| x.resident_triangles > budget)
        .count();
    let most = streamed
        .stream_log()
        .iter()
        .map(|x| x.resident_triangles)
        .max()
        .unwrap_or(0);
    let scale_pass = ratio >= 5.0 && over_budget == 0;
    say(format!(
        "1. scale: {:.2}× the budget; at most {} resident triangles, {} of {} frames over budget: {}",
        ratio,
        most,
        over_budget,
        streamed.stream_log().len(),
        if scale_pass { "PASS" } else { "FAIL" }
    ));
    let pass = scale_pass && speed_pass && settled_pass && flight_pass;
    say(format!(
        "M14 gate 1: {} (scale {}, frame rate {}, settled coverage {}, coverage in flight {})",
        if pass { "PASS" } else { "FAIL" },
        if scale_pass { "pass" } else { "fail" },
        if speed_pass { "pass" } else { "fail" },
        if settled_pass { "pass" } else { "fail" },
        if flight_pass { "pass" } else { "fail" },
    ));
    let name = if dev {
        "m14_gate1_dev.txt"
    } else {
        "m14_gate1.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
