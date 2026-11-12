//! M14 gate 3: unique geometry beyond CPU memory, produced on demand, within
//! a 60 fps frame budget.
//!
//! Gates 1 and 2 streamed camera levels correctly, but every level of every
//! rock sat packed in CPU memory (2.7 GB for 256 rocks), standing in for a
//! disk, and frames with many loads reached 29–34 ms. A world worth
//! streaming does not fit in CPU memory either.
//!
//! Method:
//! - A scene can give its camera levels through a `LevelSource`, called on
//!   worker threads as levels are wanted (the stand-in for reading a disk;
//!   here the rocks are generated from their seeds). Produced levels wait
//!   in a CPU cache with a byte cap, the least recently used dropped first.
//! - A streaming time budget (the user's decision, 2026-11-12): loads build
//!   at most `frame_ms` of structures a frame, at a calibrated cost per
//!   triangle (about 30 ns on the M4, M14 gate 2), so loading never pushes
//!   a frame past the 60 fps budget; deferred loads are reported.
//! - Everything else is gate 2's (paged pool, level chains, feedback).
//!
//! Scene: 64 × 64 = 4,096 unique rocks 6 m apart (a 384 m square) with gate
//! 1's levels (finest 65,536 triangles); lighting levels resident; camera
//! budget 8 × 2²⁰ triangles; half a pixel of error; 1280×720, the view
//! cache on, hardware camera rays. Three flights of 1,800 frames (gate 1's
//! paths over the larger field, about 13 m/s at 60 fps).
//!
//! Reference: there is no fully resident renderer at this size. At each
//! settled view, a renderer built with every rock at its *ideal* level for
//! that view (the coarsest within half a pixel at its nearest distance, if
//! in the view; its lighting level otherwise), all resident. Gates 1–2
//! measured the ideal levels against the finest (0.06–0.10× one ray's
//! coverage error); this gate measures streaming's convergence to them.
//!
//! Pass criterion, fixed before the run:
//! 1. **Scale:** the rocks' finest levels hold at least 30× the camera
//!    budget's triangles, and more bytes packed than the machine's memory;
//!    resident camera-level triangles never exceed the budget, and the CPU
//!    level cache never exceeds its cap (every frame).
//! 2. **Frame budget (60 fps):** on each flight, rendering as fast as it
//!    can, the mean frame and the 99th percentile at most 16.7 ms, the
//!    worst frame at most 33.3 ms.
//! 3. **Settled:** at each flight's last view, once streaming and the cache
//!    are quiet, the coverage error against the ideal-level reference at
//!    most 0.25× one ray's (gate 1's measure, 4×4 samples).
//!
//! Reported: loads and production (latency from request to ready, cache
//! hits), loads deferred by the time budget, the share of seen rocks
//! showing a coarser level than wanted in flight, memory (GPU allocations,
//! the CPU cache), and the settled colour difference.
//!
//! `cargo run --release -p room3d --bin m14_gate3`
//!
//! `M14_DEV=1` runs a development check (16 × 16 rocks, short flights),
//! whose numbers are not the gate's; `M14_SIZE=1` sizes the gate scene's
//! ideal working set along the flights on the CPU alone (disclosed: at most
//! 3.33 million triangles before the run, against the 8.4 million budget).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, Levels, MaskConeKernel, StreamOptions,
    StreamStats, wgpu,
};
use room3d::rocks::Rocks;

const W: u32 = 1280;
const H: u32 = 720;
const FOV: f32 = 50.0;
/// Packed bytes of a camera level's triangle (positions and normals, three
/// corners each; chart coordinates; chart).
const PACKED_BYTES: u64 = 124;

#[derive(Clone, Copy)]
struct Size {
    side: usize,
    flight: usize,
    budget: u32,
    cache: u64,
}

const GATE: Size = Size {
    side: 64,
    flight: 1800,
    budget: 8 << 20,
    cache: 2 << 30,
};
const DEV: Size = Size {
    side: 16,
    flight: 300,
    budget: 3 << 20,
    cache: 256 << 20,
};

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
            || log[log.len() - 16..].iter().all(|s| {
                s.loads == 0 && s.evictions == 0 && s.waiting_production == 0 && s.deferred == 0
            });
        if cache_quiet && stream_quiet && f >= 16 {
            r.finish();
            return f;
        }
    }
    r.finish();
    cap
}
fn offsets() -> Vec<(f32, f32)> {
    let o = [-0.375, -0.125, 0.125, 0.375];
    o.iter()
        .flat_map(|&y| o.iter().map(move |&x| (x, y)))
        .collect()
}
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

/// The machine's memory (bytes), from `sysctl hw.memsize` (0 if unknown).
fn memory_bytes() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Each rock's ideal level for `camera` (from 1; 0: its lighting level):
/// the coarsest whose error is within `pixel_error` pixels at its nearest
/// distance if any of it is in the view, as the feedback chooses from a
/// pixel's hit; its lighting level if outside the view.
fn ideal_levels(rocks: &Rocks, camera: Camera, pixel_error: f32) -> Vec<usize> {
    let forward = (camera.target - camera.eye).normalize();
    let half = (0.5 * camera.vertical_fov_degrees).to_radians();
    let aspect = W as f32 / H as f32;
    let diagonal = (half.tan() * (1.0 + aspect * aspect).sqrt()).atan();
    let per_meter = 2.0 * half.tan() / H as f32;
    rocks
        .positions
        .iter()
        .zip(&rocks.radii)
        .enumerate()
        .map(|(k, (p, &r))| {
            let Some(levels) = &rocks.scene.geometry_levels[k + 1] else {
                return 0;
            };
            let to = p.as_vec3() - camera.eye;
            let d = to.length();
            let angular = (r / d.max(r)).asin();
            if to.normalize().dot(forward) < (diagonal + angular).min(std::f32::consts::PI).cos() {
                return 0;
            }
            let allowed = pixel_error * per_meter * (d - r).max(0.05);
            let errors: Vec<f32> = std::iter::once(levels.base_error)
                .chain(levels.finer.iter().map(|l| l.error))
                .collect();
            errors
                .iter()
                .position(|&e| e <= allowed)
                .unwrap_or(errors.len() - 1)
        })
        .collect()
}

/// The scene with each rock's camera levels cut to its ideal level (the
/// finest kept is what a resident renderer loads).
fn ideal_scene(rocks: &Rocks, ideal: &[usize]) -> InstancedScene {
    let mut scene = rocks.scene.clone();
    for (k, &level) in ideal.iter().enumerate() {
        let slot = &mut scene.geometry_levels[k + 1];
        if level == 0 {
            *slot = None;
        } else if let Some(Levels { finer, .. }) = slot {
            finer.truncate(level);
        }
    }
    scene
}

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    sorted[((sorted.len() * p) / 100).min(sorted.len() - 1)]
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
    let device = kernel.device().clone();

    let built = Instant::now();
    let rocks = Rocks::on_demand(size.side, 6.0);
    let half = rocks.half as f32;
    let finest: u64 = rocks
        .scene
        .geometry_levels
        .iter()
        .flatten()
        .map(|l| l.finer.last().map_or(0, |f| f.triangles as u64))
        .sum();
    let packed = finest * PACKED_BYTES;
    let memory = memory_bytes();
    let ratio = finest as f64 / size.budget as f64;
    say(format!(
        "scene: {} unique rocks over a {:.0} m square, lighting levels built in {:.1} s; finest levels at most {} \
         triangles, {:.1} GB packed, against {:.1} GB of memory; camera budget {} triangles ({:.1}×)",
        size.side * size.side,
        2.0 * half,
        built.elapsed().as_secs_f64(),
        finest,
        packed as f64 / 1e9,
        memory as f64 / 1e9,
        size.budget,
        ratio,
    ));
    if std::env::var_os("M14_SIZE").is_some() {
        // Sizing (development): the ideal levels' triangles along each
        // flight, every 30th frame, on the CPU alone.
        for k in 0..3 {
            let most = (0..size.flight)
                .step_by(30)
                .map(|f| {
                    let camera =
                        room3d::rocks::flight(k, f as f32 / (size.flight - 1) as f32, half, FOV);
                    ideal_levels(&rocks, camera, 0.5)
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| **l > 0)
                        .map(|(i, &l)| {
                            rocks.scene.geometry_levels[i + 1].as_ref().unwrap().finer[l - 1]
                                .triangles as u64
                        })
                        .sum::<u64>()
                })
                .max()
                .unwrap_or(0);
            println!("flight {k}: ideal working set at most {most} triangles");
        }
        return Ok(());
    }
    let options = StreamOptions {
        budget_triangles: size.budget,
        cpu_cache_bytes: size.cache,
        ..Default::default()
    };
    let lights = [rocks.sun];
    let before = gpu_bytes(&device);
    let mut r = InstancedRenderer::with_streaming(
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
    );
    r.wait();
    let created = gpu_bytes(&device);

    // 2. The frame budget, first: each flight settled at its first view,
    // then flown as fast as it renders.
    let mut budget_pass = true;
    for k in 0..3 {
        let camera =
            |f: usize| room3d::rocks::flight(k, f as f32 / (size.flight - 1) as f32, half, FOV);
        let warmup = settle(&mut r, camera(0), &target, 6000);
        let skip = r.stream_log().len();
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            r.frame(camera(f), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let mean = begin.elapsed() / size.flight as u32;
        let log: Vec<StreamStats> = r.stream_log()[skip..].to_vec();
        let mut sorted = times.clone();
        sorted.sort();
        let (p99, worst) = (percentile(&sorted, 99), *sorted.last().unwrap());
        let ok = ms(mean) <= 16.7 && ms(p99) <= 16.7 && ms(worst) <= 33.3;
        budget_pass &= ok;
        let sum = |f: &dyn Fn(&StreamStats) -> u64| log.iter().map(f).sum::<u64>();
        let produced = sum(&|s| s.produced as u64);
        let seen = sum(&|s| s.seen as u64).max(1);
        say(format!(
            "2. flight {k}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms (at most 16.7, 16.7, 33.3): {} (settled first in {warmup} frames)",
            ms(mean),
            ms(p99),
            ms(worst),
            if ok { "PASS" } else { "FAIL" }
        ));
        say(format!(
            "   {} loads ({} triangles), {} evictions, {} deferred by the time budget; {} levels produced, {:.1} frames \
             from request on average, {} wants waiting for production in all; seen rocks showing a coarser level than \
             wanted {:.2}% on average; at most {} resident triangles, CPU cache at most {:.0} MB",
            sum(&|s| s.loads as u64),
            sum(&|s| s.loaded_triangles),
            sum(&|s| s.evictions as u64),
            sum(&|s| s.deferred as u64),
            produced,
            sum(&|s| s.production_frames as u64) as f64 / produced.max(1) as f64,
            sum(&|s| s.waiting_production as u64),
            100.0 * sum(&|s| s.wanting as u64) as f64 / seen as f64,
            log.iter().map(|s| s.resident_triangles).max().unwrap_or(0),
            log.iter().map(|s| s.cpu_cache_bytes).max().unwrap_or(0) as f64 / 1e6,
        ));
    }
    say(format!(
        "memory: GPU allocations {:.0} MB at creation (pool, lighting levels, cache), {:.0} MB more after the flights",
        (created - before) as f64 / 1e6,
        gpu_bytes(&device).saturating_sub(created) as f64 / 1e6
    ));

    // 3. Settled, against the ideal-level reference.
    let mut settled_pass = true;
    for k in 0..3 {
        let last = room3d::rocks::flight(k, 1.0, half, FOV);
        let frames = settle(&mut r, last, &target, 6000);
        let (ours, ours_color) = samples(&mut r, last, &target);
        let ideal = ideal_levels(&rocks, last, options.pixel_error);
        let counts = [1, 2, 3].map(|l| ideal.iter().filter(|&&i| i == l).count());
        let made = Instant::now();
        let scene = ideal_scene(&rocks, &ideal);
        let mut reference = InstancedRenderer::with_streaming(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            &scene,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            StreamOptions {
                resident: true,
                ..Default::default()
            },
        );
        let making = made.elapsed();
        settle(&mut reference, last, &target, 6000);
        reference.frame(last, &target);
        reference.finish();
        let centre = reference.read_visibility();
        let (theirs, theirs_color) = samples(&mut reference, last, &target);
        drop(reference);
        let error = coverage_error(&ours, &theirs);
        let one_ray = coverage_error(&[centre], &theirs);
        let ratio = error / one_ray.max(1e-12);
        let ok = ratio <= 0.25;
        settled_pass &= ok;
        say(format!(
            "3. flight {k} settled (in {frames} frames): coverage error {:.4}% against one ray's {:.4}%: {:.3}× \
             (at most 0.25×): {}; colour difference {:.3}% (reported); reference: {} / {} / {} rocks at levels 1 / 2 / 3, \
             made in {:.1} s",
            100.0 * error,
            100.0 * one_ray,
            ratio,
            if ok { "PASS" } else { "FAIL" },
            100.0 * relative_rms(&ours_color, &theirs_color),
            counts[0],
            counts[1],
            counts[2],
            making.as_secs_f64()
        ));
    }
    let log = r.stream_log();
    let over = log
        .iter()
        .filter(|s| s.resident_triangles > size.budget as u64)
        .count();
    let over_cache = log
        .iter()
        .filter(|s| s.cpu_cache_bytes > size.cache)
        .count();
    let scale_pass = ratio >= 30.0 && packed > memory && over == 0 && over_cache == 0;
    say(format!(
        "1. scale: {:.1}× the budget (at least 30×), {:.1} GB packed against {:.1} GB of memory; {} of {} frames over the \
         budget, {} over the CPU cache's cap: {}",
        ratio,
        packed as f64 / 1e9,
        memory as f64 / 1e9,
        over,
        log.len(),
        over_cache,
        if scale_pass { "PASS" } else { "FAIL" }
    ));
    let pass = scale_pass && budget_pass && settled_pass;
    say(format!(
        "M14 gate 3: {} (scale {}, frame budget {}, settled {})",
        if pass { "PASS" } else { "FAIL" },
        if scale_pass { "pass" } else { "fail" },
        if budget_pass { "pass" } else { "fail" },
        if settled_pass { "pass" } else { "fail" },
    ));
    let name = if dev {
        "m14_gate3_dev.txt"
    } else {
        "m14_gate3.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
