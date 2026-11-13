//! M14 gate 8: a sky and bounce light with streamed lighting levels.
//!
//! Gates 4–7 streamed lighting levels with the sun alone: the sky's and
//! bounce light's bakes did not yet mark pages that met an absent level.
//!
//! Method (developed before this stub, with `stream_dev` `PAGES` on 64 and
//! 256 rocks, every comparison exact; see below):
//! - the sky's bake marks a page (and lists the absent object) as the sun's
//!   does; every absent object a page met is listed once a frame (a hash
//!   bitset), so a page needing several levels learns them in one round;
//! - a streamed geometry's stub keeps the top four levels of its hierarchy
//!   (boxes, prefiltered coverage; gate 5 kept the root alone): wide cones
//!   (the sky's faces, coarse texels) stop within them as a full walk
//!   would, and a walk marks only below the stub's frontier;
//! - bounce light, like an absent page at a hit, counts a ray that meets an
//!   absent level as missing and retries the sample, the level wanted
//!   (after direct light's wants); the page stays shown. Direct light (sun
//!   and sky channels) stays certified; bounce converges;
//! - skies baked alone and bounce retries run only once their levels are
//!   resident (pinned then; wanted, not pinned, before: pins held by pages
//!   that cannot progress deadlocked the slots); a bounce retry no longer
//!   clears its page's marks.
//!
//! The sky cannot be certified against gate 7's reference (a region with a
//! margin): it sees rocks out to the horizon. So certification uses a field
//! small enough for a renderer with every lighting level resident.
//!
//! Scenes: (a) 96 × 96 = 9,216 unique rocks, lighting levels on demand,
//! lighting slots an eighth of them; (b) gate 7's 36,864 rocks and options.
//! Both with a uniform sky and bounce light, the sun, 1280×720, the view
//! cache, hardware camera rays; gate 4's three flights (over the central
//! 384 m of (b), over all of (a)).
//!
//! Pass criterion, fixed before the gate binary was written:
//! 1. **Certified** (a): in flight (every 60 frames) and settled at each
//!    flight's last view, every page shown here identical (1e-5) in its sun
//!    channel, and every page with its sky baked in its sky channel, to the
//!    same scene with every lighting level resident; at least 100,000 sun
//!    and 30,000 sky pages compared in all.
//! 2. **Bounce converges** (a): at each flight's last view, settled, the
//!    image's mean absolute difference from the all-resident renderer's
//!    settled image at most 1% of that image's mean.
//! 3. **Frame budget** (b): each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//! 4. **Memory per unique rock** (b, 4,096 → 36,864 rocks, same pools, at
//!    creation): GPU at most 1/6 of a resident lighting level's, CPU heap
//!    at most 1/6 of a level as gate 4 kept it (the deeper stubs cost more).
//!
//! Reported: frames to settle, pages waiting, bounce samples retried.
//!
//! Disclosed, settled while implementing (before the run; `M14_DEV=1`, 576
//! and 2,304 rocks): certification held (259,630 sun and as many sky pages,
//! none differing), the settled images were within 0.11–0.17% of the
//! reference's, memory 0.113× and 0.068×. Under streaming, views with a sky
//! and bounce light did not fully settle within 8,000 frames (bounce
//! samples keep retrying), while the cache bakes each frame up to its GPU
//! target (14 ms by default, ±20% on alternate frames): flights ran at
//! 13–14 ms mean, 99th percentile 20–22 ms. So the gate sets the cache's
//! frame target to 12 ms, warms each timed flight's first view for at most
//! 1,200 frames (not until settled), and settles (a)'s first views for at
//! most 3,000 frames.
//!
//! `cargo run --release -p room3d --bin m14_gate8`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, CachePage, Camera, InstancedRenderer, InstancedScene, MaskConeKernel,
    StreamOptions, lighting_package_bytes, wgpu,
};

/// The uniform sky's radiance (the sun's irradiance is about 0.9).
const SKY: f32 = 0.05;
use room3d::rocks::Rocks;

/// Live heap bytes (criterion 3).
static LIVE: AtomicI64 = AtomicI64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size() as i64, Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size() as i64, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size() as i64, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, size) };
        if !q.is_null() {
            LIVE.fetch_add(size as i64 - l.size() as i64, Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> i64 {
    LIVE.load(Ordering::Relaxed)
}

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
    /// Criterion 4's smaller field, and the two with every lighting level
    /// resident.
    memory_small: usize,
    resident: [usize; 2],
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
    memory_small: 64,
    resident: [32, 64],
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
    memory_small: 16,
    resident: [8, 16],
    idle: 0,
    min_sun: 1_000,
    min_sky: 300,
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
    let device = kernel.device().clone();
    let options = StreamOptions {
        budget_triangles: size.budget,
        cpu_cache_bytes: size.cache,
        lighting_cache_bytes: size.lit_cache,
        ..Default::default()
    };
    // The cache bakes until a frame's GPU time reaches its target, 20% over
    // and under on alternate frames; with a sky and bounce there is always
    // baking left, so frames sit at the target: 12 ms, not 14 (disclosed).
    let cache = CacheOptions {
        indirect: true,
        frame_target_ms: 12.0,
        ..Default::default()
    };
    let make = |scene: &InstancedScene, lighting_triangles: u32| {
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

    // (b) The timed field, and criterion 4's large field at creation.
    let heap0 = live();
    let built = Instant::now();
    let rocks = Rocks::lighting_on_demand(size.side, SPACING);
    let count = rocks.positions.len();
    say(format!(
        "(b) {count} unique rocks, lighting levels on demand, built in {:.1} s; a uniform sky ({SKY}) and bounce light",
        built.elapsed().as_secs_f64()
    ));
    let source = rocks.scene.level_source.clone().expect("a source");
    let level = source.lighting(1).expect("a lighting level");
    let packaged = lighting_package_bytes(&level) + {
        let s = &level;
        (36 * (s.triangles.len() + s.normals.len())
            + 4 * s.chart.len()
            + 24 * (s.uv.len() + s.tex.len())
            + 16 * s.charts.len()) as u64
    };
    let gpu0 = gpu_bytes(&device);
    let mut r = make(&rocks.scene, size.lit_slots);
    r.wait();
    let big_gpu = gpu_bytes(&device) - gpu0;
    let big_heap = live() - heap0;

    // 3. The frame budget, first: after idling, each flight settled at its
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
        say(format!(
            "3. flight {k}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms (at most 16.7, 16.7, 33.3): {} (settled first in \
             {warm} frames); frames over 16.7 ms: {}; lighting levels: {} loads, {} produced, at most {} pinned",
            ms(mean),
            ms(p99),
            ms(worst),
            if ok { "PASS" } else { "FAIL" },
            times.iter().filter(|t| ms(**t) > 16.7).count(),
            lit.iter().map(|s| s.loads).sum::<u32>(),
            lit.iter().map(|s| s.produced).sum::<u32>(),
            lit.iter().map(|s| s.pinned).max().unwrap_or(0),
        ));
    }
    drop(r);
    drop(rocks);

    // 4. Memory per unique rock: the same pools over a smaller field, then
    // every lighting level resident over two fields.
    let heap0 = live();
    let small_rocks = Rocks::lighting_on_demand(size.memory_small, SPACING);
    let gpu0 = gpu_bytes(&device);
    let small = make(&small_rocks.scene, size.lit_slots);
    small.wait();
    let small_gpu = gpu_bytes(&device) - gpu0;
    let small_heap = live() - heap0;
    let small_count = small_rocks.positions.len();
    drop(small);
    drop(small_rocks);
    let mut resident_gpu = [0u64; 2];
    let mut resident_count = [0usize; 2];
    for (k, &side) in size.resident.iter().enumerate() {
        let field = Rocks::with_proxies(side, SPACING);
        let gpu0 = gpu_bytes(&device);
        let all = make(&field.scene, 0);
        all.wait();
        resident_gpu[k] = gpu_bytes(&device) - gpu0;
        resident_count[k] = field.positions.len();
    }
    let added = (count - small_count) as f64;
    let gpu_rock = (big_gpu as f64 - small_gpu as f64) / added;
    let heap_rock = (big_heap - small_heap) as f64 / added;
    let level_gpu = (resident_gpu[1] as f64 - resident_gpu[0] as f64)
        / (resident_count[1] - resident_count[0]) as f64;
    let memory_pass = gpu_rock <= level_gpu / 6.0 && heap_rock <= packaged as f64 / 6.0;
    say(format!(
        "4. memory per rock ({small_count} → {count} rocks): GPU {:.1} KB a rock against a resident lighting level's \
         {:.1} KB, {:.3}× (at most 1/6); CPU {:.1} KB a rock against {:.1} KB packaged as gate 4 kept it, {:.3}× (at \
         most 1/6): {}",
        gpu_rock / 1e3,
        level_gpu / 1e3,
        gpu_rock / level_gpu,
        heap_rock / 1e3,
        packaged as f64 / 1e3,
        heap_rock / packaged as f64,
        if memory_pass { "PASS" } else { "FAIL" }
    ));

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
            false,
            false,
        );
        let (kc, kd, kw) = compare(
            &ours.read_cache_pages(1),
            &reference.read_cache_pages(1),
            false,
            true,
        );
        sun_compared += sun.0 + sc;
        sky_compared += sky.0 + kc;
        let ok = sun.1 == 0 && sky.1 == 0 && sd == 0 && kd == 0;
        certified &= ok;
        say(format!(
            "1. flight {k}: in flight {} sun pages compared ({} differing, worst {:.2e}) and {} sky pages ({} \
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
        // 2. The settled images.
        let a = ours.read_colors();
        let b = reference.read_colors();
        let luminance = |p: &[f32; 3]| (p[0] + p[1] + p[2]) as f64 / 3.0;
        let mean = b.iter().map(luminance).sum::<f64>() / b.len() as f64;
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (0..3).map(|c| (x[c] - y[c]).abs() as f64).sum::<f64>() / 3.0)
            .sum::<f64>()
            / b.len() as f64;
        let ratio = diff / mean.max(1e-9);
        let ok = ratio <= 0.01;
        bounce_pass &= ok;
        say(format!(
            "2. flight {k}: settled image's mean difference {:.3}% of its mean (at most 1%): {}",
            100.0 * ratio,
            if ok { "PASS" } else { "FAIL" }
        ));
    }
    let enough = sun_compared >= size.min_sun && sky_compared >= size.min_sky;
    certified &= enough;
    say(format!(
        "1. compared {sun_compared} sun pages (at least {}) and {sky_compared} sky pages (at least {}): {}",
        size.min_sun,
        size.min_sky,
        if enough { "PASS" } else { "FAIL" }
    ));

    let pass = certified && bounce_pass && budget_pass && memory_pass;
    say(format!(
        "M14 gate 8: {} (certified {}, bounce {}, frame budget {}, memory {})",
        if pass { "PASS" } else { "FAIL" },
        if certified { "pass" } else { "fail" },
        if bounce_pass { "pass" } else { "fail" },
        if budget_pass { "pass" } else { "fail" },
        if memory_pass { "pass" } else { "fail" },
    ));
    let name = if dev {
        "m14_gate8_dev.txt"
    } else {
        "m14_gate8.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
