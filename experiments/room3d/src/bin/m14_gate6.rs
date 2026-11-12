//! M14 gate 6: gate 5's world within 60 fps.
//!
//! Gate 5 produced lighting levels on demand over 36,864 unique rocks and
//! stayed certified and small (27.7 KB of GPU memory a rock), but flew at
//! 26–39 ms a frame. It was not the lighting churn: with 16× the lighting
//! slots frames stayed at 29–47 ms. A profile of the render thread at
//! 16,384 rocks (`sample`, `stream_dev` `GATE5=128`) found work that grows
//! with the number of objects or of uploads, not with the view:
//! - about 25%: lighting uploads through `queue.write_buffer`, which
//!   allocates a new Metal buffer for every call (a dozen a level), and
//!   about 7% more freeing them;
//! - about 13%: every submission that traces the top-level structure builds
//!   a new Metal residency set holding every bottom-level structure (one
//!   per rock's proxy, plus loaded levels) and commits it;
//! - about 12%: the camera streamer's own work, scanning every geometry's
//!   levels to trim the CPU cache (once per level dropped) and to choose an
//!   eviction (once per eviction).
//!
//! Method:
//! - lighting levels upload through a staging belt into the frame's
//!   encoder (as camera levels already do);
//! - the patched wgpu (`third_party/`) keeps one residency set across
//!   submissions and updates it by difference, committing only when the
//!   set of structures changed;
//! - the streamer keeps lists of the levels held in CPU memory and of the
//!   geometries with levels resident, and scans only those.
//!
//! Scene, options, flights, reference and criteria: gate 5's, unchanged
//! (36,864 unique rocks, camera and lighting levels produced on demand,
//! 491,520 triangles of lighting slots, a 256 MB lighting cache, an 8 ×
//! 2²⁰-triangle camera budget, the sun, 1280×720; gate 4's three flights
//! over the central 384 m).
//!
//! Pass criterion, fixed before the implementation (gate 5's four):
//! 1. **Scale:** lighting levels at least 64× the slots' triangles and 16×
//!    the CPU cache's bytes; neither ever exceeded; the scene holds no
//!    rock's lighting triangles.
//! 2. **Certified beyond one renderer:** every page shown in flight (every
//!    60 frames) and settled at each flight's last view, on a rock within
//!    200 m of the centre or on the ground inside that square, identical
//!    (1e-5) to a reference holding the rocks within 212 m with every
//!    lighting level resident; at least 100,000 compared; the longest sun
//!    shadow within the 12 m margin.
//! 3. **Memory per unique rock** (4,096 → 36,864 rocks, same pools, at
//!    creation): GPU at most 1/6 of a resident lighting level's, CPU heap
//!    at most 1/6 of a level as gate 4 kept it.
//! 4. **Frame budget:** each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//!
//! Reported besides: per frame, the CPU time of camera streaming and of
//! lighting updates, the wait for the GPU, and the GPU's camera and bake
//! passes.
//!
//! Disclosed, settled while implementing (before the run; `stream_dev`
//! `GATE5` on 16,384 and 36,864 rocks, gate 5's flights for 900 frames
//! each, right after building the field; `PAGES` and a settled image
//! against gate 5's code):
//! - the three changes above took the flights from 26–39 ms to about 14
//!   and 20 ms; the top level's build then dominated: wgpu re-resolved,
//!   re-encoded and re-listed all 36,864 instances on every build and
//!   validated and declared them on every submission. So also:
//!   - the patched wgpu-core resolves and encodes only instances that
//!     changed since the TLAS's last build (no gain alone);
//!   - **clusters of proxies:** objects showing their proxy are seen
//!     through clusters of 32 (in Morton order of position), one
//!     structure and one top-level instance each, built once at creation;
//!     an object showing a finer level gets one of a fixed number of
//!     top-level slots (enough for the budget's smallest levels), and its
//!     cluster is rebuilt with its triangles collapsed onto one vertex (its
//!     bounds unchanged). Retired cluster structures are reused. A hit on a
//!     cluster names its object through a member list in the objects'
//!     buffer (`hw_object`). The top level holds about 5,200 instances
//!     instead of 36,864, and per-rock proxy structures are gone. At
//!     16,384 rocks, measured in the same thermal state: 6.6 and 9.2 ms
//!     with clusters of 32, 12.6 and 17.5 ms with clusters of one;
//!   - the streamer keeps lists of the geometries the latest feedback
//!     saw, of those holding levels and of the produced levels in CPU
//!     memory, and a running total of resident triangles, instead of
//!     scanning every geometry several times a frame;
//!   - `StreamOptions::fit_budget` (on by default) relaxes the pixel error
//!     the feedback asks for while the view's wanted levels exceed the
//!     budget; in development it never engaged here (3.7–6.0 million of 8
//!     million wanted);
//! - at 36,864 rocks, right after the build: about 12 ms in flight 0 and
//!   17–19 ms in flight 1 (14 camera loads a frame, inherent to the larger
//!   field's far boundary where rocks first want a finer level);
//! - the settled image differs from gate 5's code in 386 of 921,600 pixels
//!   (lines where a few rocks meet the ground: proxies placed in the world
//!   on the CPU rather than by the instance's transform); every cache page
//!   compared in `PAGES` matched.
//!
//! `cargo run --release -p room3d --bin m14_gate6`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec2, Vec3};
use continuum_gpu::{
    CacheOptions, CachePage, Camera, InstancedRenderer, InstancedScene, MaskConeKernel,
    StreamOptions, lighting_package_bytes, wgpu,
};
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
    side: usize,
    flight: usize,
    /// The flights' half width (gate 4's field), the compared region's,
    /// and the reference's margin beyond it.
    flight_half: f32,
    region: f64,
    margin: f64,
    budget: u32,
    cache: u64,
    lit_slots: u32,
    lit_cache: u64,
    every: usize,
    /// The smaller field for criterion 3, and the two with every lighting
    /// level resident.
    small: usize,
    resident: [usize; 2],
    idle: u64,
    min_compared: usize,
}

const GATE: Size = Size {
    side: 192,
    flight: 1800,
    flight_half: 192.0,
    region: 200.0,
    margin: 12.0,
    budget: 8 << 20,
    cache: 2 << 30,
    lit_slots: 491_520,
    lit_cache: 256 << 20,
    every: 60,
    small: 64,
    resident: [32, 64],
    idle: 120,
    min_compared: 100_000,
};
const DEV: Size = Size {
    side: 48,
    flight: 300,
    flight_half: 60.0,
    region: 68.0,
    margin: 12.0,
    budget: 3 << 20,
    cache: 256 << 20,
    lit_slots: 61_440,
    lit_cache: 32 << 20,
    every: 30,
    small: 16,
    resident: [8, 16],
    idle: 0,
    min_compared: 1_000,
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

/// A page's key across scenes: the rock's index in the whole field plus 1
/// (0: the ground), its chart within its geometry, its mip and page.
type Key = (u32, u32, u32, [u32; 2]);

/// Names a scene's pages across scenes, and says which lie in the
/// compared region.
struct Keys {
    /// Per object: the key's first part; per geometry: its first chart.
    object: Vec<u32>,
    first_chart: Vec<u32>,
}

impl Keys {
    fn new(rocks: &Rocks) -> Keys {
        let scene = &rocks.scene;
        let mut first_chart = Vec::with_capacity(scene.geometries.len());
        let mut c = 0u32;
        for g in &scene.geometries {
            first_chart.push(c);
            c += g.charts.len() as u32;
        }
        Keys {
            object: scene
                .instances
                .iter()
                .map(|i| {
                    if i.geometry == 0 {
                        0
                    } else {
                        rocks.ids[i.geometry - 1] as u32 + 1
                    }
                })
                .collect(),
            first_chart: scene
                .instances
                .iter()
                .map(|i| first_chart[i.geometry])
                .collect(),
        }
    }

    fn key(&self, p: &CachePage) -> Key {
        let o = p.object as usize;
        (self.object[o], p.chart - self.first_chart[o], p.mip, p.page)
    }
}

/// Which pages lie in the compared region: rocks whose centres lie within
/// `half` of the centre, and ground pages whose texels' box (two texels
/// wider on each side) does.
struct Region {
    half: f64,
    /// Rock centres in the whole field.
    centres: Vec<DVec3>,
    /// The ground's chart: its size, and the map from chart coordinates to
    /// the ground plane (origin and the two axes' steps).
    size: Vec2,
    origin: Vec3,
    du: Vec3,
    dv: Vec3,
}

impl Region {
    fn new(field: &Rocks, half: f64) -> Region {
        let ground = &field.scene.geometries[0];
        let (t, q) = (ground.triangles[0], ground.uv[0]);
        // Solve p = origin + du·u + dv·v on the first triangle.
        let (e1, e2) = (q[1] - q[0], q[2] - q[0]);
        let det = e1.x * e2.y - e1.y * e2.x;
        let (p1, p2) = (t[1] - t[0], t[2] - t[0]);
        let du = (p1 * e2.y - p2 * e1.y) / det;
        let dv = (p2 * e1.x - p1 * e2.x) / det;
        Region {
            half,
            centres: field.positions.clone(),
            size: ground.charts[0],
            origin: t[0] - du * q[0].x - dv * q[0].y,
            du,
            dv,
        }
    }

    fn inside(&self, key: &Key, p: &CachePage) -> bool {
        let h = self.half as f32;
        if key.0 > 0 {
            let c = self.centres[key.0 as usize - 1];
            return c.x.abs() <= self.half && c.y.abs() <= self.half;
        }
        let n = (self.size / p.texel).ceil();
        let range = |k: u32, n: f32| {
            (
                ((k * 8) as f32 - 2.0).max(0.0) / n,
                ((k * 8 + 11) as f32 / n).min(1.0),
            )
        };
        let (u0, u1) = range(p.page[0], n.x);
        let (v0, v1) = range(p.page[1], n.y);
        [(u0, v0), (u1, v0), (u0, v1), (u1, v1)]
            .iter()
            .all(|&(u, v)| {
                let w = self.origin + self.du * u + self.dv * v;
                w.x.abs() <= h && w.y.abs() <= h
            })
    }
}

/// Pages of `ours` (only those shown, if `shown_only`) in the region and
/// shown in `theirs`: how many, how many differ beyond 1e-5 (relative,
/// absolute below 1), the largest difference, and how many were rocks'.
fn compare(
    ours: &[CachePage],
    our_keys: &Keys,
    theirs: &[CachePage],
    their_keys: &Keys,
    region: &Region,
    shown_only: bool,
) -> (usize, usize, f32, usize) {
    let theirs: HashMap<Key, &Vec<f32>> = theirs
        .iter()
        .filter(|p| p.shown)
        .map(|p| (their_keys.key(p), &p.texels))
        .collect();
    let (mut common, mut differing, mut worst, mut rocks) = (0, 0, 0.0f32, 0);
    for p in ours.iter().filter(|p| p.shown || !shown_only) {
        let key = our_keys.key(p);
        let Some(t) = theirs.get(&key) else {
            continue;
        };
        if !region.inside(&key, p) {
            continue;
        }
        common += 1;
        rocks += (key.0 > 0) as usize;
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
    (common, differing, worst, rocks)
}

/// Bytes of a surface's data (as a scene holds it).
fn surface_bytes(s: &continuum::mesh::Surface) -> u64 {
    (36 * (s.triangles.len() + s.normals.len())
        + 4 * s.chart.len()
        + 24 * (s.uv.len() + s.tex.len())
        + 8 * s.charts.len()
        + 8 * s.chart_grid.len()) as u64
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
    let options = StreamOptions {
        budget_triangles: size.budget,
        cpu_cache_bytes: size.cache,
        lighting_triangles: size.lit_slots,
        lighting_cache_bytes: size.lit_cache,
        ..Default::default()
    };
    let make = |scene: &InstancedScene, lighting_triangles: u32| {
        InstancedRenderer::with_streaming(
            &kernel,
            format,
            W,
            H,
            1,
            &[room3d::rocks::Rocks::sun()],
            scene,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            StreamOptions {
                lighting_triangles,
                ..options
            },
        )
    };

    // The scene, and the renderer at creation (criterion 3's large field).
    let heap0 = live();
    let built = Instant::now();
    let rocks = Rocks::lighting_on_demand(size.side, SPACING);
    let build_s = built.elapsed().as_secs_f64();
    let count = rocks.positions.len();
    let empty = rocks.scene.geometries[1..]
        .iter()
        .all(|g| g.triangles.is_empty());
    // Every rock's lighting level has the same layout (a 32 × 16 grid):
    // its size from the first, checked on a sample.
    let source = rocks.scene.level_source.clone().expect("a source");
    let level = source.lighting(1).expect("a lighting level");
    let level_tris = level.triangles.len() as u64;
    let package = lighting_package_bytes(&level);
    let packaged = package + surface_bytes(&level);
    for g in (1..=count).step_by((count / 61).max(1)) {
        let l = source.lighting(g).expect("a lighting level");
        assert_eq!(
            l.triangles.len() as u64,
            level_tris,
            "rock lighting levels alike"
        );
        assert_eq!(lighting_package_bytes(&l), package, "rock packages alike");
    }
    let lighting = level_tris * count as u64;
    say(format!(
        "scene: {count} unique rocks with proxies over a {:.0} m square, built in {build_s:.1} s; the scene holds no \
         rock's lighting triangles: {empty}; lighting levels {lighting} triangles ({level_tris} a rock, {:.0} KB \
         packaged, {:.0} KB with its surface as gate 4 kept it); lighting slots {} triangles, CPU cache {} MB; camera \
         budget {}",
        2.0 * rocks.half,
        package as f64 / 1e3,
        packaged as f64 / 1e3,
        size.lit_slots,
        size.lit_cache >> 20,
        size.budget,
    ));
    let gpu0 = gpu_bytes(&device);
    let made = Instant::now();
    let mut r = make(&rocks.scene, size.lit_slots);
    r.wait();
    let make_s = made.elapsed().as_secs_f64();
    let big_gpu = gpu_bytes(&device) - gpu0;
    let big_heap = live() - heap0;
    say(format!(
        "renderer made in {make_s:.1} s (stubs written from every lighting level produced once): GPU {:.0} MB, CPU \
         heap (scene and renderer) {:.0} MB",
        big_gpu as f64 / 1e6,
        big_heap as f64 / 1e6
    ));

    // 4. The frame budget, first: after idling, each flight settled at its
    // first view, then flown as fast as it renders.
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
    let mut budget_pass = true;
    for k in 0..3 {
        let warm = settle(&mut r, camera(k, 0), &target, 8000);
        let skip = r.lighting_log().0.len();
        let (cache_skip, stream_skip) = (r.cache_log().len(), r.stream_log().len());
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
        budget_pass &= ok;
        let lit = &r.lighting_log().0[skip..];
        let cache = &r.cache_log()[cache_skip..];
        let stream = &r.stream_log()[stream_skip..];
        let avg = |v: &mut dyn Iterator<Item = f64>, n: usize| v.sum::<f64>() / n.max(1) as f64;
        say(format!(
            "   per frame: camera streaming CPU {:.2} ms ({:.1} loads), lighting CPU {:.2} ms, cache planning {:.2} ms, \
             waiting for the GPU {:.2} ms; pixel error at most {:.2}",
            avg(&mut stream.iter().map(|s| s.cpu_ms as f64), stream.len()),
            avg(&mut stream.iter().map(|s| s.loads as f64), stream.len()),
            avg(&mut lit.iter().map(|s| s.cpu_ms as f64), lit.len()),
            avg(
                &mut cache.iter().map(|c| c.planning.as_secs_f64() * 1e3),
                cache.len()
            ),
            avg(
                &mut cache.iter().map(|c| c.waiting.as_secs_f64() * 1e3),
                cache.len()
            ),
            stream.iter().map(|s| s.pixel_error).fold(0.0f32, f32::max),
        ));
        let produced: u32 = lit.iter().map(|s| s.produced).sum();
        let latency: u32 = lit.iter().map(|s| s.production_frames).sum();
        say(format!(
            "4. flight {k}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms (at most 16.7, 16.7, 33.3): {} (settled first in \
             {warm} frames); lighting levels: {} produced ({:.1} frames from request on average), {} loads, {} \
             evictions, {} refused, at most {} pinned, CPU cache at most {:.0} MB",
            ms(mean),
            ms(p99),
            ms(worst),
            if ok { "PASS" } else { "FAIL" },
            produced,
            latency as f64 / produced.max(1) as f64,
            lit.iter().map(|s| s.loads).sum::<u32>(),
            lit.iter().map(|s| s.evictions).sum::<u32>(),
            lit.iter().map(|s| s.refused).sum::<u32>(),
            lit.iter().map(|s| s.pinned).max().unwrap_or(0),
            lit.iter().map(|s| s.cpu_cache_bytes).max().unwrap_or(0) as f64 / 1e6,
        ));
    }

    // 2. Certified beyond one renderer: each flight replayed beside a
    // reference holding the region and its margin, every lighting level
    // resident.
    let sun = Rocks::sun().as_rect().expect("a rectangular sun");
    let sun_centre = sun.corner + (sun.edge_u + sun.edge_v) * 0.5;
    let h = size.region as f32;
    let min_elevation = [(-h, -h), (h, -h), (-h, h), (h, h)]
        .iter()
        .map(|&(x, y)| {
            let d = sun_centre - Vec3::new(x, y, 0.0);
            (d.z / d.truncate().length()).atan()
        })
        .fold(f32::MAX, f32::min);
    // Each rock's lighting level (what the bakes see) by its extent: the
    // largest horizontal distance from its centre, its lowest and highest
    // points.
    let across = rocks.extents.iter().map(|e| e[0]).fold(0.0f32, f32::max);
    let top = rocks.extents.iter().map(|e| e[2]).fold(f32::MIN, f32::max);
    let bottom = rocks.extents.iter().map(|e| e[1]).fold(0.0f32, f32::min);
    // A texel within `across` of a compared rock's centre (or on the
    // ground inside the region) is shadowed only by rocks its sun ray
    // meets: those whose extent lies within (top - bottom) / tan(elevation)
    // of it horizontally (the sun's half degree widens the ray by under 1%
    // of its run).
    let reach = 2.0 * across + 1.01 * (top - bottom) / min_elevation.tan();
    let reach_pass = (size.margin as f32) > reach;
    let region_rocks = Rocks::region(size.side, SPACING, size.region + size.margin);
    let our_keys = Keys::new(&rocks);
    let their_keys = Keys::new(&region_rocks);
    let region = Region::new(&rocks, size.region);
    let reference_gpu0 = gpu_bytes(&device);
    let mut reference = make(&region_rocks.scene, 0);
    reference.wait();
    say(format!(
        "2. reference: {} rocks within {:.0} m, every lighting level resident ({:.0} MB of GPU memory); the longest sun \
         shadow reaches {reach:.2} m (sun at least {:.1}° up over the region, rocks reaching at most {across:.2} m from their \
         centres, {:.2} m from lowest point to highest) against a {:.0} m margin: {}",
        region_rocks.positions.len(),
        size.region + size.margin,
        (gpu_bytes(&device) - reference_gpu0) as f64 / 1e6,
        min_elevation.to_degrees(),
        top - bottom,
        size.margin,
        if reach_pass { "PASS" } else { "FAIL" },
    ));
    let mut certified = reach_pass;
    let mut compared = 0usize;
    for k in 0..3 {
        settle(&mut r, camera(k, 0), &target, 8000);
        settle(&mut reference, camera(k, 0), &target, 8000);
        let (mut common, mut differing, mut worst, mut hidden, mut rock_pages) =
            (0, 0, 0.0f32, 0, 0);
        for f in 0..size.flight {
            r.frame(camera(k, f), &target);
            reference.frame(camera(k, f), &target);
            if f % size.every == 0 {
                let ours = r.read_cache_pages(0);
                hidden += ours.iter().filter(|p| !p.shown).count();
                let theirs = reference.read_cache_pages(0);
                let (c, d, w, rp) = compare(&ours, &our_keys, &theirs, &their_keys, &region, true);
                common += c;
                differing += d;
                worst = worst.max(w);
                rock_pages += rp;
            }
        }
        let frames = settle(&mut r, camera(k, size.flight - 1), &target, 8000);
        settle(&mut reference, camera(k, size.flight - 1), &target, 8000);
        let ours = r.read_cache_pages(0);
        let theirs = reference.read_cache_pages(0);
        let (sc, sd, sw, srp) = compare(&ours, &our_keys, &theirs, &their_keys, &region, false);
        compared += common + sc;
        let ok = differing == 0 && sd == 0;
        certified &= ok;
        say(format!(
            "2. flight {k}: in flight {common} shown pages in the region compared ({rock_pages} on rocks), {differing} \
             differing (worst {worst:.2e}), {hidden} hidden waiting; settled (in {frames} frames) {sc} common pages in \
             the region ({srp} on rocks) of {} here and {} there, {sd} differing (worst {sw:.2e}): {}",
            ours.len(),
            theirs.len(),
            if ok { "PASS" } else { "FAIL" }
        ));
    }
    let enough = compared >= size.min_compared;
    certified &= enough;
    say(format!(
        "2. compared {compared} pages in all (at least {}): {}",
        size.min_compared,
        if enough { "PASS" } else { "FAIL" }
    ));
    drop(reference);
    drop(region_rocks);

    // 1. Scale.
    let (lit, _) = r.lighting_log();
    let capacity = lit.first().map_or(0, |s| s.capacity_triangles);
    let over_slots = lit
        .iter()
        .filter(|s| s.resident_triangles > size.lit_slots as u64)
        .count();
    let over_cache = lit
        .iter()
        .filter(|s| s.cpu_cache_bytes > size.lit_cache)
        .count();
    let produced: u64 = lit.iter().map(|s| s.produced as u64).sum();
    let peak_cache = lit.iter().map(|s| s.cpu_cache_bytes).max().unwrap_or(0);
    let slot_ratio = lighting as f64 / size.lit_slots as f64;
    let cache_ratio = (package * count as u64) as f64 / size.lit_cache as f64;
    let scale_pass = slot_ratio >= 64.0
        && cache_ratio >= 16.0
        && over_slots == 0
        && over_cache == 0
        && capacity <= size.lit_slots as u64
        && empty;
    say(format!(
        "1. scale: lighting levels {slot_ratio:.1}× the slots' triangles (slots hold {capacity}) and {cache_ratio:.1}× \
         the CPU cache's bytes ({:.2} GB packaged); frames over the slots {over_slots}, over the cache {over_cache} of \
         {}; {produced} lighting levels produced in all, CPU cache at most {:.0} MB; the scene holds no lighting \
         triangles: {empty}: {}",
        (package * count as u64) as f64 / 1e9,
        lit.len(),
        peak_cache as f64 / 1e6,
        if scale_pass { "PASS" } else { "FAIL" }
    ));
    drop(r);
    drop(rocks);

    // 3. Memory per unique rock: the same pools over the smaller field,
    // then every lighting level resident over two fields.
    let heap0 = live();
    let small_rocks = Rocks::lighting_on_demand(size.small, SPACING);
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
        "3. memory per rock ({small_count} → {count} rocks: GPU {:.0} → {:.0} MB, CPU heap {:.0} → {:.0} MB): GPU {:.1} \
         KB a rock against a resident lighting level's {:.1} KB ({} → {} rocks: {:.0} → {:.0} MB), {:.3}× (at most \
         1/6); CPU {:.1} KB a rock against {:.1} KB packaged as gate 4 kept it, {:.3}× (at most 1/6): {}",
        small_gpu as f64 / 1e6,
        big_gpu as f64 / 1e6,
        small_heap as f64 / 1e6,
        big_heap as f64 / 1e6,
        gpu_rock / 1e3,
        level_gpu / 1e3,
        resident_count[0],
        resident_count[1],
        resident_gpu[0] as f64 / 1e6,
        resident_gpu[1] as f64 / 1e6,
        gpu_rock / level_gpu,
        heap_rock / 1e3,
        packaged as f64 / 1e3,
        heap_rock / packaged as f64,
        if memory_pass { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "every lighting level resident at {count} rocks would take about {:.1} GB of GPU memory beyond this renderer's",
        level_gpu * count as f64 / 1e9
    ));

    let pass = scale_pass && certified && memory_pass && budget_pass;
    say(format!(
        "M14 gate 6: {} (scale {}, certified {}, memory {}, frame budget {})",
        if pass { "PASS" } else { "FAIL" },
        if scale_pass { "pass" } else { "fail" },
        if certified { "pass" } else { "fail" },
        if memory_pass { "pass" } else { "fail" },
        if budget_pass { "pass" } else { "fail" },
    ));
    let name = if dev {
        "m14_gate6_dev.txt"
    } else {
        "m14_gate6.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
