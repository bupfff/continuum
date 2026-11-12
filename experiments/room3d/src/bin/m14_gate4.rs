//! M14 gate 4: lighting levels streamed, the cache still certified.
//!
//! Gate 3 streamed camera levels beyond CPU memory, but every geometry's
//! lighting level (what mask cones, the cache's bakes and bounce light see)
//! stayed resident with its hierarchy and its hardware structure: about
//! 350 B a triangle on the GPU, 1.36 GB for 4,096 rocks before the cache,
//! growing with every unique object.
//!
//! Method:
//! - The camera never sees lighting levels: each streamed geometry has an
//!   always-resident *proxy* level (tens of triangles) in the camera pool,
//!   its stand-in until camera levels arrive. Lighting levels need no
//!   hardware structure.
//! - Lighting levels live in a slot pool (triangles, normals, hierarchy
//!   nodes and items rebased on upload, node coverage, chart tables) within
//!   a lighting budget. An absent geometry's objects point at a resident
//!   *stub*: its hierarchy's root (bounds and prefiltered coverage). A cone
//!   that the root alone answers (missed, or splatted at its footprint)
//!   needs nothing more: the full walk would stop there too.
//! - A cone or bounce ray that would descend into a stub, or a texel on an
//!   absent geometry, makes its page incomplete: the page is not shown, the
//!   geometry is requested, and the page is baked again once every
//!   lighting level it needs is resident (those pinned while it waits, as
//!   M10 gate 24's bounce pages).
//!
//! Scene: gate 3's rock field (camera levels produced on demand), each rock
//! with a proxy level, lighting levels produced on demand too, a lighting
//! budget an eighth of all lighting levels; the sun; 1280×720, the view
//! cache, hardware camera rays; three flights.
//!
//! Pass criterion, fixed before the implementation:
//! 1. **Scale:** all lighting levels hold at least 8× the lighting budget's
//!    triangles; resident lighting triangles never exceed it.
//! 2. **Certified:** at each flight's last view, settled, every cache page
//!    resident both here and in a reference renderer (every lighting level
//!    resident, the same camera levels) holds the same texel values (within
//!    1e-5 relative), and no pixel in any frame of the flights was shaded
//!    from a page baked against a stub.
//! 3. **Memory:** GPU memory at creation at most half that of the same
//!    renderer with every lighting level resident (gate 3's way).
//! 4. **Frame budget:** each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms.
//!
//! Reported: lighting loads, incomplete pages and retries, pinned levels,
//! settled frames, the settled images' difference from the reference.
//!
//! Disclosed, settled while implementing (before the run; `stream_dev`
//! with `PAGES` and `STILL` on 64 and 256 rocks, and `M14_DEV=1`):
//! - lighting levels stay packaged in CPU memory (streamed on the GPU
//!   only), not produced on demand as the scene line above says;
//! - the lighting pool's policy: pages wait for every level their bakes
//!   found absent (remembered per page across re-bakes); pages asked for on
//!   an absent level are kept pending and planned once it is resident;
//!   only as many levels as the bake budget can use are wanted; wanted
//!   levels are protected oldest-waiting first while the slots hold them,
//!   and loaded levels stay at least 4 frames (earlier versions thrashed or
//!   deadlocked at an eighth);
//! - a page's mark carries its frame (a slot's previous page's mark had
//!   hidden its next page for good);
//! - `read_cache_pages` read pages at the wrong stride without a sky (it
//!   always assumed `lights + sky` channels); fixed;
//! - the hardware scene uploads no vertices for proxied geometries.
//!
//! The development run gave a memory ratio of 0.821× at 256 rocks (the
//! cache and the camera pool, common to both, dominate there).
//!
//! `cargo run --release -p room3d --bin m14_gate4`
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
use room3d::rocks::Rocks;

const W: u32 = 1280;
const H: u32 = 720;
const FOV: f32 = 50.0;

#[derive(Clone, Copy)]
struct Size {
    side: usize,
    flight: usize,
    budget: u32,
    cache: u64,
    every: usize,
}

const GATE: Size = Size {
    side: 64,
    flight: 1800,
    budget: 8 << 20,
    cache: 2 << 30,
    every: 60,
};
const DEV: Size = Size {
    side: 16,
    flight: 300,
    budget: 3 << 20,
    cache: 256 << 20,
    every: 30,
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
/// level, for 16 frames each (at most `cap` frames). Frames rendered.
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
        let lit_quiet = waiting == 0 && lit.last().is_none_or(|s| s.loads == 0);
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

/// Pages of `ours` (only those shown, if `shown_only`) also in `theirs`:
/// how many, how many differ beyond 1e-5 (relative, absolute below 1),
/// and the largest difference.
fn compare(ours: &[CachePage], theirs: &[CachePage], shown_only: bool) -> (usize, usize, f32) {
    let theirs: HashMap<(u32, u32, u32, [u32; 2]), &Vec<f32>> = theirs
        .iter()
        .filter(|p| p.shown)
        .map(|p| ((p.object, p.chart, p.mip, p.page), &p.texels))
        .collect();
    let (mut common, mut differing, mut worst) = (0, 0, 0.0f32);
    for p in ours.iter().filter(|p| p.shown || !shown_only) {
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
    let rocks = Rocks::with_proxies(size.side, 6.0);
    let half = rocks.half as f32;
    let lighting: u64 = rocks.scene.geometries[1..]
        .iter()
        .map(|g| g.triangles.len() as u64)
        .sum();
    let lit_budget = (lighting / 8) as u32;
    say(format!(
        "scene: {} unique rocks with proxies over a {:.0} m square, built in {:.1} s; lighting levels {} triangles, \
         lighting budget {} (an eighth); camera budget {}",
        size.side * size.side,
        2.0 * half,
        built.elapsed().as_secs_f64(),
        lighting,
        lit_budget,
        size.budget
    ));
    let lights = [rocks.sun];
    let options = StreamOptions {
        budget_triangles: size.budget,
        cpu_cache_bytes: size.cache,
        ..Default::default()
    };
    let make = |scene: &InstancedScene, lighting_triangles: u32| {
        InstancedRenderer::with_streaming(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            scene,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            StreamOptions {
                lighting_triangles,
                ..options
            },
        )
    };

    // 3. Memory at creation: this renderer, then gate 3's way (no proxies,
    // every lighting level resident with its structure), both alive.
    let before = gpu_bytes(&device);
    let mut r = make(&rocks.scene, lit_budget);
    r.wait();
    let ours = gpu_bytes(&device) - before;
    let gate3_bytes = {
        let mut plain = rocks.scene.clone();
        for l in plain.geometry_levels.iter_mut().flatten() {
            l.proxy = None;
        }
        let mid = gpu_bytes(&device);
        let g = make(&plain, 0);
        g.wait();
        gpu_bytes(&device) - mid
    };
    let memory_ratio = ours as f64 / gate3_bytes as f64;
    let memory_pass = memory_ratio <= 0.5;

    // 4. The frame budget, first: each flight settled at its first view,
    // then flown as fast as it renders.
    let camera = |k: usize, f: usize| {
        room3d::rocks::flight(k, f as f32 / (size.flight - 1) as f32, half, FOV)
    };
    let mut budget_pass = true;
    for k in 0..3 {
        let warm = settle(&mut r, camera(k, 0), &target, 8000);
        let skip = r.lighting_log().0.len();
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
        say(format!(
            "4. flight {k}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms (at most 16.7, 16.7, 33.3): {} (settled first in \
             {warm} frames); lighting levels: {} loads, {} evictions, {} refused, at most {} pinned",
            ms(mean),
            ms(p99),
            ms(worst),
            if ok { "PASS" } else { "FAIL" },
            lit.iter().map(|s| s.loads).sum::<u32>(),
            lit.iter().map(|s| s.evictions).sum::<u32>(),
            lit.iter().map(|s| s.refused).sum::<u32>(),
            lit.iter().map(|s| s.pinned).max().unwrap_or(0),
        ));
    }

    // 2. Certified: replaying each flight beside a renderer with every
    // lighting level resident, every page shown here equals its page
    // there (every `every` frames), and settled at the last view, every
    // page both hold.
    let mut reference = make(&rocks.scene, 0);
    let mut certified = true;
    for k in 0..3 {
        settle(&mut r, camera(k, 0), &target, 8000);
        settle(&mut reference, camera(k, 0), &target, 8000);
        let (mut common, mut differing, mut worst, mut hidden) = (0, 0, 0.0f32, 0);
        for f in 0..size.flight {
            r.frame(camera(k, f), &target);
            reference.frame(camera(k, f), &target);
            if f % size.every == 0 {
                let ours = r.read_cache_pages(0);
                hidden += ours.iter().filter(|p| !p.shown).count();
                let (c, d, w) = compare(&ours, &reference.read_cache_pages(0), true);
                common += c;
                differing += d;
                worst = worst.max(w);
            }
        }
        let frames = settle(&mut r, camera(k, size.flight - 1), &target, 8000);
        settle(&mut reference, camera(k, size.flight - 1), &target, 8000);
        let ours = r.read_cache_pages(0);
        let theirs = reference.read_cache_pages(0);
        let (sc, sd, sw) = compare(&ours, &theirs, false);
        let ok = differing == 0 && sd == 0;
        certified &= ok;
        say(format!(
            "2. flight {k}: in flight {common} shown pages compared, {differing} differing (worst {worst:.2e}), {hidden} \
             hidden waiting; settled (in {frames} frames) {sc} common pages of {} here and {} there, {sd} differing \
             (worst {sw:.2e}): {}",
            ours.len(),
            theirs.len(),
            if ok { "PASS" } else { "FAIL" }
        ));
    }
    drop(reference);

    let (lit, _) = r.lighting_log();
    let capacity = lit.first().map_or(0, |s| s.capacity_triangles);
    let over = lit
        .iter()
        .filter(|s| s.resident_triangles > lit_budget as u64)
        .count();
    let scale_ratio = lighting as f64 / lit_budget as f64;
    let scale_pass = scale_ratio >= 8.0 && over == 0 && capacity <= lit_budget as u64;
    say(format!(
        "1. scale: lighting levels {:.2}× the lighting budget (slots hold {capacity} triangles); {over} of {} frames over: {}",
        scale_ratio,
        lit.len(),
        if scale_pass { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "3. memory at creation: {:.0} MB against gate 3's way {:.0} MB: {:.3}× (at most 0.5×): {}",
        ours as f64 / 1e6,
        gate3_bytes as f64 / 1e6,
        memory_ratio,
        if memory_pass { "PASS" } else { "FAIL" }
    ));
    let pass = scale_pass && certified && memory_pass && budget_pass;
    say(format!(
        "M14 gate 4: {} (scale {}, certified {}, memory {}, frame budget {})",
        if pass { "PASS" } else { "FAIL" },
        if scale_pass { "pass" } else { "fail" },
        if certified { "pass" } else { "fail" },
        if memory_pass { "pass" } else { "fail" },
        if budget_pass { "pass" } else { "fail" },
    ));
    let name = if dev {
        "m14_gate4_dev.txt"
    } else {
        "m14_gate4.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
