//! Development tool for M14 (not a gate). Flies gate 1's flights over the
//! rock field with a streamed renderer and reports, per flight, the
//! working set (the levels each view wanted) and why levels loaded and
//! were evicted. With no budget given, the budget holds every finest level
//! (sizing a budget); with one, the policy runs as in a gate.
//!
//! `cargo run --release -p room3d --bin stream_dev -- [side] [amplitude] [falloff] [frames] [budget] [flights]`

use continuum::glam::DVec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, StreamOptions, wgpu};
use room3d::rocks::{Detail, Rocks, flight};

/// Bytes the GPU device has allocated (Metal's `currentAllocatedSize`).
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Ok(side) = std::env::var("SPIKES") {
        // Gate 6's spike (M14 gate 7): its field, options and three flights
        // in order (each settled at its first view, then 1,800 frames), each
        // frame's wall time and parts; the slowest frames of each flight.
        let side: usize = side.parse().unwrap_or(192);
        let kernel = MaskConeKernel::new().expect("a GPU");
        kernel.set_prefilter(1.0);
        kernel.set_sky_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_sky(continuum::glam::Vec3::ZERO);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = kernel
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1280,
                    height: 720,
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
        let rocks = Rocks::lighting_on_demand(side, 6.0);
        let mut r = InstancedRenderer::with_streaming(
            &kernel,
            format,
            1280,
            720,
            1,
            &[Rocks::sun()],
            &rocks.scene,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            StreamOptions {
                budget_triangles: 8 << 20,
                cpu_cache_bytes: 2 << 30,
                lighting_triangles: 491_520,
                ..Default::default()
            },
        );
        r.wait();
        let half = if side >= 64 { 192.0 } else { side as f32 * 1.5 };
        for k in 0..3 {
            let camera = |f: usize| flight(k, f as f32 / 1799.0, half, 50.0);
            for _ in 0..600 {
                r.frame(camera(0), &target);
            }
            r.finish();
            let (c0, s0, l0) = (
                r.cache_log().len(),
                r.stream_log().len(),
                r.lighting_log().0.len(),
            );
            let mut times = Vec::with_capacity(1800);
            for f in 0..1800 {
                let call = std::time::Instant::now();
                r.frame(camera(f), &target);
                times.push(call.elapsed().as_secs_f64() * 1e3);
            }
            r.finish();
            let cache = r.cache_log()[c0..].to_vec();
            let stream = r.stream_log()[s0..].to_vec();
            let lit = r.lighting_log().0[l0..].to_vec();
            let mut order: Vec<usize> = (0..times.len()).collect();
            order.sort_by(|&a, &b| times[b].total_cmp(&times[a]));
            let mut sorted = times.clone();
            sorted.sort_by(f64::total_cmp);
            println!(
                "flight {k}: mean {:.2} ms, p99 {:.2}, worst {:.2}; frames over 16.7 ms: {}, over 33.3 ms: {}",
                times.iter().sum::<f64>() / times.len() as f64,
                sorted[sorted.len() * 99 / 100],
                sorted[sorted.len() - 1],
                times.iter().filter(|&&t| t > 16.7).count(),
                times.iter().filter(|&&t| t > 33.3).count(),
            );
            // The cache's and the lighting pool's logs are a frame or two
            // behind the call; the streamer's is the call's own.
            for &f in order.iter().take(8) {
                let st = stream.get(f).copied().unwrap_or_default();
                let li = lit.get(f).copied().unwrap_or_default();
                let ca = cache.get(f);
                println!(
                    "  frame {f}: {:.2} ms; camera streaming {:.2} ms ({:.2}/{:.2}/{:.2}/{:.2}), {} loads {} evictions, \
                     {} produced; lighting {:.2} ms, {} loads, {} produced; cache planning {:.2} ms, waiting {:.2} ms, \
                     {} pages baked",
                    times[f],
                    st.cpu_ms,
                    st.stage_ms[0],
                    st.stage_ms[1],
                    st.stage_ms[2],
                    st.stage_ms[3],
                    st.loads,
                    st.evictions,
                    st.produced,
                    li.cpu_ms,
                    li.loads,
                    li.produced,
                    ca.map_or(0.0, |c| c.planning.as_secs_f64() * 1e3),
                    ca.map_or(0.0, |c| c.waiting.as_secs_f64() * 1e3),
                    ca.map_or(0, |c| c.pages_baked),
                );
            }
        }
        return;
    }
    if let Ok(side) = std::env::var("GATE5") {
        // Where gate 5's frame time goes (M14): its field and options,
        // flights 0 and 1 (900 frames each, settled first) with the gate's
        // lighting slots, 4× and 16× them; per frame, the wall time and its
        // parts.
        let side: usize = side.parse().unwrap_or(192);
        let kernel = MaskConeKernel::new().expect("a GPU");
        kernel.set_prefilter(1.0);
        kernel.set_sky_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_sky(continuum::glam::Vec3::ZERO);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = kernel
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1280,
                    height: 720,
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
        let built = std::time::Instant::now();
        let rocks = Rocks::lighting_on_demand(side, 6.0);
        println!(
            "{} rocks built in {:.0} s",
            side * side,
            built.elapsed().as_secs_f64()
        );
        let frames = 900usize;
        let slot_list: Vec<u32> = std::env::var("GATE5_SLOTS")
            .ok()
            .map_or(vec![491_520, 4 * 491_520, 16 * 491_520], |v| {
                v.split(',').filter_map(|x| x.parse().ok()).collect()
            });
        for slots in slot_list {
            let mut r = InstancedRenderer::with_streaming(
                &kernel,
                format,
                1280,
                720,
                1,
                &[Rocks::sun()],
                &rocks.scene,
                DVec3::ZERO,
                Some(CacheOptions::default()),
                StreamOptions {
                    budget_triangles: 8 << 20,
                    cpu_cache_bytes: 2 << 30,
                    lighting_triangles: slots,
                    ..Default::default()
                },
            );
            for k in 0..2 {
                let camera = |f: usize| flight(k, f as f32 / 1799.0, 192.0, 50.0);
                for _ in 0..600 {
                    r.frame(camera(0), &target);
                }
                r.finish();
                let (c0, s0, l0) = (
                    r.cache_log().len(),
                    r.stream_log().len(),
                    r.lighting_log().0.len(),
                );
                let begin = std::time::Instant::now();
                for f in 0..frames {
                    r.frame(camera(f), &target);
                }
                r.finish();
                let wall = begin.elapsed().as_secs_f64() * 1000.0 / frames as f64;
                let cache = &r.cache_log()[c0..];
                let stream = &r.stream_log()[s0..];
                let lit = &r.lighting_log().0[l0..];
                let mean =
                    |v: &mut dyn Iterator<Item = f64>, n: usize| v.sum::<f64>() / n.max(1) as f64;
                let timed: Vec<_> = cache.iter().filter(|c| c.timed).collect();
                println!(
                    "slots {slots}, flight {k}: wall {wall:.2} ms; cache planning {:.2} ms, waiting {:.2} ms, GPU bake {:.2} \
                     ms, GPU camera {:.2} ms, pages baked {:.0}, requests {:.0}; camera streaming CPU {:.2} ms, loads \
                     {:.2}, wanted {:.2} M triangles, pixel error {:.2} (most {:.2}); lighting CPU {:.2} ms, loads {:.1}, \
                     refused {:.0}, waiting production {:.1}; pages waiting {}; streaming stages {:.2} / {:.2} / {:.2} / {:.2} \
                     ms",
                    mean(
                        &mut cache.iter().map(|c| c.planning.as_secs_f64() * 1000.0),
                        cache.len()
                    ),
                    mean(
                        &mut cache.iter().map(|c| c.waiting.as_secs_f64() * 1000.0),
                        cache.len()
                    ),
                    mean(&mut timed.iter().map(|c| c.gpu_bake_ms as f64), timed.len()),
                    mean(
                        &mut timed.iter().map(|c| c.gpu_render_ms as f64),
                        timed.len()
                    ),
                    mean(&mut cache.iter().map(|c| c.pages_baked as f64), cache.len()),
                    mean(&mut cache.iter().map(|c| c.requests as f64), cache.len()),
                    mean(&mut stream.iter().map(|s| s.cpu_ms as f64), stream.len()),
                    mean(&mut stream.iter().map(|s| s.loads as f64), stream.len()),
                    mean(
                        &mut stream.iter().map(|s| s.wanted_triangles as f64 / 1e6),
                        stream.len()
                    ),
                    mean(
                        &mut stream.iter().map(|s| s.pixel_error as f64),
                        stream.len()
                    ),
                    stream.iter().map(|s| s.pixel_error).fold(0.0f32, f32::max),
                    mean(&mut lit.iter().map(|s| s.cpu_ms as f64), lit.len()),
                    mean(&mut lit.iter().map(|s| s.loads as f64), lit.len()),
                    mean(&mut lit.iter().map(|s| s.refused as f64), lit.len()),
                    mean(
                        &mut lit.iter().map(|s| s.waiting_production as f64),
                        lit.len()
                    ),
                    r.lighting_log().1,
                    mean(
                        &mut stream.iter().map(|s| s.stage_ms[0] as f64),
                        stream.len()
                    ),
                    mean(
                        &mut stream.iter().map(|s| s.stage_ms[1] as f64),
                        stream.len()
                    ),
                    mean(
                        &mut stream.iter().map(|s| s.stage_ms[2] as f64),
                        stream.len()
                    ),
                    mean(
                        &mut stream.iter().map(|s| s.stage_ms[3] as f64),
                        stream.len()
                    ),
                );
            }
        }
        return;
    }
    if std::env::var_os("RECORDS").is_some() {
        // GPU memory per unique rock at creation, with proxies and lighting
        // levels streamed into fixed slots (gate 4's way) against every
        // lighting level resident: what a rock's resident record costs.
        let kernel = MaskConeKernel::new().expect("a GPU");
        kernel.set_prefilter(1.0);
        kernel.set_hardware_rays(true);
        kernel.set_sky(continuum::glam::Vec3::ZERO);
        let device = kernel.device().clone();
        let sides: Vec<usize> = std::env::var("RECORDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(vec![16, 32, 64], |s| vec![64, s]);
        for side in sides {
            let rocks = Rocks::with_proxies(side, 6.0);
            for slots in [491_520u32, 0]
                .into_iter()
                .take(if side > 64 { 1 } else { 2 })
            {
                let before = gpu_bytes(&device);
                let r = InstancedRenderer::with_streaming(
                    &kernel,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    1280,
                    720,
                    1,
                    &[rocks.sun],
                    &rocks.scene,
                    DVec3::ZERO,
                    Some(CacheOptions::default()),
                    StreamOptions {
                        budget_triangles: 8 << 20,
                        lighting_triangles: slots,
                        ..Default::default()
                    },
                );
                r.wait();
                let bytes = gpu_bytes(&device) - before;
                println!(
                    "{} rocks, lighting slots {slots}: {:.1} MB",
                    side * side,
                    bytes as f64 / 1e6
                );
            }
        }
        return;
    }
    if std::env::var_os("MEMORY").is_some() {
        // GPU memory against the number of unique rocks (on demand, a small
        // camera budget): what each resident lighting level costs.
        let kernel = MaskConeKernel::new().expect("a GPU");
        kernel.set_prefilter(1.0);
        kernel.set_hardware_rays(true);
        let device = kernel.device().clone();
        for side in [16usize, 32, 64] {
            let rocks = Rocks::on_demand(side, 6.0);
            let lit: usize = rocks
                .scene
                .geometries
                .iter()
                .map(|g| g.triangles.len())
                .sum();
            for cache in [false, true] {
                let before = gpu_bytes(&device);
                let r = InstancedRenderer::with_streaming(
                    &kernel,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    1280,
                    720,
                    1,
                    &[rocks.sun],
                    &rocks.scene,
                    DVec3::ZERO,
                    cache.then(CacheOptions::default),
                    StreamOptions {
                        budget_triangles: 1 << 16,
                        ..Default::default()
                    },
                );
                r.wait();
                let bytes = gpu_bytes(&device) - before;
                println!(
                    "{} rocks, {lit} lighting triangles, cache {cache}: {:.1} MB ({:.0} B a lighting triangle; scene buffers {:.1} MB)",
                    side * side,
                    bytes as f64 / 1e6,
                    bytes as f64 / lit as f64,
                    r.scene_bytes() as f64 / 1e6
                );
            }
        }
        return;
    }
    let arg = |i: usize, default: f64| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default);
    let side = arg(1, 16.0) as usize;
    let detail = Detail {
        amplitude: arg(2, 0.16) as f32,
        falloff: arg(3, 1.8) as f32,
    };
    let frames = arg(4, 600.0) as usize;
    let budget = arg(5, 0.0) as u32;
    let flights = arg(6, 3.0) as usize;
    let kernel = MaskConeKernel::new().expect("a GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let (w, h) = (1280, 720);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
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
    let rocks = if std::env::var_os("PROXIES").is_some() {
        Rocks::with_proxies(side, 6.0)
    } else {
        Rocks::with_detail(side, 6.0, detail)
    };
    let half = rocks.half as f32;
    let finest = rocks.finest_triangles();
    let n = rocks.scene.geometry_levels.iter().flatten().count() as f32;
    let mean = |f: &dyn Fn(&continuum_gpu::Levels) -> f32| {
        rocks
            .scene
            .geometry_levels
            .iter()
            .flatten()
            .map(f)
            .sum::<f32>()
            / n
    };
    println!(
        "{side}×{side} rocks, amplitude {}, falloff {}: finest {finest} triangles; mean errors {:.1} / {:.1} / {:.1} mm",
        detail.amplitude,
        detail.falloff,
        1e3 * mean(&|l| l.base_error),
        1e3 * mean(&|l| l.finer[0].error),
        1e3 * mean(&|l| l.finer[1].error),
    );
    let budget = if budget == 0 {
        finest as u32 + (1 << 20)
    } else {
        budget
    };
    let mut r = InstancedRenderer::with_streaming(
        &kernel,
        format,
        w,
        h,
        1,
        &[rocks.sun],
        &rocks.scene,
        DVec3::ZERO,
        Some(CacheOptions::default()),
        StreamOptions {
            budget_triangles: budget,
            lighting_triangles: std::env::var("LIGHTING")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            ..Default::default()
        },
    );
    let sun = rocks.sun;
    let scene_for_baseline = rocks.scene;
    if std::env::var_os("BASELINE").is_some() {
        // Streaming's fixed cost: a still view, everything it wants loaded,
        // against the fully resident renderer, five interleaved rounds.
        let mut resident = InstancedRenderer::with_streaming(
            &kernel,
            format,
            w,
            h,
            1,
            &[sun],
            &scene_for_baseline,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            StreamOptions {
                resident: true,
                ..Default::default()
            },
        );
        let view = flight(0, 1.0, half, 50.0);
        for r in [&mut r, &mut resident] {
            for _ in 0..300 {
                r.frame(view, &target);
            }
            r.finish();
        }
        let mut sums = [0.0f64; 2];
        for _ in 0..5 {
            for (j, r) in [&mut resident, &mut r].into_iter().enumerate() {
                let t = std::time::Instant::now();
                for _ in 0..120 {
                    r.frame(view, &target);
                }
                r.finish();
                sums[j] += t.elapsed().as_secs_f64() * 1e3 / 120.0;
            }
        }
        println!(
            "baseline: resident {:.3} ms, streamed {:.3} ms a frame ({:.3}×)",
            sums[0] / 5.0,
            sums[1] / 5.0,
            sums[1] / sums[0]
        );
        return;
    }
    if std::env::var_os("STILL").is_some() {
        // A still view: every 50 frames, what the cache and the lighting
        // pool did.
        let view = flight(0, 1.0, half, 50.0);
        for f in 1..=600 {
            r.frame(view, &target);
            if f % 50 == 0 {
                r.finish();
                let c = r.cache_log();
                let last = &c[c.len().saturating_sub(50)..];
                let (lit, waiting) = r.lighting_log();
                let l = &lit[lit.len().saturating_sub(50)..];
                println!(
                    "frame {f}: fallback {} at wanted {}; requests {}, baked {}, resident pages {}; lighting loads {} evictions {} refused {} pinned (last) {}; waiting {waiting}",
                    last.last().map_or(0, |s| s.fallback_pixels),
                    last.last().map_or(0, |s| s.at_wanted),
                    last.iter().map(|s| s.requests).sum::<u32>(),
                    last.iter().map(|s| s.pages_baked).sum::<u32>(),
                    last.last().map_or(0, |s| s.resident),
                    l.iter().map(|s| s.loads).sum::<u32>(),
                    l.iter().map(|s| s.evictions).sum::<u32>(),
                    l.iter().map(|s| s.refused).sum::<u32>(),
                    l.last().map_or(0, |s| s.pinned),
                );
            }
        }
        return;
    }
    if std::env::var_os("PAGES").is_some() {
        // Pages resident in both the streamed renderer above and one with
        // every lighting level resident, at a settled view, compared.
        let mut reference = InstancedRenderer::with_streaming(
            &kernel,
            format,
            w,
            h,
            1,
            &[sun],
            &scene_for_baseline,
            DVec3::ZERO,
            Some(CacheOptions::default()),
            StreamOptions {
                budget_triangles: budget,
                ..Default::default()
            },
        );
        let view = flight(0, 1.0, half, 50.0);
        for x in [&mut r, &mut reference] {
            for _ in 0..600 {
                x.frame(view, &target);
            }
            x.finish();
        }
        let ours = r.read_cache_pages(0);
        let theirs: std::collections::HashMap<(u32, u32, u32, [u32; 2]), Vec<f32>> = reference
            .read_cache_pages(0)
            .into_iter()
            .map(|p| ((p.object, p.chart, p.mip, p.page), p.texels))
            .collect();
        let (mut common, mut differing, mut worst) = (0, 0, 0.0f32);
        let mut by_object: std::collections::BTreeMap<u32, usize> = Default::default();
        for p in ours.iter().filter(|p| p.shown) {
            if let Some(t) = theirs.get(&(p.object, p.chart, p.mip, p.page)) {
                common += 1;
                let d = p
                    .texels
                    .iter()
                    .zip(t)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if d > 1e-5 {
                    differing += 1;
                    *by_object.entry(p.object).or_default() += 1;
                }
                worst = worst.max(d);
            }
        }
        println!(
            "pages: ours {}, reference {}, common {common}, differing {differing}, worst {worst}; by object {:?}",
            ours.len(),
            theirs.len(),
            by_object.iter().take(12).collect::<Vec<_>>()
        );
        return;
    }
    if let Ok(path) = std::env::var("SNAPSHOT") {
        // A view settled (or not, with SNAPSHOT_RAW), saved as a PNG.
        let view = flight(0, 1.0, half, 50.0);
        let frames = if std::env::var_os("SNAPSHOT_RAW").is_some() {
            1
        } else {
            400
        };
        for _ in 0..frames {
            r.frame(view, &target);
        }
        r.finish();
        let px = r.read_colors();
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                room3d::to_rgb(continuum::glam::Vec3::from_array(*p)),
            );
        }
        image.save(std::path::Path::new(&path)).expect("saved");
        println!("saved {path}; {:?}", r.stream_log().last());
        let (lit, waiting) = r.lighting_log();
        println!(
            "lighting: {:?}; waiting pages {waiting}; over the run {} loads, {} evictions, {} refused, most pinned {}",
            lit.last(),
            lit.iter().map(|s| s.loads).sum::<u32>(),
            lit.iter().map(|s| s.evictions).sum::<u32>(),
            lit.iter().map(|s| s.refused).sum::<u32>(),
            lit.iter().map(|s| s.pinned).max().unwrap_or(0)
        );
        return;
    }
    if std::env::var_os("LOAD_COST").is_some() {
        // What a load costs: settle at flight 0's last view, then five times
        // evict everything and render, waiting for each frame, until quiet.
        let view = flight(0, 1.0, half, 50.0);
        for _ in 0..200 {
            r.frame(view, &target);
        }
        r.finish();
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let (mut quiet, mut quiet_n) = (0.0f64, 0usize);
        let (mut busy, mut loads, mut tris, mut cpu, mut frames) =
            (0.0f64, 0u64, 0u64, 0.0f64, 0usize);
        for _ in 0..5 {
            r.stream_evict_all();
            let mut calm = 0;
            while calm < 30 {
                let t = std::time::Instant::now();
                r.frame(view, &target);
                r.finish();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let s = *r.stream_log().last().unwrap();
                if s.loads == 0 {
                    calm += 1;
                    quiet += ms;
                    quiet_n += 1;
                } else {
                    calm = 0;
                    busy += ms;
                    frames += 1;
                    loads += s.loads as u64;
                    tris += s.loaded_triangles;
                    cpu += s.cpu_ms as f64;
                    rows.push((s.loads as f64, s.loaded_triangles as f64, ms));
                }
            }
        }
        let base = quiet / quiet_n as f64;
        let extra = busy - base * frames as f64;
        // Least squares: extra time = a · loads + b · triangles.
        let (mut sll, mut slt, mut stt, mut sly, mut sty) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(l, t, ms) in &rows {
            let y = ms - base;
            let t = t / 1e6;
            sll += l * l;
            slt += l * t;
            stt += t * t;
            sly += l * y;
            sty += t * y;
        }
        let det = sll * stt - slt * slt;
        let a = (sly * stt - sty * slt) / det;
        let b = (sll * sty - slt * sly) / det;
        println!("fit: {a:.3} ms per load plus {b:.1} ms per million triangles");
        println!(
            "load cost: quiet frame {base:.2} ms; {frames} frames with {loads} loads ({tris} triangles) took {extra:.1} ms more:              {:.3} ms a load, {:.1} ns a triangle; streaming CPU {:.1} ms of it",
            extra / loads as f64,
            1e6 * extra / tris as f64,
            cpu
        );
        return;
    }
    for k in 0..flights {
        let skip = r.stream_log().len();
        let cskip = r.cache_log().len();
        let lskip = r.lighting_log().0.len();
        let begin = std::time::Instant::now();
        for f in 0..frames {
            r.frame(
                flight(k, f as f32 / (frames - 1) as f32, half, 50.0),
                &target,
            );
        }
        r.finish();
        let mean = begin.elapsed().as_secs_f64() * 1e3 / frames as f64;
        let c = &r.cache_log()[cskip..];
        let n = c.len().max(1) as f64;
        let lit = &r.lighting_log().0[lskip..];
        println!(
            "flight {k}: {mean:.2} ms a frame; GPU bake {:.2} ms, camera {:.2} ms, CPU planning {:.2} ms; pages baked {:.1} a frame, \
             budget {:.0}; lighting loads {} evictions {} refused {}; waiting pages now {}",
            c.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n,
            c.iter().map(|s| s.gpu_render_ms as f64).sum::<f64>() / n,
            c.iter()
                .map(|s| s.planning.as_secs_f64() * 1e3)
                .sum::<f64>()
                / n,
            c.iter().map(|s| s.pages_baked as f64).sum::<f64>() / n,
            c.iter().map(|s| s.budget as f64).sum::<f64>() / n,
            lit.iter().map(|s| s.loads).sum::<u32>(),
            lit.iter().map(|s| s.evictions).sum::<u32>(),
            lit.iter().map(|s| s.refused).sum::<u32>(),
            r.lighting_log().1,
        );
        let log = &r.stream_log()[skip..];
        let sum = |f: &dyn Fn(&continuum_gpu::StreamStats) -> u64| log.iter().map(f).sum::<u64>();
        let most = log.iter().map(|s| s.wanted_triangles).max().unwrap_or(0);
        println!(
            "flight {k}: {} triangles loaded again within 600 frames",
            sum(&|s| s.repeat_triangles)
        );
        println!(
            "flight {k}: working set at most {most} triangles, {} on average; {} loads ({} triangles: {} upgrades, {} reloads within 120 frames), \
             {} evictions ({} seen within 30 frames, {} over-detailed)",
            sum(&|s| s.wanted_triangles) / log.len() as u64,
            sum(&|s| s.loads as u64),
            sum(&|s| s.loaded_triangles),
            sum(&|s| s.upgrades as u64),
            sum(&|s| s.reloads as u64),
            sum(&|s| s.evictions as u64),
            sum(&|s| s.evicted_recent as u64),
            sum(&|s| s.evicted_over as u64),
        );
    }
}
