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
    if std::env::var_os("MEMORY").is_some() {
        // GPU memory against the number of unique rocks (on demand, a small
        // camera budget): what each resident lighting level costs.
        let kernel = MaskConeKernel::new().expect("a GPU");
        kernel.set_prefilter(1.0);
        kernel.set_hardware_rays(true);
        let device = kernel.device().clone();
        for side in [16usize, 32, 64] {
            let rocks = Rocks::on_demand(side, 6.0);
            let lit: usize = rocks.scene.geometries.iter().map(|g| g.triangles.len()).sum();
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
    let rocks = Rocks::with_detail(side, 6.0, detail);
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
        for f in 0..frames {
            r.frame(
                flight(k, f as f32 / (frames - 1) as f32, half, 50.0),
                &target,
            );
        }
        r.finish();
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
