//! Development tool for M14 (not a gate): sizes the rock field's working
//! set. Flies gate 1's three flights with a streamed renderer whose budget
//! holds every finest level, and reports the levels each view wanted (the
//! working set) and the level errors.
//!
//! `cargo run --release -p room3d --bin stream_dev -- [side] [amplitude] [falloff] [frames]`

use continuum::glam::DVec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, StreamOptions, wgpu};
use room3d::rocks::{Detail, Rocks, flight};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: f64| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default);
    let side = arg(1, 16.0) as usize;
    let detail = Detail {
        amplitude: arg(2, 0.16) as f32,
        falloff: arg(3, 1.2) as f32,
    };
    let frames = arg(4, 600.0) as usize;
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
            budget_triangles: finest as u32 + (1 << 20),
            ..Default::default()
        },
    );
    drop(rocks);
    for k in 0..3 {
        let skip = r.stream_log().len();
        for f in 0..frames {
            r.frame(
                flight(k, f as f32 / (frames - 1) as f32, half, 50.0),
                &target,
            );
        }
        r.finish();
        let log = &r.stream_log()[skip..];
        let most = log.iter().map(|s| s.wanted_triangles).max().unwrap_or(0);
        let avg = log.iter().map(|s| s.wanted_triangles).sum::<u64>() / log.len() as u64;
        let last = log.last().map_or(0, |s| s.wanted_triangles);
        println!(
            "flight {k}: working set at most {most} triangles, {avg} on average, {last} at the last view ({:.2}× / {:.2}× of finest)",
            finest as f64 / most.max(1) as f64,
            finest as f64 / avg.max(1) as f64,
        );
    }
}
