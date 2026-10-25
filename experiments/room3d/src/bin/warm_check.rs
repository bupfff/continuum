//! Quick check (not a gate): the million-boulder field's sun-only cache warmed
//! at the standing viewpoint, as M4 gate 11's hardware row, with the current
//! build. Prints the ray mode, frames until warm and the warm frame time.
//! `--sky` adds the field's sky, `--lights` uses three colored lights;
//! `--flight` then flies gate 8's path once.

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;

fn main() {
    let sky = std::env::args().any(|a| a == "--sky");
    let kernel = MaskConeKernel::new_with(continuum_gpu::KernelOptions {
        count_work: std::env::var("COUNT").is_ok(),
        without_packets: std::env::var("SOLO").is_ok(),
        ..Default::default()
    })
    .expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    kernel.set_sky(if sky { field.sky() } else { Vec3::ZERO });
    // `--lights`: the field's three colored lights (M5 gate 6) instead of
    // the sun.
    let lights = if std::env::args().any(|a| a == "--lights") {
        let all = field.colored_lights(origin, 1000.0);
        match std::env::var("LIGHTS") {
            Ok(v) => v
                .split(',')
                .filter_map(|k| k.parse::<usize>().ok())
                .map(|k| all[k])
                .collect(),
            Err(_) => all.to_vec(),
        }
    } else {
        vec![continuum_gpu::Light::white(light, field.radiance)]
    };
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
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        1280,
        720,
        1,
        &lights,
        &field.scene,
        origin,
        CacheOptions {
            pages: std::env::var("PAGES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1 << 19),
            ..Default::default()
        },
    );
    println!("hardware rays: {}", r.hardware_rays());
    let start = Instant::now();
    let mut frames = 0;
    loop {
        r.frame(camera, &target);
        frames += 1;
        let latest = r.cache_stats().unwrap().frame as usize;
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            break;
        }
    }
    r.finish();
    println!(
        "warm after {frames} frames, {:.1} s",
        start.elapsed().as_secs_f64()
    );
    let budget: Vec<u32> = r.cache_log().iter().map(|s| s.budget).collect();
    println!(
        "budget first/mid/last: {} {} {}",
        budget[0],
        budget[budget.len() / 2],
        budget[budget.len() - 1]
    );
    let t = Instant::now();
    for _ in 0..60 {
        r.frame(camera, &target);
    }
    r.finish();
    println!(
        "warm frame {:.2} ms",
        t.elapsed().as_secs_f64() * 1e3 / 60.0
    );
    let p = r.frame_profiled(camera, &target);
    println!("profile: {p:?}");
    if std::env::args().any(|a| a == "--flight") {
        drop(r);
        // Gate 8's flight from an empty cache, once.
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            1280,
            720,
            1,
            &lights,
            &field.scene,
            origin,
            CacheOptions {
                pages: std::env::var("PAGES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1 << 19),
                ..Default::default()
            },
        );
        let forward = (camera.target - camera.eye).normalize();
        let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
        let mut times = Vec::new();
        let start = Instant::now();
        for f in 0..600 {
            let t = f as f32 / 599.0;
            let d = (45.0 * t).to_radians();
            let eye = Vec3::new(camera.eye.x, camera.eye.y, 1.7 + 148.3 * t) + flat * 300.0 * t;
            let view = continuum_gpu::Camera {
                eye,
                target: eye + flat * d.cos() - Vec3::Z * d.sin(),
                vertical_fov_degrees: camera.vertical_fov_degrees,
            };
            let call = Instant::now();
            r.frame(view, &target);
            times.push(call.elapsed().as_secs_f64() * 1e3);
        }
        r.finish();
        let mean = start.elapsed().as_secs_f64() * 1e3 / 600.0;
        if std::env::var("SLOW").is_ok() {
            let log = r.cache_log();
            let mut order: Vec<usize> = (0..600).collect();
            order.sort_by(|&a, &b| times[b].total_cmp(&times[a]));
            for &f in order.iter().take(8) {
                let s = log[f];
                println!(
                    "  slow frame {f}: {:.2} ms, planning {:.2} ms, waiting {:.2} ms, GPU bake {:.2} ms, camera {:.2} ms, spent {}, pages {}, sky samples {}",
                    times[f],
                    s.planning.as_secs_f64() * 1e3,
                    s.waiting.as_secs_f64() * 1e3,
                    s.gpu_bake_ms,
                    s.gpu_render_ms,
                    s.spent,
                    s.pages_baked,
                    s.sky_samples
                );
            }
        }
        times.sort_by(f64::total_cmp);
        let log = r.cache_log();
        let fallback = log
            .iter()
            .map(|s| s.fallback_pixels as f64 / (1280.0 * 720.0))
            .sum::<f64>()
            / log.len() as f64;
        let samples: u64 = log.iter().map(|s| s.sky_samples as u64).sum();
        if std::env::var("FRAMES").is_ok() {
            let from: usize = std::env::var("FRAMES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(480);
            for s in &log[from..(from + 120).min(600)] {
                println!(
                    "    frame {}: spent {}, pages {}, texels {}, sky samples {}, GPU bake {:.2} ms, camera {:.2} ms, work {:?}",
                    s.frame,
                    s.spent,
                    s.pages_baked,
                    s.texels_baked,
                    s.sky_samples,
                    s.gpu_bake_ms,
                    s.gpu_render_ms,
                    s.work
                );
            }
        }
        for (k, chunk) in log.chunks(100).enumerate() {
            let fb = chunk
                .iter()
                .map(|s| s.fallback_pixels as f64 / (1280.0 * 720.0))
                .sum::<f64>()
                / chunk.len() as f64;
            let n = chunk.len() as f64;
            let budget = chunk.iter().map(|s| s.budget as f64).sum::<f64>() / n;
            let bake = chunk.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n;
            let render = chunk.iter().map(|s| s.gpu_render_ms as f64).sum::<f64>() / n;
            println!(
                "  frames {}–{}: fallback {:.2}%, budget {budget:.0}, GPU bake {bake:.2} ms, camera passes {render:.2} ms",
                100 * k,
                100 * k + chunk.len() - 1,
                100.0 * fb
            );
        }
        println!(
            "flight: mean {mean:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, sky samples {samples}",
            times[593],
            times[599],
            100.0 * fallback
        );
    }
}
