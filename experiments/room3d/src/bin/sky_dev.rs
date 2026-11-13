//! Sky warm-up development (not a gate): M5 gate 9's scene and viewpoints,
//! the cache warmed from empty. Reports frames until warm and until within
//! 5% of the warm image, sky samples, and where the bake's GPU time goes
//! (passes timed one by one on every `PROFILE`th frame).
//!
//! `cargo run --release -p room3d --bin sky_dev -- [view 0-3] [fixed=N]
//! [boulders=N] [cones] [profile] [trace] [save=path] [compare=path]`

use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const EVERY: usize = 20;
const PROFILE: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let view_index: usize = args
        .iter()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(1);
    let arg = |name: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(name))
            .map(|v| v.parse::<u32>().unwrap())
    };
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let field = Field::new(arg("boulders=").unwrap_or(1_000_000) as usize, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
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
    let options = match arg("fixed=") {
        Some(n) => CacheOptions {
            bake_budget: n,
            frame_target_ms: 0.0,
            ..Default::default()
        },
        None => CacheOptions::default(),
    };
    let options = CacheOptions {
        sky_rays: !args.iter().any(|a| a == "cones"),
        refine_levels: arg("refine=").unwrap_or(3),
        ..options
    };
    let cached = || {
        InstancedRenderer::with_cache(
            &kernel,
            format,
            W,
            H,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            options,
        )
    };
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |ahead: f32, height: f32, down: f32| {
        let d = down.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(0.0, 1.7, 0.0)),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ];
    let (name, camera) = views[view_index.min(3)];
    let warm = |r: &mut InstancedRenderer, reference: Option<&[[f32; 3]]>, profile: bool| {
        let mut within = None;
        let mut passes: Vec<(&'static str, Duration)> = Vec::new();
        let mut profiled = 0usize;
        let mut frames = None;
        for k in 1..=60_000usize {
            if profile && k % PROFILE == 0 {
                let p = r.frame_profiled(camera, &target);
                profiled += 1;
                for &(label, d) in &p.bake_passes {
                    if label.is_empty() {
                        continue;
                    }
                    match passes.iter_mut().find(|e| e.0 == label) {
                        Some(e) => e.1 += d,
                        None => passes.push((label, d)),
                    }
                }
            } else {
                r.frame(camera, &target);
            }
            if let Some(reference) = reference
                && within.is_none()
                && k % EVERY == 0
            {
                r.finish();
                if relative_rmse(&r.read_colors(), reference) <= 0.05 {
                    within = Some(k);
                }
            }
            let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
            if latest >= 16
                && r.cache_log()[latest - 16..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0)
            {
                frames = Some(latest - 16);
                break;
            }
        }
        r.finish();
        (frames, within, r.read_colors(), passes, profiled)
    };
    let mut first = cached();
    let (frames, _, image, _, _) = warm(&mut first, None, false);
    if args.iter().any(|a| a == "trace1") {
        for s in first.cache_log().iter().step_by(25).take(30) {
            println!(
                "  first run frame {}: budget {}, cap {}, spent {}, sky {}, GPU bake {:.2} ms, render {:.2} ms, planning {:.2} ms, waiting {:.2} ms",
                s.frame,
                s.budget,
                s.budget_cap,
                s.spent,
                s.sky_samples,
                s.gpu_bake_ms,
                s.gpu_render_ms,
                s.planning.as_secs_f64() * 1e3,
                s.waiting.as_secs_f64() * 1e3
            );
        }
    }
    let path_arg = |name: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(name))
            .map(String::from)
    };
    if let Some(path) = path_arg("save=") {
        let bytes: Vec<u8> = image
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
    }
    if let Some(path) = path_arg("compare=") {
        let bytes = std::fs::read(&path).unwrap();
        let other: Vec<[f32; 3]> = bytes
            .as_chunks::<12>()
            .0
            .iter()
            .map(|c| {
                let f = |i: usize| f32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap());
                [f(0), f(1), f(2)]
            })
            .collect();
        println!(
            "warm image against {path}: {:.3}% (relative RMSE)",
            100.0 * relative_rmse(&image, &other)
        );
    }
    let mut r = cached();
    let profile = args.iter().any(|a| a == "profile");
    let (frames2, within, _, passes, profiled) = warm(&mut r, Some(&image), profile);
    let log = r.cache_log();
    let n = log.len().max(1) as f64;
    let sky: u64 = log.iter().map(|s| s.sky_samples as u64).sum();
    let texels: u64 = log.iter().map(|s| s.texels_baked as u64).sum();
    let bake: f64 = log.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n;
    let budget: f64 = log.iter().map(|s| s.budget as f64).sum::<f64>() / n;
    println!(
        "{name} ({}): warm in {frames:?} frames (second run {frames2:?}), within 5% at {within:?}; sky samples {sky}, sun texels {texels}; mean budget {budget:.0}, GPU bake {bake:.2} ms",
        if arg("fixed=").is_some() {
            "fixed budget"
        } else {
            "time budget"
        }
    );
    if args.iter().any(|a| a == "trace") {
        for s in log.iter().step_by(200).take(40) {
            println!(
                "  frame {}: budget {}, cap {}, spent {}, sky {}, GPU bake {:.2} ms, render {:.2} ms, planning {:.2} ms, waiting {:.2} ms",
                s.frame,
                s.budget,
                s.budget_cap,
                s.spent,
                s.sky_samples,
                s.gpu_bake_ms,
                s.gpu_render_ms,
                s.planning.as_secs_f64() * 1e3,
                s.waiting.as_secs_f64() * 1e3
            );
        }
    }
    for (label, d) in &passes {
        println!(
            "  pass {label}: {:.3} ms per profiled frame",
            d.as_secs_f64() * 1e3 / profiled.max(1) as f64
        );
    }
}
