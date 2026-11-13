//! M9 development: where a frame's time goes with movers (not a gate).
//! `cargo run --release -p room3d --bin m9_dev -- <moving 0|1> <frames>`
//! [sun] [real: a real sun's speed] [nocap: no frame-period cap] [profile]
//! [720] [trace] [save]

use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let moving = args.get(1).is_some_and(|a| a == "1");
    let frames: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(200);
    let hw = !args.iter().any(|a| a == "compute");
    let cache = !args.iter().any(|a| a == "nocache");
    let (w, h) = if args.iter().any(|a| a == "720") {
        (1280, 720)
    } else {
        (640, 360)
    };
    let kernel = if args.iter().any(|a| a == "work") {
        MaskConeKernel::new_with(continuum_gpu::KernelOptions {
            count_work: true,
            ..Default::default()
        })
        .unwrap()
    } else {
        MaskConeKernel::new().unwrap()
    };
    kernel.set_prefilter(1.0);
    kernel.set_hardware_rays(hw);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, _) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0);
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let sun_mode = args.iter().any(|a| a == "sun");
    let real = args.iter().any(|a| a == "real");
    let options = CacheOptions {
        frame_budget_ms: if args.iter().any(|a| a == "nocap") {
            0.0
        } else {
            CacheOptions::default().frame_budget_ms
        },
        ..Default::default()
    };
    let options = match args.iter().find_map(|a| a.strip_prefix("budget=")) {
        Some(n) => CacheOptions {
            frame_target_ms: 0.0,
            bake_budget: n.parse().unwrap(),
            ..options
        },
        None => options,
    };
    let options = match args.iter().find_map(|a| a.strip_prefix("max=")) {
        Some(n) => CacheOptions {
            bake_budget: n.parse().unwrap(),
            ..options
        },
        None => options,
    };
    let movable: Vec<usize> = if sun_mode {
        Vec::new()
    } else {
        (1..=65).collect()
    };
    let center = (field.scene.instances[0].position - origin).as_vec3();
    let sun_at = |f: usize| {
        let mut l = lights[0];
        if let continuum_gpu::Emitter::Rect(mut rect) = l.emitter {
            let per_frame = if real { 1.0 / 14_400.0 } else { 1.0 / 240.0 };
            let r = Mat3::from_rotation_z((f as f32 * per_frame).to_radians());
            rect.corner = center + r * (rect.corner - center);
            rect.edge_u = r * rect.edge_u;
            rect.edge_v = r * rect.edge_v;
            l.emitter = continuum_gpu::Emitter::Rect(rect);
        }
        l
    };
    let mut r = InstancedRenderer::with_movers(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        w,
        h,
        1,
        &lights,
        &field.scene,
        origin,
        cache.then_some(options),
        &movable,
        if sun_mode { &[0] } else { &[] },
    );
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let profiled = args.iter().any(|a| a == "profile");
    let profile_from: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("profile_from="))
        .map_or(0, |n| n.parse().unwrap());
    let mut prof = [Duration::ZERO; 5];
    let mut passes: Vec<(&'static str, Duration)> = Vec::new();
    let mut set = Duration::ZERO;
    let mut call = Duration::ZERO;
    let begin = Instant::now();
    for f in 0..frames {
        let t = f as f32 / 599.0;
        let eye = Vec3::new(ground.eye.x, ground.eye.y, 1.7 + 148.3 * t) + flat * 300.0 * t;
        let d = (45.0 * t).to_radians();
        let camera = Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: 50.0,
        };
        let s = Instant::now();
        if moving && sun_mode {
            r.set_light(0, sun_at(f));
        }
        if moving && !sun_mode {
            for &m in &movable {
                let i = &field.scene.instances[m];
                let a = f as f64 * 0.02 + m as f64;
                r.set_instance_transform(
                    m,
                    Mat3::from_rotation_z(f as f32 / 60.0) * i.rotation,
                    i.position + DVec3::new(a.cos(), a.sin(), 0.0),
                );
            }
        }
        set += s.elapsed();
        let s = Instant::now();
        if profiled && f >= profile_from {
            let p = r.frame_profiled(camera, &target);
            prof[0] += p.planning;
            prof[1] += p.bake;
            prof[2] += p.visibility;
            prof[3] += p.shading;
            prof[4] += p.readback;
            for (k, &(label, d)) in p.bake_passes.iter().enumerate() {
                if label.is_empty() {
                    continue;
                }
                if let Some(e) = passes.iter_mut().find(|e| e.0 == label) {
                    e.1 += d;
                } else {
                    let _ = k;
                    passes.push((label, d));
                }
            }
        } else {
            r.frame(camera, &target);
        }
        call += s.elapsed();
    }
    if profiled {
        let f = |d: Duration| d.as_secs_f64() * 1e3 / (frames - profile_from) as f64;
        println!(
            "profiled: planning {:.2}, bake {:.2}, visibility {:.2}, shading {:.2}, readback {:.2} ms",
            f(prof[0]),
            f(prof[1]),
            f(prof[2]),
            f(prof[3]),
            f(prof[4])
        );
        for (label, d) in &passes {
            println!("  bake pass {label}: {:.3} ms", f(*d));
        }
    }
    r.finish();
    let total = begin.elapsed();
    if args.iter().any(|a| a == "save") {
        let pixels = r.read_colors();
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in pixels.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        let name = format!(
            "m9_dev_{}_{frames}.png",
            if moving { "moving" } else { "still" }
        );
        image
            .save(
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("out")
                    .join(name),
            )
            .unwrap();
    }
    let log = r.cache_log();
    let trace_step: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("step="))
        .map_or(10, |n| n.parse().unwrap());
    if args.iter().any(|a| a == "trace") {
        for s in log.iter().step_by(trace_step) {
            println!(
                "frame {}: ledger pages {} texels {}, checked {}, mixed {}, held {}, grew {}, unbounded {}, whole {}; budget {}, spent {}, baked {} pages, GPU bake {:.2} ms, camera {:.2} ms, rebaked {} pages {} texels, candidates {}, invalidated {}, missed {}, visible {}, fallback {}",
                s.frame,
                s.ledger_pages,
                s.ledger_texels,
                s.ledger_checked,
                s.ledger_mixed,
                s.ledger_held,
                s.ledger_grew,
                s.ledger_unbounded,
                s.ledger_whole,
                s.budget,
                s.spent,
                s.pages_baked,
                s.gpu_bake_ms,
                s.gpu_render_ms,
                s.rebaked_pages,
                s.rebaked_texels,
                s.candidates,
                s.invalidated,
                s.missed,
                s.visible_pages,
                s.fallback_pixels
            );
            if args.iter().any(|a| a == "work") {
                println!("  work {:?}", s.work);
            }
        }
    }
    let n = log.len().max(1) as f64;
    let mean = |f: &dyn Fn(&continuum_gpu::CacheStats) -> f64| log.iter().map(f).sum::<f64>() / n;
    println!(
        "moving {moving} hw {hw} cache {cache}: {:.2} ms per frame (set {:.3} ms, frame call {:.2} ms); GPU bake {:.2} ms, camera {:.2} ms; planning {:.2} ms, waiting {:.2} ms; budget {:.0}; rebaked {:.1}, visible {:.0}",
        total.as_secs_f64() * 1e3 / frames as f64,
        set.as_secs_f64() * 1e3 / frames as f64,
        call.as_secs_f64() * 1e3 / frames as f64,
        mean(&|s| s.gpu_bake_ms as f64),
        mean(&|s| s.gpu_render_ms as f64),
        mean(&|s| s.planning.as_secs_f64() * 1e3),
        mean(&|s| s.waiting.as_secs_f64() * 1e3),
        mean(&|s| s.budget as f64),
        mean(&|s| s.rebaked_pages as f64),
        mean(&|s| s.visible_pages as f64),
    );
}
