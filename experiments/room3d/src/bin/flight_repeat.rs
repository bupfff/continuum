//! Diagnostic (not a gate): gate 8's flight over the million-boulder field
//! several times in one process, each from a new cache, to see whether
//! flights slow down within a process (M5 gate 8's flights did). `--three`
//! flies with gate 6's three lights, otherwise the sun and the field's
//! sphere lights. `WARMUPS=n` first warms and drops n caches at the standing
//! viewpoint, as the gates do before their flights.
//! `cargo run --release -p room3d --bin flight_repeat -- [flights]`

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::field::Field;

fn main() {
    let flights: usize = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(3);
    let warmups: usize = std::env::var("WARMUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky(Vec3::ZERO);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let lights: Vec<Light> = if std::env::args().any(|a| a == "--three") {
        field.colored_lights(origin, 1000.0).to_vec()
    } else {
        let mut all = vec![Light::white(sun, field.radiance)];
        all.extend(field.sphere_lights(origin));
        all
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
    let cached = || {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            1280,
            720,
            1,
            &lights,
            &field.scene,
            origin,
            CacheOptions::default(),
        )
    };
    for _ in 0..warmups {
        let mut r = cached();
        for _ in 0..400 {
            r.frame(ground, &target);
        }
        r.finish();
    }
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    for run in 0..flights {
        let mut r = cached();
        let mut times = Vec::new();
        let start = Instant::now();
        for f in 0..600 {
            let t = f as f32 / 599.0;
            let d = (45.0 * t).to_radians();
            let eye = Vec3::new(ground.eye.x, ground.eye.y, 1.7 + 148.3 * t) + flat * 300.0 * t;
            let camera = Camera {
                eye,
                target: eye + flat * d.cos() - Vec3::Z * d.sin(),
                vertical_fov_degrees: ground.vertical_fov_degrees,
            };
            let call = Instant::now();
            r.frame(camera, &target);
            times.push(call.elapsed().as_secs_f64() * 1e3);
        }
        r.finish();
        let mean = start.elapsed().as_secs_f64() * 1e3 / 600.0;
        let log = r.cache_log();
        let n = log.len() as f64;
        let camera_ms = log.iter().map(|s| s.gpu_render_ms as f64).sum::<f64>() / n;
        let bake_ms = log.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n;
        let budget = log.iter().map(|s| s.budget as f64).sum::<f64>() / n;
        times.sort_by(f64::total_cmp);
        println!(
            "flight {}: mean {mean:.2} ms, p99 {:.2} ms, GPU camera {camera_ms:.2} ms, bake {bake_ms:.2} ms, budget {budget:.0}, cache {:.0} MB",
            run + 1,
            times[593],
            r.cache_bytes() as f64 / 1e6
        );
    }
}
