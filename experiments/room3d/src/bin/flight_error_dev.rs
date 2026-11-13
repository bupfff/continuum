//! The error while flying (development, not a gate): M9's flight over the
//! million-boulder field with sun and sky, default options; every `EVERY`th
//! frame the flying image against a settled reference at the same camera (a
//! second renderer warmed there until 16 quiet frames). Reports the mean and
//! largest relative RMSE, and fallback.
//!
//! `cargo run --release -p room3d --bin flight_error_dev -- [nolater]
//! [cones] [frames=N] [every=N] [boulders=N]`

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const FLIGHT: usize = 600;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    for k in 1..=60_000usize {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return k;
        }
    }
    60_000
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: usize| {
        args.iter()
            .find_map(|a| a.strip_prefix(name))
            .map_or(default, |v| v.parse().unwrap())
    };
    let frames = arg("frames=", FLIGHT);
    let every = arg("every=", 60);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let field = Field::new(arg("boulders=", 1_000_000), DVec3::ZERO);
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
    let options = CacheOptions {
        sky_later: !args.iter().any(|a| a == "nolater"),
        sky_rays: !args.iter().any(|a| a == "cones"),
        ..Default::default()
    };
    let renderer = || {
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
    let camera = |f: usize| {
        let t = f as f32 / (FLIGHT - 1) as f32;
        let d = (45.0 * t).to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, 1.7 + 148.3 * t) + flat * 300.0 * t;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let mut r = renderer();
    let mut errors = Vec::new();
    let start = std::time::Instant::now();
    for f in 0..frames {
        let g = f * FLIGHT / frames;
        r.frame(camera(g), &target);
        if (f + 1) % every == 0 {
            r.finish();
            let flying = r.read_colors();
            let fallback = r.cache_stats().map_or(0, |s| s.fallback_pixels) as f64 / (W * H) as f64;
            let mut reference = renderer();
            let settled = settle(&mut reference, camera(g), &target);
            reference.finish();
            let e = relative_rmse(&flying, &reference.read_colors());
            println!(
                "frame {}: {:.2}% from settled ({settled} frames to settle), fallback {:.1}%",
                f + 1,
                100.0 * e,
                100.0 * fallback
            );
            errors.push((e, fallback));
        }
    }
    let n = errors.len().max(1) as f64;
    println!(
        "sky later {}, sky rays {}: mean {:.2}% from settled, largest {:.2}%, fallback {:.1}% at the checks ({:.0} s)",
        options.sky_later,
        options.sky_rays,
        100.0 * errors.iter().map(|e| e.0).sum::<f64>() / n,
        100.0 * errors.iter().map(|e| e.0).fold(0.0, f64::max),
        100.0 * errors.iter().map(|e| e.1).sum::<f64>() / n,
        start.elapsed().as_secs_f64()
    );
}
