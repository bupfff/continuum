//! Diagnostics for the view-dependent lighting cache (not a gate): where a
//! frame's time goes on the 1,000,000-boulder field, and how far the cached
//! and the per-pixel images are from a supersampled reference (4×4 exact
//! cones per pixel, box filtered: the pixel's area, not its center).
//! `cargo run --release -p room3d --bin vcache_profile -- [--boulders N] [--compute] [--size W H]`

use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, FrameProfile, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const SS: u32 = 4;

fn stages(p: &FrameProfile) -> [Duration; 5] {
    [
        p.bake,
        p.visibility,
        p.visibility_submit,
        p.shading,
        p.readback,
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count = args
        .iter()
        .position(|a| a == "--boulders")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(1_000_000usize);
    let hardware = !args.iter().any(|a| a == "--compute");
    let size = args.iter().position(|a| a == "--size");
    let dim = |k: usize, default: u32| {
        size.and_then(|i| args.get(i + k)?.parse().ok())
            .unwrap_or(default)
    };
    let (w, h) = (dim(1, 1280), dim(2, 720));
    let kernel = MaskConeKernel::new().expect("GPU");
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let field = Field::new(count, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(1.7, 0.0)),
        ("10 m up", view(10.0, 15.0)),
        ("40 m up", view(40.0, 30.0)),
        ("150 m up", view(150.0, 45.0)),
    ];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
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
    println!(
        "{} boulders, {} rays, {w}×{h}. Stage times (ms, each stage waited for): bake | visibility | shading | display + readback + planning",
        field.boulders(),
        if hardware { "hardware" } else { "compute" }
    );
    for (name, camera) in views {
        kernel.set_hardware_rays(hardware);
        kernel.set_prefilter(1.0);
        let mut r = InstancedRenderer::with_cache(
            &kernel,
            format,
            w,
            h,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            CacheOptions::default(),
        );
        // A frame with a full bake budget: the second from an empty cache.
        r.frame(camera, &target);
        let busy = r.frame_profiled(camera, &target);
        let baked = r.cache_stats().unwrap();
        for _ in 0..300 {
            r.frame(camera, &target);
        }
        let mut warm = [Duration::ZERO; 5];
        for _ in 0..10 {
            let t = r.frame_profiled(camera, &target);
            for (k, d) in stages(&t).into_iter().enumerate() {
                warm[k] += d / 10;
            }
        }
        let busy = stages(&busy);
        let cached = r.read_colors();
        drop(r);
        // Per-pixel stages for comparison: the single camera pass.
        let mut per_pixel = InstancedRenderer::new(
            &kernel,
            format,
            w,
            h,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
        );
        let pp = stages(&per_pixel.frame_profiled(camera, &target));
        drop(per_pixel);
        kernel.set_prefilter(0.0);
        let (exact, _, _) =
            kernel.render_instanced(camera, w, h, light, &field.scene, origin, field.radiance, 1);
        let (big, _, _) = kernel.render_instanced(
            camera,
            w * SS,
            h * SS,
            light,
            &field.scene,
            origin,
            field.radiance,
            1,
        );
        let mut truth = vec![[0.0f32; 3]; (w * h) as usize];
        for y in 0..h * SS {
            for x in 0..w * SS {
                let p = big[(y * w * SS + x) as usize];
                let t = &mut truth[((y / SS) * w + x / SS) as usize];
                for c in 0..3 {
                    t[c] += p[c] / (SS * SS) as f32;
                }
            }
        }
        println!(
            "{name}: busy frame {:.1} | {:.1} ({:.1} in submit) | {:.1} | {:.1} ({} texels baked); warm {:.1} | {:.1} ({:.1} in submit) | {:.1} | {:.1}; per-pixel visibility {:.1} ({:.1} in submit)",
            ms(busy[0]),
            ms(busy[1]),
            ms(busy[2]),
            ms(busy[3]),
            ms(busy[4]),
            baked.texels_baked,
            ms(warm[0]),
            ms(warm[1]),
            ms(warm[2]),
            ms(warm[3]),
            ms(warm[4]),
            ms(pp[1]),
            ms(pp[2]),
        );
        println!(
            "   vs 4×4 supersampled exact: cached {:.2}%, per-pixel exact {:.2}%; cached vs per-pixel exact {:.2}%",
            100.0 * relative_rmse(&cached, &truth),
            100.0 * relative_rmse(&exact, &truth),
            100.0 * relative_rmse(&cached, &exact),
        );
    }
}
