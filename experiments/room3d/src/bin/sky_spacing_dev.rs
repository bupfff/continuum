//! Development measurement (not a gate): the view cache's sky and self spacings against
//! warm-up cost and accuracy, on the million-boulder field with sky only, at
//! gate 8's viewpoints. For each variant: a whole warm-up at a fixed budget
//! with every frame profiled (bake time summed), then the warm image against
//! the per-pixel sky at 1 sample per pixel and at 4×4 per pixel (the area
//! integral). Variants are (sky spacing, self spacing) pairs, as
//! `VARIANTS="0,0;0,0.01"`.
//! `cargo run --release -p room3d --bin sky_spacing_dev -- [W H]`

use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

fn main() {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h) = (*args.first().unwrap_or(&640), *args.get(1).unwrap_or(&360));
    // (sky spacing, self spacing) pairs, as VARIANTS="0,0;0,0.01".
    let variants: Vec<(f32, f32)> = std::env::var("VARIANTS")
        .ok()
        .map(|v| {
            v.split(';')
                .filter_map(|p| {
                    let (a, b) = p.split_once(',')?;
                    Some((a.parse().ok()?, b.parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(0.0, 0.0), (0.0, 0.01)]);
    let stage: u32 = std::env::var("STAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_profile_stage(stage);
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let sky = field.sky();
    kernel.set_sky(sky);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down: f32| {
        let d = down.to_radians();
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
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    for (name, camera) in views {
        kernel.set_hardware_rays(false);
        let one = kernel
            .render_instanced(camera, w, h, light, &field.scene, origin, 0.0, 1)
            .0;
        let (big, _, _) =
            kernel.render_instanced(camera, w * 4, h * 4, light, &field.scene, origin, 0.0, 1);
        let mut area = vec![[0.0f32; 3]; (w * h) as usize];
        for y in 0..(h * 4) as usize {
            for x in 0..(w * 4) as usize {
                let p = big[y * (w * 4) as usize + x];
                let o = &mut area[(y / 4) * w as usize + x / 4];
                for c in 0..3 {
                    o[c] += p[c] / 16.0;
                }
            }
        }
        drop(big);
        kernel.set_hardware_rays(true);
        println!(
            "{name}: per-pixel 1 spp vs 4×4: {:.2}%",
            100.0 * relative_rmse(&one, &area)
        );
        for &(spacing, own) in &variants {
            let mut r = InstancedRenderer::with_cache(
                &kernel,
                format,
                w,
                h,
                1,
                light,
                &field.scene,
                origin,
                0.0,
                CacheOptions {
                    frame_target_ms: 0.0,
                    sky_spacing: spacing,
                    self_spacing: own,
                    sky_sample_cost: 1,
                    ..Default::default()
                },
            );
            let mut bake = Duration::ZERO;
            let mut frames = 0;
            loop {
                bake += r.frame_profiled(camera, &target).bake;
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
            let log = r.cache_log();
            let texels: u64 = log.iter().map(|s| s.texels_baked as u64).sum();
            let samples: u64 = log.iter().map(|s| s.sky_samples as u64).sum();
            let image = r.read_colors();
            println!(
                "  sky spacing {spacing} m, self spacing {own} m: {frames} frames, bake {:.0} ms ({texels} texels, {samples} sky samples), vs 1 spp {:.2}%, vs 4×4 {:.2}%",
                ms(bake),
                100.0 * relative_rmse(&image, &one),
                100.0 * relative_rmse(&image, &area)
            );
        }
    }
}
