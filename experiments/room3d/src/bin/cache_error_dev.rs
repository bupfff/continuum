//! Development measurement (not a gate): the view cache's warm image against
//! the per-pixel image (1 sample per pixel, the same lights) on the
//! million-boulder field at gate 8's viewpoints, 1280×720 unless given.
//! Sun only by default; `SKY=1` adds the sky, `SKY=only` lights by the sky
//! alone; `LIGHTS=3` uses the three colored lights of M5 gate 6,
//! `LIGHTS=sphere` the sun with the field's sphere lights; `RATIO=0`
//! caches irradiance instead of light ratios. Prints
//! frames until warm and the relative RMSE.
//! `cargo run --release -p room3d --bin cache_error_dev -- [W H] [VIEWS=0,1]`

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

fn main() {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h) = (*args.first().unwrap_or(&1280), *args.get(1).unwrap_or(&720));
    let sky_mode = std::env::var("SKY").unwrap_or_default();
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(if sky_mode.is_empty() {
        Vec3::ZERO
    } else {
        field.sky()
    });
    // LIGHTS=3: the field's three colored lights (M5 gate 6) for the sun.
    let lights: Vec<Light> = if std::env::var("LIGHTS").is_ok_and(|v| v == "3") {
        field.colored_lights(origin, 1000.0).to_vec()
    } else if std::env::var("LIGHTS").is_ok_and(|v| v == "sphere") {
        // The sun with the field's point light and spot (M5 gate 8).
        let mut all = vec![Light::white(light, field.radiance)];
        all.extend(field.sphere_lights(origin));
        all
    } else {
        vec![Light::white(
            light,
            if sky_mode == "only" {
                0.0
            } else {
                field.radiance
            },
        )]
    };
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
    let picked: Vec<usize> = std::env::var("VIEWS")
        .map(|v| v.split(',').filter_map(|k| k.parse().ok()).collect())
        .unwrap_or_else(|_| (0..views.len()).collect());
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
    for (k, (name, camera)) in views.into_iter().enumerate() {
        if !picked.contains(&k) {
            continue;
        }
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &lights,
            &field.scene,
            origin,
            CacheOptions {
                light_ratio: std::env::var("RATIO").map_or(true, |v| v != "0"),
                ..Default::default()
            },
        );
        let mut frames = 0;
        loop {
            r.frame(camera, &target);
            frames += 1;
            let latest = r.cache_stats().unwrap().frame as usize;
            if (latest >= 16
                && r.cache_log()[latest - 16..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0))
                || frames > 40_000
            {
                break;
            }
        }
        r.finish();
        let cached = r.read_colors();
        drop(r);
        let exact = kernel
            .render_instanced_lights(camera, w, h, &lights, &field.scene, origin, 1)
            .0;
        println!(
            "{name}: warm after {frames} frames, cached vs per-pixel {:.2}%",
            100.0 * relative_rmse(&cached, &exact)
        );
    }
}
