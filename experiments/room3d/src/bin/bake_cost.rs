//! Diagnostic (not a gate): what a view-cache bake costs against the number
//! of texels baked, for sun texels and sky texels, on the million-boulder
//! field. Each budget runs from an empty cache with a fixed budget (no frame
//! target), each frame profiled, so the bake pass is timed on its own.
//! If the time grows with the texel count, the bake is throughput-bound; if
//! it barely grows, each texel's latency dominates and the GPU idles.
//! `cargo run --release -p room3d --bin bake_cost -- [150] [--sky]
//! [--budgets 1000,16000] [--stage N] [--count] [--solo] [--no-cull] [--warm]
//! [--sky-cost N] [--light k]`

use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, KernelOptions, Light, MaskConeKernel, wgpu,
};
use room3d::field::Field;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let high = args.iter().any(|a| a == "150");
    let sky_only = args.iter().any(|a| a == "--sky");
    let warm_up = args.iter().any(|a| a == "--warm");
    let sky_sample_cost = args
        .iter()
        .position(|a| a == "--sky-cost")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(CacheOptions::default().sky_sample_cost);
    let budgets: Vec<u32> = args
        .iter()
        .position(|a| a == "--budgets")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]);
    let stage = args
        .iter()
        .position(|a| a == "--stage")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(0u32);
    // Counting work changes the kernel: its timings are not comparable.
    let count = args.iter().any(|a| a == "--count");
    let kernel = MaskConeKernel::new_with(KernelOptions {
        count_work: count,
        without_packets: args.iter().any(|a| a == "--solo"),
        ..Default::default()
    })
    .expect("GPU");
    kernel.set_profile_stage(stage);
    kernel.set_sky_horizon_cull(!args.iter().any(|a| a == "--no-cull"));
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let camera = if high {
        let eye = Vec3::new(ground.eye.x, ground.eye.y, 150.0);
        let d = 45f32.to_radians();
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    } else {
        ground
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
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    // `--light k`: one of the field's colored lights (M5 gate 6) instead of
    // the sun, at its own radiance.
    let colored = args
        .iter()
        .position(|a| a == "--light")
        .and_then(|i| args.get(i + 1)?.parse::<usize>().ok())
        .map(|k| field.colored_lights(origin, 1000.0)[k]);
    let lit = |radiance: f32| match colored {
        Some(l) => Light {
            radiance: l.radiance * (radiance > 0.0) as u32 as f32,
            ..l
        },
        None => Light::white(light, radiance),
    };
    for (name, radiance, sky) in [
        ("sun", field.radiance, Vec3::ZERO),
        ("sky", 0.0, field.sky()),
    ] {
        if sky_only && name == "sun" {
            continue;
        }
        kernel.set_sky(sky);
        println!("{name}: budget | texels per frame | bake ms per frame | µs per texel");
        if warm_up {
            // A whole warm-up at a fixed budget: every frame profiled.
            let budget = budgets[budgets.len() - 1];
            let options = CacheOptions {
                bake_budget: budget,
                frame_target_ms: 0.0,
                sky_sample_cost,
                ..Default::default()
            };
            let mut r = InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                1280,
                720,
                1,
                &[lit(radiance)],
                &field.scene,
                origin,
                options,
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
            let texels: u64 = r.cache_log().iter().map(|s| s.texels_baked as u64).sum();
            println!(
                "  warm-up at budget {budget}: {frames} frames, {texels} texels, bake {:.0} ms in all, {:.2} µs per texel",
                ms(bake),
                ms(bake) * 1e3 / texels.max(1) as f64
            );
            continue;
        }
        for &budget in &budgets {
            let options = CacheOptions {
                bake_budget: budget,
                frame_target_ms: 0.0,
                sky_sample_cost,
                ..Default::default()
            };
            let mut r = InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                1280,
                720,
                1,
                &[lit(radiance)],
                &field.scene,
                origin,
                options,
            );
            // Warm-up frames fill the request pipeline; then 12 profiled.
            for _ in 0..4 {
                r.frame_profiled(camera, &target);
            }
            let mut bake = Duration::ZERO;
            let first = r.cache_log().len();
            for _ in 0..12 {
                bake += r.frame_profiled(camera, &target).bake;
            }
            r.finish();
            let log = &r.cache_log()[first..first + 12];
            let texels: u64 = log.iter().map(|s| s.texels_baked as u64).sum();
            println!(
                "  {budget:>6} | {:>7.0} | {:>8.2} | {:>7.2}",
                texels as f64 / 12.0,
                ms(bake) / 12.0,
                ms(bake) * 1e3 / texels.max(1) as f64
            );
            if count {
                let mut work = [0u64; 12];
                for s in log {
                    for (w, &x) in work.iter_mut().zip(&s.work) {
                        *w += x as u64;
                    }
                }
                let per = |k: usize| work[k] as f64 / texels.max(1) as f64;
                println!(
                    "      per texel: cones {:.2}, object nodes {:.1} (splatted {:.1}), meshes {:.1}, triangle nodes {:.1} (splatted {:.1}), triangles {:.1} (rasterized {:.1})",
                    per(7),
                    per(0),
                    per(1),
                    per(2),
                    per(3),
                    per(4),
                    per(5),
                    per(6)
                );
                println!(
                    "      packets visited per texel: object nodes {:.1}, triangle nodes {:.1}; own object: triangle nodes {:.1}, entered {:.2}",
                    per(8),
                    per(9),
                    per(10),
                    per(11)
                );
            }
        }
    }
}
