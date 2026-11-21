//! Development (a first impression): how fast a view's lighting fills in
//! through `Renderer` with its defaults: the frames and seconds until under
//! 1% and 0.1% of the pixels are shaded without their page (unshadowed
//! fallback), and until settled. The boulder field (`WU_BOULDERS`,
//! 1,000,000) at `WU_W`×`WU_H` (2560×1440), `WU_SCALE` (1), with the sky
//! unless `WU_SKY=0`, bounce light unless `WU_BOUNCE=0`. Not a gate.

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{MaskConeKernel, Renderer, RendererOptions, wgpu};
use room3d::field::Field;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let (w, h): (u32, u32) = (env("WU_W", 2560), env("WU_H", 1440));
    let scale: u32 = env("WU_SCALE", 1);
    let sky = env("WU_SKY", 1) == 1;
    let bounce = env("WU_BOUNCE", 1) == 1;
    let field = Field::new(env("WU_BOULDERS", 1_000_000), DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    let lights = [continuum_gpu::Light::rect(
        light,
        Vec3::splat(field.radiance),
    )];
    let kernel = MaskConeKernel::new().expect("GPU");
    let mut r = Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (w, h),
        &field.scene,
        origin,
        &lights,
        RendererOptions {
            scale,
            sky: sky.then_some(Vec3::new(0.6, 0.75, 1.0) * 0.3),
            bounce,
            cache_options: continuum_gpu::CacheOptions {
                bounce_settles: env("WU_SETTLES", 1) == 1,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let target = r
        .kernel()
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
    let pixels = (w / scale) as f64 * (h / scale) as f64;
    let start = Instant::now();
    let mut marks: [Option<(usize, f64)>; 2] = [None, None];
    let mut settled = None;
    let frames: usize = env("WU_FRAMES", 6000);
    for f in 1..=frames {
        r.render(camera, &target);
        r.wait();
        let s = r.stats().expect("the cache");
        let share = s.fallback_pixels as f64 / pixels;
        let t = start.elapsed().as_secs_f64();
        for (k, bar) in [0.01, 0.001].into_iter().enumerate() {
            if marks[k].is_none() && f > 4 && share < bar {
                marks[k] = Some((f, t));
            }
        }
        if f % 30 == 0 && std::env::var_os("WU_TRACE").is_some() {
            println!(
                "frame {f} ({t:.1} s): fallback {:.2}%, requests {}, pages baked {}, budget {}, resident {}, evicted {}, bounce samples {} ({} incomplete)",
                100.0 * share,
                s.requests,
                s.pages_baked,
                s.budget,
                s.resident,
                s.evicted,
                s.indirect_samples,
                s.indirect_incomplete
            );
        }
        if f > 32 && r.is_settled(32) {
            settled = Some((f, t));
            break;
        }
    }
    println!(
        "sky {sky}, bounce {bounce}, scale {scale}: fallback under 1% at {:?}, under 0.1% at {:?}, settled at {:?} (frame, seconds)",
        marks[0], marks[1], settled
    );
}
