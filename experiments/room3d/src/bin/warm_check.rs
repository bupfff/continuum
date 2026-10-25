//! Quick check (not a gate): the million-boulder field's sun-only cache warmed
//! at the standing viewpoint, as M4 gate 11's hardware row, with the current
//! build. Prints the ray mode, frames until warm and the warm frame time.

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;

fn main() {
    let sky = std::env::args().any(|a| a == "--sky");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    kernel.set_sky(if sky { field.sky() } else { Vec3::ZERO });
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
    let mut r = InstancedRenderer::with_cache(
        &kernel,
        format,
        1280,
        720,
        1,
        light,
        &field.scene,
        origin,
        field.radiance,
        CacheOptions::default(),
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
}
