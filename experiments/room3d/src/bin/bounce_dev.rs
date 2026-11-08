//! Development tool for M10 gate 22's speed miss (not a gate): the warm
//! frame with bounce light on against off, on `Textured::with_lamps`, with
//! all five lights and with the first three (bounce light's old limit),
//! alternated over three rounds.
//!
//! `cargo run --release -p room3d --bin bounce_dev`

use std::time::{Duration, Instant};

use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 frames rendered since the call requested and baked
/// nothing; the frames it took.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> usize {
    let start = r.cache_log().len();
    for k in 1..=50_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return k;
        }
    }
    panic!("the view never settled");
}

/// The median of 30 warm frames, each waited for.
fn warm(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> Duration {
    let mut times: Vec<Duration> = (0..30)
        .map(|_| {
            let start = Instant::now();
            r.frame(camera, target);
            r.wait();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[15]
}

fn main() {
    let kernel = MaskConeKernel::new().expect("GPU");
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let t = Textured::new().with_lamps();
    kernel.set_sky(t.sky);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
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
    for n in [5usize, 3] {
        let renderer = |bounce: bool| {
            InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                W,
                H,
                1,
                &t.lights[..n],
                &t.scene,
                t.origin,
                CacheOptions {
                    indirect: bounce,
                    ..Default::default()
                },
            )
        };
        let mut off = renderer(false);
        let mut on = renderer(true);
        settle(&mut off, t.camera, &target);
        settle(&mut on, t.camera, &target);
        for round in 0..3 {
            let a = ms(warm(&mut off, t.camera, &target));
            let b = ms(warm(&mut on, t.camera, &target));
            println!(
                "{n} lights, round {round}: off {a:.2} ms, on {b:.2} ms ({:.3}×)",
                b / a
            );
        }
    }
}
