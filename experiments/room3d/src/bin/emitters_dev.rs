//! Development tool for M10 gate 23's unsettled view (not a gate): the
//! glowing panel's faces as lights, settled, then textures read for a
//! quarter of a pixel at gate 23's first sub-pixel offset; cache statistics
//! every 250 frames. `nobounce` turns bounce light off.
//!
//! `cargo run --release -p room3d --bin emitters_dev`

use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::textured::Textured;

use std::time::{Duration, Instant};

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

#[allow(dead_code)]
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
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let mut t = Textured::new().with_glow();
    let made = t.scene.emitters_to_lights(t.origin, 6);
    t.lights.extend(made);
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
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        W,
        H,
        1,
        &t.lights,
        &t.scene,
        t.origin,
        CacheOptions {
            indirect: !std::env::args().any(|a| a == "nobounce"),
            ..Default::default()
        },
    );
    let frames = settle(&mut r, t.camera, &target);
    println!("settled in {frames} frames");
    r.set_footprint(0.25);
    r.set_pixel_offset(-0.375, -0.375);
    for k in 1..=3000 {
        r.frame(t.camera, &target);
        if k % 250 == 0 {
            r.wait();
            let s = r.cache_stats().unwrap();
            println!(
                "frame {k}: requests {}, pages baked {}, texels {}, resident {}, evicted {}, fallback {}, budget {}, bounce samples {}",
                s.requests,
                s.pages_baked,
                s.texels_baked,
                s.resident,
                s.evicted,
                s.fallback_pixels,
                s.budget,
                s.indirect_samples
            );
        }
    }
}
