//! Development (M22, bounce light that settles): a still scene settled with
//! bounce light on, then every resident page's bounce samples re-baked in
//! place (`verify_bounce`): pages whose bounce an in-place re-bake changes
//! by more than ε (0.01) of the largest bounce value. Not a gate.
//!
//! `SD_SCENE`: `sun` (M6's textured scene, sun only, as M22's flights;
//! default), `lamps` (M10 gate 22's night scene) or `sky` (the textured
//! scene with its sky). `SD_W`/`SD_H` (640×360); `SD_FIXED=1` a fixed bake
//! budget (`frame_target_ms` 0), else the default time budget; `SD_OLD=1`
//! without `CacheOptions::bounce_settles`.

use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    let start = r.cache_log().len();
    for n in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0 && s.indirect_samples == 0)
        {
            return n;
        }
    }
    20_000
}

fn main() {
    let w: u32 = env("SD_W", 640);
    let h: u32 = env("SD_H", 360);
    let scene: String = env("SD_SCENE", "sun".to_string());
    let fixed = std::env::var_os("SD_FIXED").is_some();
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
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
    let mut t = match scene.as_str() {
        "lamps" => Textured::new().with_lamps(),
        _ => Textured::new(),
    };
    if scene == "sun" {
        t.sky = Vec3::ZERO;
    }
    kernel.set_sky(t.sky);
    let mut options = CacheOptions {
        indirect: true,
        ..Default::default()
    };
    if fixed {
        options.frame_target_ms = 0.0;
    }
    // `SD_OLD=1`: bounce light as before M22 gate 3.
    options.bounce_settles = std::env::var_os("SD_OLD").is_none();
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel, format, w, h, 1, &t.lights, &t.scene, t.origin, options,
    );
    let start = Instant::now();
    let frames = settle(&mut r, t.camera, &target);
    let secs = start.elapsed().as_secs_f64();
    let log = r.cache_log();
    let samples: u64 = log.iter().map(|s| s.indirect_samples as u64).sum();
    let pages: u64 = log.iter().map(|s| s.pages_baked as u64).sum();
    let resident = r.cache_stats().map_or(0, |s| s.resident);
    if std::env::var_os("SD_TAIL").is_some() {
        // Activity in windows of 250 frames.
        for (k, w) in r.cache_log().chunks(250).enumerate() {
            let sum = |f: &dyn Fn(&continuum_gpu::CacheStats) -> u32| {
                w.iter().map(|s| f(s) as u64).sum::<u64>()
            };
            println!(
                "frames {:5}-: requests {:6} baked {:6} samples {:7} incomplete {:6} evicted {:6} budget {:6}",
                k * 250,
                sum(&|s| s.requests),
                sum(&|s| s.pages_baked),
                sum(&|s| s.indirect_samples),
                sum(&|s| s.indirect_incomplete),
                sum(&|s| s.evicted),
                w.last().map_or(0, |s| s.budget)
            );
        }
    }
    let c = r.verify_bounce(0.01);
    println!(
        "{scene}{} {w}x{h}{}: settled in {frames} frames ({secs:.1} s); {pages} pages and {samples} bounce samples baked, {resident} resident; over eps {} of {} pages (changed {}), largest change {:.3e} of {:.3e}",
        if options.bounce_settles { "" } else { " (old)" },
        if fixed { " fixed budget" } else { "" },
        c.over_epsilon,
        c.pages,
        c.changed,
        c.largest_change,
        c.largest_value
    );
}
