//! Development: bake cost with the foliage cards' back faces exactly behind
//! the front ones against 1 mm behind (M10 gate 6's finding). Not a gate.
//! Usage: backface_dev <back offset m> [opaque] [nosun] [nosky] [noindirect]

use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::textured::Textured;
use std::time::Instant;

const W: u32 = 1280;
const H: u32 = 720;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let back: f32 = args.get(1).map_or(0.0, |a| a.parse().unwrap());
    let has = |f: &str| args.iter().any(|a| a == f);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(if has("noprefilter") { 0.0 } else { 1.0 });
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(!has("nohw"));
    let mut t = Textured::new().with_foliage_back(back);
    if has("opaque") {
        t.scene.geometry_alpha_cutoff.clear();
    }
    if has("nosun") {
        t.lights.clear();
    }
    if !has("nosky") {
        kernel.set_sky(t.sky);
    }
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
    if let Some(name) = args.iter().find_map(|a| a.strip_prefix("pixel=")) {
        let mut r =
            InstancedRenderer::new_lights(&kernel, format, W, H, 1, &t.lights, &t.scene, t.origin);
        r.frame(t.camera, &target);
        r.finish();
        let px = r.read_colors();
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % W as usize,
                i / W as usize,
                room3d::to_rgb(continuum::glam::Vec3::from_array(*p)),
            );
        }
        image
            .save(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("out/{name}.png")))
            .unwrap();
        return;
    }
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
            indirect: !has("noindirect"),
            ..Default::default()
        },
    );
    let begin = Instant::now();
    for _ in 0..200 {
        r.frame(t.camera, &target);
    }
    r.finish();
    let log = r.cache_log();
    let (ms, texels, sky, pages): (f64, u64, u64, u64) = log.iter().fold((0.0, 0, 0, 0), |a, s| {
        (
            a.0 + s.gpu_bake_ms as f64,
            a.1 + s.texels_baked as u64,
            a.2 + s.sky_samples as u64,
            a.3 + s.pages_baked as u64,
        )
    });
    let last = *log.last().unwrap();
    if let Some(name) = args.iter().find_map(|a| a.strip_prefix("save=")) {
        for _ in 0..20_000 {
            r.frame(t.camera, &target);
            let log = r.cache_log();
            let latest = r
                .cache_stats()
                .map_or(0, |s| s.frame as usize)
                .min(log.len());
            if latest >= 32
                && log[latest - 32..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0)
            {
                break;
            }
        }
        r.finish();
        let px = r.read_colors();
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % W as usize,
                i / W as usize,
                room3d::to_rgb(continuum::glam::Vec3::from_array(*p)),
            );
        }
        image
            .save(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("out/{name}.png")))
            .unwrap();
    }
    let log = r.cache_log();
    if has("dump") {
        for s in log.iter().skip(20).step_by(30) {
            println!(
                "frame {} bake {:.1} ms pages {} texels {} ind_samples {} incomplete {} work {:?}",
                s.frame,
                s.gpu_bake_ms,
                s.pages_baked,
                s.texels_baked,
                s.indirect_samples,
                s.indirect_incomplete,
                s.work
            );
        }
    }
    println!(
        "back {back}: 200 frames {:.1} s; bake {ms:.0} ms GPU, {pages} pages, {texels} texels, {sky} sky samples; {:.2} us/texel; last frame requests {} baked {} fallback {}",
        begin.elapsed().as_secs_f64(),
        1000.0 * ms / texels.max(1) as f64,
        last.requests,
        last.pages_baked,
        last.fallback_pixels
    );
}
