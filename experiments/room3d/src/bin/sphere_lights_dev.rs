//! Development check (not a gate): sphere lights on the textured scene at
//! night: a point light (10 cm across) 2 m up among the boulders and a spot
//! (5 cm across, 4 m up, aimed down the scene, 20° inner and 30° outer
//! half-angles). Renders per pixel and through the view cache, and with
//! `SPP=n` against Cycles at n samples per pixel (`SS` per pixel for ours,
//! default 4).
//! `cargo run --release -p room3d --bin sphere_lights_dev -- [W H]`

use std::path::Path;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, Light, MaskConeKernel, Spot, wgpu};
use flatland::img::Image;
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h) = (*args.first().unwrap_or(&320), *args.get(1).unwrap_or(&180));
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let t = Textured::new();
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky(Vec3::ZERO);
    let point = Light::point(
        Vec3::new(-1.0, 7.0, 2.0),
        0.05,
        Vec3::new(1.0, 0.8, 0.6) * 20.0,
    );
    let spot = Light::spot(
        Vec3::new(2.0, 4.0, 4.0),
        0.025,
        Vec3::new(0.6, 0.8, 1.0) * 120.0,
        Spot {
            axis: Vec3::new(-0.3, 1.0, -1.2).normalize(),
            inner: 20f32.to_radians(),
            outer: 30f32.to_radians(),
        },
    );
    let lights = [point, spot];
    let save = |img: &[[f32; 3]], tag: &str| {
        let mut image = Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in img.iter().enumerate() {
            image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
        }
        image.save(&out_dir.join(format!("sphere_lights_{tag}.png")))
    };
    let (per_pixel, _, _) =
        kernel.render_instanced_lights(t.camera, w, h, &lights, &t.scene, t.origin, 1);
    let mean = |px: &[[f32; 3]]| {
        px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>() / (3 * px.len()) as f64
    };
    println!("per pixel: mean {:.4}", mean(&per_pixel));
    save(&per_pixel, "perpixel")?;

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
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        w,
        h,
        1,
        &lights,
        &t.scene,
        t.origin,
        CacheOptions::default(),
    );
    let mut frames = 0;
    loop {
        r.frame(t.camera, &target);
        frames += 1;
        let latest = r.cache_stats().unwrap().frame as usize;
        if (latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0))
            || frames > 20_000
        {
            break;
        }
    }
    r.finish();
    let cached = r.read_colors();
    drop(r);
    println!(
        "cache: warm after {frames} frames, cached vs per-pixel {:.2}%",
        100.0 * relative_rmse(&cached, &per_pixel)
    );
    save(&cached, "cached")?;

    if let Some(spp) = std::env::var("SPP").ok().and_then(|v| v.parse().ok()) {
        let ss: u32 = std::env::var("SS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        kernel.set_hardware_rays(false);
        let (big, _, _) = kernel.render_instanced_lights(
            t.camera,
            w * ss,
            h * ss,
            &lights,
            &t.scene,
            t.origin,
            1,
        );
        let mut ours = vec![[0.0f32; 3]; (w * h) as usize];
        for y in 0..(h * ss) as usize {
            for x in 0..(w * ss) as usize {
                let p = big[y * (w * ss) as usize + x];
                let o = &mut ours[(y / ss as usize) * w as usize + x / ss as usize];
                for c in 0..3 {
                    o[c] += p[c] / (ss * ss) as f32;
                }
            }
        }
        let reference = render_instanced_lights(
            &t.scene,
            t.origin,
            &lights,
            Vec3::ZERO,
            t.camera,
            (w, h),
            spp,
            0,
            1,
            &out_dir.join("sphere_lights_cycles"),
        )?;
        println!(
            "{ss}×{ss} per pixel vs Cycles {spp} spp: {:.2}% (means {:.4} / {:.4}, Cycles {:.1} s)",
            100.0 * relative_rmse(&ours, &reference.pixels),
            mean(&ours),
            mean(&reference.pixels),
            reference.seconds
        );
        save(&ours, "ours_ss")?;
        save(&reference.pixels, "cycles")?;
    }
    Ok(())
}
