//! Development (M18): fog with light shafts against Cycles, on the
//! textured scene with the glass block's neighbours (sun and sky). Not a
//! gate.
//!
//! `FOG_W`/`FOG_H` set the size (640×360), `FOG_SAMPLES` Cycles' samples
//! (1024), `FOG_DENSITY` (0.05), `FOG_TOP` (6), `FOG_FALLOFF` (0). Ours:
//! 4×4 sub-pixel offsets, each settled. `profile`: camera passes timed.

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Camera, Fog, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_fog};
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return;
        }
    }
    eprintln!("WARNING: not settled");
}

fn main() -> std::io::Result<()> {
    let (w, h): (u32, u32) = (env("FOG_W", 640), env("FOG_H", 360));
    let samples: u32 = env("FOG_SAMPLES", 1024);
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
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
    let mut t = Textured::new().with_transmission(true);
    let mut single = Textured::new().with_transmission(false);
    // `FOG_VIEW=sun`: the brick block lit from beside, its shadow through
    // the fog.
    if std::env::var("FOG_VIEW").is_ok_and(|v| v == "sun") {
        let eye = Vec3::new(6.0, 4.0, 1.6);
        let camera = Camera {
            eye,
            target: Vec3::new(-3.0, 9.5, 0.6),
            vertical_fov_degrees: 45.0,
        };
        t.camera = camera;
        single.camera = camera;
    }
    kernel.set_sky(t.sky);
    let fog = Fog {
        density: env("FOG_DENSITY", 0.05),
        height_falloff: env("FOG_FALLOFF", 0.0),
        top: env("FOG_TOP", 6.0),
        albedo: Vec3::splat(0.9),
        anisotropy: 0.3,
        ..Default::default()
    };
    let make = |fog: Option<Fog>| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions {
                fog,
                ..Default::default()
            },
        );
        settle(&mut r, t.camera, &target);
        r
    };
    if std::env::args().any(|a| a == "profile") {
        let mut r = make(Some(fog));
        let mut per: Vec<Vec<f64>> = Vec::new();
        for _ in 0..30 {
            let p = r.frame_profiled(t.camera, &target);
            for (k, d) in p.passes.iter().enumerate().filter(|(_, d)| !d.is_zero()) {
                if per.len() <= k {
                    per.push(Vec::new());
                }
                per[k].push(d.as_secs_f64() * 1000.0);
            }
        }
        for v in &mut per {
            v.sort_by(f64::total_cmp);
        }
        println!(
            "camera passes (median ms, each waited for): {:?}",
            per.iter()
                .map(|v| (v[15] * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        return Ok(());
    }
    let timed = |r: &mut InstancedRenderer| {
        let mut v: Vec<f64> = (0..30)
            .map(|_| {
                let s = Instant::now();
                r.frame(t.camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v[15]
    };
    let supersampled = |r: &mut InstancedRenderer| {
        let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        for k in 0..16 {
            let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
            r.set_pixel_offset(dx, dy);
            settle(r, t.camera, &target);
            r.finish();
            for (s, p) in sum.iter_mut().zip(r.read_colors()) {
                for c in 0..3 {
                    s[c] += p[c] / 16.0;
                }
            }
        }
        sum
    };
    let mut foggy = make(Some(fog));
    let mut clear = make(None);
    let (t_fog, t_clear) = (timed(&mut foggy), timed(&mut clear));
    let ours = supersampled(&mut foggy);
    let none = supersampled(&mut clear);
    let reference = render_instanced_fog(
        &single.scene,
        single.origin,
        &single.lights,
        single.sky,
        (&fog, -1.0, 500.0),
        single.camera,
        (w, h),
        samples,
        1,
        1,
        &out_dir.join(format!(
            "fog_dev_cycles_{}",
            std::env::var("FOG_VIEW").unwrap_or_default()
        )),
    )?;
    let mean = |px: &[[f32; 3]]| {
        px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>() / (3 * px.len()) as f64
    };
    println!(
        "{w}x{h}, Cycles {samples} samples ({:.0} s): ours {:.2}%, without fog {:.2}%; means: ours {:+.2}%, without {:+.2}%",
        reference.seconds,
        100.0 * relative_rmse(&ours, &reference.pixels),
        100.0 * relative_rmse(&none, &reference.pixels),
        100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&none) / mean(&reference.pixels) - 1.0)
    );
    println!(
        "warm frame: fog {t_fog:.2} ms, clear {t_clear:.2} ms ({:.3}x, +{:.2} ms)",
        t_fog / t_clear,
        t_fog - t_clear
    );
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!(
            "fog_dev_{}{name}.png",
            std::env::var("FOG_VIEW").unwrap_or_default()
        )))
    };
    save("ours", &ours)?;
    save("clear", &none)?;
    save("cycles", &reference.pixels)?;
    Ok(())
}
