//! Development (M16): thin transmissive glass against Cycles, on the
//! textured scene with M10 gate 16's panes as glass
//! (`Textured::with_transmission`). Not a gate.
//!
//! `TR_W`/`TR_H` set the size (640×360), `TR_SAMPLES` Cycles' samples
//! (1024). Ours: 4×4 sub-pixel offsets, each settled, bounce light on.

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, MaskConeKernel, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
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
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return n;
        }
    }
    panic!("never settled");
}

fn main() -> std::io::Result<()> {
    let (w, h): (u32, u32) = (env("TR_W", 640), env("TR_H", 360));
    let samples: u32 = env("TR_SAMPLES", 1024);
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
    // `solids`: the glass sphere and block (volumes); the third image is
    // the same glass drawn thin, not as blending.
    let solids = std::env::args().any(|a| a == "solids");
    let (t, single) = if solids {
        (
            Textured::new().with_glass_solids(),
            Textured::new().with_glass_solids(),
        )
    } else {
        (
            Textured::new().with_transmission(true),
            Textured::new().with_transmission(false),
        )
    };
    kernel.set_sky(t.sky);
    let mut opaque = t.scene.clone();
    opaque.geometry_transmission.clear();
    let mut blended = t.scene.clone();
    if solids {
        blended.geometry_volumes.clear();
    } else {
        blended.geometry_blend = t
            .scene
            .geometry_transmission
            .iter()
            .map(|x| x.map(|tr| 1.0 - tr))
            .collect();
    }
    let render = |scene: &InstancedScene| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            scene,
            t.origin,
            CacheOptions {
                indirect: true,
                ..Default::default()
            },
        );
        settle(&mut r, t.camera, &target);
        let mut times: Vec<f64> = (0..30)
            .map(|_| {
                let s = Instant::now();
                r.frame(t.camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        times.sort_by(f64::total_cmp);
        let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        for k in 0..16 {
            let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
            r.set_pixel_offset(dx, dy);
            settle(&mut r, t.camera, &target);
            r.finish();
            for (s, p) in sum.iter_mut().zip(r.read_colors()) {
                for c in 0..3 {
                    s[c] += p[c] / 16.0;
                }
            }
        }
        (sum, times[15])
    };
    if std::env::args().any(|a| a == "profile") {
        for (name, scene) in [("glass", &t.scene), ("opaque", &opaque)] {
            let mut r = InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                w,
                h,
                1,
                &t.lights,
                scene,
                t.origin,
                CacheOptions {
                    indirect: true,
                    ..Default::default()
                },
            );
            settle(&mut r, t.camera, &target);
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
                "{name}: camera passes (median ms, each waited for): {:?}; edge pixels {}",
                per.iter()
                    .map(|v| (v[15] * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>(),
                r.cache_stats().map_or(0, |s| s.edge_pixels)
            );
        }
        return Ok(());
    }
    let (ours, t_ours) = render(&t.scene);
    let (flat, t_flat) = render(&opaque);
    let (blend, t_blend) = render(&blended);
    let reference = render_instanced_lights(
        &single.scene,
        single.origin,
        &single.lights,
        single.sky,
        single.camera,
        (w, h),
        samples,
        1,
        1,
        &out_dir.join(if solids {
            "transmission_dev_solids_cycles"
        } else {
            "transmission_dev_cycles"
        }),
    )?;
    let mean_all = ours
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64)
        .sum::<f64>()
        / (3 * ours.len()) as f64;
    let changed: Vec<bool> = ours
        .iter()
        .zip(&flat)
        .map(|(a, b)| (0..3).any(|c| ((a[c] - b[c]) as f64).abs() > 0.01 * mean_all))
        .collect();
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(&changed)
            .filter(|(_, c)| **c)
            .map(|(p, _)| *p)
            .collect()
    };
    let region = keep(&reference.pixels);
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    println!(
        "{w}x{h}, Cycles {samples} samples ({:.0} s): ours {:.2}%, opaque {:.2}%, {} {:.2}%",
        reference.seconds,
        100.0 * relative_rmse(&ours, &reference.pixels),
        100.0 * relative_rmse(&flat, &reference.pixels),
        if solids { "thin" } else { "as alpha blending" },
        100.0 * relative_rmse(&blend, &reference.pixels)
    );
    println!(
        "over the {} pixels transmission changes: ours {:.2}%, opaque {:.2}%, blending {:.2}%; means over them: ours {:+.2}%, opaque {:+.2}%, blending {:+.2}%",
        changed.iter().filter(|c| **c).count(),
        100.0 * relative_rmse(&keep(&ours), &region),
        100.0 * relative_rmse(&keep(&flat), &region),
        100.0 * relative_rmse(&keep(&blend), &region),
        100.0 * (mean(&keep(&ours)) / mean(&region) - 1.0),
        100.0 * (mean(&keep(&flat)) / mean(&region) - 1.0),
        100.0 * (mean(&keep(&blend)) / mean(&region) - 1.0)
    );
    // The same with every channel clamped at 4× the image's mean (a
    // highlight of the sun covers a few pixels at its full radiance).
    let cap = 4.0 * mean_all as f32;
    let clamp =
        |img: &[[f32; 3]]| -> Vec<[f32; 3]> { img.iter().map(|p| p.map(|c| c.min(cap))).collect() };
    let rc = clamp(&reference.pixels);
    println!(
        "clamped at 4x the mean: ours {:.2}%, opaque {:.2}%, other {:.2}%; over the changed pixels: ours {:.2}%, other {:.2}%; pixels above the cap: ours {}, Cycles {}",
        100.0 * relative_rmse(&clamp(&ours), &rc),
        100.0 * relative_rmse(&clamp(&flat), &rc),
        100.0 * relative_rmse(&clamp(&blend), &rc),
        100.0 * relative_rmse(&keep(&clamp(&ours)), &keep(&rc)),
        100.0 * relative_rmse(&keep(&clamp(&blend)), &keep(&rc)),
        ours.iter().filter(|p| p.iter().any(|&c| c > cap)).count(),
        reference
            .pixels
            .iter()
            .filter(|p| p.iter().any(|&c| c > cap))
            .count()
    );
    println!(
        "warm frame: ours {t_ours:.2} ms, opaque {t_flat:.2} ms ({:.3}x), blending {t_blend:.2} ms",
        t_ours / t_flat
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
        let tag = if solids { "solids_" } else { "" };
        image.save(&out_dir.join(format!("transmission_dev_{tag}{name}.png")))
    };
    save("ours", &ours)?;
    save("opaque", &flat)?;
    save("blend", &blend)?;
    save("cycles", &reference.pixels)?;
    Ok(())
}
