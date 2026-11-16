//! M16 gate 4: surface lobes (a dielectric's reflectance and a clear coat).
//!
//! Method (DESIGN §3.4, "Transmission", surface lobes):
//! `InstancedScene::geometry_specular` (index, factor, tint; glTF's
//! `KHR_materials_ior` and `KHR_materials_specular`) sets the specular
//! layer's reflectance at normal and grazing incidence;
//! `InstancedScene::geometry_clearcoat` (glTF's `KHR_materials_clearcoat`)
//! adds a coat (index 1.5): its own specular from each light and a
//! reflection ray, over what lies under it dimmed by its directional
//! albedo.
//!
//! Scene: M6's textured scene with the ground rough (0.6), the boulders a
//! rough dielectric (0.6) under a smooth clear coat (factor 1, roughness
//! 0.05) and the brick block of index 2.0 with its reflectance tinted
//! (1, 0.75, 0.5), roughness 0.3 (`Textured::with_lobes`). 1280×720, bounce
//! light on, cache settled (32 quiet frames). Ours: 4×4 sub-pixel offsets,
//! each settled, textures for a quarter pixel. Cycles: 4,096 samples, one
//! bounce, the Principled BSDF's IOR, Specular IOR Level (half the factor),
//! Specular Tint, Coat Weight and Roughness (Coat IOR 1.5).
//!
//! The bars were set with development numbers known (DESIGN §3.4: 0.58×
//! over the changed pixels, 1.18× the frame, at 640×360).
//!
//! A view that does not settle within 20,000 frames is reported and the
//! gate goes on.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, five rounds of 30 warm
//!    frames each (each waited for), the scene without the lobes then with
//!    them; the median over rounds of the ratio of their medians at most
//!    1.25×.
//! 2. **Accuracy,** every channel of both images clamped at 4× the
//!    reference's mean (the coat's sun highlights cover a few pixels at the
//!    sun's radiance): relative RMSE from Cycles at most 6%, and over the
//!    pixels the lobes change (against the scene without them, by more
//!    than 1% of the image's mean) at most 0.65× the error of ignoring
//!    them.
//!
//! Reported: the unclamped errors; the means over the changed pixels;
//! pixels above the clamp.
//!
//! Development mode (`M16_DEV=1`): 640×360, Cycles at 256 samples.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, MaskConeKernel, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

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
    eprintln!("WARNING: not settled after 20,000 frames");
    UNSETTLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    20_000
}

static UNSETTLED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M16_DEV").is_some();
    let (w, h, samples) = if dev {
        (640, 360, 256)
    } else {
        (1280, 720, 4096)
    };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        std::io::stdout().flush().ok();
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    say(format!(
        "M16 gate 4{}: {w}x{h}, adapter {}",
        if dev { " (development mode)" } else { "" },
        kernel.adapter_name
    ));
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
    let t = Textured::new().with_lobes();
    kernel.set_sky(t.sky);
    let mut opaque = t.scene.clone();
    opaque.geometry_specular.clear();
    opaque.geometry_clearcoat.clear();
    let make = |scene: &InstancedScene| {
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
        r
    };
    let timed = |r: &mut InstancedRenderer| {
        let mut v: Vec<f64> = (0..30)
            .map(|_| {
                let s = Instant::now();
                r.frame(t.camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        median(&mut v)
    };

    // 1. Speed.
    let mut glass = make(&t.scene);
    let mut flat = make(&opaque);
    let mut ratios = Vec::new();
    let mut rounds = String::new();
    for _ in 0..5 {
        let a = timed(&mut flat);
        let b = timed(&mut glass);
        ratios.push(b / a);
        rounds.push_str(&format!(" {b:.2}/{a:.2}"));
    }
    let ratio = median(&mut ratios);
    let c1 = ratio <= 1.25;
    say(format!(
        "1. speed: with / without the lobes {ratio:.3}x (at most 1.25x): {} (rounds, ms:{rounds})",
        verdict(c1)
    ));

    // 2. Accuracy.
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
        r.set_pixel_offset(0.0, 0.0);
        r.set_footprint(1.0);
        sum
    };
    let ours = supersampled(&mut glass);
    let without = supersampled(&mut flat);
    drop(glass);
    drop(flat);
    let reference = render_instanced_lights(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (w, h),
        samples,
        1,
        1,
        &out_dir.join(if dev {
            "m16_gate4_dev_cycles"
        } else {
            "m16_gate4_cycles"
        }),
    )?;
    let mean_all = ours
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64)
        .sum::<f64>()
        / (3 * ours.len()) as f64;
    let changed: Vec<bool> = ours
        .iter()
        .zip(&without)
        .map(|(a, b)| (0..3).any(|c| ((a[c] - b[c]) as f64).abs() > 0.01 * mean_all))
        .collect();
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(&changed)
            .filter(|(_, c)| **c)
            .map(|(p, _)| *p)
            .collect()
    };
    let ref_mean = reference
        .pixels
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64)
        .sum::<f64>()
        / (3 * reference.pixels.len()) as f64;
    let cap = (4.0 * ref_mean) as f32;
    let clamp =
        |img: &[[f32; 3]]| -> Vec<[f32; 3]> { img.iter().map(|p| p.map(|c| c.min(cap))).collect() };
    let rc = clamp(&reference.pixels);
    let (co, cw) = (clamp(&ours), clamp(&without));
    let whole = relative_rmse(&co, &rc);
    let region = keep(&rc);
    let (e_ours, e_flat) = (
        relative_rmse(&keep(&co), &region),
        relative_rmse(&keep(&cw), &region),
    );
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    let c2 = whole <= 0.06 && e_ours <= 0.65 * e_flat;
    say(format!(
        "2. accuracy against Cycles ({:.0} s, {}), clamped at 4x its mean: {:.2}% (at most 6%; ignoring the lobes {:.2}%); over the {} pixels the lobes change: {:.2}% against {:.2}% ignoring them ({:.3}x, at most 0.65x): {}",
        reference.seconds,
        reference.device,
        100.0 * whole,
        100.0 * relative_rmse(&cw, &rc),
        changed.iter().filter(|c| **c).count(),
        100.0 * e_ours,
        100.0 * e_flat,
        e_ours / e_flat,
        verdict(c2)
    ));
    say(format!(
        "   unclamped: {:.2}% (ignoring the lobes {:.2}%), over the changed pixels {:.2}% ({:.2}%); means over them: ours {:+.2}%, ignoring {:+.2}%; pixels above the clamp: ours {}, Cycles {}",
        100.0 * relative_rmse(&ours, &reference.pixels),
        100.0 * relative_rmse(&without, &reference.pixels),
        100.0 * relative_rmse(&keep(&ours), &keep(&reference.pixels)),
        100.0 * relative_rmse(&keep(&without), &keep(&reference.pixels)),
        100.0 * (mean(&keep(&ours)) / mean(&keep(&reference.pixels)) - 1.0),
        100.0 * (mean(&keep(&without)) / mean(&keep(&reference.pixels)) - 1.0),
        ours.iter().filter(|p| p.iter().any(|&c| c > cap)).count(),
        reference
            .pixels
            .iter()
            .filter(|p| p.iter().any(|&c| c > cap))
            .count()
    ));
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("m16_gate4_{name}.png")))
    };
    save("ours", &ours)?;
    save("opaque", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "views that did not settle within 20,000 frames: {}",
        UNSETTLED.load(std::sync::atomic::Ordering::Relaxed)
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}: {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m16_gate4_dev.txt"
        } else {
            "m16_gate4.txt"
        }),
        report,
    )?;
    Ok(())
}
