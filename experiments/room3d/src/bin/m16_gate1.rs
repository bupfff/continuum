//! M16 gate 1: thin transmissive glass.
//!
//! Method (DESIGN §3.4, "Transmission"): `InstancedScene::geometry_transmission`
//! (glTF's `KHR_materials_transmission` without a volume). Of the light a
//! surface's specular layer does not reflect, the factor passes straight
//! through tinted by the base colour, the rest is diffuse; the specular
//! layer keeps its full reflection. Camera rays see through it (alpha
//! blending's layered records); cones weigh its triangles by what passes
//! at their angle (Fresnel, Schlick's), untinted.
//!
//! Scene: M6's textured scene with M10 gate 16's two panes as smooth thin
//! glass (`Textured::with_transmission`: blue-tinted, transmission 0.9;
//! green, 0.7), double-sided for Continuum and one-sided for Cycles (as
//! gate 16). 1280×720, bounce light on, cache settled (32 quiet frames).
//! Ours: 4×4 sub-pixel offsets, each settled, textures for a quarter pixel.
//! Cycles: 4,096 samples, one bounce, its reference material (the opaque
//! Principled BSDF mixed by one less the factor with its specular layer
//! plus a Transparent BSDF passing the base colour times one less
//! Schlick's Fresnel reflectance).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, five rounds of 30 warm
//!    frames each (each waited for), the panes opaque then transmissive;
//!    the median over rounds of the ratio of their medians at most 1.3×.
//! 2. **Accuracy:** relative RMSE from Cycles at most 7%, and over the
//!    pixels transmission changes (against the panes opaque, by more than
//!    1% of the image's mean) at most 0.5× the error of the panes drawn as
//!    alpha blending with opacity one less the factor.
//!
//! Reported: the panes opaque against Cycles; the means over the changed
//! pixels.
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
    panic!("never settled");
}

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
        "M16 gate 1{}: {w}x{h}, adapter {}",
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
    let t = Textured::new().with_transmission(true);
    let single = Textured::new().with_transmission(false);
    kernel.set_sky(t.sky);
    let mut opaque = t.scene.clone();
    opaque.geometry_transmission.clear();
    let mut blended = t.scene.clone();
    blended.geometry_blend = t
        .scene
        .geometry_transmission
        .iter()
        .map(|x| x.map(|tr| 1.0 - tr))
        .collect();
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
    let c1 = ratio <= 1.3;
    say(format!(
        "1. speed: transmissive / opaque panes {ratio:.3}x (at most 1.3x): {} (rounds, ms:{rounds})",
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
    let as_blend = supersampled(&mut make(&blended));
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
        &out_dir.join(if dev {
            "m16_gate1_dev_cycles"
        } else {
            "m16_gate1_cycles"
        }),
    )?;
    let whole = relative_rmse(&ours, &reference.pixels);
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
    let region = keep(&reference.pixels);
    let (e_ours, e_blend, e_flat) = (
        relative_rmse(&keep(&ours), &region),
        relative_rmse(&keep(&as_blend), &region),
        relative_rmse(&keep(&without), &region),
    );
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    let c2 = whole <= 0.07 && e_ours <= 0.5 * e_blend;
    say(format!(
        "2. accuracy against Cycles ({:.0} s, {}): {:.2}% (at most 7%; panes opaque {:.2}%, as blending {:.2}%); over the {} pixels transmission changes: {:.2}% against blending's {:.2}% ({:.3}x, at most 0.5x; opaque {:.2}%); means over them: ours {:+.2}%, blending {:+.2}%, opaque {:+.2}%: {}",
        reference.seconds,
        reference.device,
        100.0 * whole,
        100.0 * relative_rmse(&without, &reference.pixels),
        100.0 * relative_rmse(&as_blend, &reference.pixels),
        changed.iter().filter(|c| **c).count(),
        100.0 * e_ours,
        100.0 * e_blend,
        e_ours / e_blend,
        100.0 * e_flat,
        100.0 * (mean(&keep(&ours)) / mean(&region) - 1.0),
        100.0 * (mean(&keep(&as_blend)) / mean(&region) - 1.0),
        100.0 * (mean(&keep(&without)) / mean(&region) - 1.0),
        verdict(c2)
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
        image.save(&out_dir.join(format!("m16_gate1_{name}.png")))
    };
    save("ours", &ours)?;
    save("opaque", &without)?;
    save("blend", &as_blend)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {}: {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m16_gate1_dev.txt"
        } else {
            "m16_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
