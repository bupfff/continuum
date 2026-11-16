//! M18 gate 1: fog with light shafts.
//!
//! Method (DESIGN §3.4, "Fog with light shafts"): `CacheOptions::fog`. The
//! first light's visibility in the air comes from a world-space grid
//! around the eye (128 × 128 × 32 cells to the fog's top, snapped to its
//! cells), each cell traced toward it by a hardware ray and kept while the
//! eye stays in its cells; per frame, froxel columns (8 × 8 pixels, 64
//! slices) integrate the fog front to back (each slice's optical depth
//! exact, the light gathered at its middle: the first light's through its
//! visibility, the others' unshadowed, the sky's), and each pixel takes
//! what the fog lets through to its hit plus the light gathered. Surfaces
//! get each light dimmed by the fog.
//!
//! Scene: M6's textured scene with M10 gate 16's panes as thin glass
//! (`Textured::with_transmission`: double-sided for Continuum, one-sided
//! for Cycles), sun and sky, bounce light off. Views: (a) its camera in an
//! even layer (0.05 per metre, top 6 m, albedo 0.9, anisotropy 0.3); (b)
//! from beside the brick block (eye (6, 4, 1.6) toward (−3, 9.5, 0.6)) in a
//! layer falling with height (0.08 at the ground, 0.3 per metre, top 6 m,
//! albedo 0.9, anisotropy 0.3). 1280×720, cache settled. Ours: 4×4
//! sub-pixel offsets, each settled. Cycles: 4,096 samples, one bounce, a
//! box 500 m across from 1 m below the ground to the top holding Volume
//! Scatter and Volume Absorption at the fog's density with its albedo as
//! colour, no volume bounces.
//!
//! The bars were set with development numbers known (DESIGN §3.4: 0.218×
//! and 0.151× the error without fog, 0.80 ms at 1280×720).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost** (first): on view (a), fog adds at most 1.0 ms to the warm
//!    frame (medians of five interleaved rounds of 30 frames, each waited
//!    for).
//! 2. **Accuracy,** in each view: relative RMSE from Cycles at most 12%,
//!    and at most 0.35× that of the image without fog.
//!
//! Reported: the image's mean against Cycles' with and without fog; views
//! that did not settle.
//!
//! Development mode (`M18_DEV=1`): 640×360, Cycles at 256 samples.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Camera, Fog, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_fog};
use room3d::textured::Textured;

static UNSETTLED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
    eprintln!("WARNING: not settled after 20,000 frames");
    UNSETTLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M18_DEV").is_some();
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
        "M18 gate 1{}: {w}x{h}, adapter {}",
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
    let beside = Camera {
        eye: Vec3::new(6.0, 4.0, 1.6),
        target: Vec3::new(-3.0, 9.5, 0.6),
        vertical_fov_degrees: 45.0,
    };
    let views = [
        (
            "(a) even layer",
            t.camera,
            Fog {
                density: 0.05,
                height_falloff: 0.0,
                top: 6.0,
                albedo: Vec3::splat(0.9),
                anisotropy: 0.3,
                ..Default::default()
            },
        ),
        (
            "(b) falling with height, from beside",
            beside,
            Fog {
                density: 0.08,
                height_falloff: 0.3,
                top: 6.0,
                albedo: Vec3::splat(0.9),
                anisotropy: 0.3,
                ..Default::default()
            },
        ),
    ];
    let make = |fog: Option<Fog>, camera: Camera| {
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
        settle(&mut r, camera, &target);
        r
    };

    // 1. Cost, view (a).
    let (_, camera_a, fog_a) = views[0];
    let mut foggy = make(Some(fog_a), camera_a);
    let mut clear = make(None, camera_a);
    let timed = |r: &mut InstancedRenderer| {
        let mut v: Vec<f64> = (0..30)
            .map(|_| {
                let s = Instant::now();
                r.frame(camera_a, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        median(&mut v)
    };
    let mut diffs = Vec::new();
    let mut rounds = String::new();
    for _ in 0..5 {
        let off = timed(&mut clear);
        let on = timed(&mut foggy);
        diffs.push(on - off);
        rounds.push_str(&format!(" {on:.3}/{off:.3}"));
    }
    drop(foggy);
    drop(clear);
    let added = median(&mut diffs);
    let c1 = added <= 1.0;
    say(format!(
        "1. cost: fog adds {added:.3} ms (at most 1.0): {} (rounds, with/without ms:{rounds})",
        verdict(c1)
    ));

    // 2. Accuracy.
    let supersampled = |r: &mut InstancedRenderer, camera: Camera| {
        let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        for k in 0..16 {
            let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
            r.set_pixel_offset(dx, dy);
            settle(r, camera, &target);
            r.finish();
            for (s, p) in sum.iter_mut().zip(r.read_colors()) {
                for c in 0..3 {
                    s[c] += p[c] / 16.0;
                }
            }
        }
        sum
    };
    let mut c2 = true;
    for (k, (name, camera, fog)) in views.iter().enumerate() {
        let ours = supersampled(&mut make(Some(*fog), *camera), *camera);
        let none = supersampled(&mut make(None, *camera), *camera);
        let reference = render_instanced_fog(
            &single.scene,
            single.origin,
            &single.lights,
            single.sky,
            (fog, -1.0, 500.0),
            *camera,
            (w, h),
            samples,
            1,
            1,
            &out_dir.join(format!(
                "m18_gate1{}_cycles_{k}",
                if dev { "_dev" } else { "" }
            )),
        )?;
        let (e, e0) = (
            relative_rmse(&ours, &reference.pixels),
            relative_rmse(&none, &reference.pixels),
        );
        let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
        let ok = e <= 0.12 && e <= 0.35 * e0;
        c2 &= ok;
        say(format!(
            "2. {name} ({:.0} s, {}): {:.2}% from Cycles (at most 12%), without fog {:.2}% ({:.3}x, at most 0.35x); means: ours {:+.2}%, without {:+.2}%: {}",
            reference.seconds,
            reference.device,
            100.0 * e,
            100.0 * e0,
            e / e0,
            100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0),
            100.0 * (mean(&none) / mean(&reference.pixels) - 1.0),
            verdict(ok)
        ));
        let save = |tag: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
            let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                image.set(
                    i % w as usize,
                    i / w as usize,
                    room3d::to_rgb(Vec3::from_array(*p)),
                );
            }
            image.save(&out_dir.join(format!("m18_gate1_{k}_{tag}.png")))
        };
        save("ours", &ours)?;
        save("clear", &none)?;
        save("cycles", &reference.pixels)?;
    }
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
            "m18_gate1_dev.txt"
        } else {
            "m18_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
