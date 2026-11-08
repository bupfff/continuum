//! M10 gate 10: materials accuracy (M10 gate 2 re-run after a fix).
//!
//! M10 gate 2 failed on accuracy: 8.94% from Cycles against 6%. Diagnosis
//! (candidate 5 of DESIGN §3.8, `M10_DEV=ss` in `m10_gate2`, full size,
//! accuracy only, against gate 2's cached reference; disclosed: these
//! numbers were seen before this gate was written):
//! - Today's code before any change: 9.04%.
//! - Pixels where Cycles exceeds 1 (sun highlights on the metal boulders,
//!   0.24% of pixels) carried 75% of the squared error; the rest of the
//!   image was 4.67% from Cycles. Highlight positions agreed (2,177 of
//!   2,217 pixels); their energy was 3.5% low, and 5.4% low where Cycles
//!   exceeds 10.
//! - Reflection rays kept above the surface (shading normals can turn the
//!   mirror direction below it): 9.04%, no change.
//! - Eight rays across the glossy lobe instead of one: 8.85%. Not the
//!   cause, not kept.
//! - **The cause:** `specular_light` evaluated GGX's D and V at the
//!   widened roughness α′ *and* multiplied by Karis's energy normalization
//!   (α/α′)², which lowers a small light's peak twice. Karis (2013)
//!   evaluates D at α and normalizes once. With that: 5.85% (highlights
//!   1.9% low, 40% of the squared error).
//!
//! Fixes in this gate: D and V at the material's roughness with the
//! normalization once; reflection rays kept just above the geometric
//! surface (as Cycles' ensure_valid_reflection).
//!
//! Method, scene and the criteria below are gate 2's, unchanged:
//!
//! Method (in the cached path):
//! - **Materials:** per geometry, metallic and roughness (glTF's model:
//!   F0 = 0.04 for dielectrics, the base color for metals; GGX microfacets,
//!   α = roughness²). Geometries without one stay Lambertian.
//! - **Light from lights:** GGX specular evaluated toward each light's
//!   nearest point to the reflection direction (a representative point,
//!   Karis 2013, the lobe widened by the light's angular size), times the
//!   cached shadowed share (the diffuse visibility stands in for the
//!   specular's).
//! - **Glossy reflections:** one ray per pixel along the reflection
//!   direction. The hit's light is read from the cache at the mip matching
//!   the lobe's width there (a roughness cone, prefiltered by the cache's
//!   mips rather than sampled, so no noise). A miss sees the sky. Weighted
//!   by the split-sum's directional albedo (Karis's analytic fit).
//! - **Diffuse:** the base color times (1 − metallic), less what the
//!   specular layer reflects.
//!
//! Scene: the textured scene of M6, its camera, 1280×720, sun and sky,
//! bounce light on. Materials: the ground dielectric, roughness 0.6; the
//! boulders metallic, roughness 0.3; the brick block dielectric, roughness
//! 0.4.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 6%
//!    relative RMSE of instanced Cycles (Principled BSDF, GGX, one bounce,
//!    4,096 samples per pixel), and at most 0.6× the error of the same
//!    scene rendered Lambertian against that reference.
//! 2. **Speed:** the warm frame at most 2× the Lambertian scene's (median
//!    of 30 frames each waited for, bounce light on in both).
//!
//! Reported, not gated: the specular share of the image; the error with
//! materials but without reflection rays.
//!
//! Details fixed before the first run: as M10 gate 1 (settled: 32 quiet
//! frames; 4×4 supersampling over sub-pixel offsets, each settled; Cycles
//! at 4,096 samples per pixel, its one-pixel box filter). "Without
//! reflection rays" keeps the materials and the specular from lights.
//!
//! `cargo run --release -p room3d --bin m10_gate10`
//!
//! `M10_DEV=1` saves the settled image alone (no Cycles): a development
//! check, whose numbers are not the gate's.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 consecutive frames requested and baked nothing; the
/// frames it took.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> usize {
    let mut frames = 0;
    loop {
        r.frame(camera, target);
        frames += 1;
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
            return frames;
        }
        assert!(frames < 50_000, "the view never settled");
    }
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

/// The view supersampled 4×4: each sub-pixel offset's frame after it
/// settles, averaged.
fn supersampled(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> Vec<[f32; 3]> {
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        settle(r, camera, target);
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
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let lambert = Textured::new();
    let t = Textured::new().with_materials();
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
    let renderer = |scene: &Textured, glossy_rays: bool| {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &scene.lights,
            &scene.scene,
            scene.origin,
            CacheOptions {
                indirect: true,
                glossy_rays,
                ..Default::default()
            },
        )
    };
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % W as usize,
                i / W as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("m10_gate10_{name}.png")))
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(&t, true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate10_dev.png");
        return Ok(());
    }

    // 2. Speed, first.
    let mut plain = renderer(&lambert, true);
    let plain_frames = settle(&mut plain, t.camera, &target);
    let plain_warm = warm(&mut plain, t.camera, &target);
    let mut shiny = renderer(&t, true);
    let shiny_frames = settle(&mut shiny, t.camera, &target);
    let shiny_warm = warm(&mut shiny, t.camera, &target);
    let c2 = ms(shiny_warm) <= 2.0 * ms(plain_warm);
    say(format!(
        "2. speed: warm {:.2} ms with materials, {:.2} ms Lambertian ({:.2}×, at most 2×); settled in {shiny_frames} and {plain_frames} frames: {}",
        ms(shiny_warm),
        ms(plain_warm),
        ms(shiny_warm) / ms(plain_warm),
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 1. Accuracy against Cycles with the materials.
    let with = supersampled(&mut shiny, t.camera, &target);
    let without = supersampled(&mut plain, t.camera, &target);
    let mut flat = renderer(&t, false);
    settle(&mut flat, t.camera, &target);
    let no_rays = supersampled(&mut flat, t.camera, &target);
    let reference = render_instanced_lights(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (W, H),
        4096,
        1,
        1,
        &out_dir.join("m10_gate2_cycles"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    let lambertian = relative_rmse(&without, &reference.pixels);
    let c1 = ours <= 0.06 && ours <= 0.6 * lambertian;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "1. accuracy against Cycles with materials ({:.0} s, {}): {:.2}% (at most 6%), Lambertian {:.2}% (ours at most 0.6×: {:.2}×); means: ours {:+.2}%, Lambertian {:+.2}%: {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * lambertian,
        ours / lambertian,
        100.0 * (mean(&with) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&without) / mean(&reference.pixels) - 1.0),
        if c1 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "reported: without reflection rays {:.2}% (mean {:+.2}%); materials change the image's mean by {:+.2}%",
        100.0 * relative_rmse(&no_rays, &reference.pixels),
        100.0 * (mean(&no_rays) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&with) / mean(&without) - 1.0)
    ));
    save("ours", &with)?;
    save("lambertian", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m10_gate10.txt"), report)?;
    Ok(())
}
