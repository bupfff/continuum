//! M10 gate 1: one bounce of light in the view-dependent cache.
//!
//! The instanced renderer lights with direct light only; one bounce ran
//! only in the lightmap path (M2 gate 6). Outdoors, the light the ground
//! and objects reflect fills shadows; without it they are too dark.
//!
//! Method:
//! - **Samples per page.** Each page holds a 3×3 grid of indirect samples
//!   (at its texels 0, 4 and 8 along each axis), read bilinearly by
//!   shading. Bounce light varies slowly; a sample costs hundreds of rays.
//! - **A hemicube of bins.** A sample casts one ray through the center of
//!   each of 192 bins of a hemicube around its normal (M2 gate 6's), each
//!   weighted by its projected solid angle.
//! - **Direct light at the hits from the cache.** Each hit's outgoing light
//!   is its albedo (textured) over π times its direct light, read from the
//!   cache's page at the hit at a mip matched to the bin's footprint (each
//!   light's ratio times its unshadowed light there, and the sky's channel).
//!   A hit with no page resident requests one and counts as missing; the
//!   page's samples re-bake once the requested pages have arrived.
//! - **Per light and the sky, in RGB,** per unit radiance, so re-coloring a
//!   light still bakes nothing.
//!
//! Scene: the textured scene of M6 (`Textured::new()`, sun and sky), its
//! camera, 1280×720, cache defaults with bounce light on.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4 (16 frames
//!    over sub-pixel offsets), within 5% relative RMSE of instanced Cycles
//!    with one diffuse bounce, and at most half the error of the same image
//!    without bounce light against that reference.
//! 2. **Speed:** the warm frame with bounce light at most 1.25× the warm
//!    frame without (median of 30 frames each waited for); the view settles
//!    in at most 3× the frames.
//! 3. **Free re-coloring:** re-coloring the sun on a settled view bakes
//!    nothing over the next 30 frames.
//!
//! Reported, not gated: the bounce's share of the image; samples per page
//! re-baked for missing hits; the error of the direct-only image against
//! direct-only Cycles, as a control.
//!
//! Details fixed before the first run:
//! - "Settled": 32 consecutive frames with nothing requested or baked (a
//!   frame's requests sample one pixel in 16, so 32 frames see each twice).
//! - Supersampling: 16 frames whose camera rays cross each pixel at the
//!   centers of a 4×4 grid (`set_pixel_offset`), textures sampled for a
//!   quarter pixel, each after its view settles; averaged. Cycles at 4,096
//!   samples per pixel, its one-pixel box filter.
//!
//! `cargo run --release -p room3d --bin m10_gate1`
//!
//! `M10_DEV=1` runs criterion 2 alone at 1280×720 (no Cycles): a
//! development check, whose numbers are not the gate's.

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
    let t = Textured::new();
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
    let renderer = |indirect: bool| {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions {
                indirect,
                ..Default::default()
            },
        )
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    // 2. Speed, first: settling and warm frames, without and with.
    let mut plain = renderer(false);
    let plain_frames = settle(&mut plain, t.camera, &target);
    let plain_warm = warm(&mut plain, t.camera, &target);
    let mut bounce = renderer(true);
    let bounce_frames = settle(&mut bounce, t.camera, &target);
    let bounce_warm = warm(&mut bounce, t.camera, &target);
    bounce.finish();
    let log = bounce.cache_log();
    let samples: u64 = log.iter().map(|s| s.indirect_samples as u64).sum();
    let incomplete: u64 = log.iter().map(|s| s.indirect_incomplete as u64).sum();
    let c2 = ms(bounce_warm) <= 1.25 * ms(plain_warm) && bounce_frames <= 3 * plain_frames;
    say(format!(
        "2. speed: warm {:.2} ms with bounce light, {:.2} ms without ({:.2}×, at most 1.25×); settled in {bounce_frames} frames, {plain_frames} without ({:.2}×, at most 3×): {}",
        ms(bounce_warm),
        ms(plain_warm),
        ms(bounce_warm) / ms(plain_warm),
        bounce_frames as f64 / plain_frames as f64,
        if c2 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "reported: {samples} bounce samples baked while settling, {incomplete} met missing hits"
    ));
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        if std::env::var_os("M10_TRACE").is_some() {
            if std::env::var_os("M10_SUPER").is_some() {
                supersampled(&mut bounce, t.camera, &target);
            }
            settle(&mut bounce, t.camera, &target);
            let before = bounce.cache_log().len();
            if std::env::var_os("M10_RECOLOR").is_some() {
                bounce.set_light_radiance(0, t.lights[0].radiance * Vec3::new(1.0, 0.7, 0.5));
            }
            for _ in 0..200 {
                bounce.frame(t.camera, &target);
            }
            bounce.finish();
            for s in &bounce.cache_log()[before..] {
                if s.pages_baked > 0 || s.requests > 0 {
                    println!(
                        "frame {}: baked {} pages, {} requests, {} samples, {} incomplete, evicted {}",
                        s.frame,
                        s.pages_baked,
                        s.requests,
                        s.indirect_samples,
                        s.indirect_incomplete,
                        s.evicted
                    );
                }
            }
        }
        return Ok(());
    }

    // 1. Accuracy against Cycles with one bounce.
    let with = supersampled(&mut bounce, t.camera, &target);
    let without = supersampled(&mut plain, t.camera, &target);
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
        &out_dir.join("m10_gate1_cycles_1b"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    let direct = relative_rmse(&without, &reference.pixels);
    let c1 = ours <= 0.05 && ours <= 0.5 * direct;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "1. accuracy against Cycles with one bounce ({:.0} s, {}): {:.2}% with bounce light (at most 5%), {:.2}% without (ours at most half: {:.2}×); means: ours {:+.2}%, without {:+.2}% against Cycles: {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * direct,
        ours / direct,
        100.0 * (mean(&with) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&without) / mean(&reference.pixels) - 1.0),
        if c1 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "reported: bounce light adds {:.2}% to the image",
        100.0 * (mean(&with) / mean(&without) - 1.0)
    ));
    let control = render_instanced_lights(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (W, H),
        4096,
        0,
        1,
        &out_dir.join("m10_gate1_cycles_0b"),
    )?;
    say(format!(
        "reported, control: without bounce light against Cycles without bounces: {:.2}%",
        100.0 * relative_rmse(&without, &control.pixels)
    ));
    for (name, px) in [
        ("bounce", &with),
        ("direct", &without),
        ("cycles_1b", &reference.pixels),
    ] {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % W as usize,
                i / W as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("m10_gate1_{name}.png")))?;
    }

    // 3. Free re-coloring.
    settle(&mut bounce, t.camera, &target);
    let before = bounce.cache_log().len();
    bounce.set_light_radiance(0, t.lights[0].radiance * Vec3::new(1.0, 0.7, 0.5));
    for _ in 0..30 {
        bounce.frame(t.camera, &target);
    }
    bounce.finish();
    let baked: u32 = bounce.cache_log()[before..]
        .iter()
        .map(|s| s.pages_baked)
        .sum();
    let c3 = baked == 0;
    say(format!(
        "3. re-coloring the sun: {baked} pages baked over 30 frames: {}",
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m10_gate1.txt"), report)?;
    Ok(())
}
