//! M10 gate 5: normal maps.
//!
//! glTF content carries surface detail in tangent-space normal maps; without
//! them, tiles and bricks shade flat.
//!
//! Method (in the cached path):
//! - `InstancedScene::geometry_normal_maps`: per geometry, a texture holding
//!   a tangent-space normal map (linear; OpenGL's convention, as glTF
//!   specifies: green toward the image's top), uploaded to its own linear
//!   layered texture with mip levels.
//! - The tangent frame per triangle, from its texture coordinates'
//!   derivatives (exact for planar texture mappings; MikkTSpace differs on
//!   curved meshes), orthogonalized against the interpolated normal.
//! - The mapped normal replaces the interpolated one for light from lights
//!   (the cached shadowed share times the unshadowed light at the mapped
//!   normal), specular and reflections. The cached sky and bounce light stay
//!   at the interpolated normal.
//!
//! Scene: the textured scene of M6 (sun and sky), its camera, 1280×720,
//! bounce light on, Lambertian. Normal maps, generated to match the color
//! textures: the ground's stone tiles beveled toward their mortar lines,
//! and the block's bricks likewise (bevels 12% of a tile's width, sloped
//! 30°).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 4%
//!    relative RMSE of instanced Cycles with the same normal maps (Normal
//!    Map node, tangent space, one bounce, 4,096 samples per pixel), and at
//!    most 0.6× the error of the same scene rendered ignoring the normal
//!    maps against that reference.
//! 2. **Speed:** the warm frame at most 1.15× the scene's without normal
//!    maps (median of 30 frames each waited for, bounce light on in both).
//!
//! Reported, not gated: Cycles with against without the normal maps (the
//! effect's size).
//!
//! Details fixed before the first run: as M10 gate 2 (settled: 32 quiet
//! frames after the call; 4×4 supersampling over sub-pixel offsets, each
//! settled; Cycles at 4,096 samples per pixel, its one-pixel box filter).
//!
//! `cargo run --release -p room3d --bin m10_gate5`
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
    let flat = Textured::new();
    let t = Textured::new().with_normal_maps();
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
    let renderer = |scene: &Textured| {
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
        image.save(&out_dir.join(format!("m10_gate5_{name}.png")))
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(&t);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate5_dev.png");
        return Ok(());
    }

    // 2. Speed, first.
    let mut plain = renderer(&flat);
    let plain_frames = settle(&mut plain, t.camera, &target);
    let plain_warm = warm(&mut plain, t.camera, &target);
    let mut mapped = renderer(&t);
    let mapped_frames = settle(&mut mapped, t.camera, &target);
    let mapped_warm = warm(&mut mapped, t.camera, &target);
    let c2 = ms(mapped_warm) <= 1.15 * ms(plain_warm);
    say(format!(
        "2. speed: warm {:.2} ms with normal maps, {:.2} ms without ({:.2}×, at most 1.15×); settled in {mapped_frames} and {plain_frames} frames: {}",
        ms(mapped_warm),
        ms(plain_warm),
        ms(mapped_warm) / ms(plain_warm),
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 1. Accuracy against Cycles with the normal maps.
    let with = supersampled(&mut mapped, t.camera, &target);
    let without = supersampled(&mut plain, t.camera, &target);
    let cycles = |scene: &Textured, dir: &str| {
        render_instanced_lights(
            &scene.scene,
            scene.origin,
            &scene.lights,
            scene.sky,
            scene.camera,
            (W, H),
            4096,
            1,
            1,
            &out_dir.join(dir),
        )
    };
    let reference = cycles(&t, "m10_gate5_cycles")?;
    let ours = relative_rmse(&with, &reference.pixels);
    let ignoring = relative_rmse(&without, &reference.pixels);
    let c1 = ours <= 0.04 && ours <= 0.6 * ignoring;
    say(format!(
        "1. accuracy against Cycles with normal maps ({:.0} s, {}): {:.2}% (at most 4%), ignoring them {:.2}% (ours at most 0.6×: {:.2}×): {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * ignoring,
        ours / ignoring,
        if c1 { "PASS" } else { "FAIL" }
    ));
    let reference_flat = cycles(&flat, "m10_gate5_cycles_flat")?;
    say(format!(
        "reported: Cycles without normal maps against with: {:.2}%",
        100.0 * relative_rmse(&reference_flat.pixels, &reference.pixels)
    ));
    save("ours", &with)?;
    save("ignoring", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m10_gate5.txt"), report)?;
    Ok(())
}
