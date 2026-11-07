//! M10 gate 3: bounce light and reflections without request cascades.
//!
//! M10 gates 1 and 2 left the cache restless. Pages requested only for
//! light at bounce or reflection hits were baked with bounce samples of
//! their own, whose rays requested more pages, and so on through the scene.
//! Pages seen only in reflections were evicted and requested again. With
//! materials, a view took 4,266 frames to settle against 621; after the
//! supersampling, requests resumed and failed gate 1's re-coloring check.
//!
//! The fix: requests from bounce and reflection lookups are secondary.
//! They are planned after pixels' requests and baked without bounce
//! samples, and the fill clears what a previous page in the slot left. A
//! page shown by a pixel without samples asks for them, and the CPU
//! schedules them. Pages read through reflections are marked used.
//!
//! Disclosed: the fix and a development check (`M10_DEV=1` of gates 1 and
//! 2: settled in 602 and 585 frames) came before this stub was written.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Re-coloring:** gate 1's check again, on its scene and procedure
//!    (settle, supersample 4×4, settle, re-color the sun): no page baked
//!    over the next 30 frames.
//! 2. **Settling with materials:** gate 2's scene (materials, bounce light)
//!    settles in at most 1.5× the frames of the same scene, Lambertian,
//!    without bounce light.
//! 3. **No loss of accuracy:** against the same Cycles references,
//!    gate 1's image within 5% (it was 1.79%) and gate 2's within 0.5
//!    points of its 8.94%.
//!
//! `cargo run --release -p room3d --bin m10_gate3`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

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
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let plain_scene = Textured::new();
    let shiny_scene = Textured::new().with_materials();
    kernel.set_sky(plain_scene.sky);
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
    let renderer = |scene: &Textured, indirect: bool| {
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
                indirect,
                ..Default::default()
            },
        )
    };
    let reference = |scene: &Textured, dir: &str| {
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
    let camera = plain_scene.camera;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let start = Instant::now();

    // 1. Gate 1's re-coloring check, with its procedure.
    let mut bounce = renderer(&plain_scene, true);
    settle(&mut bounce, camera, &target);
    let with = supersampled(&mut bounce, camera, &target);
    settle(&mut bounce, camera, &target);
    let before = bounce.cache_log().len();
    bounce.set_light_radiance(0, plain_scene.lights[0].radiance * Vec3::new(1.0, 0.7, 0.5));
    for _ in 0..30 {
        bounce.frame(camera, &target);
    }
    bounce.finish();
    let baked: u32 = bounce.cache_log()[before..]
        .iter()
        .map(|s| s.pages_baked)
        .sum();
    let c1 = baked == 0;
    say(format!(
        "1. re-coloring: {baked} pages baked over 30 frames: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));

    // 2. Settling with materials.
    let mut shiny = renderer(&shiny_scene, true);
    let shiny_frames = settle(&mut shiny, camera, &target);
    let mut plain = renderer(&plain_scene, false);
    let plain_frames = settle(&mut plain, camera, &target);
    let c2 = shiny_frames as f64 <= 1.5 * plain_frames as f64;
    say(format!(
        "2. settling: {shiny_frames} frames with materials and bounce light, {plain_frames} Lambertian without ({:.2}×, at most 1.5×): {}",
        shiny_frames as f64 / plain_frames as f64,
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 3. Accuracy against the same references.
    let shiny_image = supersampled(&mut shiny, camera, &target);
    let r1 = reference(&plain_scene, "m10_gate1_cycles_1b")?;
    let r2 = reference(&shiny_scene, "m10_gate2_cycles")?;
    let (e1, e2) = (
        relative_rmse(&with, &r1.pixels),
        relative_rmse(&shiny_image, &r2.pixels),
    );
    let c3 = e1 <= 0.05 && (e2 - 0.0894).abs() <= 0.005;
    say(format!(
        "3. accuracy: gate 1's scene {:.2}% (at most 5%), gate 2's {:.2}% (within 0.5 points of 8.94%) [{} / {}]: {}",
        100.0 * e1,
        100.0 * e2,
        r1.device,
        r2.device,
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {} ({:.0} s)",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" },
        start.elapsed().as_secs_f64()
    ));
    std::fs::write(out_dir.join("m10_gate3.txt"), report)?;
    Ok(())
}
