//! M10 gate 4: free re-coloring with bounce light, with a corrected settle.
//!
//! Gates 1 and 3 failed their re-coloring check (794 and 522 pages baked in
//! the 30 frames after re-coloring). The cause was the check's procedure,
//! not the renderer. Their `settle` looked at the last 32 frames already
//! read back, which, right after another settle, were that settle's quiet
//! frames, so it returned after one frame. The new view's requests arrived
//! later, inside the re-coloring window. (Each sub-pixel offset of their
//! supersampling was likewise rendered one frame after the switch.)
//!
//! Here `settle` waits for 32 quiet frames rendered after it is called.
//! Renderer as in gate 3 (secondary requests for bounce and reflection hits).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Re-coloring:** gate 1's scene and procedure (settle, supersample
//!    4×4, settle, re-color the sun), with the corrected settle: no page
//!    baked over the next 30 frames.
//!
//! Reported, not gated: gate 1's and gate 2's images against their Cycles
//! references, supersampled with the corrected settle.
//!
//! `cargo run --release -p room3d --bin m10_gate4`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 consecutive frames rendered since the call requested
/// and baked nothing (read back two frames late); the frames it took.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> usize {
    let start = r.cache_log().len();
    let mut frames = 0;
    loop {
        r.frame(camera, target);
        frames += 1;
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        let log = r.cache_log();
        if latest >= start + 32
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

    let mut shiny = renderer(&shiny_scene, true);
    settle(&mut shiny, camera, &target);
    // 3. Accuracy against the same references.
    let shiny_image = supersampled(&mut shiny, camera, &target);
    let r1 = reference(&plain_scene, "m10_gate1_cycles_1b")?;
    let r2 = reference(&shiny_scene, "m10_gate2_cycles")?;
    let (e1, e2) = (
        relative_rmse(&with, &r1.pixels),
        relative_rmse(&shiny_image, &r2.pixels),
    );
    say(format!(
        "reported: gate 1's scene {:.2}% from Cycles with one bounce, gate 2's {:.2}% (corrected settle) [{} / {}]",
        100.0 * e1,
        100.0 * e2,
        r1.device,
        r2.device
    ));
    say(format!(
        "Verdict: 1 {} ({:.0} s)",
        if c1 { "PASS" } else { "FAIL" },
        start.elapsed().as_secs_f64()
    ));
    std::fs::write(out_dir.join("m10_gate4.txt"), report)?;
    Ok(())
}
