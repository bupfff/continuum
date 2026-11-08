//! M10 gate 19: alpha blending's speed.
//!
//! M10 gate 16 passed accuracy (4.71% from Cycles) and failed speed: 1.33×
//! the frame with the panes opaque, against 1.3×. Development runs before
//! this gate (`blend_dev`, disclosed; each camera pass submitted alone):
//! blending adds two full-screen passes, `hw_edges` (0.96 ms alone) and
//! `vc_shade_edges` (0.90 ms alone). With their work removed they still
//! cost about 0.25 ms each in the warm frame (3.77 against 3.24 ms):
//! launching a workgroup per 8×8 pixels of these large kernels, most of
//! which have nothing to do. The layer traces themselves cost about
//! 0.13 ms, and the list's global atomic nothing measurable (a
//! per-workgroup append did not change the frame). CPU encoding is 0.12 ms
//! against 0.09 ms.
//!
//! Method:
//! - **Layers in the visibility pass.** `hw_visibility_blend` traces on
//!   through blended surfaces where the camera ray's hit is blended, and
//!   lists the pixel, as `hw_edges` did. With blending and no silhouette
//!   rays, the `hw_edges` pass is gone; with both, it lists silhouettes
//!   only.
//! - **Shading only the list.** A one-thread pass sizes `vc_shade_edges`'s
//!   dispatch from the list's count (dispatched indirectly), instead of a
//!   workgroup per 8×8 pixels.
//!
//! Images are unchanged in principle: the same samples and weights.
//!
//! Scene and comparison are gate 16's (`Textured::with_glass(true)`; the
//! panes opaque; 1280×720, bounce light on).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) of the panes opaque and
//!    then blended; the median over rounds of the blended-to-opaque ratio
//!    at most 1.3× (gate 16's bound, gate 18's method).
//! 2. **Accuracy:** gate 16's criterion 1, unchanged: within 5% of Cycles,
//!    and over the pixels blending changes, at most 0.5× the error of the
//!    panes opaque.
//!
//! `cargo run --release -p room3d --bin m10_gate19`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 frames rendered since the call requested and baked
/// nothing; the frames it took.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> usize {
    let start = r.cache_log().len();
    for k in 1..=50_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return k;
        }
    }
    panic!("the view never settled");
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
    let t = Textured::new().with_glass(true);
    let mut opaque = Textured::new().with_glass(true);
    opaque.scene.geometry_blend.clear();
    let single = Textured::new().with_glass(false);
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
    let renderer = |blend: bool| {
        let scene = if blend { &t } else { &opaque };
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
        image.save(&out_dir.join(format!("m10_gate19_{name}.png")))
    };
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate19_dev.png");
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 1. Speed, first: five interleaved rounds (gate 18's method).
    let mut opaque_r = renderer(false);
    let mut blended = renderer(true);
    let frames = (
        settle(&mut opaque_r, t.camera, &target),
        settle(&mut blended, t.camera, &target),
    );
    let mut ratios = Vec::new();
    for round in 0..5 {
        let o = ms(warm(&mut opaque_r, t.camera, &target));
        let b = ms(warm(&mut blended, t.camera, &target));
        ratios.push(b / o);
        say(format!(
            "round {round}: panes opaque {o:.2} ms, blended {b:.2} ms ({:.3}×)",
            b / o
        ));
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[2];
    let c1 = median <= 1.3;
    say(format!(
        "1. speed: median ratio {median:.3}× over five rounds (at most 1.3×); settled in {} and {} frames: {}",
        frames.1,
        frames.0,
        verdict(c1)
    ));

    // 2. Accuracy against Cycles (gate 16's criterion 1).
    let with = supersampled(&mut blended, t.camera, &target);
    let without = supersampled(&mut opaque_r, t.camera, &target);
    let reference = render_instanced_lights(
        &single.scene,
        single.origin,
        &single.lights,
        single.sky,
        single.camera,
        (W, H),
        4096,
        1,
        1,
        &out_dir.join("m10_gate16_cycles"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    // The pixels blending changes.
    let mean_all = with
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64)
        .sum::<f64>()
        / (3 * with.len()) as f64;
    let changed: Vec<bool> = with
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
    let (ours_region, frames_region) = (
        relative_rmse(&keep(&with), &region),
        relative_rmse(&keep(&without), &region),
    );
    let ignoring = relative_rmse(&without, &reference.pixels);
    let c2 = ours <= 0.05 && ours_region <= 0.5 * frames_region;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "2. accuracy against Cycles ({:.0} s, {}): {:.2}% (at most 5%; panes opaque {:.2}%); over the {} pixels blending changes: {:.2}% against {:.2}% ({:.2}×, at most 0.5×); means: ours {:+.2}%, opaque {:+.2}%: {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * ignoring,
        changed.iter().filter(|&&c| c).count(),
        100.0 * ours_region,
        100.0 * frames_region,
        ours_region / frames_region,
        100.0 * (mean(&with) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&without) / mean(&reference.pixels) - 1.0),
        verdict(c2)
    ));
    save("ours", &with)?;
    save("opaque", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(out_dir.join("m10_gate19.txt"), report)?;
    Ok(())
}
