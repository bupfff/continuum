//! M10 gate 20: texture wrap modes (glTF samplers).
//!
//! Textures have always repeated. glTF samplers also clamp to the edge and
//! mirror, per axis.
//!
//! Method (`InstancedScene::texture_wrap`, per texture and axis: repeat,
//! clamp or mirror):
//! - Each object's modes for its four texture slots (base color, emissive,
//!   normal map, metallic-roughness) ride in a word per object appended to
//!   the object-texture buffer (no new binding).
//! - Every texture read applies them to the coordinates before sampling
//!   with the repeating sampler: clamp keeps them half a texel (at the
//!   coarser of the two mip levels read) inside the edges, so filtering
//!   never reaches the far edge; mirror folds them into [0, 1] first
//!   (repeating every two tiles) and then clamps the same way, which
//!   repeats the edge texel at each fold, as glTF's mirrored repeat does.
//!   The box filter's taps, opacity, normal maps, metallic-roughness and
//!   emission all go through it.
//! - Alpha cutouts (cut on the CPU) index texels with the same modes.
//! - glTF import reads each texture's sampler (`wrapS`, `wrapT`); save
//!   writes them.
//!
//! Scene: M6's textured scene, its camera, sun and sky, 1280×720, bounce
//! light on, with the ground's tiles clamped (its coordinates run from −7.5
//! to 7.5, so most of the ground shows the edge texels), the boulders'
//! lichen and the block's bricks mirrored. The comparison is the same scene
//! with every texture repeating.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) of the repeating scene and
//!    then the wrapped one; the median over rounds of the ratio at most
//!    1.1× (gate 18's method).
//! 2. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles (one bounce, 4,096 samples per
//!    pixel; each image texture's extension EXTEND or MIRROR) and at most
//!    0.5× the error of the repeating scene against that reference.
//!
//! `cargo run --release -p room3d --bin m10_gate20`

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
    let t = Textured::new().with_wrap();
    let flat = Textured::new();
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
    let renderer = |wrapped: bool| {
        let scene = if wrapped { &t } else { &flat };
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
        image.save(&out_dir.join(format!("m10_gate20_{name}.png")))
    };
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate20_dev.png");
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 1. Speed, first: five interleaved rounds (gate 18's method).
    let mut plain = renderer(false);
    let mut tex = renderer(true);
    let frames = (
        settle(&mut plain, t.camera, &target),
        settle(&mut tex, t.camera, &target),
    );
    let mut ratios = Vec::new();
    for round in 0..5 {
        let p = ms(warm(&mut plain, t.camera, &target));
        let w = ms(warm(&mut tex, t.camera, &target));
        ratios.push(w / p);
        say(format!(
            "round {round}: repeating {p:.2} ms, wrapped {w:.2} ms ({:.3}×)",
            w / p
        ));
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[2];
    let c1 = median <= 1.1;
    say(format!(
        "1. speed: median ratio {median:.3}× over five rounds (at most 1.1×); settled in {} (wrapped) and {} (repeating) frames: {}",
        frames.1,
        frames.0,
        verdict(c1)
    ));

    // 2. Accuracy against Cycles.
    let with = supersampled(&mut tex, t.camera, &target);
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
        &out_dir.join("m10_gate20_cycles"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    let ignoring = relative_rmse(&without, &reference.pixels);
    let c2 = ours <= 0.05 && ours <= 0.5 * ignoring;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "2. accuracy against Cycles ({:.0} s, {}): {:.2}% (at most 5%), repeating {:.2}% (ours at most 0.5×: {:.2}×); means: ours {:+.2}%, repeating {:+.2}%: {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * ignoring,
        ours / ignoring,
        100.0 * (mean(&with) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&without) / mean(&reference.pixels) - 1.0),
        verdict(c2)
    ));
    save("ours", &with)?;
    save("repeating", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(out_dir.join("m10_gate20.txt"), report)?;
    Ok(())
}
