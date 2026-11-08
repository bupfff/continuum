//! M10 gate 17: vertex colors.
//!
//! glTF meshes often carry a color per vertex (`COLOR_0`) multiplying the
//! base color, with or without a texture. Until now the importer dropped it.
//!
//! Method (`InstancedScene::geometry_colors`): per geometry, a linear RGB
//! color per triangle corner. It rides in the triangle buffer's unused w
//! (no new binding: the kernel is at Metal's 29 buffers per stage), as an
//! exact integer: 1 + an sRGB byte per channel (0: none, white). Shading
//! decodes the corners to linear, interpolates them barycentrically, and
//! multiplies the base color (and its texture) by the result, in the
//! cache's shading, reflections and bounce samples. Alpha cutouts carry
//! colors into the triangles they cut. glTF import reads `COLOR_0`'s RGB
//! (alpha ignored).
//!
//! Scene (`Textured::with_vertex_colors`): M6's textured scene, its camera,
//! sun and sky, 1280×720, bounce light on. The boulders get a smooth color
//! field over their corners, multiplying their lichen texture; the brick
//! block loses its texture and gets a color per corner (untextured path).
//! The comparison is the same scene with the colors ignored.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles (one bounce, 4,096 samples per
//!    pixel; the colors as a float corner attribute multiplying the base
//!    color) and at most 0.5× the error with the colors ignored.
//! 2. **Speed** (first): the warm frame at most 1.1× the colors ignored
//!    (median of 30 frames each waited for).
//!
//! Details as M10 gate 15 (settled: 32 quiet frames after the call; 4×4
//! supersampling over sub-pixel offsets, each settled; Cycles' one-pixel box
//! filter).
//!
//! `cargo run --release -p room3d --bin m10_gate17`
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
    let t = Textured::new().with_vertex_colors();
    let mut flat = Textured::new().with_vertex_colors();
    flat.scene.geometry_colors.clear();
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
    let renderer = |colored: bool| {
        let scene = if colored { &t } else { &flat };
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
        image.save(&out_dir.join(format!("m10_gate17_{name}.png")))
    };
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate17_dev.png");
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 2. Speed, first.
    let mut plain = renderer(false);
    let plain_frames = settle(&mut plain, t.camera, &target);
    let plain_warm = warm(&mut plain, t.camera, &target);
    let mut tex = renderer(true);
    let tex_frames = settle(&mut tex, t.camera, &target);
    let tex_warm = warm(&mut tex, t.camera, &target);
    let c2 = ms(tex_warm) <= 1.1 * ms(plain_warm);
    say(format!(
        "2. speed: warm {:.2} ms with vertex colors, {:.2} ms ignoring them ({:.2}×, at most 1.1×); settled in {tex_frames} and {plain_frames} frames: {}",
        ms(tex_warm),
        ms(plain_warm),
        ms(tex_warm) / ms(plain_warm),
        verdict(c2)
    ));

    // 1. Accuracy against Cycles.
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
        &out_dir.join("m10_gate17_cycles"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    let ignoring = relative_rmse(&without, &reference.pixels);
    let c1 = ours <= 0.05 && ours <= 0.5 * ignoring;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "1. accuracy against Cycles ({:.0} s, {}): {:.2}% (at most 5%), the colors ignored {:.2}% (ours at most 0.5×: {:.2}×); means: ours {:+.2}%, ignored {:+.2}%: {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * ignoring,
        ours / ignoring,
        100.0 * (mean(&with) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&without) / mean(&reference.pixels) - 1.0),
        verdict(c1)
    ));
    save("ours", &with)?;
    save("ignored", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(out_dir.join("m10_gate17.txt"), report)?;
    Ok(())
}
