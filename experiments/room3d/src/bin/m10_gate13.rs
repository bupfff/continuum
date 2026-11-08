//! M10 gate 13: MikkTSpace tangents for normal maps.
//!
//! Normal maps (M10 gate 5) bend the normal in each triangle's own frame:
//! the direction of increasing u on the triangle, made orthogonal to the
//! interpolated normal. glTF's normal maps are authored for MikkTSpace's
//! per-corner tangents, which Blender and Cycles use too. On a coarse curved
//! mesh the two frames differ from triangle to triangle, and the bumps look
//! faceted or turned.
//!
//! Method: per geometry with a normal map and texture coordinates, MikkTSpace
//! tangents (`bevy_mikktspace`, on the texture coordinates with v flipped to
//! Blender's convention, as the Cycles export writes them) are stored per
//! triangle corner in the normals buffer's fourth component, as an exact
//! integer code (octahedral, 11 bits per axis; its sign the handedness; 0
//! none). Shading interpolates them and keeps them orthogonal to the
//! normal; without codes it keeps the triangle's frame.
//!
//! Scene (`Textured::with_curved_normal_maps`): M10 gate 5's normal maps
//! (the ground's tiles and the block's bricks), and three low-poly spheres
//! (16 × 8 segments) wearing the tiles' map. Sun and sky, its camera,
//! 1280×720, bounce light on.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles (one bounce, 4,096 samples per
//!    pixel; its normal maps through MikkTSpace); and over the pixels the
//!    tangents change (where the two configurations' images differ by more
//!    than 1% of the image's mean in some channel), at most 0.8× the error
//!    of each triangle's own frame against that reference.
//! 2. **Speed** (first): the warm frame at most 1.1× the triangles' own
//!    frames (median of 30 frames each waited for).
//!
//! Details as M10 gate 7 (settled: 32 quiet frames after the call; 4×4
//! supersampling over sub-pixel offsets, each settled; Cycles' one-pixel box
//! filter).
//!
//! `cargo run --release -p room3d --bin m10_gate13`
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
    let t = Textured::new().with_curved_normal_maps();
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
    let renderer = |mikktspace: bool| {
        kernel.set_mikktspace(mikktspace);
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
        image.save(&out_dir.join(format!("m10_gate13_{name}.png")))
    };
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate13_dev.png");
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 2. Speed, first.
    let mut own = renderer(false);
    let own_frames = settle(&mut own, t.camera, &target);
    let own_warm = warm(&mut own, t.camera, &target);
    let mut mikk = renderer(true);
    let mikk_frames = settle(&mut mikk, t.camera, &target);
    let mikk_warm = warm(&mut mikk, t.camera, &target);
    let c2 = ms(mikk_warm) <= 1.1 * ms(own_warm);
    say(format!(
        "2. speed: warm {:.2} ms with MikkTSpace tangents, {:.2} ms with triangles' frames ({:.2}×, at most 1.1×); settled in {mikk_frames} and {own_frames} frames: {}",
        ms(mikk_warm),
        ms(own_warm),
        ms(mikk_warm) / ms(own_warm),
        verdict(c2)
    ));

    // 1. Accuracy against Cycles.
    let with = supersampled(&mut mikk, t.camera, &target);
    let without = supersampled(&mut own, t.camera, &target);
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
        &out_dir.join("m10_gate13_cycles"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    // The pixels the tangents change.
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
    let c1 = ours <= 0.05 && ours_region <= 0.8 * frames_region;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "1. accuracy against Cycles ({:.0} s, {}): {:.2}% (at most 5%; triangles' frames {:.2}%); over the {} pixels the tangents change: {:.2}% against {:.2}% ({:.2}×, at most 0.8×); means: ours {:+.2}%, frames {:+.2}%: {}",
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
        verdict(c1)
    ));
    save("ours", &with)?;
    save("frames", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(out_dir.join("m10_gate13.txt"), report)?;
    Ok(())
}
