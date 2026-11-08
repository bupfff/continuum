//! M10 gate 14: emissive textures.
//!
//! Emission has been one color per geometry. glTF's screens, signs and
//! lamps use an emissive map: a texture multiplying the emissive factor.
//!
//! Method (`InstancedScene::geometry_emissive_textures`): per geometry, an
//! sRGB texture in the base color array; its layer per object in the
//! materials buffer (and with the emission words bounce samples read). The
//! cache's shading samples it through the texture coordinates at the
//! pixel's footprint; reflections and bounce samples at their footprints.
//! glTF import maps `emissiveTexture` (the base color's coordinates); the
//! Cycles reference multiplies its emission by the same image.
//!
//! Scene (`Textured::with_sign`): M6's textured scene at night (M5 gate 8's
//! point and spot lights, the sky at 2%) with a double-sided sign 5 m ahead
//! (1.2 × 0.9 m), emission (4, 4, 4) times three colored stripes. 1280×720,
//! bounce light on. The comparison is the same scene with the texture
//! ignored (emission the factor alone).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy over surfaces:** the settled cached image, supersampled
//!    4×4, within 10% relative RMSE of instanced Cycles (one bounce, 4,096
//!    samples per pixel) and at most 0.5× the error with the texture
//!    ignored, both over the pixels not set apart: pixels where Cycles
//!    exceeds 8 in a channel (twice the sign's peak radiance: the night
//!    lights' disks), and their eight neighbours.
//! 2. **Speed** (first): the warm frame at most 1.1× the texture ignored
//!    (median of 30 frames each waited for).
//!
//! Details as M10 gate 7 (settled: 32 quiet frames after the call; 4×4
//! supersampling over sub-pixel offsets, each settled; Cycles' one-pixel box
//! filter).
//!
//! `cargo run --release -p room3d --bin m10_gate14`
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
    let t = Textured::new().with_sign();
    let mut flat = Textured::new().with_sign();
    flat.scene.geometry_emissive_textures.clear();
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
    let renderer = |textured: bool| {
        let scene = if textured { &t } else { &flat };
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
        image.save(&out_dir.join(format!("m10_gate14_{name}.png")))
    };
    if std::env::var("M10_DEV").as_deref() == Ok("ss") {
        // Diagnosis after the gate: where ours and Cycles differ.
        println!("DIAGNOSTIC RUN: not the gate's numbers");
        let mut r = renderer(true);
        settle(&mut r, t.camera, &target);
        let ours = supersampled(&mut r, t.camera, &target);
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
            &out_dir.join("m10_gate14_cycles"),
        )?;
        let bright = |p: &[f32; 3]| p.iter().cloned().fold(0.0f32, f32::max) > 1.0;
        let (mut se_b, mut se_r, mut n_b, mut sum_o, mut sum_c) =
            (0.0f64, 0.0f64, 0usize, 0.0f64, 0.0f64);
        for (o, c) in ours.iter().zip(&reference.pixels) {
            let d: f64 = (0..3).map(|k| ((o[k] - c[k]) as f64).powi(2)).sum();
            if bright(c) && c.iter().all(|&x| x < 8.0) {
                se_b += d;
                n_b += 1;
                sum_o += (o[0] + o[1] + o[2]) as f64;
                sum_c += (c[0] + c[1] + c[2]) as f64;
            } else {
                se_r += d;
            }
        }
        println!(
            "sign-bright pixels (Cycles above 1, below 8): {n_b}, {:.1}% of the squared error; ours/Cycles over them {:.4}; the rest {:.1}%",
            100.0 * se_b / (se_b + se_r),
            sum_o / sum_c,
            100.0 * se_r / (se_b + se_r)
        );
        std::fs::write(
            out_dir.join("m10_gate14_ours.f32"),
            ours.iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        return Ok(());
    }
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate14_dev.png");
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
        "2. speed: warm {:.2} ms with the emissive texture, {:.2} ms ignoring it ({:.2}×, at most 1.1×); settled in {tex_frames} and {plain_frames} frames: {}",
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
        &out_dir.join("m10_gate14_cycles"),
    )?;
    // Over surfaces: pixels where the reference exceeds 8 (the night lights'
    // disks), and their eight neighbours, are set apart.
    let sees: Vec<bool> = reference
        .pixels
        .iter()
        .map(|p| p.iter().any(|&c| c > 8.0))
        .collect();
    let mut apart = sees.clone();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if !sees[(y * W as i32 + x) as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < W as i32 && ny < H as i32 {
                        apart[(ny * W as i32 + nx) as usize] = true;
                    }
                }
            }
        }
    }
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(&apart)
            .filter(|(_, a)| !**a)
            .map(|(p, _)| *p)
            .collect()
    };
    let kept_reference = keep(&reference.pixels);
    let ours = relative_rmse(&keep(&with), &kept_reference);
    let ignoring = relative_rmse(&keep(&without), &kept_reference);
    let all_pixels = (
        relative_rmse(&with, &reference.pixels),
        relative_rmse(&without, &reference.pixels),
    );
    let c1 = ours <= 0.10 && ours <= 0.5 * ignoring;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "1. accuracy against Cycles over surfaces ({:.0} s, {}; {} pixels apart): {:.2}% (at most 10%), the texture ignored {:.2}% (ours at most 0.5×: {:.2}×); all pixels {:.2}% and {:.2}%; means: ours {:+.2}%, ignored {:+.2}%: {}",
        reference.seconds,
        reference.device,
        apart.iter().filter(|&&a| a).count(),
        100.0 * ours,
        100.0 * ignoring,
        ours / ignoring,
        100.0 * all_pixels.0,
        100.0 * all_pixels.1,
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
    std::fs::write(out_dir.join("m10_gate14.txt"), report)?;
    Ok(())
}
