//! M5 gate 12: environment light (gate 11 re-run after a fix).
//!
//! Gate 11 failed (22.64% from Cycles, 1.16× the frame): surfaces' sky
//! irradiance from the environment's nine harmonics was right, but camera
//! and reflection rays that missed saw its radiance through them, half the
//! true value near the horizon (a step nine terms cannot follow). Fix (made
//! before this binary; only the GPU tests ran on it): rays that miss read
//! the environment's image itself, an extra layer of the base color texture
//! array (sRGB over its peak radiance: limited dynamic range for what is
//! seen directly); the irradiance is evaluated unrolled. Criteria and scene
//! are gate 11's.
//!
//!
//! The sky has been one uniform radiance. Outdoor scenes have skies brighter
//! at the horizon and bluer at the zenith, and HDR environments.
//!
//! Method (`MaskConeKernel::set_environment`): the environment's radiance
//! above the horizon as nine spherical harmonics. The view cache still keeps
//! each texel's sky visibility for a uniform sky; shading multiplies it by
//! the environment's irradiance at the pixel's normal over the uniform open
//! sky's there (π(1 + n_z)/2), as lights' cached ratios are: exact with
//! nothing in the way, approximate where occlusion depends on direction.
//! Camera and reflection rays that miss see the environment through its
//! harmonics. Bounce light and per-pixel rendering use its mean radiance.
//!
//! Scene: M6's textured scene (the sun; its camera, 1280×720; bounce light
//! off), the sky a gradient from the horizon, (1.3, 1.2, 1.05) times the
//! scene's uniform sky, to the zenith, (0.55, 0.72, 1.25) times it, linear
//! in the sine of elevation, black below the horizon. The comparison is the
//! uniform sky at the environment's mean radiance above the horizon.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles with the same environment image
//!    (direct light only, 4,096 samples per pixel), and at most 0.6× the
//!    error of the uniform sky against that reference.
//! 2. **Speed** (first): the warm frame at most 1.05× the uniform sky's
//!    (median of 30 frames each waited for).
//!
//! Details as M10 gate 7 (settled: 32 quiet frames after the call; 4×4
//! supersampling over sub-pixel offsets, each settled; Cycles' one-pixel box
//! filter).
//!
//! `cargo run --release -p room3d --bin m5_gate12`
//!
//! `M5_DEV=1` saves the settled image alone (no Cycles): a development
//! check, whose numbers are not the gate's.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Environment, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_environment};
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
    let t = Textured::new();
    let (horizon, zenith) = (
        t.sky * Vec3::new(1.3, 1.2, 1.05),
        t.sky * Vec3::new(0.55, 0.72, 1.25),
    );
    let environment = Environment::from_fn(512, 256, |d| {
        if d.z <= 0.0 {
            Vec3::ZERO
        } else {
            horizon * (1.0 - d.z) + zenith * d.z
        }
    });
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
    let renderer = |env: bool| {
        // The environment, or the uniform sky at its mean (which
        // `set_environment` also sets).
        kernel.set_environment(Some(&environment));
        if !env {
            let mean = kernel.sky();
            kernel.set_environment(None);
            kernel.set_sky(mean);
        }
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
        image.save(&out_dir.join(format!("m5_gate12_{name}.png")))
    };
    if std::env::var_os("M5_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m5_gate12_dev.png");
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 2. Speed, first.
    let mut uniform_r = renderer(false);
    let uniform_frames = settle(&mut uniform_r, t.camera, &target);
    let uniform_warm = warm(&mut uniform_r, t.camera, &target);
    let mut env_r = renderer(true);
    let env_frames = settle(&mut env_r, t.camera, &target);
    let env_warm = warm(&mut env_r, t.camera, &target);
    let c2 = ms(env_warm) <= 1.05 * ms(uniform_warm);
    say(format!(
        "2. speed: warm {:.2} ms with the environment, {:.2} ms with the uniform sky ({:.2}×, at most 1.05×); settled in {env_frames} and {uniform_frames} frames: {}",
        ms(env_warm),
        ms(uniform_warm),
        ms(env_warm) / ms(uniform_warm),
        verdict(c2)
    ));

    // 1. Accuracy against Cycles.
    let with = supersampled(&mut env_r, t.camera, &target);
    let without = supersampled(&mut uniform_r, t.camera, &target);
    let reference = render_instanced_environment(
        &t.scene,
        t.origin,
        &t.lights,
        &environment,
        t.camera,
        (W, H),
        4096,
        0,
        1,
        &out_dir.join("m5_gate12_cycles"),
    )?;
    let ours = relative_rmse(&with, &reference.pixels);
    let ignoring = relative_rmse(&without, &reference.pixels);
    let c1 = ours <= 0.05 && ours <= 0.6 * ignoring;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "1. accuracy against Cycles ({:.0} s, {}): {:.2}% (at most 5%), the uniform sky {:.2}% (ours at most 0.6×: {:.2}×); means: ours {:+.2}%, uniform {:+.2}%: {}",
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
    save("uniform", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(out_dir.join("m5_gate12.txt"), report)?;
    Ok(())
}
