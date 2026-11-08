//! M10 gate 23: emissive faces as area lights.
//!
//! Emissive geometry lights its surroundings through bounce light (M10
//! gate 12), whose 3×3 samples a page cannot resolve an emitter's sharp
//! shadows or its light's fall-off near it: 7.46% from Cycles against a 5%
//! bound. Rectangular lights through mask cones are exact.
//!
//! Method (`InstancedScene::emitters_to_lights`): each geometry with
//! uniform emission (no emissive texture) whose triangles all pair into
//! planar rectangles, at most a given number, becomes one one-sided
//! rectangular light per rectangle and instance, facing out of the
//! rectangle's front, at the emission's radiance, 1 mm in front of it. The
//! geometry stays (it still occludes, from behind and around) with its
//! emission set to zero, so its light is not counted twice: the lights show
//! it to the camera, light the scene directly, and bounce (any number of
//! lights, M10 gate 22).
//!
//! Scene: M10 gate 12's (M6's textured scene at night with a glowing box
//! panel, 1.2 × 0.08 × 0.9 m, emission (3, 1.86, 0.9): six faces, so six
//! lights besides the two night lights), its camera, 1280×720, bounce light
//! on. The comparison is gate 12's method: the panel's emission through
//! bounce light. Cycles renders the original scene (the emissive box), as
//! in gate 12.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) with emission through
//!    bounce and then with the faces as lights; the median over rounds of
//!    the ratio at most 1.5× (six lights more to shade).
//! 2. **Accuracy over surfaces:** the settled cached image, supersampled
//!    4×4, within 5% relative RMSE of instanced Cycles (one bounce, 4,096
//!    samples per pixel) and at most 0.5× the error of emission through
//!    bounce, both over the pixels not set apart: where Cycles exceeds 1% of
//!    the brightest light's radiance in a channel (the night lights' disks
//!    and the panel's faces), and their eight neighbours (gate 12's rule).
//!
//! `cargo run --release -p room3d --bin m10_gate23`
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
    // Gate 12's scene, and the same with the panel's faces made lights.
    let t = Textured::new().with_glow();
    let mut faces = Textured::new().with_glow();
    let made = faces.scene.emitters_to_lights(faces.origin, 6);
    let made_count = made.len();
    faces.lights.extend(made);
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
    let renderer = |as_lights: bool| {
        let scene = if as_lights { &faces } else { &t };
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
        image.save(&out_dir.join(format!("m10_gate23_{name}.png")))
    };
    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let mut r = renderer(true);
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate23_dev.png");
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };

    // 1. Speed, first: five interleaved rounds (gate 18's method).
    say(format!("the panel's faces made {made_count} lights"));
    let mut plain = renderer(false);
    let mut lit = renderer(true);
    let frames = (
        settle(&mut plain, t.camera, &target),
        settle(&mut lit, t.camera, &target),
    );
    let mut ratios = Vec::new();
    for round in 0..5 {
        let p = ms(warm(&mut plain, t.camera, &target));
        let b = ms(warm(&mut lit, t.camera, &target));
        ratios.push(b / p);
        say(format!(
            "round {round}: emission through bounce {p:.2} ms, faces as lights {b:.2} ms ({:.3}×)",
            b / p
        ));
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[2];
    let c1 = median <= 1.5;
    say(format!(
        "1. speed: median ratio {median:.3}× over five rounds (at most 1.5×); settled in {} (faces as lights) and {} (through bounce) frames: {}",
        frames.1,
        frames.0,
        verdict(c1)
    ));

    // 2. Accuracy against Cycles, over surfaces.
    let with = supersampled(&mut lit, t.camera, &target);
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
        &out_dir.join("m10_gate23_cycles"),
    )?;
    // Pixels where the reference exceeds 1% of the brightest light's
    // radiance (the lamps' disks), and their eight neighbours, are set apart.
    let peak = t
        .lights
        .iter()
        .map(|l| l.radiance.max_element())
        .fold(0.0f32, f32::max);
    let sees: Vec<bool> = reference
        .pixels
        .iter()
        .map(|p| p.iter().any(|&c| c > 0.01 * peak))
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
    let c2 = ours <= 0.05 && ours <= 0.5 * ignoring;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    say(format!(
        "2. accuracy against Cycles over surfaces ({:.0} s, {}; {} pixels apart): {:.2}% (at most 5%), emission through bounce {:.2}% (ours at most 0.5×: {:.2}×); means: ours {:+.2}%, through bounce {:+.2}%: {}",
        reference.seconds,
        reference.device,
        apart.iter().filter(|&&a| a).count(),
        100.0 * ours,
        100.0 * ignoring,
        ours / ignoring,
        100.0 * (mean(&keep(&with)) / mean(&kept_reference) - 1.0),
        100.0 * (mean(&keep(&without)) / mean(&kept_reference) - 1.0),
        verdict(c2)
    ));

    save("ours", &with)?;
    save("through_bounce", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        verdict(c1),
        verdict(c2),
        verdict(c1 && c2)
    ));
    std::fs::write(out_dir.join("m10_gate23.txt"), report)?;
    Ok(())
}
