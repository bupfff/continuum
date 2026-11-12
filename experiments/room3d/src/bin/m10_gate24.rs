//! M10 gate 24: the cache settles with bounce light (M10 gate 23 re-run).
//!
//! M10 gate 23 (emissive faces as lights) never reached its accuracy
//! criterion: with bounce light on, the view never settled at a sub-pixel
//! offset (50,000 frames). Diagnosis after it (`emitters_dev` with
//! `CONTINUUM_TRACE_BAKES`): a page's bounce samples complete only when
//! every one of their hits finds a resident page at once; hit pages off
//! screen, read by no pixel, aged out after 64 frames before the rest had
//! arrived (the bake budget at its floor bakes few a frame), so the
//! waiting page was retried forever (one page 552 times in 3,000 frames).
//! A liveness bug in the certified cache, not specific to emissive faces.
//!
//! Fix (made while diagnosing, before this stub): pages baked for bounce
//! hits (secondary requests) stay resident while any page's bounce samples
//! wait, at most a quarter of the pool; once none wait they age out as
//! before. In development the same view settled about 1,000 frames after
//! the offset and stayed quiet.
//!
//! Scene, renderers, Cycles reference and accuracy criterion: gate 23's
//! (M10 gate 12's scene, the textured scene at night with a glowing panel
//! whose six faces become lights, against emission through bounce light;
//! Cycles at 4,096 samples per pixel, one bounce, gate 11's cached render
//! of the same job).
//!
//! Pass criterion, fixed before the run:
//! 1. **Liveness:** with bounce light on, both renderers (faces as lights,
//!    and emission through bounce) settle (32 frames asking for and baking
//!    nothing) at the first view and then at each of the 16 sub-pixel
//!    offsets of the 4×4 supersampling (textures read for a quarter of a
//!    pixel), each within 20,000 frames.
//! 2. **Accuracy over surfaces** (gate 23's criterion 2): the settled cached
//!    image with the faces as lights, supersampled 4×4, within 5% relative
//!    RMSE of Cycles, and at most 0.5× the error of emission through
//!    bounce, over the pixels not set apart (gate 12's rule).
//!
//! Reported: frames to settle per offset; the pages resident when settled.
//!
//! `cargo run --release -p room3d --bin m10_gate24`
//!
//! `M10_DEV=1` runs a development check (the first view and two offsets,
//! no Cycles), whose numbers are not the gate's.

use std::fmt::Write as _;
use std::path::Path;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Frames a view may take to settle (criterion 1).
const SETTLE_CAP: usize = 20_000;

/// Renders until 32 frames rendered since the call requested and baked
/// nothing; the frames it took, or None after `SETTLE_CAP`.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> Option<usize> {
    let start = r.cache_log().len();
    for k in 1..=SETTLE_CAP {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return Some(k);
        }
    }
    None
}

/// The view supersampled 4×4: each sub-pixel offset's frame after it
/// settles, averaged; with the frames each offset took to settle (None:
/// not within `SETTLE_CAP`, its frame averaged anyway).
fn supersampled(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
    offsets: usize,
) -> (Vec<[f32; 3]>, Vec<Option<usize>>) {
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    let mut frames = Vec::new();
    r.set_footprint(0.25);
    for k in 0..offsets {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        frames.push(settle(r, camera, target));
        r.finish();
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    (sum, frames)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
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
        image.save(&out_dir.join(format!("m10_gate24_{name}.png")))
    };
    let dev = std::env::var_os("M10_DEV").is_some();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say(
            "DEVELOPMENT RUN (the first view and two offsets, no Cycles): not the gate's numbers"
                .into(),
        );
    }
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    say(format!("the panel's faces made {made_count} lights"));

    // 1. Liveness: each renderer settles at its first view, then at every
    // offset of the supersampling.
    let offsets = if dev { 2 } else { 16 };
    let mut lit = renderer(true);
    let first_lit = settle(&mut lit, t.camera, &target);
    let (with, lit_frames) = supersampled(&mut lit, t.camera, &target, offsets);
    let lit_resident = lit.cache_stats().map_or(0, |s| s.resident);
    drop(lit);
    let mut plain = renderer(false);
    let first_plain = settle(&mut plain, t.camera, &target);
    let (without, plain_frames) = supersampled(&mut plain, t.camera, &target, offsets);
    let plain_resident = plain.cache_stats().map_or(0, |s| s.resident);
    drop(plain);
    let show = |v: &[Option<usize>]| -> String {
        v.iter()
            .map(|f| f.map_or("never".to_string(), |f| f.to_string()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let all = |first: Option<usize>, v: &[Option<usize>]| {
        first.is_some() && v.iter().all(Option::is_some)
    };
    let c1 = all(first_lit, &lit_frames) && all(first_plain, &plain_frames);
    say(format!(
        "1. liveness (each within {SETTLE_CAP} frames): faces as lights settled in {} frames, then per offset {} \
         ({lit_resident} pages resident); through bounce in {} frames, then per offset {} ({plain_resident} pages): {}",
        first_lit.map_or("never".to_string(), |f| f.to_string()),
        show(&lit_frames),
        first_plain.map_or("never".to_string(), |f| f.to_string()),
        show(&plain_frames),
        verdict(c1)
    ));
    if dev {
        save("dev", &with)?;
        std::fs::write(out_dir.join("m10_gate24_dev.txt"), report)?;
        return Ok(());
    }

    // 2. Accuracy against Cycles, over surfaces.
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
        // Gate 11's work directory: the same job, rendered then (re-rendered
        // if anything differs).
        &out_dir.join("m10_gate11_cycles"),
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
    std::fs::write(out_dir.join("m10_gate24.txt"), report)?;
    Ok(())
}
