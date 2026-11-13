//! M10 gate 25: more than one bounce (Landmark 2).
//!
//! The cache gathers one bounce of light: a bounce sample sees the direct
//! light of the pages its rays hit. That is enough outdoors (on M6's
//! textured scene Cycles' 1 and 4 bounces differ by 0.65–0.93%), but in the
//! showcase's closed gallery, lit by one skylight panel, light past the first
//! bounce is two fifths of it: Cycles' mean radiance is 0.188 with one
//! bounce and 0.318 with twelve, and the cache's one bounce is 45% from the
//! latter (3.6% from Cycles' one bounce).
//!
//! Now (`CacheOptions::bounces`): a bounce sample also gathers the bounce
//! light of the pages it hits, and records how many bounces it holds, one
//! more than the fewest its hits hold. A page whose hits hold too few asks
//! for them with samples of their own (one bounce fewer, down to pages with
//! none) and re-bakes its samples until they do (a capped number of times).
//!
//! Pass criteria, fixed before the run:
//! 1. **Accuracy indoors:** the showcase room (its static surfaces, the
//!    skylight panel, no sky), 640×360, the cache settled at 4×4 sub-pixel
//!    offsets, against instanced Cycles at 12 bounces and 4,096 samples per
//!    pixel: with 8 bounces, relative RMSE at most 10% and at most 0.25× the
//!    one bounce's.
//! 2. **Settling:** with 8 bounces, every one of the 16 offsets settles (32
//!    frames requesting and baking nothing) within 2,000 frames.
//! 3. **Outdoors unharmed:** M6's textured scene by day, the same way,
//!    against Cycles at 12 bounces: with 8 bounces at most the one bounce's
//!    error plus 0.1 point.
//!
//! Reported: 2 and 4 bounces; frames and bounce samples to settle from empty.
//!
//! **Disclosed (before the run):** developed with `bounces_dev`: the room
//! with 1, 2, 3, 4 and 8 bounces was 45.36, 26.60, 11.73, 7.89–9.02 and
//! 6.54% from Cycles at 12 bounces (mean radiance 0.186, 0.242, 0.286,
//! 0.295–0.298, 0.303 against 0.318; with 4 bounces the result moved by
//! about 1% between runs, from the bake order); from empty 1, 4 and 8
//! bounces settled in 31, 44 and 79 frames with 27,243, 75,348 and 154,773
//! bounce samples. The textured scene with several bounces was not tried
//! before the development mode, which passed (2×2 offsets: room 8 bounces
//! 6.61% against 45.36%, worst offset 85 frames; textured scene 3.05%
//! against 3.23%).
//!
//! `M10_DEV=1`: a development check with 2×2 offsets; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m10_gate25`

use std::fmt::Write as _;
use std::path::Path;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::showcase::Showcase;
use room3d::textured::Textured;

const W: u32 = 640;
const H: u32 = 360;

/// The showcase's static surfaces as an instanced scene, its light, and
/// its first camera.
fn room() -> (InstancedScene, Vec<Light>, Camera) {
    let s = Showcase::new();
    let n = s.first_mover;
    let scene = InstancedScene {
        geometries: s.surfaces[..n].to_vec(),
        instances: (0..n)
            .map(|k| Instance {
                geometry: k,
                rotation: Mat3::IDENTITY,
                scale: 1.0,
                position: DVec3::ZERO,
                albedo: s.albedo[k],
            })
            .collect(),
        textures: Vec::new(),
        geometry_textures: Vec::new(),
        geometry_materials: Vec::new(),
        geometry_emission: Vec::new(),
        geometry_emissive_textures: Vec::new(),
        geometry_blend: Vec::new(),
        geometry_normal_maps: Vec::new(),
        geometry_metallic_roughness: Vec::new(),
        geometry_alpha_cutoff: Vec::new(),
        geometry_colors: Vec::new(),
        texture_wrap: Vec::new(),
        geometry_levels: Vec::new(),
        level_source: None,
    };
    (
        scene,
        vec![Light::white(s.light, s.radiance)],
        s.camera(0.0),
    )
}

struct Settled {
    image: Vec<[f32; 3]>,
    /// The most frames any offset took to settle (None: one never did).
    worst: Option<usize>,
    /// From empty, the first offset: frames and bounce samples.
    first_frames: usize,
    first_samples: u64,
}

/// The cache with `bounces`, settled at `side`×`side` sub-pixel offsets
/// (32 quiet frames each, at most 2,000), averaged.
#[allow(clippy::too_many_arguments)]
fn settled(
    kernel: &MaskConeKernel,
    target: &wgpu::TextureView,
    scene: &InstancedScene,
    origin: DVec3,
    lights: &[Light],
    sky: Vec3,
    camera: Camera,
    bounces: u32,
    side: usize,
) -> Settled {
    kernel.set_sky(sky);
    let mut r = InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        lights,
        scene,
        origin,
        CacheOptions {
            indirect: true,
            frame_target_ms: 0.0,
            bounces,
            ..Default::default()
        },
    );
    let n = side * side;
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    let mut worst = Some(0usize);
    let (mut first_frames, mut first_samples) = (0, 0);
    r.set_footprint(1.0 / side as f32);
    for k in 0..n {
        let step = 1.0 / side as f32;
        let (dx, dy) = (
            (k % side) as f32 * step - 0.5 + 0.5 * step,
            (k / side) as f32 * step - 0.5 + 0.5 * step,
        );
        r.set_pixel_offset(dx, dy);
        let start = r.cache_log().len();
        let mut frames = None;
        for f in 1..=2_000usize {
            r.frame(camera, target);
            let log = r.cache_log();
            if log.len() >= start + 32
                && log[log.len() - 32..]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0)
            {
                frames = Some(f - 32);
                break;
            }
        }
        r.finish();
        worst = match (worst, frames) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };
        if k == 0 {
            first_frames = frames.unwrap_or(2_000);
            first_samples = r
                .cache_log()
                .iter()
                .map(|s| s.indirect_samples as u64)
                .sum();
        }
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / n as f32;
            }
        }
    }
    Settled {
        image: sum,
        worst,
        first_frames,
        first_samples,
    }
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M10_DEV").is_some();
    let side = if dev { 2 } else { 4 };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say("DEVELOPMENT RUN: 2×2 offsets; not the gate's numbers".into());
    }
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let cycles = &out_dir.join("m10_gate25_cycles");

    // 1–2. The room.
    let (scene, lights, camera) = room();
    let reference = render_instanced_lights(
        &scene,
        DVec3::ZERO,
        &lights,
        Vec3::ZERO,
        camera,
        (W, H),
        4096,
        12,
        1,
        cycles,
    )?;
    let mut errors = Vec::new();
    let mut eight_worst = None;
    for bounces in [1u32, 2, 4, 8] {
        let s = settled(
            &kernel,
            &target,
            &scene,
            DVec3::ZERO,
            &lights,
            Vec3::ZERO,
            camera,
            bounces,
            side,
        );
        let e = relative_rmse(&s.image, &reference.pixels);
        let mean =
            s.image.iter().flatten().map(|&v| v as f64).sum::<f64>() / (3 * s.image.len()) as f64;
        say(format!(
            "room, {bounces} bounce(s): {:.2}% from Cycles at 12 bounces (mean radiance {mean:.4}); from empty settled in {} frames with {} bounce samples; worst offset {}",
            100.0 * e,
            s.first_frames,
            s.first_samples,
            s.worst
                .map_or("never (2,000)".into(), |f| format!("{f} frames"))
        ));
        if bounces == 8 {
            eight_worst = s.worst;
        }
        errors.push((bounces, e));
    }
    let reference_mean = reference
        .pixels
        .iter()
        .flatten()
        .map(|&v| v as f64)
        .sum::<f64>()
        / (3 * reference.pixels.len()) as f64;
    say(format!(
        "room: Cycles at 12 bounces, mean radiance {reference_mean:.4}"
    ));
    let (one, eight) = (errors[0].1, errors[3].1);
    let c1 = eight <= 0.10 && eight <= 0.25 * one;
    say(format!(
        "1. accuracy indoors: 8 bounces {:.2}% against one bounce's {:.2}% ({:.3}×; at most 10% and 0.25×): {}",
        100.0 * eight,
        100.0 * one,
        eight / one,
        if c1 { "PASS" } else { "FAIL" }
    ));
    let c2 = eight_worst.is_some_and(|f| f <= 2_000);
    say(format!(
        "2. settling: every offset settled with 8 bounces within 2,000 frames: {}",
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 3. The textured scene by day.
    let t = Textured::new();
    let reference = render_instanced_lights(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (W, H),
        4096,
        12,
        1,
        cycles,
    )?;
    let mut day = [0.0f64; 2];
    for (k, bounces) in [1u32, 8].into_iter().enumerate() {
        let s = settled(
            &kernel, &target, &t.scene, t.origin, &t.lights, t.sky, t.camera, bounces, side,
        );
        day[k] = relative_rmse(&s.image, &reference.pixels);
        say(format!(
            "textured scene by day, {bounces} bounce(s): {:.2}% from Cycles at 12 bounces; from empty settled in {} frames with {} bounce samples; worst offset {}",
            100.0 * day[k],
            s.first_frames,
            s.first_samples,
            s.worst
                .map_or("never (2,000)".into(), |f| format!("{f} frames"))
        ));
    }
    let c3 = day[1] <= day[0] + 0.001;
    say(format!(
        "3. outdoors unharmed: 8 bounces {:.2}% against one bounce's {:.2}% (at most +0.1 point): {}",
        100.0 * day[1],
        100.0 * day[0],
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m10_gate25_dev.txt"
        } else {
            "m10_gate25.txt"
        }),
        report,
    )?;
    Ok(())
}
