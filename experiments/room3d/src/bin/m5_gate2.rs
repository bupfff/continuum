//! M5 gate 2: sky light against Cycles, with the reference fixed.
//!
//! M5 gate 1 failed (sun and sky 5.12% from Cycles; sky alone 2.74%), but
//! its diagnosis found two faults in the reference, not the sky:
//! - Cycles integrates each pixel's area (Blender 5.2 applies at least a
//!   one-pixel box filter whatever width is asked), while our renders took one
//!   sample per pixel center, so every silhouette counted as error.
//! - Cycles loses 1.9% of a 0.5° quad sun 3 km away, but matches the analytic
//!   irradiance at 1 km and closer.
//!
//! This gate repeats gate 1 with both fixed, and nothing else changed in the
//! renderer: our renders take 4×4 samples per pixel (rendered at 4× the
//! resolution and box filtered), and the outdoor scene's sun is 1 km away
//! (`Outdoor::with_sun_at(1000.0)`, same angular size, radiance and
//! direction). The sky's radiance follows from the sun's irradiance as before.
//!
//! Scene: `room3d::outdoor` at 640×360, Lambertian, direct light only, compute
//! rays. Reference: Cycles (Blender 5.2), direct light only, world color = sky
//! radiance, one-pixel box filter, 4096 samples per pixel, two seeds (noise
//! reported as the RMSE between them / √2).
//!
//! Pass criterion, fixed before the first run:
//! 1. Sky only (sun off): exact cones, 4×4 per pixel, within 3% relative RMSE
//!    of Cycles (seed 1).
//! 2. Sun and sky: exact cones and prefiltered cones (1 bin), 4×4 per pixel,
//!    each within 3% of Cycles (seed 1).
//!
//! Reported: render times, Cycles' noise, the same comparisons at one sample
//! per pixel, and per-surface ratios (sun and sky) in surface interiors.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::MaskConeKernel;
use flatland::img::Image;
use room3d::cycles::{Reference, relative_rmse, render_with_sky};
use room3d::outdoor::Outdoor;
use room3d::to_rgb;

const W: u32 = 640;
const H: u32 = 360;
const SS: u32 = 4;
const SAMPLES: u32 = 4096;

fn save_png(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
}

/// Box filters an image of `SS`× the resolution down to `W × H`.
fn downsample(big: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; (W * H) as usize];
    for y in 0..H * SS {
        for x in 0..W * SS {
            let p = big[(y * W * SS + x) as usize];
            let o = &mut out[((y / SS) * W + x / SS) as usize];
            for c in 0..3 {
                o[c] += p[c] / (SS * SS) as f32;
            }
        }
    }
    out
}

/// Mean of the green channel over rows `r` and columns `c`.
fn region_mean(img: &[[f32; 3]], r: std::ops::Range<u32>, c: std::ops::Range<u32>) -> f64 {
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for y in r {
        for x in c.clone() {
            sum += img[(y * W + x) as usize][1] as f64;
            n += 1.0;
        }
    }
    sum / n
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let s = |d: Duration| d.as_secs_f64();
    let scene = Outdoor::with_sun_at(1000.0);
    kernel.set_hardware_rays(false);
    let render = |radiance: f32, sky: Vec3, prefilter: f32, ss: u32| {
        kernel.set_sky(sky);
        kernel.set_prefilter(prefilter);
        let (pixels, time, _) = kernel.render_instanced(
            scene.camera,
            W * ss,
            H * ss,
            scene.light,
            &scene.scene,
            DVec3::ZERO,
            radiance,
            1,
        );
        let pixels = if ss == 1 { pixels } else { downsample(&pixels) };
        (pixels, time)
    };
    let (sky_ss, sky_ss_t) = render(0.0, scene.sky, 0.0, SS);
    let (both_ss, both_ss_t) = render(scene.radiance, scene.sky, 0.0, SS);
    let (pre_ss, pre_ss_t) = render(scene.radiance, scene.sky, 1.0, SS);
    let (sun_ss, _) = render(scene.radiance, Vec3::ZERO, 0.0, SS);
    let (sky_1, _) = render(0.0, scene.sky, 0.0, 1);
    let (both_1, _) = render(scene.radiance, scene.sky, 0.0, 1);
    let (pre_1, _) = render(scene.radiance, scene.sky, 1.0, 1);

    let (surfaces, albedo) = scene.flattened();
    let work = out_dir.join("m5_gate2_cycles");
    let cycles = |radiance: f32, seed: u32| {
        render_with_sky(
            &surfaces,
            &albedo,
            scene.light,
            radiance,
            scene.sky,
            scene.camera,
            (W, H),
            SAMPLES,
            0,
            seed,
            &work,
        )
    };
    let sky_ref = [cycles(0.0, 1)?, cycles(0.0, 2)?];
    let both_ref = [cycles(scene.radiance, 1)?, cycles(scene.radiance, 2)?];
    let noise = |r: &[Reference; 2]| relative_rmse(&r[0].pixels, &r[1].pixels) / 2f64.sqrt();
    let sun_ref: Vec<[f32; 3]> = both_ref[0]
        .pixels
        .iter()
        .zip(&sky_ref[0].pixels)
        .map(|(b, k)| [b[0] - k[0], b[1] - k[1], b[2] - k[2]])
        .collect();

    for (name, img) in [
        ("sky_ss", &sky_ss),
        ("both_ss", &both_ss),
        ("both_cycles", &both_ref[0].pixels),
    ] {
        save_png(&out_dir.join(format!("m5_gate2_{name}.png")), img)?;
    }
    let e = |img: &[[f32; 3]], r: &[Reference; 2]| relative_rmse(img, &r[0].pixels);
    let sky_err = e(&sky_ss, &sky_ref);
    let both_err = e(&both_ss, &both_ref);
    let pre_err = e(&pre_ss, &both_ref);
    let pass = sky_err <= 0.03 && both_err <= 0.03 && pre_err <= 0.03;

    let mut out = String::new();
    writeln!(out, "# M5 gate 2 results\n").unwrap();
    writeln!(
        out,
        "{}, outdoor scene with the sun at 1 km ({} instances), {W}×{H}, compute rays. Cycles: {} spp, direct light only, one-pixel box filter, device {}, {:.1} s per render.\n",
        kernel.adapter_name,
        scene.scene.instances.len(),
        SAMPLES,
        both_ref[0].device,
        both_ref[0].seconds
    )
    .unwrap();
    writeln!(
        out,
        "| Render | 4×4 per pixel vs Cycles | 1 per pixel vs Cycles | Time (4×4) |\n|---|---|---|---|"
    )
    .unwrap();
    for (name, ss, one, t) in [
        (
            "Sky only, exact cones",
            sky_err,
            e(&sky_1, &sky_ref),
            sky_ss_t,
        ),
        (
            "Sun and sky, exact cones",
            both_err,
            e(&both_1, &both_ref),
            both_ss_t,
        ),
        (
            "Sun and sky, prefiltered cones (1 bin)",
            pre_err,
            e(&pre_1, &both_ref),
            pre_ss_t,
        ),
    ] {
        writeln!(
            out,
            "| {name} | {:.2}% | {:.2}% | {:.1} s |",
            100.0 * ss,
            100.0 * one,
            s(t)
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nCycles' own noise (RMSE between seeds / √2): sky only {:.2}%, sun and sky {:.2}%.\n",
        100.0 * noise(&sky_ref),
        100.0 * noise(&both_ref)
    )
    .unwrap();
    writeln!(
        out,
        "Surface interiors, ours / Cycles (green channel, 4×4 per pixel):\n\n| Region | Sun | Sky |\n|---|---|---|"
    )
    .unwrap();
    for (name, r, c) in [
        ("wall", 120..150, 300..480),
        ("pillar", 40..140, 170..200),
        ("ground", 300..360, 0..250),
    ] {
        writeln!(
            out,
            "| {name} | {:.4} | {:.4} |",
            region_mean(&sun_ss, r.clone(), c.clone())
                / region_mean(&sun_ref, r.clone(), c.clone()),
            region_mean(&sky_ss, r.clone(), c.clone())
                / region_mean(&sky_ref[0].pixels, r.clone(), c.clone())
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n## Verdict\n\nSky only (exact), and sun and sky (exact and prefiltered), 4×4 per pixel, each within 3% of Cycles → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m5_gate2_results.md"), &out)?;
    Ok(())
}
