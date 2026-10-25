//! M5 gate 1: sky light through mask cones, against Cycles.
//!
//! Outdoor scenes have had black shadows: the only light was the sun. A
//! uniform sky (constant radiance over the whole sphere of directions, like
//! Cycles' world color) now lights every receiver through six mask cones, one
//! per cube face around the receiver:
//! - Each face is a 90° pyramid with 8×8 bins, its far plane beyond the
//!   scene. Faces wholly below the receiver's horizon are skipped.
//! - Bins straddling the horizon have their projected solid angle clipped to
//!   it (Lambert's edge integrals over the clipped polygon).
//! - Within a mesh, coverage adds as for the sun. Across meshes, sky cones
//!   combine coverage as independent occluders, 1 − Π(1 − c), because their
//!   11° bins often hold several partial occluders. Sun cones keep the maximum.
//! - Camera rays that miss see the sky's radiance.
//! - This gate covers per-pixel cones only (the lighting caches come later).
//!
//! Scene ("outdoor", `room3d::outdoor`): a 400 m ground, 60 boulders of the
//! field's geometry within 30 m of the camera (scales 0.6–2.0, some
//! touching), a wall and a pillar; the field's sun (0.5° wide, 30° up); a
//! blue-white sky whose unoccluded irradiance is about a quarter of the
//! sun's. Lambertian, direct light only. 640×360.
//!
//! Reference: Cycles (Blender 5.2), direct light only (no bounces), world
//! color = sky radiance, 4096 samples per pixel, two seeds; its own noise is
//! reported as the RMSE between the seeds divided by √2.
//!
//! Pass criterion, fixed before the first run:
//! 1. Sky only (sun off): per-pixel exact cones within 3% relative RMSE of
//!    Cycles (seed 1).
//! 2. Sun and sky: per-pixel exact cones within 3%, and prefiltered cones
//!    (1 bin) within 3%, of Cycles (seed 1).
//!
//! Reported: render times, Cycles' noise, and as a diagnostic the same renders
//! with sky cones combining coverage by the maximum.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::MaskConeKernel;
use flatland::img::Image;
use room3d::cycles::{relative_rmse, render_with_sky};
use room3d::outdoor::Outdoor;
use room3d::to_rgb;

const W: u32 = 640;
const H: u32 = 360;
const SAMPLES: u32 = 4096;

fn save_png(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let scene = Outdoor::new();
    // Compute rays: watertight, the reference-quality path.
    kernel.set_hardware_rays(false);
    kernel.set_sky(scene.sky);
    let render = |radiance: f32, prefilter: f32, max: bool| {
        kernel.set_prefilter(prefilter);
        kernel.set_sky_combination_max(max);
        let (pixels, time, _) = kernel.render_instanced(
            scene.camera,
            W,
            H,
            scene.light,
            &scene.scene,
            DVec3::ZERO,
            radiance,
            3,
        );
        kernel.set_sky_combination_max(false);
        (pixels, time)
    };
    let (sky_exact, sky_exact_t) = render(0.0, 0.0, false);
    let (sky_max, _) = render(0.0, 0.0, true);
    let (both_exact, both_exact_t) = render(scene.radiance, 0.0, false);
    let (both_pre, both_pre_t) = render(scene.radiance, 1.0, false);
    let (both_max, _) = render(scene.radiance, 0.0, true);

    // Development check: our renders only, no Cycles and no verdict.
    if std::env::args().any(|a| a == "--ours-only") {
        for (name, img, t) in [
            ("sky_exact", &sky_exact, sky_exact_t),
            ("both_exact", &both_exact, both_exact_t),
            ("both_prefiltered", &both_pre, both_pre_t),
        ] {
            save_png(&out_dir.join(format!("m5_dev_{name}.png")), img)?;
            println!("{name}: {:.1} ms", ms(t));
        }
        println!(
            "max vs independent: sky {:.2}%, both {:.2}%",
            100.0 * relative_rmse(&sky_max, &sky_exact),
            100.0 * relative_rmse(&both_max, &both_exact)
        );
        return Ok(());
    }
    let (surfaces, albedo) = scene.flattened();
    let work = out_dir.join("m5_gate1_cycles");
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
    let noise = |r: &[room3d::cycles::Reference; 2]| {
        relative_rmse(&r[0].pixels, &r[1].pixels) / 2f64.sqrt()
    };

    for (name, img) in [
        ("sky_exact", &sky_exact),
        ("both_exact", &both_exact),
        ("both_prefiltered", &both_pre),
        ("sky_cycles", &sky_ref[0].pixels),
        ("both_cycles", &both_ref[0].pixels),
    ] {
        save_png(&out_dir.join(format!("m5_gate1_{name}.png")), img)?;
    }
    let e = |img: &[[f32; 3]], r: &[room3d::cycles::Reference; 2]| relative_rmse(img, &r[0].pixels);
    let rows = [
        ("Sky only, exact cones", e(&sky_exact, &sky_ref), Some(sky_exact_t)),
        ("Sky only, exact, max combination (diagnostic)", e(&sky_max, &sky_ref), None),
        ("Sun and sky, exact cones", e(&both_exact, &both_ref), Some(both_exact_t)),
        ("Sun and sky, prefiltered cones (1 bin)", e(&both_pre, &both_ref), Some(both_pre_t)),
        ("Sun and sky, exact, max combination (diagnostic)", e(&both_max, &both_ref), None),
    ];
    let pass = rows[0].1 <= 0.03 && rows[2].1 <= 0.03 && rows[3].1 <= 0.03;

    let mut out = String::new();
    writeln!(out, "# M5 gate 1 results\n").unwrap();
    writeln!(
        out,
        "{}, outdoor scene ({} instances), {W}×{H}, compute rays (best of 3). Cycles: {} spp, direct light only, device {}, {:.1} s per render.\n",
        kernel.adapter_name,
        scene.scene.instances.len(),
        SAMPLES,
        both_ref[0].device,
        both_ref[0].seconds
    )
    .unwrap();
    writeln!(out, "| Render | Relative RMSE vs Cycles | Time |\n|---|---|---|").unwrap();
    for (name, err, time) in &rows {
        writeln!(
            out,
            "| {name} | {:.2}% | {} |",
            100.0 * err,
            time.map_or("".to_string(), |t| format!("{:.1} ms", ms(t)))
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nCycles' own noise (RMSE between seeds / √2): sky only {:.2}%, sun and sky {:.2}%.",
        100.0 * noise(&sky_ref),
        100.0 * noise(&both_ref)
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nSky only (exact), and sun and sky (exact and prefiltered), each within 3% of Cycles → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m5_gate1_results.md"), &out)?;
    Ok(())
}
