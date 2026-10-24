//! M4 gate 6: hardware ray queries for camera rays.
//!
//! Flying above the boulder field was unplayable: from 150 m up, a 1080p frame
//! took about 500 ms, nearly all of it camera rays through the compute
//! hierarchy (gates 3–5 could not lower that floor). The M4's ray-tracing
//! hardware is reachable through wgpu's (experimental) ray queries. Camera rays
//! now go through a hardware visibility pass where supported: one bottom-level
//! structure per geometry, one top-level instance per object (with its
//! transform and index), writing the same visibility buffer as the compute
//! pass. Shading (mask cones, caches) is unchanged and stays in compute; the
//! compute hierarchy remains the fallback.
//!
//! Scene: the 1,000,000-boulder field, per-pixel prefiltered mask cones, best
//! of 5. Viewpoints: standing; 10, 40 and 150 m up looking 0°, 15°, 30° and
//! 45° down.
//!
//! Pass criterion, fixed before the first run:
//! 1. Agreement: at 1920×1080 from the 150 m viewpoint, the hardware render is
//!    within 0.5% relative RMSE of the compute-ray render, and at least 99.5%
//!    of pixels are within 1e-3 of peak (intersection arithmetic differs).
//! 2. Speed: at 1920×1080 from 150 m up, the frame is at least 4× faster than
//!    with compute rays.
//! 3. Playable: at 1280×720, every viewpoint renders in at most 16.7 ms.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, MaskConeKernel};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    if !kernel.has_hardware_rays() {
        println!("This GPU or backend has no ray queries; the gate does not apply.");
        return Ok(());
    }
    kernel.set_prefilter(1.0);
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let dir = flat * d.cos() - Vec3::Z * d.sin();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + dir,
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(1.7, 0.0)),
        ("10 m up", view(10.0, 15.0)),
        ("40 m up", view(40.0, 30.0)),
        ("150 m up", view(150.0, 45.0)),
    ];
    let render = |camera: Camera, w: u32, h: u32, hardware: bool| {
        kernel.set_hardware_rays(hardware);
        let (pixels, time, _) =
            kernel.render_instanced(camera, w, h, light, &field.scene, origin, field.radiance, 5);
        (pixels, time)
    };

    // 1 and 2: 150 m up at 1080p.
    let top = views[3].1;
    let (compute_px, compute_time) = render(top, 1920, 1080, false);
    let (hw_px, hw_time) = render(top, 1920, 1080, true);
    let rmse = relative_rmse(&hw_px, &compute_px);
    let peak = compute_px.iter().flatten().copied().fold(0.0, f32::max);
    let close = hw_px
        .iter()
        .zip(&compute_px)
        .filter(|(a, b)| {
            (Vec3::from_array(**a) - Vec3::from_array(**b))
                .abs()
                .max_element()
                <= 1e-3 * peak
        })
        .count() as f64
        / compute_px.len() as f64;
    let speedup = compute_time.as_secs_f64() / hw_time.as_secs_f64();
    let mut image = Image::new(1920, 1080, [0; 3]);
    for (i, p) in hw_px.iter().enumerate() {
        image.set(i % 1920, i / 1920, to_rgb(Vec3::from_array(*p)));
    }
    image.save(&out_dir.join("m4_gate6_150m.png"))?;

    // 3: every viewpoint at 720p, hardware rays.
    let mut rows = Vec::new();
    for (name, camera) in views {
        let (_, hw) = render(camera, 1280, 720, true);
        let (_, compute) = render(camera, 1280, 720, false);
        rows.push((name, hw, compute));
    }
    kernel.set_hardware_rays(true);
    let playable = rows.iter().all(|r| ms(r.1) <= 16.7);
    let pass = rmse <= 0.005 && close >= 0.995 && speedup >= 4.0 && playable;
    let mut out = String::new();
    writeln!(out, "# M4 gate 6 results\n").unwrap();
    writeln!(
        out,
        "1,000,000 boulders on {}, per-pixel prefiltered mask cones, best of 5.\n\n\
         1. 150 m up, 1920×1080, hardware vs compute rays: relative RMSE {:.3}%, pixels within 1e-3 of peak {:.2}%.\n\
         2. 150 m up, 1920×1080: compute rays {:.1} ms, hardware rays {:.1} ms → {speedup:.1}× faster.\n\
         3. 1280×720:\n",
        kernel.adapter_name,
        100.0 * rmse,
        100.0 * close,
        ms(compute_time),
        ms(hw_time),
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | Hardware rays | Compute rays |\n|---|---|---|"
    )
    .unwrap();
    for (name, hw, compute) in &rows {
        writeln!(
            out,
            "| {name} | {:.1} ms | {:.1} ms |",
            ms(*hw),
            ms(*compute)
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n## Verdict\n\nAgrees with compute rays, at least 4× faster from 150 m, and every viewpoint at most 16.7 ms at 720p → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate6_results.md"), &out)?;
    Ok(())
}
