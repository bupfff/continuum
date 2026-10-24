//! M2 gate 3: 960×540 at 60 fps on the M4 Air.
//!
//! Gate 2's profile showed shared cone work dominating the frame: per-pixel bin
//! weights (256 `atan2` calls) and triangle clipping through polygon copies. This
//! gate optimizes those without changing results.
//!
//! Pass criterion, fixed before the first run: the render scene at 960×540 with
//! the hierarchy renders in at most 16.7 ms (best of 5), and matches the
//! pre-optimization kernel within 1e-5 of peak brightness at every pixel.
//!
//! Run once with `capture` before optimizing to save the reference render.

use std::fmt::Write as _;
use std::path::Path;

use continuum::glam::Vec3;
use continuum_gpu::{Camera, MaskConeKernel, RectLight};
use room3d::render_scene;

const W: u32 = 960;
const H: u32 = 540;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let reference_path = out_dir.join("m2_gate3_reference.f32");
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let scene = render_scene();
    let l = &scene.room.light;
    let (pixels, time) = kernel.render(
        Camera {
            eye: scene.eye,
            target: scene.target,
            vertical_fov_degrees: scene.vertical_fov_degrees,
        },
        W,
        H,
        RectLight {
            corner: l.corner,
            edge_u: l.edge_u,
            edge_v: l.edge_v,
            one_sided: false,
        },
        &scene.room.meshes,
        &scene.albedo,
        scene.radiance,
        true,
        5,
    );
    let ms = time.as_secs_f64() * 1000.0;
    if std::env::args().any(|a| a == "capture") {
        let bytes: Vec<u8> = pixels
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&reference_path, bytes)?;
        println!("Captured reference render ({W}×{H}) in {ms:.2} ms.");
        return Ok(());
    }
    let raw = std::fs::read(&reference_path)?;
    let reference: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let peak = reference.iter().copied().fold(0.0, f32::max);
    let max_diff = pixels
        .iter()
        .zip(reference.as_chunks::<3>().0)
        .map(|(p, r)| {
            (Vec3::from_array(*p) - Vec3::from_array(*r))
                .abs()
                .max_element()
                / peak
        })
        .fold(0.0, f32::max);
    let pass = ms <= 16.7 && max_diff <= 1e-5;
    let mut out = String::new();
    writeln!(out, "# M2 gate 3 results\n").unwrap();
    writeln!(
        out,
        "Render scene at {W}×{H} on {} with the mesh hierarchy: {ms:.2} ms (best of 5). \
         Max difference from the pre-optimization kernel: {max_diff:.1e} of peak.\n\n## Verdict\n\n\
         At most 16.7 ms and within 1e-5 → {}.",
        kernel.adapter_name,
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m2_gate3_results.md"), &out)?;
    Ok(())
}
