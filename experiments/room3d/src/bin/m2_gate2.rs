//! M2 gate 2: a hierarchy on the GPU.
//!
//! The first GPU render tested every triangle twice per pixel (camera ray, then
//! the cone). Here a mesh-level bounding volume hierarchy is traversed by both:
//! rays with a slab test, cones with a conservative box test. Coverage is still
//! summed per mesh, so the result must not change.
//!
//! Pass criterion, fixed before the first run: on the render scene at 1920×1080,
//! (1) the hierarchy render matches the brute-force render within 1e-5 of peak
//! brightness at every pixel, and (2) it is at least 3× faster. A scene with
//! 3,000 extra boxes is reported for information.

use std::fmt::Write as _;
use std::path::Path;

use continuum::cone3::box_mesh;
use continuum::glam::Vec3;
use continuum_gpu::{Camera, MaskConeKernel, RectLight};
use room3d::{RenderScene, render_scene, with_rubble};

fn compare(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    let peak = a.iter().flatten().copied().fold(0.0, f32::max);
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            (Vec3::from_array(*x) - Vec3::from_array(*y))
                .abs()
                .max_element()
                / peak
        })
        .fold(0.0, f32::max)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let run = |scene: &RenderScene, w: u32, h: u32, hierarchy: bool| {
        let l = &scene.room.light;
        kernel.render(
            Camera {
                eye: scene.eye,
                target: scene.target,
                vertical_fov_degrees: scene.vertical_fov_degrees,
            },
            w,
            h,
            RectLight {
                corner: l.corner,
                edge_u: l.edge_u,
                edge_v: l.edge_v,
                one_sided: false,
            },
            &scene.room.meshes,
            &scene.albedo,
            scene.radiance,
            hierarchy,
            5,
        )
    };
    let mut out = String::new();
    writeln!(out, "# M2 gate 2 results\n").unwrap();
    writeln!(
        out,
        "GPU: {}. Times are the best of 5, submit to completion.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(out, "| Scene | Meshes | Triangles | Resolution | Brute force | Hierarchy | Speedup | Max difference |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|").unwrap();

    let mut verdict = None;
    let mut rubble = render_scene();
    let extra = with_rubble();
    rubble
        .room
        .meshes
        .splice(34..34, extra.meshes[34..].iter().cloned());
    rubble.albedo.splice(
        34..34,
        std::iter::repeat_n(Vec3::new(0.5, 0.5, 0.5), extra.meshes.len() - 34),
    );
    let _ = box_mesh;
    for (name, scene, w, h) in [
        ("Render scene", render_scene(), 1920u32, 1080u32),
        ("+ 3,000 boxes", rubble, 480, 270),
    ] {
        let (brute, tb) = run(&scene, w, h, false);
        let (fast, th) = run(&scene, w, h, true);
        let diff = compare(&brute, &fast);
        let speedup = tb.as_secs_f64() / th.as_secs_f64();
        writeln!(
            out,
            "| {name} | {} | {} | {w}×{h} | {:.1} ms | {:.1} ms | {speedup:.1}× | {diff:.1e} |",
            scene.room.meshes.len(),
            scene.room.meshes.iter().map(|m| m.len()).sum::<usize>(),
            tb.as_secs_f64() * 1000.0,
            th.as_secs_f64() * 1000.0,
        )
        .unwrap();
        if verdict.is_none() {
            verdict = Some(diff <= 1e-5 && speedup >= 3.0);
        }
    }
    writeln!(
        out,
        "\n## Verdict\n\nRender scene at 1080p: identical within 1e-5 and at least 3× faster → {}.",
        if verdict == Some(true) {
            "PASS"
        } else {
            "FAIL"
        }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m2_gate2_results.md"), &out)?;
    Ok(())
}
