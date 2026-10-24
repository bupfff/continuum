//! M3 gate 1: per-mesh triangle hierarchies on the GPU.
//!
//! Until now the GPU hierarchy stopped at meshes: a ray or cone that reached a
//! mesh tested every one of its triangles, which is fine for 12-triangle boxes
//! and hopeless for real meshes. Each mesh now gets its own triangle hierarchy
//! (two levels: meshes, then triangles).
//!
//! Scene: the M2 render scene plus a finely tessellated sphere (200 × 100
//! segments, about 40,000 triangles; center (0.3, 0.72, 0.45), radius 0.08, as
//! in `with_sphere`). Per-pixel mask-cone render (no cache) at 960×540 with the
//! mesh hierarchy on.
//!
//! Pass criterion, fixed before the first run: (1) with per-mesh triangle
//! hierarchies, every pixel is within 1e-5 of peak brightness of the render
//! that tests each touched mesh's triangles one by one; (2) it is at least 10×
//! faster (best of 5, warm).

use std::fmt::Write as _;
use std::path::Path;

use continuum::cone3::sphere_mesh;
use continuum::glam::Vec3;
use continuum_gpu::{Camera, Hierarchy, MaskConeKernel, RectLight};
use flatland::img::Image;
use room3d::{render_scene, to_rgb};

const W: u32 = 960;
const H: u32 = 540;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let mut scene = render_scene();
    scene
        .room
        .meshes
        .push(sphere_mesh(Vec3::new(0.3, 0.72, 0.45), 0.08, 200, 100));
    scene.albedo.push(Vec3::new(0.75, 0.30, 0.25));
    let triangles: usize = scene.room.meshes.iter().map(Vec::len).sum();
    let l = &scene.room.light;
    let light = RectLight {
        corner: l.corner,
        edge_u: l.edge_u,
        edge_v: l.edge_v,
        one_sided: false,
    };
    let camera = Camera {
        eye: scene.eye,
        target: scene.target,
        vertical_fov_degrees: scene.vertical_fov_degrees,
    };
    let render = |hierarchy| {
        kernel.render(
            camera,
            W,
            H,
            light,
            &scene.room.meshes,
            &scene.albedo,
            scene.radiance,
            hierarchy,
            5,
        )
    };
    let (flat, flat_time) = render(Hierarchy::Meshes);
    let (tree, tree_time) = render(Hierarchy::Triangles);
    let peak = flat.iter().flatten().copied().fold(0.0, f32::max);
    let max_diff = flat
        .iter()
        .zip(&tree)
        .map(|(a, b)| {
            (Vec3::from_array(*a) - Vec3::from_array(*b))
                .abs()
                .max_element()
                / peak
        })
        .fold(0.0, f32::max);
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in tree.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(&out_dir.join("m3_gate1.png"))?;

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let speedup = flat_time.as_secs_f64() / tree_time.as_secs_f64();
    let pass = max_diff <= 1e-5 && speedup >= 10.0;
    let mut out = String::new();
    writeln!(out, "# M3 gate 1 results\n").unwrap();
    writeln!(
        out,
        "Render scene plus a 200×100 sphere: {} meshes, {triangles} triangles. Per-pixel mask cones at {W}×{H} on {}, best of 5.\n",
        scene.room.meshes.len(),
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "- Mesh hierarchy only (each touched mesh's triangles one by one): {:.1} ms.\n\
         - With per-mesh triangle hierarchies: {:.2} ms → {speedup:.1}× faster.\n\
         - Max difference: {max_diff:.1e} of peak.",
        ms(flat_time),
        ms(tree_time)
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nWithin 1e-5 and at least 10× faster → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m3_gate1_results.md"), &out)?;
    Ok(())
}
