//! M4 gate 1: prefiltered cones (scale of detail).
//!
//! M3 gate 1 found that a cone's cost grows with the triangles really inside
//! it. Now a cone that reaches a node of a mesh's triangle hierarchy whose
//! projected size is at most one bin stops descending and splats the node: its
//! projected box, weighted by the fraction of that box the node's front-facing
//! triangles cover from the cone's direction (sampled at 26 directions per
//! node, summed bottom-up). Cone cost then follows bins, not triangles.
//!
//! Scene: the M2 render scene plus the sphere of M3 gate 1 at four
//! tessellations: 200×100, 400×200, 800×400 and 1600×800 segments (about 40k,
//! 160k, 640k and 2.56M triangles). Per-pixel mask cones at 960×540.
//!
//! Pass criterion, fixed before the first run: (1) at every tessellation, the
//! prefiltered render is within 1% relative RMSE (RMSE over the mean, linear
//! RGB) of the exact render (no prefiltering); (2) the prefiltered render at
//! 2.56M triangles takes at most 1.5× the prefiltered render at 40k triangles
//! (best of 5, warm).

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::cone3::sphere_mesh;
use continuum::glam::Vec3;
use continuum_gpu::{Camera, Hierarchy, MaskConeKernel, RectLight};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
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
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let base = render_scene();
    let l = &base.room.light;
    let light = RectLight {
        corner: l.corner,
        edge_u: l.edge_u,
        edge_v: l.edge_v,
        one_sided: false,
    };
    let camera = Camera {
        eye: base.eye,
        target: base.target,
        vertical_fov_degrees: base.vertical_fov_degrees,
    };
    let render = |meshes: &[Vec<[Vec3; 3]>], albedo: &[Vec3], prefilter: f32, runs: usize| {
        kernel.set_prefilter(prefilter);
        let out = kernel.render(
            camera,
            W,
            H,
            light,
            meshes,
            albedo,
            base.radiance,
            Hierarchy::Triangles,
            runs,
        );
        kernel.set_prefilter(0.0);
        out
    };
    let (_, empty_time) = render(&base.room.meshes, &base.albedo, 1.0, 5);
    let mut rows = Vec::new();
    let mut last = Vec::new();
    for (segments, rings) in [(200, 100), (400, 200), (800, 400), (1600, 800)] {
        let mut meshes = base.room.meshes.clone();
        meshes.push(sphere_mesh(
            Vec3::new(0.3, 0.72, 0.45),
            0.08,
            segments,
            rings,
        ));
        let mut albedo = base.albedo.clone();
        albedo.push(Vec3::new(0.75, 0.30, 0.25));
        let triangles = meshes.last().unwrap().len();
        let (exact, exact_time) = render(&meshes, &albedo, 0.0, 1);
        let (pre, pre_time) = render(&meshes, &albedo, 1.0, 5);
        let rmse = relative_rmse(&pre, &exact);
        println!(
            "{triangles} triangles: exact {:.1} ms, prefiltered {:.2} ms, RMSE {:.3}%",
            ms(exact_time),
            ms(pre_time),
            100.0 * rmse
        );
        rows.push((triangles, exact_time, pre_time, rmse));
        last = vec![pre, exact];
    }
    let gap = 8;
    let (w, h) = (W as usize, H as usize);
    let mut sheet = Image::new(3 * w + 4 * gap, h + 2 * gap, [28; 3]);
    let mean =
        last[1].iter().flatten().map(|&x| x as f64).sum::<f64>() / (3 * last[1].len()) as f64;
    for (i, (a, b)) in last[0].iter().zip(&last[1]).enumerate() {
        let (x, y) = (i % w, i / w);
        let (a, b) = (Vec3::from_array(*a), Vec3::from_array(*b));
        sheet.set(gap + x, gap + y, to_rgb(a));
        sheet.set(2 * gap + w + x, gap + y, to_rgb(b));
        let d = ((a - b).abs().max_element() / mean as f32 * 255.0 * 4.0).min(255.0) as u8;
        sheet.set(3 * gap + 2 * w + x, gap + y, [d, d, d]);
    }
    sheet.save(&out_dir.join("m4_gate1.png"))?;

    let accurate = rows.iter().all(|r| r.3 <= 0.01);
    let ratio = rows[3].2.as_secs_f64() / rows[0].2.as_secs_f64();
    let pass = accurate && ratio <= 1.5;
    let mut out = String::new();
    writeln!(out, "# M4 gate 1 results\n").unwrap();
    writeln!(
        out,
        "Render scene plus a sphere at four tessellations, per-pixel mask cones at {W}×{H} on {}. \
         Prefiltered: threshold 1 bin. Without the sphere: {:.2} ms (prefiltered).\n",
        kernel.adapter_name,
        ms(empty_time)
    )
    .unwrap();
    writeln!(out, "| Sphere triangles | Exact | Prefiltered (best of 5) | Relative RMSE vs exact |\n|---|---|---|---|").unwrap();
    for (t, e, p, r) in &rows {
        writeln!(
            out,
            "| {t} | {:.1} ms | {:.2} ms | {:.3}% |",
            ms(*e),
            ms(*p),
            100.0 * r
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nPrefiltered time at {} vs {} triangles: {ratio:.2}×.\n\n## Verdict\n\n\
         Every tessellation within 1% and at most 1.5× from 40k to 2.56M triangles → {}.",
        rows[3].0,
        rows[0].0,
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate1_results.md"), &out)?;
    Ok(())
}
