//! M4 gate 2: ten billion instanced triangles in a large world (scale of
//! amount and size).
//!
//! Objects reference shared geometry with a rigid transform and a uniform
//! scale; rays and cones move into each object's space, where prefiltered
//! cones (gate 1) apply. Object positions are kept in f64 on the CPU and
//! uploaded relative to a render origin near the camera.
//!
//! Scene: a ground plane with a field of instances of one detailed procedural
//! object (about 10,000 triangles) on a jittered 2 m grid with random rotation
//! and scale, lit by a sun-sized (0.5°) one-sided rectangular light. Per-pixel
//! mask cones at 1920×1080 from a camera at standing height looking over the
//! field.
//!
//! Pass criterion, fixed before the first run:
//! 1. Correctness: with 64 instances and exact cones, the instanced render
//!    matches the same scene with every instance's triangles flattened into
//!    world space: relative RMSE at most 0.1%, and at least 99.9% of pixels
//!    within 1e-4 of peak.
//! 2. Scale: with 1,000,000 instances (about 10^10 triangles), the frame takes
//!    at most 2× the frame with 1,000 instances (about 10^7 triangles), same
//!    camera, best of 5.
//! 3. Large world: the 1,000,000-instance field moved 64 km from the world
//!    origin renders within 0.1% relative RMSE of the field at the origin, from
//!    the same relative camera.
//!
//! Frame times and GPU memory are reported.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Instance, InstancedScene, MaskConeKernel};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::to_rgb;

const W: u32 = 1920;
const H: u32 = 1080;

/// Renders a field from its camera with the given render origin.
fn render(
    kernel: &MaskConeKernel,
    field: &Field,
    scene: &InstancedScene,
    origin: DVec3,
    runs: usize,
) -> (Vec<[f32; 3]>, Duration, u64) {
    let (camera, light) = field.relative(origin);
    kernel.render_instanced(camera, W, H, light, scene, origin, field.radiance, runs)
}

/// The same scene with every instance's triangles moved into world space
/// (relative to `origin`) and placed with the identity.
fn flattened(scene: &InstancedScene, origin: DVec3) -> InstancedScene {
    let mut geometries = Vec::new();
    let mut instances = Vec::new();
    for (k, i) in scene.instances.iter().enumerate() {
        let g = &scene.geometries[i.geometry];
        let t = (i.position - origin).as_vec3();
        geometries.push(Surface {
            triangles: g
                .triangles
                .iter()
                .map(|tri| tri.map(|p| i.rotation * p * i.scale + t))
                .collect(),
            normals: g
                .normals
                .iter()
                .map(|n| n.map(|v| (i.rotation * v).normalize()))
                .collect(),
            ..g.clone()
        });
        instances.push(Instance {
            geometry: k,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: origin,
            albedo: i.albedo,
        });
    }
    InstancedScene {
        geometries,
        instances,
    }
}

fn save(pixels: &[[f32; 3]], path: &Path) -> std::io::Result<()> {
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

    // 1: instanced vs flattened, exact cones.
    let small = Field::new(64, DVec3::ZERO);
    let origin = small.origin_near_camera();
    let (instanced, _, _) = render(&kernel, &small, &small.scene, origin, 1);
    let (flat, _, _) = render(&kernel, &small, &flattened(&small.scene, origin), origin, 1);
    let rmse_flat = relative_rmse(&instanced, &flat);
    let peak = flat.iter().flatten().copied().fold(0.0, f32::max);
    let close = instanced
        .iter()
        .zip(&flat)
        .filter(|(a, b)| {
            (Vec3::from_array(**a) - Vec3::from_array(**b))
                .abs()
                .max_element()
                <= 1e-4 * peak
        })
        .count() as f64
        / flat.len() as f64;

    // 2: scale, prefiltered cones.
    kernel.set_prefilter(1.0);
    let few = Field::new(1_000, DVec3::ZERO);
    let (_, few_time, few_bytes) = render(&kernel, &few, &few.scene, few.origin_near_camera(), 5);
    let many = Field::new(1_000_000, DVec3::ZERO);
    let (reference, many_time, many_bytes) =
        render(&kernel, &many, &many.scene, many.origin_near_camera(), 5);
    save(&reference, &out_dir.join("m4_gate2_field.png"))?;
    drop(many);

    // 3: the same field 64 km from the world origin.
    let far = Field::new(1_000_000, DVec3::new(64_000.0, 0.0, 0.0));
    let (moved, _, _) = render(&kernel, &far, &far.scene, far.origin_near_camera(), 1);
    let rmse_far = relative_rmse(&moved, &reference);
    // For contrast only: the same, uploaded relative to the world origin.
    let (no_rebase, _, _) = render(&kernel, &far, &far.scene, DVec3::ZERO, 1);
    let rmse_no_rebase = relative_rmse(&no_rebase, &reference);
    save(&no_rebase, &out_dir.join("m4_gate2_no_rebase.png"))?;

    let ratio = many_time.as_secs_f64() / few_time.as_secs_f64();
    let pass = rmse_flat <= 0.001 && close >= 0.999 && ratio <= 2.0 && rmse_far <= 0.001;
    let tris = |f: &Field| f.triangles();
    let few_tris = tris(&few);
    let far_tris = tris(&far);
    let mut out = String::new();
    writeln!(out, "# M4 gate 2 results\n").unwrap();
    writeln!(
        out,
        "Boulder field on {}: one 9,800-triangle boulder geometry, a sun-sized light, per-pixel mask cones at {W}×{H}.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "1. 64 instances, exact cones, instanced vs flattened: relative RMSE {:.4}%, pixels within 1e-4 of peak {:.3}%.\n\
         2. Prefiltered cones (1 bin): 1,000 boulders ({few_tris} triangles) {:.1} ms, {:.1} MB; \
         1,000,000 boulders ({far_tris} triangles) {:.1} ms, {:.1} MB → {ratio:.2}× (best of 5).\n\
         3. The million-boulder field 64 km from the origin, rebased: relative RMSE {:.4}% vs at the origin. \
         Without rebasing (for contrast): {:.2}%.",
        100.0 * rmse_flat,
        100.0 * close,
        ms(few_time),
        few_bytes as f64 / 1e6,
        ms(many_time),
        many_bytes as f64 / 1e6,
        100.0 * rmse_far,
        100.0 * rmse_no_rebase,
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nMatches flattened, at most 2× for 1000× the triangles, and unchanged 64 km out → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate2_results.md"), &out)?;
    Ok(())
}
