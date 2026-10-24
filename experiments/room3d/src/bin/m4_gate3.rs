//! M4 gate 3: nearest-first ray traversal (scale of amount, for camera rays).
//!
//! M4 gate 2 failed its scaling criterion (3.04× for 1000× the triangles), and
//! a diagnostic showed camera rays, not cones, set the cost. Rays walked both
//! hierarchies (objects, and each object's triangles) in a fixed child order.
//! Now, at every interior node, a ray tests both children's boxes and visits
//! the one it enters first, pushing the other only if it may still hold a
//! nearer hit.
//!
//! Scene and camera: gate 2's boulder field, per-pixel mask cones (prefiltered,
//! 1 bin) at 1920×1080.
//!
//! Pass criterion, fixed before the first run:
//! 1. Same output: with 1,000,000 boulders, every pixel is within 1e-5 of peak
//!    of the render with the fixed child order (the nearest hit and its tie rule
//!    do not depend on traversal order).
//! 2. Scale: 1,000,000 boulders take at most 2× the frame of 1,000 boulders
//!    (best of 5).
//! 3. Speed: the 1,000,000-boulder frame is at least 1.5× faster than with the
//!    fixed child order (best of 5).

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::MaskConeKernel;
use room3d::field::Field;

const W: u32 = 1920;
const H: u32 = 1080;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let render = |field: &Field, nearest_first: bool| {
        kernel.set_nearest_first(nearest_first);
        let origin = field.origin_near_camera();
        let (camera, light) = field.relative(origin);
        kernel.render_instanced(camera, W, H, light, &field.scene, origin, field.radiance, 5)
    };
    let few = Field::new(1_000, DVec3::ZERO);
    let (_, few_time, _) = render(&few, true);
    let many = Field::new(1_000_000, DVec3::ZERO);
    let (ordered, many_time, _) = render(&many, true);
    let (fixed, fixed_time, _) = render(&many, false);
    kernel.set_nearest_first(true);

    let peak = fixed.iter().flatten().copied().fold(0.0, f32::max);
    let max_diff = ordered
        .iter()
        .zip(&fixed)
        .map(|(a, b)| {
            (Vec3::from_array(*a) - Vec3::from_array(*b))
                .abs()
                .max_element()
                / peak
        })
        .fold(0.0, f32::max);
    let scale = many_time.as_secs_f64() / few_time.as_secs_f64();
    let speedup = fixed_time.as_secs_f64() / many_time.as_secs_f64();
    let pass = max_diff <= 1e-5 && scale <= 2.0 && speedup >= 1.5;
    let mut out = String::new();
    writeln!(out, "# M4 gate 3 results\n").unwrap();
    writeln!(
        out,
        "Boulder field on {} at {W}×{H}, per-pixel prefiltered mask cones, best of 5.\n\n\
         1. 1,000,000 boulders, nearest-first vs fixed order: max difference {max_diff:.1e} of peak.\n\
         2. Nearest-first: 1,000 boulders {:.1} ms, 1,000,000 boulders {:.1} ms → {scale:.2}×.\n\
         3. 1,000,000 boulders: fixed order {:.1} ms → nearest-first is {speedup:.2}× faster.",
        kernel.adapter_name,
        ms(few_time),
        ms(many_time),
        ms(fixed_time),
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nSame output, at most 2× for 1000× the triangles, and at least 1.5× faster → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate3_results.md"), &out)?;
    Ok(())
}
