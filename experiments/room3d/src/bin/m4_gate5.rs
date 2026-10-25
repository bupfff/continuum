//! M4 gate 5: traversal stacks in workgroup memory.
//!
//! M4 gate 4 showed that ray traversal itself, not register pressure from the
//! cone code, limits camera rays. Every traversal (rays and cones, through
//! object and triangle hierarchies) uses a 48-entry stack in a per-thread array
//! indexed at runtime, which Apple GPUs tend to keep in slow memory. Now each
//! invocation's stack is a slice of a workgroup array (64 invocations × 48
//! entries per workgroup).
//!
//! Pass criterion, fixed before the first run (1920×1080, best of 5, both
//! variants in the same session):
//! 1. Same output: the 1,000,000-boulder field (per-pixel prefiltered cones)
//!    and the showcase's cached render match the per-thread-stack versions
//!    within 1e-5 of peak at every pixel.
//! 2. The 1,000,000-boulder frame is at least 1.3× faster.
//!
//! Also reported: the showcase's cached render, M2 gate 3's cone-heavy
//! per-pixel render, and the showcase bake.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, KernelOptions, Lightmap, MaskConeKernel, RectLight};
use room3d::field::Field;
use room3d::render_scene;
use room3d::showcase::{Showcase, TEXEL};

fn max_diff(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    let peak = b.iter().flatten().copied().fold(0.0, f32::max);
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

struct Run {
    field: Vec<[f32; 3]>,
    field_time: Duration,
    cached: Vec<[f32; 3]>,
    cached_time: Duration,
    cones_time: Duration,
    bake: Duration,
}

fn run(workgroup_stacks: bool, field: &Field, still: &Showcase) -> Run {
    let kernel = MaskConeKernel::new_with(KernelOptions {
        workgroup_stacks,
        ..Default::default()
    })
    .expect("GPU");
    kernel.set_prefilter(1.0);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    let (field_px, field_time, _) = kernel.render_instanced(
        camera,
        1920,
        1080,
        light,
        &field.scene,
        origin,
        field.radiance,
        5,
    );
    kernel.set_prefilter(0.0);
    let meshes = still.meshes();
    let mut map = Lightmap::new(&still.surfaces, TEXEL, |_| true);
    let bake = kernel.bake(&mut map, still.light, &meshes);
    let (cached, cached_time) = kernel.render_cached(
        still.camera(0.0),
        1920,
        1080,
        still.light,
        &meshes,
        &still.albedo,
        still.radiance,
        &map,
        5,
    );
    let room = render_scene();
    let l = &room.room.light;
    let (_, cones_time) = kernel.render(
        Camera {
            eye: room.eye,
            target: room.target,
            vertical_fov_degrees: room.vertical_fov_degrees,
        },
        960,
        540,
        RectLight {
            corner: l.corner,
            edge_u: l.edge_u,
            edge_v: l.edge_v,
            one_sided: false,
        },
        &room.room.meshes,
        &room.albedo,
        room.radiance,
        true,
        5,
    );
    Run {
        field: field_px,
        field_time,
        cached,
        cached_time,
        cones_time,
        bake,
    }
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let field = Field::new(1_000_000, DVec3::ZERO);
    let still = Showcase::new().without_movers();
    let thread = run(false, &field, &still);
    let group = run(true, &field, &still);
    let (field_diff, cached_diff) = (
        max_diff(&group.field, &thread.field),
        max_diff(&group.cached, &thread.cached),
    );
    let speedup = thread.field_time.as_secs_f64() / group.field_time.as_secs_f64();
    let pass = field_diff <= 1e-5 && cached_diff <= 1e-5 && speedup >= 1.3;
    let mut out = String::new();
    writeln!(out, "# M4 gate 5 results\n").unwrap();
    writeln!(
        out,
        "Apple M4, 1920×1080 unless noted, best of 5; per-thread stacks vs workgroup stacks.\n\n\
         1. Max difference: field {field_diff:.1e} of peak, showcase {cached_diff:.1e}.\n\
         2. 1,000,000 boulders: per-thread {:.1} ms, workgroup {:.1} ms → {speedup:.2}× faster.\n\n\
         Also: showcase cached render {:.2} → {:.2} ms; M2 render scene per-pixel cones at 960×540 {:.1} → {:.1} ms; \
         showcase bake {:.1} → {:.1} ms.",
        ms(thread.field_time),
        ms(group.field_time),
        ms(thread.cached_time),
        ms(group.cached_time),
        ms(thread.cones_time),
        ms(group.cones_time),
        ms(thread.bake),
        ms(group.bake),
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nSame output and at least 1.3× faster on the field → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate5_results.md"), &out)?;
    Ok(())
}
