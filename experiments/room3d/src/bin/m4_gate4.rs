//! M4 gate 4: a visibility buffer (splitting camera rays from shading).
//!
//! M4 gate 3 left a high floor: 72 ms at 1080p for 1,000 boulders, mostly
//! camera rays. One large shader did both the ray cast and the mask cones, and
//! the cone state (accumulator, coverage bins, clip polygons) costs registers,
//! so few rays run at once. Now a lean pass finds each pixel's nearest hit and
//! writes it to a visibility buffer; a second pass shades from it.
//!
//! Pass criterion, fixed before the first run (all at 1920×1080, best of 5,
//! split vs single pass in the same session):
//! 1. Same output: on the 1,000,000-boulder field (per-pixel prefiltered cones)
//!    and on the showcase (cached render, direct and indirect light), every
//!    pixel of the split render is within 1e-5 of peak of the single pass.
//! 2. The 1,000,000-boulder frame is at least 1.3× faster split.
//! 3. The showcase's cached render is no more than 1.05× slower split.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Lightmap, LiveRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;
use room3d::showcase::{PROBES, Showcase, TEXEL};

const W: u32 = 1920;
const H: u32 = 1080;

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

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;

    // The million-boulder field, per-pixel prefiltered cones.
    kernel.set_prefilter(1.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    let field_frame = |split: bool| {
        kernel.set_visibility_buffer(split);
        let (pixels, time, _) =
            kernel.render_instanced(camera, W, H, light, &field.scene, origin, field.radiance, 5);
        (pixels, time)
    };
    let (field_single, field_single_time) = field_frame(false);
    let (field_split, field_split_time) = field_frame(true);
    drop(field);
    kernel.set_prefilter(0.0);

    // The showcase's cached render (static, direct and indirect light).
    let still = Showcase::new().without_movers();
    let meshes = still.meshes();
    let mut map = Lightmap::new(&still.surfaces, TEXEL, |_| true).with_probes(PROBES);
    kernel.bake(&mut map, still.light, &meshes);
    let live = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        still.light,
        &meshes,
        &still.albedo,
        still.radiance,
        &map,
    );
    live.wait();
    map.indirect = live.read_probes();
    let camera = still.camera(0.0);
    let cached_frame = |split: bool| {
        kernel.set_visibility_buffer(split);
        kernel.render_cached(
            camera,
            W,
            H,
            still.light,
            &meshes,
            &still.albedo,
            still.radiance,
            &map,
            5,
        )
    };
    let (cached_single, cached_single_time) = cached_frame(false);
    let (cached_split, cached_split_time) = cached_frame(true);
    kernel.set_visibility_buffer(true);

    let (field_diff, cached_diff) = (
        max_diff(&field_split, &field_single),
        max_diff(&cached_split, &cached_single),
    );
    let speedup = field_single_time.as_secs_f64() / field_split_time.as_secs_f64();
    let cached_ratio = cached_split_time.as_secs_f64() / cached_single_time.as_secs_f64();
    let pass = field_diff <= 1e-5 && cached_diff <= 1e-5 && speedup >= 1.3 && cached_ratio <= 1.05;
    let mut out = String::new();
    writeln!(out, "# M4 gate 4 results\n").unwrap();
    writeln!(
        out,
        "{} at {W}×{H}, best of 5, single pass vs visibility pass + shading pass.\n\n\
         1. Max difference, split vs single: field {field_diff:.1e} of peak, showcase {cached_diff:.1e}.\n\
         2. 1,000,000 boulders: single {:.1} ms, split {:.1} ms → {speedup:.2}× faster.\n\
         3. Showcase cached render: single {:.2} ms, split {:.2} ms → {cached_ratio:.2}×.",
        kernel.adapter_name,
        ms(field_single_time),
        ms(field_split_time),
        ms(cached_single_time),
        ms(cached_split_time),
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nSame output, at least 1.3× faster on the field, at most 1.05× on the showcase → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate4_results.md"), &out)?;
    Ok(())
}
