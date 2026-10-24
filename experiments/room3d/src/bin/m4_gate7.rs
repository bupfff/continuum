//! M4 gate 7: cones that pay only for what they see.
//!
//! M4 gate 6's corrected diagnostic found the field's frame dominated by
//! per-pixel sun cones, at a cost nearly independent of what the cone contains:
//! every cone summed 64 bin weights (4 edge integrals each, with `atan2`) even
//! when nothing covered it. Now a cone's visible light is the whole window's
//! weight (one 4-edge integral) minus the weights of the bins with coverage,
//! the same quantity rearranged, so an open cone costs 4 integrals, not 256.
//! And a node is projected for splatting only after a cheap angular-size test
//! says it could be small enough.
//!
//! Pass criterion, fixed before the first run:
//! 1. Same output: the showcase's cached render and per-pixel render (1920×1080)
//!    and the 1,000,000-boulder field (1280×720, from 150 m up) are within 1e-4
//!    of peak of the pre-change kernel at every pixel (the sum's rounding
//!    changes).
//! 2. Playable: the 1,000,000-boulder field at 1280×720 renders in at most 33.3
//!    ms (30 fps) from each viewpoint of gate 6 (standing; 10, 40, 150 m up),
//!    with prefiltered cones and the faster of compute or hardware rays (best
//!    of 5).
//!
//! Also reported: whether each viewpoint reaches 16.7 ms, and the bake times.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, Lightmap, LiveRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;
use room3d::showcase::{PROBES, Showcase, TEXEL};

fn save(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let bytes: Vec<u8> = pixels
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, bytes)
}

fn load(path: &Path) -> std::io::Result<Vec<[f32; 3]>> {
    let raw = std::fs::read(path)?;
    let values: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(values.as_chunks::<3>().0.to_vec())
}

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
    let capture = std::env::args().any(|a| a == "capture");
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;

    // The showcase: cached render (direct and indirect) and per-pixel cones.
    let still = Showcase::new().without_movers();
    let meshes = still.meshes();
    let mut map = Lightmap::new(&still.surfaces, TEXEL, |_| true).with_probes(PROBES);
    let bake = kernel.bake(&mut map, still.light, &meshes);
    let live = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        64,
        64,
        still.light,
        &meshes,
        &still.albedo,
        still.radiance,
        &map,
    );
    live.wait();
    map.indirect = live.read_probes();
    let camera = still.camera(0.0);
    let (cached, cached_time) = kernel.render_cached(
        camera,
        1920,
        1080,
        still.light,
        &meshes,
        &still.albedo,
        still.radiance,
        &map,
        5,
    );
    let (per_pixel, per_pixel_time) = kernel.render_per_pixel(
        camera,
        1920,
        1080,
        still.light,
        &meshes,
        &still.albedo,
        still.radiance,
        &map,
        3,
    );

    // The field from 150 m up (prefiltered, compute rays for a fixed reference).
    kernel.set_prefilter(1.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(1.7, 0.0)),
        ("10 m up", view(10.0, 15.0)),
        ("40 m up", view(40.0, 30.0)),
        ("150 m up", view(150.0, 45.0)),
    ];
    let render = |camera: Camera, hardware: bool, runs: usize| {
        kernel.set_hardware_rays(hardware);
        let (pixels, time, _) = kernel.render_instanced(
            camera,
            1280,
            720,
            light,
            &field.scene,
            origin,
            field.radiance,
            runs,
        );
        (pixels, time)
    };
    let (field_px, _) = render(views[3].1, false, 1);
    let names = ["cached", "per_pixel", "field"];
    let images = [&cached, &per_pixel, &field_px];
    if capture {
        for (name, img) in names.iter().zip(images) {
            save(&out_dir.join(format!("m4_gate7_reference_{name}.f32")), img)?;
        }
        println!(
            "Captured references. Pre-change: cached {:.2} ms, per-pixel {:.1} ms, bake {:.1} ms.",
            ms(cached_time),
            ms(per_pixel_time),
            ms(bake)
        );
        return Ok(());
    }
    let mut diffs = Vec::new();
    for (name, img) in names.iter().zip(images) {
        let reference = load(&out_dir.join(format!("m4_gate7_reference_{name}.f32")))?;
        diffs.push((name, max_diff(img, &reference)));
    }
    let mut rows = Vec::new();
    for (name, camera) in views {
        let (_, compute) = render(camera, false, 5);
        let hardware = if kernel.has_hardware_rays() {
            Some(render(camera, true, 5).1)
        } else {
            None
        };
        let best = hardware.map_or(compute, |h| h.min(compute));
        rows.push((name, compute, hardware, best));
    }
    let same = diffs.iter().all(|d| d.1 <= 1e-4);
    let playable = rows.iter().all(|r| ms(r.3) <= 33.3);
    let pass = same && playable;
    let mut out = String::new();
    writeln!(out, "# M4 gate 7 results\n").unwrap();
    writeln!(
        out,
        "{}, best of 5.\n\n1. Max difference from the pre-change kernel:",
        kernel.adapter_name
    )
    .unwrap();
    for (name, d) in &diffs {
        writeln!(out, "   - {name}: {d:.1e} of peak").unwrap();
    }
    writeln!(
        out,
        "\n   Showcase: cached render {:.2} ms, per-pixel cones {:.1} ms, bake {:.1} ms.\n\n\
         2. 1,000,000 boulders at 1280×720, prefiltered cones:\n",
        ms(cached_time),
        ms(per_pixel_time),
        ms(bake)
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | Compute rays | Hardware rays | Best | 60 fps |\n|---|---|---|---|---|"
    )
    .unwrap();
    for (name, compute, hardware, best) in &rows {
        writeln!(
            out,
            "| {name} | {:.1} ms | {} | {:.1} ms | {} |",
            ms(*compute),
            hardware.map_or("n/a".to_string(), |h| format!("{:.1} ms", ms(h))),
            ms(*best),
            if ms(*best) <= 16.7 { "yes" } else { "no" }
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n## Verdict\n\nSame output within 1e-4 and every viewpoint at most 33.3 ms at 720p → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m4_gate7_results.md"), &out)?;
    Ok(())
}
