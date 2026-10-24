//! The first render on the GPU: ray casting for visibility, then a mask cone at
//! every hit, all in one WGSL kernel (brute force over every triangle, no
//! hierarchy yet). Checked against the CPU render pixel by pixel, then timed at
//! larger sizes. Not a gate; a first data point toward M2's 1080p frame time.

use std::fmt::Write as _;
use std::path::Path;

use continuum::bvh::Bvh;
use continuum::glam::Vec3;
use continuum_gpu::{Camera, MaskConeKernel, RectLight};
use flatland::img::Image;
use room3d::{render_scene, to_rgb};

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let scene = render_scene();
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let l = &scene.room.light;
    let light = RectLight {
        corner: l.corner,
        edge_u: l.edge_u,
        edge_v: l.edge_v,
    };
    let camera = Camera {
        eye: scene.eye,
        target: scene.target,
        vertical_fov_degrees: scene.vertical_fov_degrees,
    };
    let render = |w: u32, h: u32| {
        kernel.render(
            camera,
            w,
            h,
            light,
            &scene.room.meshes,
            &scene.albedo,
            scene.radiance,
            5,
        )
    };

    // Pixel-by-pixel check against the CPU at 480×360.
    let (w, h) = (480usize, 360usize);
    let (gpu, _) = render(w as u32, h as u32);
    let cpu = scene.masked(
        &Bvh::build(&scene.room.meshes),
        &scene.primary_hits(w, h, threads),
        threads,
    );
    let peak = cpu.iter().map(|c| c.max_element()).fold(0.0, f32::max);
    let diffs: Vec<f32> = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (*c - Vec3::from_array(*g)).abs().max_element() / peak)
        .collect();
    let max_diff = diffs.iter().copied().fold(0.0, f32::max);
    let differing = diffs.iter().filter(|&&d| d > 1e-3).count();

    let mut out = String::new();
    writeln!(out, "# GPU render\n").unwrap();
    writeln!(
        out,
        "Scene: gate 1 room plus floor and walls ({} triangles), one area light, direct light only. GPU: {}.\n",
        scene.room.meshes.iter().map(|m| m.len()).sum::<usize>(),
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "Check against the CPU render at {w}×{h}: max difference {max_diff:.1e} of peak brightness; pixels over 1e-3: {differing} of {}.\n",
        w * h
    )
    .unwrap();
    writeln!(out, "| Resolution | GPU time (best of 5) |").unwrap();
    writeln!(out, "|---|---|").unwrap();
    for (rw, rh) in [(480u32, 360u32), (960, 540), (1920, 1080)] {
        let (pixels, time) = render(rw, rh);
        writeln!(out, "| {rw}×{rh} | {:.1} ms |", time.as_secs_f64() * 1000.0).unwrap();
        if rw == 1920 {
            let mut img = Image::new(rw as usize, rh as usize, [0; 3]);
            for (i, c) in pixels.iter().enumerate() {
                img.set(
                    i % rw as usize,
                    i / rw as usize,
                    to_rgb(Vec3::from_array(*c)),
                );
            }
            img.save(&out_dir.join("render_gpu_1080p.png"))?;
        }
    }
    print!("{out}");
    std::fs::write(out_dir.join("render_gpu_results.md"), &out)?;
    Ok(())
}
