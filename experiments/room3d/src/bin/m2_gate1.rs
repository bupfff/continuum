//! M2 gate 1: mask cones on the GPU.
//!
//! The core primitive (8×8 continuous mask cones with exact-area rasterization)
//! ported to a WGSL compute shader and run through wgpu (Metal on this Mac).
//! Both paths do the same work: every receiver tests every triangle (no
//! hierarchy), rasterizes the candidates, and sums coverage per mesh.
//!
//! Pass criterion, fixed before the first run: (1) every receiver's GPU result is
//! within 1e-4 of the light's weight of the CPU result; (2) the GPU is faster
//! than the CPU on all threads for the same work.
//!
//! Attempt 1 failed (1): a third of receivers differed by up to 2.7%. Two causes,
//! both fixed on CPU and GPU alike: `acos(dot)` lost precision for small bins
//! (now `atan2`), and stepping along an edge on the grid boundary could drift to
//! x = -1e-9, where floor() credited a whole row to the wrong cell (now clamped).
//! The second was a latent CPU bug that the GPU's rounding exposed.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{MaskConeKernel, RectLight};
use flatland::incremental::par_map;
use room3d::{Receiver, floor_point, gate1};

const SIZE: usize = 256;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let room = gate1();
    let n = SIZE * SIZE;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };

    let start = Instant::now();
    let cpu: Vec<(f32, f32)> = par_map(n, threads, 64, |px| {
        Receiver::new(&room, floor_point(px, SIZE))
            .map_or((0.0, 1.0), |r| (r.masked(), r.cone.total_weight()))
    });
    let cpu_time = start.elapsed();

    let receivers: Vec<(Vec3, Vec3)> = (0..n).map(|px| (floor_point(px, SIZE), Vec3::Z)).collect();
    let light = RectLight {
        corner: room.light.corner,
        edge_u: room.light.edge_u,
        edge_v: room.light.edge_v,
    };
    let (gpu, gpu_time) = kernel.irradiance(&receivers, light, &room.meshes, 5);

    let (mut max_diff, mut failures) = (0.0f32, 0usize);
    for ((c, w), g) in cpu.iter().zip(&gpu) {
        let d = (c - g).abs() / w;
        max_diff = max_diff.max(d);
        if d > 1e-4 {
            failures += 1;
        }
    }
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    let pass = failures == 0 && speedup > 1.0;
    let mut out = String::new();
    writeln!(out, "# M2 gate 1 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} floor receivers, gate 1 room ({} meshes, {} triangles), every receiver tests every triangle.\n",
        room.meshes.len(),
        room.meshes.iter().map(|m| m.len()).sum::<usize>()
    )
    .unwrap();
    writeln!(
        out,
        "- GPU: {} — {:.2} ms (best of 5, submit to completion).\n\
         - CPU: {threads} threads — {:.2} ms.\n\
         - GPU speedup: {speedup:.1}×.\n\
         - Max difference: {max_diff:.2e} of the light's weight; receivers over 1e-4: {failures}.",
        kernel.adapter_name,
        gpu_time.as_secs_f64() * 1000.0,
        cpu_time.as_secs_f64() * 1000.0,
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nMatches the CPU and is faster → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m2_gate1_results.md"), &out)?;
    Ok(())
}
