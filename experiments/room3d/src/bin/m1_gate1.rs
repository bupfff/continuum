//! M1 gate 1: do 3D mask cones beat rays on error per unit of cost?
//!
//! Irradiance on a floor of receivers from a rectangular ceiling light, occluded
//! by box meshes (two double-slat fences, a table, a cube). Every method sees the
//! same candidate triangles per receiver (the culling a hierarchy would do), and
//! is timed single-threaded on top of that shared setup.
//!
//! Pass criterion, fixed before the first run: the continuous mask cone (8×8
//! bins) has lower error than 64 jittered rays, at no more than 1/4 of their cost
//! per receiver, with zero flicker.
//!
//! Attempt 1 computed per-bin areas by clipping the projected triangle against
//! every overlapped bin (with allocations) and cost 2.15× the rays. Attempt 2
//! uses an accumulation rasterizer (exact area, each edge touches only the
//! cells it crosses, no allocation). Same criterion.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::CoverageMask;
use flatland::img;
use flatland::incremental::par_map;
use flatland::rng::Pcg32;
use room3d::{Receiver, floor_point, gate1, stratified};

const SIZE: usize = 96;
/// Stratified light samples per side for the reference (32 × 32 = 1024).
const REFERENCE_SIDE: usize = 32;

struct Render {
    name: String,
    irradiance: Vec<f32>,
    ns: f64,
}

fn rel_rmse(a: &[f32], reference: &[f32]) -> f64 {
    let mean = reference.iter().map(|&r| r as f64).sum::<f64>() / reference.len() as f64;
    let mse = a
        .iter()
        .zip(reference)
        .map(|(&x, &r)| ((x - r) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    mse.sqrt() / mean
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let room = gate1();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let n = SIZE * SIZE;

    let start = Instant::now();
    let receivers: Vec<Option<Receiver>> = (0..n)
        .map(|px| Receiver::new(&room, floor_point(px, SIZE)))
        .collect();
    let setup_ns = start.elapsed().as_nanos() as f64 / n as f64;
    let candidates: f64 = receivers
        .iter()
        .flatten()
        .map(|r| r.candidates.iter().map(|c| c.1.len()).sum::<usize>())
        .sum::<usize>() as f64
        / n as f64;

    let reference: Vec<f32> = par_map(n, threads, 64, |px| {
        receivers[px].as_ref().map_or(0.0, |r| {
            r.sampled(&room.light, stratified(REFERENCE_SIDE, || (0.5, 0.5)))
        })
    });

    let time = |name: &str, f: &dyn Fn(usize, &Receiver) -> f32| -> Render {
        let start = Instant::now();
        let irradiance: Vec<f32> = (0..n)
            .map(|px| receivers[px].as_ref().map_or(0.0, |r| f(px, r)))
            .collect();
        Render {
            name: name.to_string(),
            irradiance,
            ns: start.elapsed().as_nanos() as f64 / n as f64,
        }
    };
    let mask = time("Mask cone, 8×8 continuous", &|_, r| {
        let mut m = CoverageMask::EMPTY;
        for (_, tris) in &r.candidates {
            r.cone.add_mesh(&mut m, tris);
            if m.is_full() {
                break;
            }
        }
        r.cone.visible(&m)
    });
    let light = &room.light;
    let fixed64 = time("64 rays, fixed grid", &|_, r| {
        r.sampled(light, stratified(8, || (0.5, 0.5)))
    });
    let jittered = |k: usize, frame: u64| {
        time(&format!("{} rays, jittered", k * k), &move |px, r| {
            let mut rng = Pcg32::new(px as u64, frame);
            r.sampled(light, stratified(k, || (rng.next_f32(), rng.next_f32())))
        })
    };
    let ray_runs: Vec<(Render, Render)> = [2usize, 4, 8, 16]
        .iter()
        .map(|&k| (jittered(k, 0), jittered(k, 1)))
        .collect();

    let cost_64 = ray_runs[2].0.ns;
    let mut out = String::new();
    writeln!(out, "# M1 gate 1 results (3D)\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} floor receivers, {} box meshes ({} triangles), 0.2 × 0.2 light at height 1. \
         Reference: {} stratified light samples per receiver. Shared setup (cone + candidate culling): \
         {setup_ns:.0} ns/receiver, {candidates:.1} candidate triangles on average. Methods timed single-threaded on top.\n",
        room.meshes.len(),
        room.meshes.iter().map(|m| m.len()).sum::<usize>(),
        REFERENCE_SIDE * REFERENCE_SIDE
    )
    .unwrap();
    writeln!(
        out,
        "| Method | Rel. RMSE | Flicker | ns/receiver | Cost vs 64 jittered rays |"
    )
    .unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();
    let mut row = |r: &Render, flicker: Option<f64>| {
        let flicker = flicker.map_or("0 (deterministic)".to_string(), |f| {
            format!("{:.2}%", f * 100.0)
        });
        writeln!(
            out,
            "| {} | {:.2}% | {} | {:.0} | {:.2}× |",
            r.name,
            rel_rmse(&r.irradiance, &reference) * 100.0,
            flicker,
            r.ns,
            r.ns / cost_64
        )
        .unwrap();
    };
    row(&mask, None);
    row(&fixed64, None);
    for (a, b) in &ray_runs {
        row(a, Some(rel_rmse(&a.irradiance, &b.irradiance)));
    }

    let mask_err = rel_rmse(&mask.irradiance, &reference);
    let rays_err = rel_rmse(&ray_runs[2].0.irradiance, &reference);
    let pass = mask_err < rays_err && mask.ns <= cost_64 / 4.0;
    writeln!(
        out,
        "\n## Verdict\n\nMask cone error {:.2}% vs 64 jittered rays {:.2}%, at {:.2}× their cost (target: lower error at ≤ 0.25×, \
         zero flicker) → {}.",
        mask_err * 100.0,
        rays_err * 100.0,
        mask.ns / cost_64,
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m1_gate1_results.md"), &out)?;

    // Images: reference, mask cone, 16 jittered rays; then signed error maps.
    let exposure = 12.0;
    let shaded =
        |v: &[f32]| img::tonemap(&v.iter().map(|x| x * exposure).collect::<Vec<_>>(), SIZE);
    let mean = reference.iter().sum::<f32>() / n as f32;
    let error = |v: &[f32]| img::signed_error(v, &reference, SIZE, 0.1 * mean);
    let top = [
        shaded(&reference),
        shaded(&mask.irradiance),
        shaded(&ray_runs[1].0.irradiance),
    ];
    let bottom = [
        img::Image::new(SIZE, SIZE, [0; 3]),
        error(&mask.irradiance),
        error(&ray_runs[1].0.irradiance),
    ];
    let gap = 6;
    let mut sheet = img::Image::new(3 * SIZE + 4 * gap, 2 * SIZE + 3 * gap, [28; 3]);
    for (i, (t, b)) in top.iter().zip(&bottom).enumerate() {
        sheet.blit(t, gap + i * (SIZE + gap), gap);
        sheet.blit(b, gap + i * (SIZE + gap), 2 * gap + SIZE);
    }
    sheet.save(&out_dir.join("m1_gate1_sheet.png"))?;
    Ok(())
}
