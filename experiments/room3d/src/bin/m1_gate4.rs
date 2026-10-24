//! M1 gate 4: exact incremental direct light in 3D.
//!
//! The main mechanism from the 2D phase, now on triangle meshes: each receiver
//! caches the coverage of static geometry in its cone. Each frame, only tiles
//! whose frustum a mover's swept box can reach are recomputed, starting from the
//! cached static coverage and adding just the movers. Everything else is reused.
//!
//! Pass criterion, fixed before the first run: (1) the incremental output equals
//! a full recompute at every receiver and frame, within 1e-5 of the light's
//! weight; (2) the incremental path is at least 5× faster than recomputing every
//! receiver (hierarchy traversal plus rasterization), same thread count.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::CoverageMask;
use continuum::bvh::Bvh;
use continuum::cone3::{PyramidCone, box_mesh};
use continuum::glam::Vec3;
use flatland::incremental::par_map;
use room3d::{TileFrustum, Tri, floor_point, gate1};

const SIZE: usize = 64;
const TILE: usize = 8;
const FRAMES: u32 = 120;
const TOLERANCE: f32 = 1e-5;

/// Three boxes moving between the floor and the light.
fn movers(f: u32) -> Vec<(Vec3, Vec3)> {
    let t = f as f32;
    let cube = |c: Vec3, half: f32| (c - Vec3::splat(half), c + Vec3::splat(half));
    vec![
        cube(
            Vec3::new(0.2 + 0.6 * (0.5 + 0.5 * (t * 0.03).sin()), 0.5, 0.45),
            0.03,
        ),
        cube(
            Vec3::new(
                0.5 + 0.25 * (t * 0.02).cos(),
                0.5 + 0.25 * (t * 0.02).sin(),
                0.8,
            ),
            0.02,
        ),
        cube(Vec3::new(0.75, 0.25, 0.35 + 0.15 * (t * 0.05).sin()), 0.025),
    ]
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let room = gate1();
    let statics = Bvh::build(&room.meshes);
    let l = &room.light;
    let n = SIZE * SIZE;
    let tiles_per_side = SIZE / TILE;
    let tiles = tiles_per_side * tiles_per_side;
    let tile_pixels = |t: usize| {
        let (tx, ty) = (t % tiles_per_side, t / tiles_per_side);
        (0..TILE * TILE).map(move |i| (ty * TILE + i / TILE) * SIZE + tx * TILE + i % TILE)
    };
    let cones: Vec<Option<PyramidCone>> = (0..n)
        .map(|px| {
            PyramidCone::toward_rect_light(
                floor_point(px, SIZE),
                l.corner,
                l.edge_u,
                l.edge_v,
                Vec3::Z,
            )
        })
        .collect();
    let frustums: Vec<TileFrustum> = (0..tiles)
        .map(|t| {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for px in tile_pixels(t) {
                let p = floor_point(px, SIZE);
                (lo, hi) = (lo.min(p), hi.max(p));
            }
            TileFrustum::new([lo.x, lo.y, hi.x, hi.y], l)
        })
        .collect();

    // The same mask computation for both paths: static meshes (via the
    // hierarchy) first, then movers.
    let static_mask = |cone: &PyramidCone| {
        let mut m = CoverageMask::EMPTY;
        for (_, tris) in statics.candidates(cone).0 {
            cone.add_mesh(&mut m, &tris);
        }
        m
    };
    let add_movers = |cone: &PyramidCone, m: &mut CoverageMask, boxes: &[Vec<Tri>]| {
        for mesh in boxes {
            let tris: Vec<Tri> = mesh.iter().filter(|t| cone.may_touch(t)).copied().collect();
            if !tris.is_empty() {
                cone.add_mesh(m, &tris);
            }
        }
    };

    let start = Instant::now();
    let cache: Vec<CoverageMask> = par_map(n, threads, 64, |px| {
        cones[px].as_ref().map_or(CoverageMask::EMPTY, static_mask)
    });
    let cache_ms = start.elapsed().as_secs_f64() * 1000.0;
    let meshes_at = |f: u32| {
        movers(f)
            .into_iter()
            .map(|(a, b)| box_mesh(a, b))
            .collect::<Vec<_>>()
    };
    let boxes0 = meshes_at(0);
    let mut visible: Vec<f32> = par_map(n, threads, 64, |px| {
        cones[px].as_ref().map_or(0.0, |c| {
            let mut m = cache[px];
            add_movers(c, &mut m, &boxes0);
            c.visible(&m)
        })
    });

    let (mut full_time, mut incremental_time) = (Duration::ZERO, Duration::ZERO);
    let (mut recomputed, mut max_diff, mut failures) = (0usize, 0.0f32, 0usize);
    for f in 1..FRAMES {
        let (prev, now) = (movers(f - 1), movers(f));
        let boxes = meshes_at(f);

        let start = Instant::now();
        let reference: Vec<f32> = par_map(n, threads, 64, |px| {
            cones[px].as_ref().map_or(0.0, |c| {
                let mut m = static_mask(c);
                add_movers(c, &mut m, &boxes);
                c.visible(&m)
            })
        });
        full_time += start.elapsed();

        let start = Instant::now();
        let swept: Vec<(Vec3, Vec3)> = prev
            .iter()
            .zip(&now)
            .filter(|(a, b)| a != b)
            .map(|(a, b)| (a.0.min(b.0), a.1.max(b.1)))
            .collect();
        let dirty: Vec<usize> = (0..tiles)
            .filter(|&t| {
                swept
                    .iter()
                    .any(|&(lo, hi)| frustums[t].may_touch_box(lo, hi))
            })
            .collect();
        let updates: Vec<Vec<(usize, f32)>> = par_map(dirty.len(), threads, 1, |i| {
            tile_pixels(dirty[i])
                .map(|px| {
                    let v = cones[px].as_ref().map_or(0.0, |c| {
                        let mut m = cache[px];
                        add_movers(c, &mut m, &boxes);
                        c.visible(&m)
                    });
                    (px, v)
                })
                .collect()
        });
        for batch in updates {
            recomputed += batch.len();
            for (px, v) in batch {
                visible[px] = v;
            }
        }
        incremental_time += start.elapsed();

        for px in 0..n {
            let weight = cones[px].as_ref().map_or(1.0, |c| c.total_weight());
            let d = (visible[px] - reference[px]).abs() / weight;
            max_diff = max_diff.max(d);
            if d > TOLERANCE {
                failures += 1;
            }
        }
    }

    let frames = (FRAMES - 1) as f64;
    let (full_ms, inc_ms) = (
        full_time.as_secs_f64() * 1000.0 / frames,
        incremental_time.as_secs_f64() * 1000.0 / frames,
    );
    let speedup = full_ms / inc_ms;
    let pass = failures == 0 && speedup >= 5.0;
    let mut out = String::new();
    writeln!(out, "# M1 gate 4 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} floor receivers in {TILE}×{TILE} tiles, gate 1 room ({} static meshes) plus 3 moving boxes, \
         {FRAMES} frames, {threads} threads for both paths.\n",
        room.meshes.len()
    )
    .unwrap();
    writeln!(
        out,
        "- Static coverage cache: built once in {cache_ms:.1} ms, {:.1} MB.\n\
         - Full recompute (hierarchy traversal and rasterization for every receiver): {full_ms:.2} ms/frame.\n\
         - Incremental (dirty tiles only, from the static cache): {inc_ms:.2} ms/frame → {speedup:.1}× faster.\n\
         - Receivers recomputed per frame: {:.1}%.\n\
         - Exactness: max difference {max_diff:.1e} of the light's weight; receiver-frames over {TOLERANCE:e}: {failures}.",
        (n * std::mem::size_of::<CoverageMask>()) as f64 / 1e6,
        100.0 * recomputed as f64 / (frames * n as f64),
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nExact and at least 5× faster → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m1_gate4_results.md"), &out)?;
    Ok(())
}
