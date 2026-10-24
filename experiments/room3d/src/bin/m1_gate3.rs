//! M1 gate 3: tile-shared hierarchy traversal.
//!
//! Neighboring receivers' cones are nearly identical, so an 8×8 tile of receivers
//! walks the hierarchy once with a frustum enclosing all 64 cones (the
//! prismatoid between the tile and the light), and each receiver then filters
//! the tile's candidates with its own cone. This is the tile-frustum traversal
//! from the original design.
//!
//! Pass criterion, fixed before the first run: in both large scenes, culling cost
//! per receiver with tile-shared traversal is at most 1/3 of per-receiver
//! traversal, with identical candidate sets at every receiver.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::bvh::Bvh;
use room3d::{Receiver, Room, TileFrustum, floor_point, gate1, with_rubble, with_sphere};

const SIZE: usize = 64;
const TILE: usize = 8;

fn key(r: &Receiver) -> Vec<(usize, [u32; 9])> {
    let mut all: Vec<(usize, [u32; 9])> = r
        .candidates
        .iter()
        .flat_map(|(m, tris)| {
            tris.iter()
                .map(move |t| (*m, std::array::from_fn(|i| t[i / 3][i % 3].to_bits())))
        })
        .collect();
    all.sort_unstable();
    all
}

struct Measure {
    name: &'static str,
    triangles: usize,
    per_receiver_ns: f64,
    tile_ns: f64,
    tile_candidates: f64,
    mismatches: usize,
}

fn measure(name: &'static str, room: &Room) -> Measure {
    let bvh = Bvh::build(&room.meshes);
    let n = SIZE * SIZE;

    let start = Instant::now();
    let per_receiver: Vec<Option<Receiver>> = (0..n)
        .map(|px| Receiver::with_bvh(room, &bvh, floor_point(px, SIZE)).map(|r| r.0))
        .collect();
    let per_receiver_ns = start.elapsed().as_nanos() as f64 / n as f64;

    let tiles_per_side = SIZE / TILE;
    let start = Instant::now();
    let mut tiled: Vec<Option<Receiver>> = (0..n).map(|_| None).collect();
    let mut tile_candidates = 0usize;
    for t in 0..tiles_per_side * tiles_per_side {
        let (tx, ty) = (t % tiles_per_side, t / tiles_per_side);
        let pixels: Vec<usize> = (0..TILE * TILE)
            .map(|i| (ty * TILE + i / TILE) * SIZE + tx * TILE + i % TILE)
            .collect();
        let (mut lo, mut hi) = (
            continuum::glam::Vec3::splat(f32::MAX),
            continuum::glam::Vec3::splat(f32::MIN),
        );
        for &px in &pixels {
            let p = floor_point(px, SIZE);
            (lo, hi) = (lo.min(p), hi.max(p));
        }
        let frustum = TileFrustum::new([lo.x, lo.y, hi.x, hi.y], &room.light);
        let (shared, _) = bvh.collect(
            |a, b| frustum.may_touch_box(a, b),
            |tri| frustum.may_touch(tri),
        );
        tile_candidates += shared.len();
        for &px in &pixels {
            tiled[px] = Receiver::from_tile(room, &shared, floor_point(px, SIZE));
        }
    }
    let tile_ns = start.elapsed().as_nanos() as f64 / n as f64;

    let mismatches = per_receiver
        .iter()
        .zip(&tiled)
        .filter(|(a, b)| match (a, b) {
            (Some(a), Some(b)) => key(a) != key(b),
            (None, None) => false,
            _ => true,
        })
        .count();
    Measure {
        name,
        triangles: bvh.triangles(),
        per_receiver_ns,
        tile_ns,
        tile_candidates: tile_candidates as f64 / (tiles_per_side * tiles_per_side) as f64,
        mismatches,
    }
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let results = [
        measure("Gate 1 room", &gate1()),
        measure("+ sphere", &with_sphere()),
        measure("+ 3,000 boxes", &with_rubble()),
    ];
    let mut out = String::new();
    writeln!(out, "# M1 gate 3 results\n").unwrap();
    writeln!(out, "{SIZE}×{SIZE} floor receivers in {TILE}×{TILE} tiles, single-threaded. Culling time per receiver, including building each receiver's cone.\n").unwrap();
    writeln!(out, "| Scene | Triangles | Per-receiver traversal | Tile-shared traversal | Ratio | Tile candidates | Candidate-set mismatches |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|").unwrap();
    for m in &results {
        writeln!(
            out,
            "| {} | {} | {:.0} ns | {:.0} ns | {:.2} | {:.0} | {} |",
            m.name,
            m.triangles,
            m.per_receiver_ns,
            m.tile_ns,
            m.tile_ns / m.per_receiver_ns,
            m.tile_candidates,
            m.mismatches
        )
        .unwrap();
    }
    let pass = results[1..]
        .iter()
        .all(|m| m.tile_ns <= m.per_receiver_ns / 3.0)
        && results.iter().all(|m| m.mismatches == 0);
    writeln!(out, "\n## Verdict\n\nTile-shared ≤ 1/3 of per-receiver in both large scenes, identical candidates → {}.", if pass { "PASS" } else { "FAIL" })
        .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m1_gate3_results.md"), &out)?;
    Ok(())
}
