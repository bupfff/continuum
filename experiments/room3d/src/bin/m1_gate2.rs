//! M1 gate 2: hierarchy culling, and how cost scales with scene size.
//!
//! Each receiver's cone must find the triangles that can touch it. Testing every
//! triangle costs time linear in the scene; a bounding volume hierarchy should
//! cost far less. Three scenes: the gate 1 room (408 triangles), the room plus a
//! finely tessellated sphere (about 40,000), and the room plus 3,000 small boxes
//! (36,408).
//!
//! Pass criterion, fixed before the first run: (1) hierarchy culling gives the
//! same mask results as testing every triangle (within 1e-6 of the light's
//! weight) at every receiver; (2) hierarchy culling time per receiver in the
//! largest scene is at most 10× that of the gate 1 room (for about 100× the
//! triangles). The mask's own cost is reported, not judged: with no prefiltered
//! hierarchy yet, it grows with the triangles inside each cone.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::bvh::Bvh;
use room3d::{Receiver, Room, floor_point, gate1, with_rubble, with_sphere};

const SIZE: usize = 64;

struct Measure {
    name: &'static str,
    triangles: usize,
    build_ms: f64,
    brute_ns: f64,
    bvh_ns: f64,
    visited: f64,
    candidates: f64,
    mask_ns: f64,
    max_diff: f32,
    /// Receivers whose candidate triangle sets differ (as multisets), a diagnostic
    /// separating culling bugs from float summation order.
    set_mismatches: usize,
}

fn measure(name: &'static str, room: &Room) -> Measure {
    let n = SIZE * SIZE;
    let start = Instant::now();
    let bvh = Bvh::build(&room.meshes);
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let brute: Vec<Option<Receiver>> = (0..n)
        .map(|px| Receiver::new(room, floor_point(px, SIZE)))
        .collect();
    let brute_ns = start.elapsed().as_nanos() as f64 / n as f64;

    let start = Instant::now();
    let with_bvh: Vec<Option<(Receiver, usize)>> = (0..n)
        .map(|px| Receiver::with_bvh(room, &bvh, floor_point(px, SIZE)))
        .collect();
    let bvh_ns = start.elapsed().as_nanos() as f64 / n as f64;

    let mut mask_time = Duration::ZERO;
    let (mut visited, mut candidates, mut max_diff, mut set_mismatches) =
        (0usize, 0usize, 0.0f32, 0usize);
    let key = |r: &Receiver| {
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
    };
    for (a, b) in brute.iter().zip(&with_bvh) {
        let (Some(a), Some((b, v))) = (a, b) else {
            continue;
        };
        if key(a) != key(b) {
            set_mismatches += 1;
        }
        visited += v;
        candidates += b.candidates.iter().map(|c| c.1.len()).sum::<usize>();
        let start = Instant::now();
        let mb = b.masked();
        mask_time += start.elapsed();
        let ma = a.masked();
        max_diff = max_diff.max((ma - mb).abs() / b.cone.total_weight());
    }
    Measure {
        name,
        triangles: bvh.triangles(),
        build_ms,
        brute_ns,
        bvh_ns,
        visited: visited as f64 / n as f64,
        candidates: candidates as f64 / n as f64,
        mask_ns: mask_time.as_nanos() as f64 / n as f64,
        max_diff,
        set_mismatches,
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
    writeln!(out, "# M1 gate 2 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} floor receivers, single-threaded release build. Times per receiver.\n"
    )
    .unwrap();
    writeln!(out, "| Scene | Triangles | BVH build | Brute-force culling | BVH culling | Nodes visited | Candidate triangles | Mask (after culling) | Max difference | Candidate-set mismatches |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|").unwrap();
    for m in &results {
        writeln!(
            out,
            "| {} | {} | {:.1} ms | {:.0} ns | {:.0} ns | {:.0} | {:.0} | {:.0} ns | {:.1e} | {} |",
            m.name,
            m.triangles,
            m.build_ms,
            m.brute_ns,
            m.bvh_ns,
            m.visited,
            m.candidates,
            m.mask_ns,
            m.max_diff,
            m.set_mismatches
        )
        .unwrap();
    }
    let identical = results.iter().all(|m| m.max_diff <= 1e-6);
    let growth = results[1..]
        .iter()
        .map(|m| m.bvh_ns / results[0].bvh_ns)
        .fold(0.0, f64::max);
    let pass = identical && growth <= 10.0;
    writeln!(
        out,
        "\n## Verdict\n\nSame results as brute force: {}. Largest BVH culling growth over the gate 1 room: {:.1}× \
         (target ≤ 10× for about 100× the triangles) → {}.",
        if identical { "yes" } else { "no" },
        growth,
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m1_gate2_results.md"), &out)?;
    Ok(())
}
