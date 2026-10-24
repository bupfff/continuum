# M1 gate 3: tile-shared hierarchy traversal

Date: 2026-10-23 (overnight run). Gate 2 showed culling dominating per-receiver
cost. Neighboring receivers' cones are nearly identical, so an 8×8 tile of
receivers walks the hierarchy once, with a frustum enclosing all 64 cones (the
prismatoid between the tile and the light). Each receiver then filters the
tile's candidate triangles with its own cone.

Pass criterion, fixed before the first run: in both large scenes, culling cost
per receiver with tile-shared traversal is at most 1/3 of per-receiver
traversal, with identical candidate sets at every receiver.

**Verdict: FAIL.** Candidate sets are identical everywhere, but the saving is
only 20–24% in the large scenes (ratio 0.76–0.80, target ≤ 0.33).

## Results

64×64 floor receivers in 8×8 tiles, single-threaded; culling time per receiver,
including building each receiver's cone. Reproduce with
`cargo run --release -p room3d --bin m1_gate3`.

| Scene | Triangles | Per-receiver traversal | Tile-shared traversal | Ratio | Tile candidates | Mismatches |
|---|---|---|---|---|---|---|
| Gate 1 room | 408 | 8.5 µs | 4.2 µs | 0.50 | 51 | 0 |
| + sphere | 40,008 | 25.3 µs | 20.2 µs | 0.80 | 2,054 | 0 |
| + 3,000 boxes | 36,408 | 30.1 µs | 23.0 µs | 0.76 | 1,214 | 0 |

## Why it missed

The tile's frustum is far wider than any one cone, so it collects 2.5× more
triangles than a single receiver needs (1,214–2,054 vs 473–827). Every
receiver then tests all of them, and that flat filter now dominates.

## Likely fixes (not tried)

- **Frontier handoff**: the tile walk stops at the nodes where receivers start
  to disagree and hands that frontier to each receiver, which continues
  hierarchically instead of testing a flat triangle list.
- **Smaller tiles** (4×4) for a tighter shared frustum.
- **Cheaper per-triangle tests** (precomputed plane distances per tile).
