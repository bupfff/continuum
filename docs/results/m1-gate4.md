# M1 gate 4: exact incremental direct light in 3D

Date: 2026-10-23 (overnight run). The main mechanism from the 2D phase (cache
what static geometry contributes per cone; apply dynamic changes exactly, and
only where they can reach), now on triangle meshes.

Pass criterion, fixed before the first run: (1) the incremental output equals a
full recompute at every receiver and frame, within 1e-5 of the light's weight;
(2) the incremental path is at least 5× faster than recomputing every receiver
(hierarchy traversal plus rasterization), at the same thread count.

**Verdict: PASS.** Bit-exact (maximum difference 0) and 14.9× faster.

## Method

- Each floor receiver caches the `CoverageMask` of the static meshes in its cone
  toward the light (built once: 5.3 ms, 1.1 MB for 4,096 receivers).
- Each frame, each moved box's swept bounding box is tested against every 8×8
  tile's frustum (the prismatoid enclosing the tile's cones). Only touched tiles
  are recomputed: start from the cached static mask and add the movers.
- The full recompute does what a renderer without caches does: traverse the
  hierarchy and rasterize every candidate for every receiver.

## Results

Gate 1 room (34 static meshes) plus 3 moving boxes (sliding, orbiting near the
light, bobbing); 64×64 receivers; 120 frames; 10 threads for both paths.
Reproduce with `cargo run --release -p room3d --bin m1_gate4`.

| | |
|---|---|
| Full recompute | 2.93 ms/frame |
| Incremental | 0.20 ms/frame (**14.9× faster**) |
| Receivers recomputed per frame | 23.4% |
| Max difference from full recompute | 0.0 |

## What this shows

Exact incremental work carries over to 3D, with two sources of speed: most
tiles are untouched in a typical frame, and touched receivers skip hierarchy
traversal and all static triangles. Both are exact.

## Caveats

- Direct light from one static area light; floor receivers only.
- The static cache costs 264 bytes per cone; a GPU layout needs less.
- Tile reach is tested with each mover's swept bounding box, which is coarser
  than its true swept volume (it recomputes some tiles that did not change).
