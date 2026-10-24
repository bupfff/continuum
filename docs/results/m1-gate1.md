# M1 gate 1: mask cones vs rays in 3D

Date: 2026-10-23 (overnight run). First milestone 1 result: does the 2D gate 1
win carry over to 3D, with 8×8 bins and triangle meshes?

Pass criterion, fixed before the first run: the continuous mask cone (8×8 bins)
has lower error than 64 jittered rays, at no more than 1/4 of their cost per
receiver, with zero flicker.

**Verdict: attempt 1 failed (2.15× the cost of the rays); attempt 2 passed
(0.22×, lower error, no flicker).** Both are reported.

## What was built (`continuum::cone3`)

- **Pyramid cones**: an apex, an orthonormal frame, and a window on the plane
  one unit ahead, split into 8×8 bins (one `CoverageMask`). A cone toward a
  rectangular light uses the light's plane normal as `forward`, so the light
  projects exactly onto the window.
- **Exact per-bin coverage**: triangles are clipped to the slab between a near
  distance and the light's plane (geometry behind the light does not occlude),
  projected, clipped to the window, and rasterized with an accumulation buffer:
  each edge adds its signed swept area to the cells it crosses, and a running
  sum gives every bin's exact covered area. This is the technique font
  renderers (for example font-rs) use for anti-aliased glyphs.
- **Meshes**: within one mesh, front-facing triangles add, which is exact for
  convex meshes, so internal edges do not leak. Across meshes, bins combine by
  max, as in 2D.
- **Bin weights**: each bin's exact projected solid angle (Lambert's polygon
  formula), so visibility sums directly to irradiance.
- Tests: weights match sampled projected solid angle to 0.1%; occluded
  visibility matches ray sampling to within 2%; a mesh split by internal edges
  covers exactly; geometry behind the light does not occlude.

## Setup

96×96 floor receivers; a 0.2 × 0.2 ceiling light at height 1; 34 box meshes
(408 triangles): two double-slat fences (one aligned, one offset), a table top
and a floating cube. Reference: 1,024 stratified light samples per receiver.
Every method gets the same candidate triangles per receiver (the culling a
hierarchy would do; about 28 on average) and is timed single-threaded on top of
that shared setup. Reproduce with `cargo run --release -p room3d --bin m1_gate1`.

## Results (attempt 2)

| Method | Rel. RMSE | Flicker | ns/receiver | Cost vs 64 jittered rays |
|---|---|---|---|---|
| **Mask cone, 8×8 continuous** | **2.80%** | **0** | **1,071** | **0.22×** |
| 64 rays, fixed grid | 5.67% | 0 | 4,805 | 0.98× |
| 4 rays, jittered | 17.83% | 25.57% | 374 | 0.08× |
| 16 rays, jittered | 7.92% | 11.19% | 1,313 | 0.27× |
| 64 rays, jittered | 3.19% | 4.27% | 4,882 | 1.00× |
| 256 rays, jittered | 1.55% | 1.55% | 18,368 | 3.76× |

Attempt 1 had the same accuracy at 10,486 ns/receiver (2.15×). It clipped the
projection against every overlapped bin, with allocations; the accumulation
rasterizer is 10× faster with identical results.

## What this shows

1. **The 2D result carries over to 3D.** One 8×8 mask cone gives lower error
   than 64 random rays at about a fifth of their cost, and it is
   deterministic: no flicker, no denoiser. It is cheaper than 16 random rays
   while being 2.8× more accurate.
2. **A regular ray grid is a poor baseline**: 64 fixed rays alias against the
   regular slats (5.67% error), worse than 64 jittered rays.
3. **The remaining error is structured**, not noisy. It is mostly light
   leaking under the fences, where two separate slats partly cover the same bin
   and max-combining under-counts their union (the 2D within-bin correlation
   issue, now between meshes). Candidates: merge nearby small meshes into one
   coverage group, or use finer bins near silhouette edges.

## Caveats

- The shared setup (cone construction plus testing all 408 triangles per
  receiver, with no hierarchy) costs about 15 µs per receiver, more than either
  method. A BVH is the next infrastructure piece; it benefits both methods.
- Max-combining across meshes is approximate; light can leak where separate
  meshes share bins (see above).
- CPU only; the GPU version is milestone M2.
