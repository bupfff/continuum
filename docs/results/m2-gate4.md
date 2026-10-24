# M2 gate 4: a world-space lighting cache on the GPU

Date: 2026-10-24. Gate 3 showed that recomputing a cone for every pixel every
frame is the brute-force path. The paradigm says to cache what cannot change:
direct light on static surfaces changes only when a light or mover changes. So
it is baked into lightmap texels (one mask cone per texel, through the
hierarchy), and a frame is a camera ray per pixel plus a bilinear lookup.
Dynamic geometry (none in this scene) keeps a cone per pixel.

Pass criterion, fixed before the first run: (1) the cached render at 1920×1080
takes at most 16.7 ms (best of 5) on the M4 Air; (2) its relative RMSE against
the per-pixel cone render of the same frame is at most 2%. Bake time is
reported.

**Verdict: attempt 1 failed (2); attempt 2 passed.** 1080p in 5.66 ms (about
177 fps) at 0.58% RMSE.

## Results (attempt 2)

Apple M4 (MacBook Air). Lightmap: 751,592 texels of about 0.005 room units on
37 static box meshes. Reproduce with `cargo run --release -p room3d --bin m2_gate4`.

| | Time |
|---|---|
| Bake (one cone per texel) | 38.1 ms, once |
| Cached render, 1920×1080 | **5.66 ms** |
| Per-pixel cone render, 1920×1080 | 96.4 ms |

Relative RMSE of the cached frame against per-pixel cones: 0.58%.

## Attempt 1 and the bug it found

Attempt 1 had the same speed but 3.78% RMSE. The error map showed two causes:

1. **A bug**: faces pointing away from the light were baked with light. The
   bake used the receiver kernel, which lacked the camera path's facing rule
   (a surface facing away from the light gets no direct light). Every receiver
   now uses the same rule. This also made the bake 2× faster, since back faces
   skip the cone.
2. **Interpolation**: bilinear lookups blur shadow structure finer than a
   texel, visible as faint stripes under the fences. This remains, and is most
   of the remaining 0.58%. Texel size was not changed.

## What this shows

- With static lighting cached in world space, a 1080p frame costs about what
  camera rays cost. That is about 17× cheaper than per-pixel cones, at under
  1% error, on a fanless laptop GPU.
- This is the paradigm's first rule ("cache per cone what cannot change")
  paying off at frame level. The next step is the second rule: when something
  moves, update exactly the texels whose cones it can reach (3D gate 4 did
  that on the CPU at 14.9×).

## Caveats

- Box meshes only (texels per box face); general meshes need a texel
  parameterization (per-triangle grids or UV charts).
- 751k texels includes faces that are never visible (box bottoms, wall
  backs); culling those would cut memory and bake time.
- Lightmap resolution limits sharp contact shadows; fine shadow detail blurs.

## Rerun after the block layout (gate 5)

For gate 5, texels are now stored in 8×8 blocks, and faces are padded to whole
blocks: 821,376 entries, of which the padding is unlit and never read. Rerun on
2026-10-24: cached render 5.66 ms, 0.58% RMSE (unchanged), bake 49.8 ms (more
entries, and the run includes first-dispatch warm-up).
