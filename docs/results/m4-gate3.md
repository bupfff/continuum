# M4 gate 3: nearest-first ray traversal

Date: 2026-10-24. M4 gate 2 failed its scaling criterion: a million boulders
cost 3.04× a thousand. A diagnostic showed camera rays, not cones, set the
cost. Rays walked both hierarchies (objects, and each object's triangles) in a
fixed child order, so a ray near the horizon often descended into far boulders
before finding the near one that ends it.

**The change.** At every interior node, a ray tests both children's boxes. It
visits the one it enters first, skips a child it misses, and pushes the other
only if it may still hold a nearer hit. The nearest hit and its tie rule
(lower object, then lower triangle) do not depend on the order, so the output
should not change. The kernel setting `set_nearest_first` (on by default)
keeps the old order for comparison.

Pass criterion, fixed before the first run (committed in 4ee1875 before
implementing): gate 2's boulder field and camera, per-pixel prefiltered mask
cones at 1920×1080, best of 5.

1. **Same output.** With 1,000,000 boulders, every pixel is within 1e-5 of
   peak of the render with the fixed child order.
2. **Scale.** 1,000,000 boulders take at most 2× the frame of 1,000 boulders.
3. **Speed.** The 1,000,000-boulder frame is at least 1.5× faster than with
   the fixed order.

**Verdict: PASS.** Identical output, 1.40× for 1000× the triangles, and 2.83×
faster.

## Results

Apple M4 (MacBook Air, on battery). Reproduce with
`cargo run --release -p room3d --bin m4_gate3`.

| | Frame at 1920×1080 |
|---|---|
| 1,000 boulders (10^7 triangles), nearest-first | 72.2 ms |
| 1,000,000 boulders (10^10 triangles), nearest-first | **100.8 ms** |
| 1,000,000 boulders, fixed order | 284.8 ms |

- Max difference, nearest-first vs fixed order: **0.0**.
- Scaling for 1000× the triangles: **1.40×**.

The fixed-order frame took 284.8 ms here, against 237.9 ms in gate 2's run of
the same code. That is battery and thermal variance between runs; the
criterion compares runs made in the same session.

## What this shows

Across the three axes of scale, after gates 1–3:

- **Detail:** prefiltered cones keep a cone's cost flat as triangles inside it
  grow (gate 1).
- **Amount:** with instancing and nearest-first rays, 1000× more triangles
  (10^7 → 10^10) cost 1.4× more per frame.
- **Size:** f64 positions rebased near the camera make a field 64 km from the
  origin render identically (gate 2).

## What still limits it

- **The floor is high.** 72 ms for 1,000 boulders at 1080p is slow for about
  2 million camera rays. A likely cause: one large shader does both ray
  casting and mask cones. Cone state (an accumulator, coverage bins, clipping
  polygons) takes many registers, so fewer rays run at once. Splitting the
  frame into a visibility pass (nearest hit per pixel) and a shading pass is
  the standard fix, and the one Nanite uses.
- **Lighting is per-pixel every frame.** That is the brute-force path. Large
  worlds need the lighting cache to scale with the view, not the world: a
  virtual cache that allocates texels where they are seen, coarser with
  distance.
- **The far field aliases.** Boulders 1–2 km away are smaller than a pixel,
  so one ray per pixel flickers as the camera moves. Primary visibility
  through pixel-footprint cones that stop at sub-pixel nodes (gate 1's
  prefiltering, for the camera) would fix both aliasing and far-field cost.
