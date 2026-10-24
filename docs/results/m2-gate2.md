# M2 gate 2: a hierarchy on the GPU

Date: 2026-10-24. The first GPU render tested every triangle twice per pixel:
once for the camera ray, once for the cone. Here a mesh-level bounding volume
hierarchy (median splits, at most two meshes per leaf) is traversed by both:
rays with a slab test, cones with a conservative box test. Coverage is still
summed per mesh, so results must not change. Ray ties (coplanar faces of
different meshes) now go to the lower mesh index, so the result does not depend
on traversal order.

Pass criterion, fixed before the first run: on the render scene at 1920×1080,
(1) the hierarchy render matches the brute-force render within 1e-5 of peak
brightness at every pixel, and (2) it is at least 3× faster. A scene with 3,000
extra boxes is reported for information.

**Verdict: FAIL.** Identical output (difference 0), but only 1.6× faster.

## Results

Apple M4 (MacBook Air), best of 5. Reproduce with
`cargo run --release -p room3d --bin m2_gate2`.

| Scene | Meshes | Triangles | Resolution | Brute force | Hierarchy | Speedup |
|---|---|---|---|---|---|---|
| Render scene | 37 | 444 | 1920×1080 | 210.3 ms | 130.4 ms | 1.6× |
| + 3,000 boxes | 3,037 | 36,444 | 480×270 | 689.7 ms | 347.9 ms | 2.0× |

## Where the time goes (1080p, profiled by disabling stages)

| Stage | Brute force | Hierarchy |
|---|---|---|
| Camera rays | 41.8 ms | 5.6 ms |
| + cone setup and culling | 98.5 ms | 50.8 ms |
| + cone rasterization | 69.6 ms | 72.7 ms |

## What this shows

- The hierarchy does its job: camera rays are 7.5× faster.
- The frame is now dominated by work the hierarchy cannot remove. Each pixel
  computes its 64 bin weights from scratch (256 `atan2` calls), and each
  candidate triangle is clipped through polygon copies that are expensive in
  GPU registers. The 3× target was really a statement about shared cone cost,
  not about the hierarchy.
- Next (M2 gate 3, fixed before running): 960×540 in at most 16.7 ms on the M4
  Air, with output within 1e-5 of this kernel.
