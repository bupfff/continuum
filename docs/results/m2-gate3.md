# M2 gate 3: 960×540 at 60 fps with per-pixel cones

Date: 2026-10-24. Gate 2's profile showed shared cone work dominating the GPU
frame. This gate tried to optimize it without changing results.

Pass criterion, fixed before the first run: the render scene at 960×540 with
the hierarchy renders in at most 16.7 ms (best of 5), and matches the
pre-optimization kernel within 1e-5 of peak brightness at every pixel.

**Verdict: FAIL.** 33.6 ms → 26.0 ms (23% faster), matching the reference
within 5.8e-9, but not 16.7 ms.

## What was tried, in order (all measured with `m2_gate3`)

| Change | Time | Kept? |
|---|---|---|
| Baseline (gate 2 kernel) | 33.6 ms | — |
| Shared-corner bin weights (144 edge terms instead of 256) using ~1.8 KB of per-thread arrays | 55.4 ms | No: the arrays spill out of registers and cut occupancy |
| Skip clipping when every vertex is inside; stop when every bin is full | 28.0 ms | Yes (bit-identical) |
| Shared edges carried with small state (8-float array) | 28.1 ms | No: no gain |
| Bin weights computed at the end without an array; prefix-sum only the cells a mesh touched | 26.0 ms | Yes (within 5.8e-9) |
| Array-free path for unclipped triangles | 26.1 ms | No: no gain |

Profile at 960×540 before the last two changes: camera rays 1.6 ms, bin weights
4.6 ms, traversal and culling (with per-mesh bookkeeping) 8.6 ms,
rasterization 13.3 ms.

## What this shows

- On this GPU, per-thread arrays are the main hazard: making them bigger to do
  less arithmetic made the kernel 1.6× slower.
- Rasterizing every candidate triangle into every pixel's cone, every frame, is
  inherently expensive (roughly 8 million triangle rasterizations per frame
  here), and divergent across pixels.
- That is the brute-force approach the paradigm exists to avoid. Direct light on
  static surfaces changes only when a light or mover changes, so it belongs in
  a world-space cache updated exactly when something moves, as 3D gate 4 did on
  the CPU at 14.9×. Then a frame is camera rays plus lookups (rays alone: 1.6 ms
  at 960×540, 5.6 ms at 1080p). That is the next gate.
