# M2 gate 1: mask cones on the GPU

Date: 2026-10-23 (overnight run). The design's open GPU question was whether
the core primitive (8×8 continuous mask cones with exact-area rasterization)
works in a compute shader through wgpu. It is now ported to WGSL
(`crates/continuum-gpu/src/mask_cones.wgsl`) and runs on Metal on the M4.

Pass criterion, fixed before the first run: (1) every receiver's GPU result is
within 1e-4 of the light's weight of the CPU result; (2) the GPU is faster than
the CPU on all threads for the same work.

**Verdict: attempt 1 failed (1); attempt 2 passed.** The GPU is 7.5× faster
than 10 CPU threads and matches the CPU to within 6.4e-6 at every receiver.

## Setup

65,536 floor receivers (256×256) in the gate 1 room (34 meshes, 408 triangles).
Both paths do identical work: every receiver tests every triangle (no hierarchy
yet), rasterizes the candidates, and sums coverage per mesh. GPU time is
submit-to-completion, best of 5, excluding uploads and readback. Reproduce with
`cargo run --release -p room3d --bin m2_gate1`.

## Results

| | Time |
|---|---|
| CPU, 10 threads | 73.1 ms |
| GPU, Apple M4 (wgpu → Metal) | 9.7 ms (**7.5× faster**) |

Max difference from the CPU: 6.4e-6 of the light's weight; receivers over
1e-4: 0.

## Attempt 1 and the bugs it found

Attempt 1 was 7.3× faster but a third of receivers differed by up to 2.7%.
Comparing per-bin coverage and the raw accumulation buffers found two causes,
both now fixed on CPU and GPU alike:

1. **Precision in the bin weights**: `acos(dot(a, b))` loses precision for the
   tiny angles of small, far bins (8e-4 relative differences between CPU and
   GPU). It is now `atan2(|a × b|, a · b)`, which is stable for small angles.
2. **A latent rasterizer bug**: stepping along an edge that lies on the grid's
   left boundary can drift to x ≈ -1e-9; `floor` then gives -1 and a whole
   row's coverage is credited to the wrong cell. The CPU's rounding happened to
   avoid it in every scene so far; the GPU's did not. Stepped x is now clamped
   to the grid. The M1 gates were rerun after the fix and are unchanged.

## What this shows

- The paradigm's core primitive runs on the GPU with no special hardware or
  extensions: plain WGSL compute, private arrays (68 floats per cone), no
  64-bit integers (coverage is a float per bin).
- Even brute force (every receiver tests every triangle) is 7.5× faster than
  the CPU. Hierarchy traversal and tile-shared work on the GPU are next.

## Caveats

- One kernel and one scene; occupancy and register pressure are not tuned.
- Timing excludes uploads and readback (fine for a renderer, where data stays
  on the GPU).
- Tested on Apple M4 only. NVIDIA/Windows correctness should come from CI
  (WARP, lavapipe), which is not set up yet.
