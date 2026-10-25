# M5 gate 2: sky light against Cycles, with the reference fixed

Date: 2026-10-24. M5 gate 1 failed (sun and sky 5.12% from Cycles), and its
diagnosis blamed the reference, not the sky:

- **Pixel sampling.** Cycles integrates each pixel's area: Blender 5.2
  applies at least a one-pixel box filter. Our renders took one sample per
  pixel center.
- **The distant sun.** Cycles loses 1.9% of a 0.5° quad sun 3 km away.

This gate repeats gate 1 with both fixed, and nothing changed in the
renderer:

- our renders take 4×4 samples per pixel (rendered at 4× the resolution and
  box filtered);
- the sun is 1 km away, with the same angular size, radiance and direction.

## Pass criterion

Fixed before the first run, in 2dcddc0. At 640×360 against Cycles (4096
samples per pixel, direct light only, one-pixel box filter, seed 1):

1. **Sky only:** exact cones within 3% relative RMSE.
2. **Sun and sky:** exact and prefiltered cones, each within 3%.

**Verdict: PASS.**

## Results

Apple M4, compute rays, the outdoor scene (63 instances) with the sun at 1
km. Cycles on Metal took 36.6 s per render. Reproduce with `cargo run
--release -p room3d --bin m5_gate2`.

| Render | 4×4 per pixel vs Cycles | 1 per pixel vs Cycles | Time (4×4) |
|---|---|---|---|
| Sky only, exact cones | **0.74%** | 2.74% | 503.0 s |
| Sun and sky, exact cones | **1.19%** | 4.63% | 488.7 s |
| Sun and sky, prefiltered cones (1 bin) | **1.19%** | 4.63% | 66.5 s |

Cycles' own noise (RMSE between two seeds / √2): 0.10% sky only, 0.04% sun
and sky.

Surface interiors, ours divided by Cycles (green channel, 4×4 per pixel):

| Region | Sun | Sky |
|---|---|---|
| Wall | 1.0000 | 0.9992 |
| Pillar | 1.0000 | 1.0003 |
| Ground | 1.0001 | 0.9951 |

## What this means

- **Direct light from a sun and a uniform sky matches an independent path
  tracer.** The match is to 0.7–1.2% over whole images, and to 0.0–0.5% inside
  surfaces. This is the project's closest measured agreement with Cycles; M3
  gate 3 had 3.80%, a figure inflated by the same sampling mismatch.
- **What remains is mostly the sky on open ground (−0.5%).** Two likely
  causes, not yet separated:
  - bins treat coverage as uniform within their 11° (the cosine varies
    within a bin);
  - the combination rule across meshes.
- **One sample per pixel costs 2–3.5 points of error at silhouettes in this
  scene.** Anti-aliased primary visibility (roadmap item 4) is where that
  goes.
- **Prefiltering costs nothing here.** Exact and prefiltered agree to
  0.00%, and prefiltering is 7× faster.
- **Per-pixel sky cones are expensive.** At one sample per pixel, a 640×360
  frame takes about 7 s prefiltered and 40 s exact (gate 1's times). Real-time sky needs the
  lighting cache and, for large scenes, prefiltered coverage at the object
  level. That is the next M5 gate.
- **Lessons recorded for future references:**
  - Cycles references are area-integrated, so compare supersampled renders.
  - Keep small, bright emitters within about 1 km in Cycles scenes.
