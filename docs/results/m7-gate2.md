# M7 gate 2: texture filtering for the area integral

**PASS on both criteria.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. At most 0.8× the isotropic filter's error from the 16×16 area-integrated reference | **3.99%** against 7.11% (**0.56×**) | PASS |
| 2. Warm cached frame at most 1.15× | 2.54 ms against 2.47 ms (**1.03×**) | PASS |

Apple M4, 1280×720, release build, run once on 2026-11-07 (binary
`experiments/room3d/src/bin/m7_gate2.rs`, criteria pre-registered in
`bb1f4f0`). The isotropic figure, 7.11%, reproduces M7 gate 1's exactly.
No pixel saw a light, so none were set apart.

## Method

M7 gate 1 found that on the textured scene the error of one ray per pixel
is texture filtering, not silhouettes. Each pixel read its texture at the
mip of its footprint's longest axis, isotropically, which blurs across the
shorter axis.

Now the pixel's footprint is found from ray differentials: the camera
ray's change per pixel, projected onto the triangle's plane and mapped to
texture space. The texture is then read with four bilinear taps at the
centers of the footprint's 2×2 quarters, each at the mip of a quarter's
longest axis. That is a box filter over the pixel, anisotropic where the
footprint is.

It is the default. `InstancedRenderer::set_texture_box(false)` restores
the old filter.

## Notes

- The remaining 3.99% includes edges, shading that varies within pixels,
  and texture detail finer than four taps resolve.
- The GPU test's golden image was re-rendered on the M4 for the new
  filter. Its checkered ground moved 2.7% against the old image.
- No development runs.

## Prior art

Ray differentials (Igehy 1999), and anisotropic filtering by multiple
taps along the footprint (as hardware anisotropic filtering does). No
novelty claim is made.
