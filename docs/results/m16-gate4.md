# M16 gate 4: surface lobes (a dielectric's reflectance and a clear coat)

**PASS** on both criteria.

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Speed: warm frame, with / without the lobes | **1.134×** (5.97–6.01 ms against 5.28–5.30 ms, five rounds) | at most 1.25× | PASS |
| 2. Accuracy against Cycles, clamped at 4× its mean | **3.64%** (ignoring the lobes 4.94%) | at most 6% | PASS |
| 2. Over the 112,639 pixels the lobes change | **3.90%** against 8.48% ignoring them (**0.461×**) | at most 0.65× | PASS |

Unclamped: 44.98% (ignoring the lobes 136.60%), over the changed pixels
98.73% (300.54%): the coat's sun highlights decide it (pixels above the
clamp: ours 88, Cycles 110). Means over the changed pixels: ours +0.38%,
ignoring +0.75%. Every view settled. Cycles took 63 s (Metal, 4,096
samples).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `457ecf7`; binary
`experiments/room3d/src/bin/m16_gate4.rs`, stub committed in the commit
before; output `docs/results/m16-gate4-run.txt`), started after three
minutes with no game or indexing running and one more minute idle.

## Method

DESIGN §3.4, "Transmission" (surface lobes): `InstancedScene::geometry_specular`
(index, factor, tint) sets the specular layer's reflectance at normal and
grazing incidence; `InstancedScene::geometry_clearcoat` adds a coat of
index 1.5, its own specular from each light and a reflection ray, over
what lies under it dimmed by its directional albedo. The textured scene
with the boulders a rough dielectric under a smooth coat and the brick
block of index 2.0 tinted (`Textured::with_lobes`), bounce light on; ours
at 4×4 sub-pixel offsets against Cycles' Principled BSDF (IOR, Specular
IOR Level half the factor, Specular Tint, Coat Weight and Roughness, Coat
IOR 1.5).

## Disclosures

- The bars were set with development numbers known (0.58× over the
  changed pixels and 1.18× the frame at 640×360); the full-size run did
  better (0.461×, 1.134×).
- Both images are clamped at 4× the reference's mean, as fixed before the
  run: the smooth coat reflects the sun over a few pixels at its full
  radiance.
- Not covered: the coat's textures and normal map (glTF's), sheen.
