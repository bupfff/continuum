# M10 gate 2: metallic and rough materials, glossy reflections

**FAIL on criterion 1 (accuracy); criterion 2 (speed) passes.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 6% of Cycles, and at most 0.6× the Lambertian render's error | **8.94%**; Lambertian 119.37% (0.07×) | FAIL |
| 2. Warm frame at most 2× the Lambertian scene's | 4.98 ms against 2.74 ms (**1.82×**) | PASS |

Apple M4, 1280×720, release build, 2026-11-07 (commit `3e97f17`; binary
`experiments/room3d/src/bin/m10_gate2.rs`, criteria pre-registered in
`e6aafb2`).

## Method

Materials are set per geometry (`InstancedScene::geometry_materials`): glTF's
metallic-roughness, with GGX at α = roughness². All of this is in the
cached path:

- **Specular from lights:** evaluated toward each light's nearest point to
  the mirror direction (Karis's representative point, its lobe widened by
  the light's size), times the cached shadowed share.
- **Glossy reflections:** one ray per pixel. The hit's cached diffuse light
  is read at the mip matching the lobe's width there (a roughness cone,
  prefiltered by mips), and a miss sees the sky. It is weighted by the
  split sum's directional albedo (Karis's fit).
- **Diffuse:** times (1 − metallic), less the specular layer's albedo.

The scene is M6's textured scene with bounce light on. The ground is a
dielectric of roughness 0.6, the boulders metal of roughness 0.3, and the
brick block a dielectric of roughness 0.4. The reference is instanced
Cycles: Principled BSDF with GGX, one bounce, 4,096 samples per pixel.

## Results

- Ours: 8.94% relative RMSE, mean +0.96% against Cycles.
- Without reflection rays: 12.71% (mean −3.69%), so reflections remove
  about a third of the error.
- Lambertian: 119.37%. Metal boulders look nothing like Lambertian ones,
  so the 0.6× part passes easily.
- The views settle slowly with materials: 4,266 frames against 621.
  Reflection rays request pages for every surface seen in reflections.
  Before the run, pages read only through reflections were evicted and
  requested again (28,116 frames); now they are marked as used.

## Disclosures

- An earlier command meant for the development check also started a full
  run by mistake (`./target/release/m10_gate2 | head -1`). Its output was
  discarded unseen, but it left the Cycles reference cached, which this run
  reused ("reused" in its output). This is the first run whose numbers were
  seen.
- Development: one check (`M10_DEV=1`) of the settled image and the
  settling time, before and after the eviction fix above.

## Likely sources of the remaining error (not yet tested)

- Specular shadows use the diffuse visibility, and highlights of the sun on
  metal are sharp.
- The reflection is a single ray, blurred only by the cache's mips, and is
  missing contact detail at roughness 0.3.
- Reflected surfaces carry no specular of their own; Cycles' one bounce
  includes it.
- Representative-point specular is approximate for the sun's square.

## Caveats

- One scene. Normal maps, emission, alpha, and glTF's material import are
  not done.
- Materials apply in the cached path only. Per-pixel rendering and pixel
  cones stay Lambertian.

## Prior art

GGX (Walter et al. 2007), height-correlated Smith (Heitz 2014), the
representative point and split sum (Karis 2013, 2014). No novelty claim is
made.
