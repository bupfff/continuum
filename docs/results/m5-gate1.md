# M5 gate 1: sky light through mask cones, against Cycles

Date: 2026-10-24. The first step of roadmap item 2 (sky light and several
lights): a uniform sky lighting every receiver through mask cones.

## What was built

- **Sky cones.** A uniform sky (`MaskConeKernel::set_sky`): constant radiance
  over the whole sphere of directions, like Cycles' world color. Each receiver
  gets six mask cones, one per face of a cube of directions:
  - 90° pyramids with 8×8 bins, their far planes beyond the scene;
  - faces wholly below the receiver's horizon are skipped;
  - bins straddling the horizon have their projected solid angle clipped to
    it, using Lambert's edge integrals over the clipped polygon.
- **Combining occluders.** Across meshes, sky cones combine coverage as
  independent occluders, 1 − Π(1 − c). Sun cones keep the maximum.
- **Camera rays that miss** see the sky.
- **Scope.** Per-pixel rendering only. The lighting caches refuse a sky for
  now.
- **Test scene.** A new outdoor scene (`room3d::outdoor`): a 400 m ground, 60
  boulders, a wall, a pillar, the field's sun (0.5°, 30° up, 3 km away) and a
  blue-white sky.
- **Reference.** The Cycles reference script takes a sky (world color).

## Pass criterion

Fixed before the first run, in 8e214be. Per-pixel cones at 640×360 against
Cycles (4096 samples per pixel, direct light only, seed 1):

1. **Sky only:** exact cones within 3% relative RMSE.
2. **Sun and sky:** exact cones within 3%, and prefiltered cones within 3%.

**Verdict: FAIL.**

| Render | Relative RMSE vs Cycles | Time |
|---|---|---|
| Sky only, exact cones | **2.74%** (pass) | 41.4 s |
| Sky only, exact, max combination (diagnostic) | 2.72% | |
| Sun and sky, exact cones | **5.12%** (fail) | 39.5 s |
| Sun and sky, prefiltered cones (1 bin) | **5.12%** (fail) | 7.3 s |
| Sun and sky, exact, max combination (diagnostic) | 5.13% | |

- **Cycles' own noise** (RMSE between two seeds / √2): 0.10% sky only, 0.18%
  sun and sky.
- **Setup.** Apple M4, compute rays, best of 3. Cycles on Metal, 45.9 s per
  render.
- **Reproduce** with `cargo run --release -p room3d --bin m5_gate1`.

## Diagnosis: the sky is right; the reference was not

The component renders were saved (`sky_diagnostic`) and compared separately.

**1. Sky light matches Cycles in the interiors of surfaces.** Our sky light
divided by Cycles':

| Surface | Ours / Cycles |
|---|---|
| Wall | 0.9992 |
| Pillar | 1.0004 |
| Ground | 0.9951 |

Of the sky-only image's squared error, 96% sits in the top 1% of pixels:
silhouettes.

**2. Cycles integrates over each pixel's area.** The reference script asked
Cycles for a 0.01-pixel box filter, meaning point samples at pixel centers,
since M3. A test of a straight edge under an orthographic camera shows
otherwise:

- An edge covering 40% of a pixel renders as 0.398 with both
  `cycles.filter_width = 0.01` and `render.filter_size = 0.01`.
- Blender 5.2's Cycles always applies at least a one-pixel box.

So the references were area-integrated images: the right target for image
quality, compared against our one-sample-per-pixel renders. Every silhouette
pixel counted as error.

**3. Cycles loses 1.9% of a small sun 3 km away.** Our sun light is uniformly
2% brighter than Cycles' on every surface (wall 1.021, pillar 1.022, ground
1.019). A bare test checks Cycles against the analytic irradiance: a ground
plane under the same 0.5° quad, the distance varied, the angular size held.

| Distance | Cycles / analytic |
|---|---|
| 3,000 m | 0.9813 |
| 1,000 m | 1.0000 |
| 600 m | 1.0000 |
| 300 m | 1.0000 |
| 30 m | 1.0005 |

Our sun matches the analytic value to 0.3% in the scene, against Cycles' 2%.
This looks like float precision in Cycles' sampling of a tiny, distant mesh
light.

## What this means

- **The sky integration works:** six cube-face mask cones with
  horizon-clipped bin weights. Interiors match Cycles to 0.1–0.5%, and sky
  only passes at 2.74% even against the flawed reference.
- **The combination rule barely matters in this scene.** Max and independent
  occluders differ by 0.02 points.
- **Exact sky cones are expensive per pixel.** A 640×360 frame took 41 s
  (7.3 s prefiltered). A sky needs the lighting cache and, for large scenes,
  prefiltered object-level coverage. That is later work in M5.
- **Correction to earlier gates.** M3 gate 3's 3.80% against Cycles compared
  point samples with area-integrated pixels, so part of it was edge sampling,
  not lighting. The calibration's 0.15% was on a scene with few silhouettes.
  A note is added to `m3-gate3.md`.

## Next

M5 gate 2 repeats this comparison with the reference fixed:

- our renders supersampled 4×4 and box filtered, to match Cycles' one-pixel
  box;
- the sun 1 km away (same angular size and radiance), where Cycles matches
  the analytic irradiance.
