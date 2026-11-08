# M5 gate 11: environment light

**FAIL on both criteria.** The surfaces' environment irradiance is right.
The sky seen directly and in reflections is not: nine spherical harmonics
cannot follow a sky that drops to black at the horizon.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 5% of Cycles, and at most 0.6× the uniform sky's error | **22.64%**; uniform sky 19.92% (**1.14×**); means: ours −13.23%, uniform +1.91% | FAIL |
| 2. Warm frame at most 1.05× the uniform sky's | 3.13 ms against 2.70 ms (**1.16×**) | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (commit `7a669ba`; binary
`experiments/room3d/src/bin/m5_gate11.rs`, committed in `966a3e8`). The
implementation came before the binary; only the GPU tests had run on it.
Cycles: the same environment image as a world texture, direct light only,
4,096 samples per pixel, 62 s on Metal.

## Method

`MaskConeKernel::set_environment`:

- **The environment.** An equirectangular radiance image, as Blender reads
  it. Its radiance above the horizon is kept as nine real spherical
  harmonics.
- **Surfaces.** The view cache keeps each texel's sky visibility for a
  uniform sky. Shading multiplies it by the environment's irradiance at the
  pixel's normal over the uniform open sky's there (π(1 + n_z)/2), as with
  lights' ratios.
- **Misses.** Camera and reflection rays that miss see the environment
  through the harmonics.

Scene: M6's textured scene with the sun and a gradient sky, from (1.3,
1.2, 1.05) times the scene's sky at the horizon to (0.55, 0.72, 1.25) times
it at the zenith, black below the horizon.

## Why

The harmonics against the exact sky (CPU check of the same projection):

| Quantity | Harmonics | Exact |
|---|---|---|
| Radiance near the horizon (sine of elevation 0.02) | (0.59, 0.57, 0.56) | (1.29, 1.19, 1.05) |
| Radiance high up (sine of elevation 0.9) | (0.84, 0.97, 1.42) | (0.63, 0.77, 1.23) |
| Irradiance on an upward normal | (2.53, 2.77, 3.72) | (2.51, 2.77, 3.72) |

- **Irradiance is right; radiance is not.** Nine terms cannot follow the
  step at the horizon.
- **The sky seen directly is the error.** It fills about 40% of the frame,
  and the background came out too dark and too blue. Hence a mean 13% low.

**Speed:** 9-term evaluations with dynamic indexing of a uniform array, a
few times per pixel, likely account for the 1.16×.

## Fix, for M5 gate 12

- Camera and reflection misses read the environment image itself, as an
  extra layer of the base color texture array (normalized by its peak,
  sRGB). That needs no new binding, at the cost of limited dynamic range
  for what is seen directly.
- The irradiance is evaluated unrolled.
