# M5 gate 12: environment light (gate 11 re-run after a fix)

**PASS.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 5% of Cycles, and at most 0.6× the uniform sky's error | **1.37%**; uniform sky 19.92% (**0.07×**); means: ours +0.26%, uniform +1.91% | PASS |
| 2. Warm frame at most 1.05× the uniform sky's | 2.83 ms against 2.73 ms (**1.04×**) | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (commit `103311f`; binary
`experiments/room3d/src/bin/m5_gate12.rs`, committed in `ebbb716`). The fix
came before the binary; only the GPU tests had run on it. Cycles: the
environment image as a world texture, direct light only, 4,096 samples per
pixel, 57 s on Metal. Settled in 501 frames each.

Scene and criteria are gate 11's (`m5-gate11.md`): M6's textured scene
with the sun and a gradient sky, warm at the horizon and blue at the
zenith, black below the horizon.

## What changed since gate 11

- **Rays that miss read the environment's image itself.** Camera and
  reflection rays sample an extra layer of the base color texture array:
  the image in sRGB over its peak radiance, the peak in the light uniform.
  It needs no new binding. The dynamic range of what is seen directly is
  8-bit sRGB over the peak, enough for skies without a sun disk; the sun is
  a light.
- **The irradiance is evaluated unrolled**, not in a loop over the uniform
  array.

Surfaces keep gate 11's method: the cached sky visibility for a uniform
sky, scaled by the environment's harmonic irradiance at the pixel's normal
over the uniform open sky's.
