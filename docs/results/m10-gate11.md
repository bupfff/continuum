# M10 gate 11: emissive surfaces light their surroundings

**FAIL on accuracy, from how it scored the image.** Speed passed. The
error was decided by a few pixels where the night lights are seen directly,
not by the method: see M10 gate 12, the same run scored over surfaces.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 5% of Cycles, and at most 0.5× the error without emission lighting | **109.67%**; without 109.81% (1.00×); means +0.19% and −1.62% from Cycles' | FAIL |
| 2. Warm frame at most 1.1× without emission lighting | 2.89 ms against 2.89 ms (**1.00×**) | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `19b5e96`; binary
`experiments/room3d/src/bin/m10_gate11.rs`, criteria in `2abfad5`). Cycles:
one bounce, 4,096 samples per pixel, 90 s on Metal. Settled in 609 frames
with emission lighting, 604 without.

## Method

`CacheOptions::emission_bounce`, on by default, with bounce light:

- Bounce samples (192 rays on a hemicube, 3×3 samples a page) also gather
  the emission of the front faces they hit. Each hit adds its radiance
  times its bin's projected solid angle. No page is needed at the hit.
- That goes into a fifth channel of absolute radiance, so lights can still
  be re-colored for free (the pool's bounce storage grows by a quarter).
- The bounce pass reads emission from the per-object texture buffer, after
  each object's texture layer. Binding the materials buffer would pass
  Metal's 29-buffer limit.

Scene (`Textured::with_glow`): M6's textured scene at night (M5 gate 8's
point and spot lights, the sky at 2% of the day's), with a glowing panel
5 m ahead (1.2 × 0.9 m, radiance (3.0, 1.86, 0.9)). The Cycles reference
now exports emission (an Emission shader added to the surface's).

## Why the error is 110%

- The images agree closely by eye (`out/m10_gate11_ours.png`,
  `out/m10_gate11_cycles.png`), and their means are within 0.2%.
- The point light's disk (radiance 20) is seen directly, in an image whose
  mean is about 0.1. Differences at its few edge pixels decide the whole
  RMSE.
- Emission lighting changes the error by only 0.14 points, because those
  pixels dwarf everything else.
- M5 gates 6 and 8 scored such scenes "over surfaces" for this reason; this
  gate's criterion did not.

M10 gate 12 re-scores the same scene over surfaces, as M5 gate 8 defines
it, with gate 11's Cycles reference.
