# M5 gate 7: lights cached as ratios, shaded with each pixel's normal

Date: 2026-10-25. The view cache stored irradiance per texel, so the cosine
at each texel's normal was interpolated together with the shadow, and normal
detail finer than a texel was lost. This gate stores each light's shadowed
share instead and applies the cosine per pixel.

## What was built

- **Light channels hold ratios** (`CacheOptions::light_ratio`, on by
  default).
  - **What is stored.** A texel holds its light's shadowed share of the
    unshadowed light at the texel.
  - **Shading.** The interpolated share is multiplied by the unshadowed
    light at the pixel, from the pixel's own position and normal.
  - **Prior art.** This is a ratio estimator. Heitz, Hill and McGuire
    (2018) combine analytic unshadowed light with a stochastic shadow ratio
    in the same way.
  - **Below the horizon.** Where the light is below a texel's horizon, its
    share is one; the texel's own light is zero either way. Development runs
    with zero there were 9.56% and 17.31% at the first two viewpoints
    (below), against 6.21% and 10.63% with one.
- **Light windows that cross the horizon are clipped.**
  - **The bug.** `cone_visible` summed every bin's unclipped weight, which
    counted bins below the horizon as light.
  - **The fix.** The window and each covered bin are now clipped, as for
    sky faces.
  - **Effect.** Against gate 6's Cycles references, the three lights are at
    1.22% before and after, and the lamp alone moves from 4.69% to 4.72%.
- **Cycles references are reused** when their job and every input file are
  byte-identical (here they were rendered again: the job gained a texture
  list in M6).
- **The sky is unchanged.** Treating it the same way, with an occlusion
  vector or a ratio to a reference, did not help
  (`m5-sky-bake-diagnostics.md`).

## Pass criterion

Fixed before the first run, in 4b4ff86. Run at 1c42308, which adds sphere
lights. Rectangular lights' arithmetic is unchanged there: the spot falloff
is exactly one, and the horizon test is the same.

Setup:

- The million-boulder field at gate 8's four viewpoints, 1280×720,
  hardware rays.
- Prefilter 1 bin with object-level prefiltering; cache defaults.
- "Cache error" is the warm cached image against the per-pixel image with
  the same lights and prefiltering (1 sample per pixel).
- Both schemes are measured in the same run.

Control first: the sun-only warm frame at the standing viewpoint, with
ratios, at most 3.3 ms.

1. **Sun alone.** The cache error with ratios at most 0.6× that with
   irradiance texels, at every viewpoint.
2. **Three lights** (gate 6's). At most 0.75×, likewise.
3. **Per-pixel accuracy is kept.** The three lights per pixel (16×16 at
   320×180) within 1.30% of Cycles over surfaces.
4. **Speed with three lights and ratios.**
   - Warm: at most 16.7 ms at each viewpoint.
   - Three of gate 8's flights: each with mean at most 16.7 ms, 99th
     percentile at most 33.3 ms and fallback at most 10%.

Development runs before the gate, disclosed in the binary, measured the sun
at full size and the three lights at 640×360.

**Verdict: FAIL** (criterion 4; criteria 1–3 pass).

## Results

Apple M4 (MacBook Air), plugged in. The control was 1.94 ms, and the
DisplayLink agent used about 45% of a core, as in earlier gates. Reproduce
with `cargo run --release -p room3d --bin m5_gate7`.

| Viewpoint | Lights | Cache error, ratios | Cache error, irradiance | Ratio | Warm frame, ratios / irradiance | Frames until warm, ratios / irradiance |
|---|---|---|---|---|---|---|
| standing | sun | **6.21%** | 14.49% | ×0.43 | 2.19 / 1.73 ms | 102 / 102 |
| standing | three | **0.64%** | 1.06% | ×0.61 | 2.53 / 1.85 ms | 837 / 789 |
| 10 m up | sun | **10.63%** | 25.87% | ×0.41 | 6.65 / 4.68 ms | 215 / 135 |
| 10 m up | three | **10.28%** | 18.19% | ×0.57 | 7.16 / 3.74 ms | 6,201 / 3,541 |
| 40 m up | sun | **14.78%** | 30.88% | ×0.48 | 5.74 / 5.97 ms | 442 / 185 |
| 40 m up | three | **16.14%** | 27.53% | ×0.59 | 6.95 / 5.34 ms | 4,962 / 4,119 |
| 150 m up | sun | **21.75%** | 38.80% | ×0.56 | 6.22 / 4.98 ms | 100 / 113 |
| 150 m up | three | **25.15%** | 37.02% | ×0.68 | 5.76 / 2.94 ms | 2,125 / 1,618 |

The irradiance figures reproduce gate 5's sun-only cache errors exactly.
They also match gate 6's three-light errors to 0.01 point, though gate 6
compared with exact cones (prefilter 0).

3. **Per pixel against Cycles** (4,096 samples per pixel, rendered again,
   13.1 s): **1.22%** over surfaces. Alone: sun 0.49%, fill 4.06%, lamp
   4.72%.

Flights with the three lights and ratios, 600 frames from an empty cache:

| Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |
|---|---|---|---|---|---|---|---|
| 1 | **17.09 ms** | **36.44 ms** | 118.51 ms | 2.91% | 1,482 | 9.96 ms | 5.64 ms |
| 2 | **17.58 ms** | 30.62 ms | 74.69 ms | 3.21% | 1,497 | 10.35 ms | 6.18 ms |
| 3 | 16.21 ms | 30.65 ms | 46.56 ms | 2.35% | 1,932 | 9.07 ms | 4.27 ms |

Gate 6's flights, before ratios: means 14.05–14.85 ms, camera passes
2.25–2.47 ms on the GPU, bake budgets 2,930–3,430.

## Why criterion 4 failed

- **The camera passes cost twice as much: 4.3–6.2 ms on the GPU, against
  2.3–2.5 ms.**
  - Every pixel evaluates every light's unshadowed light, so ratios add
    per-pixel work for each light.
  - That evaluation builds the light's whole cone: its frame, window and
    bounding cone. Then it clips the window to the horizon.
- **The bake cannot give way.**
  - Its budget follows the frame target, but it fell to about 1,500 units,
    near its floor of 1,000.
  - At the floor, a bake still takes 9–10 ms. Camera passes plus bake no
    longer fit 14 ms, so frames ran at 16–18 ms.
- **The same squeeze slows warm-up.** With three lights, views took up to
  1.75× as many frames to warm (10 m up: 6,201 against 3,541).
- **Irradiance warm frames understate the old cost.** In this build the
  irradiance scheme also computes the pixel's shading normal and each
  texel's unshadowed light. Its warm frames include some of the new cost.
- **Next.** Evaluate unshadowed light per pixel from the light's corners
  directly, with Lambert's formula. That skips the cone's frame and bounding
  cone, and handles small distant lights cheaply. Then gate speed again.

## What this means

- **The cache now keeps the normal detail of every pixel for lights.** Its
  error against per-pixel cones fell by 32–59% at every viewpoint, with no
  extra memory and no extra bake work.
  - What remains is shadow edges at texel resolution, and the sky.
- **It costs shading time.**
  - Warm frames rose 0.4–3.4 ms; all stayed under 7.2 ms.
  - Flights with three lights went over budget because the bake budget sits
    at its floor while flying (gate 11's caveat).
  - The per-pixel evaluation has to get cheaper before this holds 60 Hz in
    flight with several lights.
