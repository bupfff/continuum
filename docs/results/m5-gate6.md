# M5 gate 6: several colored lights

Date: 2026-10-25. Until now every renderer took one white rectangular light.
This gate adds a list of colored lights to per-pixel rendering and to the view
cache, and measures them on the million-boulder field.

## What was built

- **A list of lights** (`Light`: a rectangle and RGB radiance, at most
  `MAX_LIGHTS` = 16) in per-pixel rendering
  (`render_instanced_lights`, `InstancedRenderer::new_lights`) and in the
  view cache (`with_cache_lights`).
  - Camera rays record which light they see.
  - A cache texel holds one channel per light (irradiance per unit radiance)
    plus the sky's, and each texel's lights are separate bake jobs, light by
    light.
  - **Re-coloring is free.** Lighting is linear in emitted power, so
    `set_light_radiance` re-weights stored channels and bakes nothing.
  - **One white light reproduces the previous images bit for bit**, cached
    and per pixel (`sun_dump`, compared with the previous build).
  - **Warm frames got faster:** 1.65 ms against 2.24 ms at gate 11's
    standing view, as shading now reads channels only once it has found a
    page.
- **Tighter light cones.** Development flights with the lamp below took
  579 ms per frame on average; with these three changes, 14 ms.
  - **Bounding-cone culling.** A light's cone has its frame on the light's
    plane, so a lamp seen at a grazing angle from 100 m gets a window far off
    its forward axis. Testing a box against the window's four planes one at a
    time then passes boxes metres off the line to the lamp, and each cone
    walked about 1,000 boulders to their triangles. Cones are now also culled
    against a circular cone around them: the axis points to the light's
    center and the half-angle reaches its farthest corner, tested against
    each box's bounding sphere. This only removes geometry that cannot touch
    the cone, so images are unchanged (sun-only images bit-identical).
  - **Nearest first.** Light cones walk the object hierarchy nearest first
    along that axis. Their meshes combine by the maximum, so the order does
    not change the result.
  - **Whole bins.** Coverage within 1e-5 of a whole bin counts as whole: a
    bin covered by several triangles sums to one only up to rounding, and a
    cone stops once every bin is whole. Sun-only images move by 4e-8
    (relative).
- **A fixed pool size.**
  - **Why.** With several channels per texel, the pool first kept its memory
    by holding fewer pages. Flights then ran out of pages and evicted while
    waiting for the GPU (15–37 ms of CPU planning).
  - **Now.** It keeps 2^19 pages, which is 170 MB per channel.
- **The test scene** (`Field::colored_lights`), placed around the camera:
  - the sun, warm, 1 km away in its usual direction;
  - a cool fill opposite it, 4° across, 20° up, giving a quarter of the
    sun's irradiance;
  - an orange lamp, 1.5 m square, 6 m up, facing down, 25 m ahead of the
    camera.
- **Cycles references take a light list**, and a Blender run that exits
  without a result is retried once. That happened once, with no error, on a
  job that rendered the next time.

## Pass criterion

Fixed before the first run, in 0c6900e. Control first: gate 11's sun-only
warm frame at most 8.03 ms.

1. Accuracy: the three lights per pixel, 16×16 samples per pixel (box
   filtered) at 320×180, within 3% relative RMSE of instanced Cycles at
   4,096 samples per pixel, over surfaces. Pixels where a sample sees a
   light, and their eight neighbours, are scored apart.
2. Re-coloring is exact and free: a warm cache with the three lights and the
   sky, given new RGB radiance for all three, bakes no page in the next 4
   frames, and its image is bit-identical to a fresh warm cache with the new
   colors.
3. Speed with the three lights: warm, each of gate 8's viewpoints at most
   16.7 ms; three of gate 8's flights, each with mean at most 16.7 ms, 99th
   percentile at most 33.3 ms and fallback at most 10%.

Flights with the three lights and the sky were reported, not gated:
development runs had measured a 99th percentile of about 34 ms there.

**Verdict: PASS.**

## Results

Apple M4 (MacBook Air), plugged in, 999,986 boulders, 1280×720, hardware
rays. Reproduce with `cargo run --release -p room3d --bin m5_gate6`.

- **Control:** 1.65 ms (gate 11: 8.03 ms).
- **1. Accuracy against Cycles:** **1.22%** for the three lights, with 483
  pixels set apart. With the sky added, 1.03%.

  | Light alone | vs Cycles |
  |---|---|
  | sun | 0.49% |
  | cool fill | 4.06% |
  | lamp | 4.69% |

  - **Why the single local lights score worse.** Their images are dark
    apart from small lit regions, so relative RMSE magnifies the error at
    the edges of small, brightly lit boulders.
  - **The error falls with more samples per pixel.** In development, the
    lamp alone was 30.4% at 4×4 samples per pixel, 11.5% at 8×8 and 4.7% at
    16×16, against Cycles at 4,096 samples.
  - **It is not a lighting error.** The lamp over bare ground matched Cycles
    to 0.78%, and the means agree to 0.1–0.7%.
- **2. Re-coloring:** the warm cache with three lights and sky (warm after
  4,687 frames) baked **0 pages** in the 4 frames after re-coloring, and its
  image was **bit-identical** to a fresh cache with the new colors.

| Viewpoint | 3. Warm, three lights | Frames until warm: three lights / sun alone | Cached vs per-pixel exact: three lights / sun alone |
|---|---|---|---|
| standing | 1.70 ms | 677 / 102 | 1.06% / 14.18% |
| 10 m up | 2.29 ms | 2,617 / 132 | 18.18% / 25.58% |
| 40 m up | 2.47 ms | 3,050 / 136 | 27.53% / 36.80% |
| 150 m up | 2.31 ms | 1,607 / 85 | 37.01% / 53.12% |

The cached-vs-exact figures are relative to each image's mean. At the
standing viewpoint the lamp, which is smooth, dominates the image's
brightness, so the three-light figure is not comparable with the sun's.

Flights with the three lights (criterion 3):

| Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |
|---|---|---|---|---|---|---|---|
| 1 | 14.05 ms | 30.97 ms | 49.99 ms | 1.79% | 2,930 | 9.52 ms | 2.47 ms |
| 2 | 14.18 ms | 29.13 ms | 50.23 ms | 1.78% | 3,048 | 9.16 ms | 2.35 ms |
| 3 | 14.85 ms | 30.75 ms | 49.28 ms | 1.73% | 3,430 | 9.10 ms | 2.25 ms |

Flights with the three lights and the sky (reported):

| Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |
|---|---|---|---|---|---|---|---|
| 1 | 14.68 ms | 35.10 ms | 43.16 ms | 7.45% | 7,022 | 10.85 ms | 3.40 ms |
| 2 | 14.70 ms | 35.23 ms | 44.55 ms | 7.59% | 6,958 | 10.95 ms | 3.47 ms |
| 3 | 14.68 ms | 34.62 ms | 44.13 ms | 7.50% | 6,981 | 10.94 ms | 3.45 ms |

The slow frames wait for the GPU behind bakes the cost model underestimated.
The worst are single pages on the ground, seen from above, whose 81 sky
samples alone take about 30 ms: gate 5's open problem.

## A correction to gates 3–5

`Field::with_sun_at(d)` places the sun `d` from the field's center, but the
camera stands at the field's corner, 1.4 km from it. With the sun at 1 km,
the camera's surroundings lie behind the one-sided sun panel. In the
references gates 3 and 4 rendered, the sun lights only a thin line at the
horizon (`out/m5_gate3_both_cycles.png`), in Cycles and in ours alike.

- **Gate 3's "sun and sky, 2.39% from Cycles"** therefore compared sky light
  plus the horizon's sun, not sun shadows near the camera.
- **Not affected:** the sky-only comparisons (gates 3, 4 and 5), and gates
  8–11, which use the sun 3 km away, whose emitting side faces the camera.
- **What tests near-field sun shadows now.** This gate places its lights
  around the camera (`colored_lights`), and the sun alone is 0.49% from
  Cycles at 16×16 samples per pixel.

## What this means

- **Several colored lights work per pixel and in the cache**, within 1.2% of
  Cycles, at the same frame rate as one sun (flights average 14–15 ms).
- **Re-coloring a light costs nothing and is exact.** The cache stores light
  per unit radiance, so any change of color or intensity re-shades in the
  same frame, bit-identical to baking from scratch. A temporal accumulator
  in UE5's Lumen has to converge again after such a change; Continuum has
  nothing to converge. (No side-by-side measurement has been made.)
- **Local and grazing lights cost more than the sun.** The three lights warm
  a view in 5–22 times the sun's frames. Cones that graze across a field
  were the worst case until bounding-cone culling and nearest-first walks.
- **Sky remains the weak spot:** with three lights and sky, flights keep
  their mean but have a 35 ms 99th percentile, and views warm slowly. The
  next step shares the sky's far field across a page.
