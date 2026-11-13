# M5 gate 13: the sky through hardware rays

**PASS.** Where camera rays use hardware ray queries, a sky sample now
traces 192 rays instead of six cube faces of mask cones. With the
library's default options, views of the million-boulder field warm with a
sky in 166–400 frames instead of 1,827–15,905, closer to Cycles at three of
four viewpoints, and flights are 5% faster.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 1. Flight mean (three each, interleaved) | rays 9.86, 9.46, 9.68 ms (mean 9.67); cones 10.26, 10.15, 10.20 (10.21): 0.947× | at most 1.05× the cones' and 16.7 ms | PASS |
| 2. Frames until warm, standing | rays **166**; cones 1,827 | at most 600 | PASS |
| 2. 10 m up | rays **360**; cones 11,278 | at most 600 | PASS |
| 2. 40 m up | rays **400**; cones 15,905 | at most 600 | PASS |
| 2. 150 m up | rays **268**; cones 7,342 | at most 600 | PASS |
| 3. Against Cycles, sun and sky, standing | rays **2.53%**; cones 4.67% (0.542×) | at most 1.05× and 3% | PASS |
| 3. 10 m up | rays 10.04%; cones 11.99% (0.837×) | at most 1.05× | PASS |
| 3. 40 m up | rays 14.59%; cones 18.08% (0.807×) | at most 1.05× | PASS |
| 3. 150 m up | rays 20.43%; cones 20.01% (1.021×) | at most 1.05× | PASS |

Reported, sky only against Cycles: rays 1.28, 6.92, 14.31, 21.75%; cones
4.17, 10.26, 18.67, 19.01%. Flights' 99th percentiles: rays 12.74–12.96 ms,
cones 13.87–14.33 ms. Fallback in flight (pixels with no page or a guessed
sky): rays 50.0–53.3%, cones 76.8–79.7%.

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-13 (stub `36dfb9f`; implementation and binary `7ac186a`, before the
run), after 90 s idle. Flights and warm-ups at 1280×720 on the field with
its own sun; accuracy at 640×360 with the sun at 1 km, the cache settled at
4×4 sub-pixel offsets against instanced Cycles at 1,024 samples per pixel
(no bounces).

## Method

`hw_vc_bake_sky` (`CacheOptions::sky_rays`, on by default where camera rays
use hardware queries): each of a sky sample's six jobs traces 32 of the 192
bins of a hemicube around the texel's normal, one ray through each bin's
center (opaque, the first hit ending it). A bin whose ray meets nothing and
points above the horizon adds its projected solid angle: the uniform sky's
light per unit radiance, the same quantity the cone faces produce. A sample
costs about 0.72 µs of GPU time against 10.3 µs through cones at 10 m up
(development, `sky_dev`), and counts as one sun texel against the budget
(`sky_ray_cost`; cones count 10, though they cost about 17).

Movers, streamed levels and alpha-blended objects keep cones: movers live in
a second structure, camera levels differ from the lighting levels a texel
lies on, and cones weigh a blended object by its opacity.

**Disclosed (before the run):** the development measurements on the full
field and the development mode's results are listed in the binary's doc
comment; the criteria were written after them. A readback every 20th frame
held the frame-period cap near its floor (up to 1,895 frames to warm in
development), so the gate's warm-ups ran undisturbed and the frames until
within 5% of the warm image were not measured.

## Caveats

- **Binary visibility per bin.** One ray per bin sees a bin as open or
  closed; cones count partial coverage. Rays are nonetheless closer to
  Cycles here (cones' binned coverage is orientation-dependent and too dark,
  M5 gate 5). Thin far geometry could alias where cones would not; a
  sub-grid of rays per bin is the lever.
- **From above, both are 10–22% from Cycles**: the cache's resolution at
  distance, which the sun channel shows too, not the sky's estimate. At 150 m
  rays are 2% (relative) further than cones.
- **Settled images depend a little on the bake order:** the standing sky-only
  error was 1.66% in development with a fixed budget and 1.28% here with
  the default one.
- **Fallback in flight stays high** (half the pixels show a guessed sky or
  no page in these fast flights, against four in five with cones): the
  error while flying is Landmark 2's next measure.
- **The frame-period cap** collapses after outlier frames (a readback every
  20 frames held it at its floor): to fix on its own.

## What the numbers say

- **Landmark 2's sky warm-up is met on this scene:** a few hundred frames
  from empty at every viewpoint with default options (4–6 s at 60 fps),
  where cones took 2–4 minutes.
- **Hardware rays now carry camera rays, bounce rays and sky rays**; cones
  keep the sun and lights (prefiltered, certified) and the hard cases
  (movers' sky, streamed levels, blended objects).
