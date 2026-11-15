# M13 gate 14: lights seen directly on the sub-pixel path

**PASS** on both criteria. One caveat: in the low view the low sun was
hidden behind a near boulder (a large object, about 10 m away), so no view
put small instances in front of a light. That case goes to gate 15.

| View | Near the lights: ours / one ray | Drawn / off | Drawn | Off |
|---|---|---|---|---|
| Low (0.4 m, 3° up) | **0.145× (5.57% / 38.46%, 1,410 px): PASS** | **1.001×: PASS** (0.998, 1.003, 1.001, 1.002, 1.001) | 9.85 ms | 9.84 ms |
| Standing (1.7 m, 3° up) | **0.205× (14.57% / 71.08%, 1,261 px): PASS** | **0.999×: PASS** (0.995, 1.000, 1.004, 0.970, 0.999) | 16.97 ms | 17.05 ms |

Bars: near the lights at most 0.5× one ray's error; drawn at most 1.03×
the frame with drawing off. Times are medians over five interleaved rounds
(20 frames to warm up, then the median of 21).

Reported:
- **Whole image:** 0.145× (low) and 0.205× (standing) one ray's error. The
  lights dominate it.
- **The rest (away from the lights):** 0.525× (2.00% / 3.80%) low and
  0.402× (4.16% / 10.34%) standing. These views were not chosen for this;
  the low eye puts the sub-pixel horizon across much of the image.
- **Mean colour near the lights:**
  - low: reference (59.561, 37.226, 19.867), ours (59.785, 37.347, 19.916);
  - standing: reference (617.610, 372.215, 215.673), ours (609.299,
    367.241, 212.810), 1.3% dark.
- **Gate 13's standing view (no light in view):** drawn 1.000× off, 17.05
  ms (gate 13's own run: 17.07 ms).
- **Settling:** the sub-pixel path settled in 514, 804 and 838 frames; the
  one-ray path in 204 and 302.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-14 (a million boulders). The run's lines are in
`docs/results/m13-gate14-run.txt`.

## Method

Lights seen directly were not drawn on the sub-pixel path. The one-ray
path draws a light where the pixel's centre sees it nearer than its hit
(`seen_light`).
- **Which lights a tile can meet:** the roots pass (one invocation per 8×8
  tile) tests each light's bounding sphere against the tile's cone. It
  stores a mask in the tile's word 379, just past its roots. The test is
  `cos a ≥ cos(asin(r/d) + h)` without inverse functions, with h the
  tile's half-diagonal plus room. State word 23 = 1 turns drawing off (the
  gate's control).
- **Bins that see a light:** in the pixels' pass, once the pixel's layers
  are known, each candidate light is tested against the pixel's cone, then
  at the 16 bins' centres with `rect_distance` (the one-ray path's test). A
  bin sees the light where it is nearer than what the bin shows behind
  small entries:
  - the large object's distance for the bins it claims;
  - else the background's 1/t across the pixel (a plane);
  - else the sky.
- **Small entries behind the light:** map records, walked nodes and pages
  farther than the nearest light over the pixel are dropped from its light
  bins (`g9_in_front`). Those in front cover the light by their coverage.
- **The resolve:** light bins take the nearest light's radiance (with a
  spot's falloff) behind their transmittance, before any other colour. The
  resolve's fast paths are skipped in those pixels.

The GPU test `sub_pixel_detail_settles_near_one_ray` now checks the drawn
lights too. At 160×90 their total is 1.05× one ray's. The surfaces'
difference is unchanged at 0.1211: light pixels and their neighbours are
left out of it, since our path draws partly covered pixels at a light's
edge.

## Scene

The million-boulder field, its ground and sky, and four lights:
- the field's sun (30° up behind the camera; the light small instances
  take);
- a low sun ahead: 0.5° across, its centre 0.3° above the horizon, 5° right
  of the camera's line, 5 km away;
- M5 gate 6's lamp: 1.5 m square, 6 m up, facing down;
- M5 gate 8's point light: 10 cm across, 3 m up.

**The reference:** per pixel the mean of 16×16 samples. Each sample is one
ray shaded by the renderer's own path, with its cache settled and lights
drawn as that path draws them. **One ray:** the centre sample. **Near the
lights:** pixels within 2 px of a light's projected bounding box.

## Disclosed

- **The low view did not test what its doc comment says.** The low sun
  (projected at about x 707, y 396) sat behind a boulder about 10 m from
  the eye, so in that view criterion 1 measured:
  - the lamp and the point light against the sky;
  - the sun hidden wholly by a large object.

  The development views had the same flaw. The "low turned 3° left" view
  moves that boulder with the sun. Standing, boulders stay below the eye,
  so no view put small instances in front of a light. Gate 15 picks a view
  where the low sun is verified to stand behind small instances before
  its stub is committed.
- **The near-light numbers equal the development run's on 10,000
  boulders** (5.57%/38.46% low, 14.57%/71.08% standing). The near-light
  pixels are the same there: the sun hidden by the same near boulder (low),
  or against the sky above the far ground (standing). Pixels 2 px below the
  standing sun do differ slightly (mean 617.610 against 617.607).
- **Developed at full size** on other views (low turned 3° left, standing
  6° up): near the lights 0.139× and 0.216×, drawn/off 1.002×.
- **The sun's edges:**
  - 4×4 bins measure an axis-aligned edge to a quarter of a pixel, so a
    light 7 px across can lose or gain up to about 3.6% of its total. In
    development the low sun read 1.4% dark and the lamp 1.6% dark.
  - Centroids match the reference within 0.03 px.
  - A sun cut by the ground's horizon (development only, `M13_SUNSET=1`)
    read 0.23× one ray's error in its box.
- **The machine:** a macOS service (`ANECompilerService`) used about one
  CPU core throughout. The cost criterion compares the two settings
  interleaved in one process.
- **No change to other paths:** the one-ray path's light test is
  unchanged. The model path (gate 11's) draws lights the same way.
