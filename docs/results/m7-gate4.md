# M7 gate 4: silhouette rays through hardware ray queries

**FAIL on criterion 2 (speed) for the field from 150 m up.** Quality
equals the compute path's in both views. The textured scene passes both
criteria. On the field, ray queries take silhouette rays from 24× the
frame to 3×: 11.08 ms, under a 60 Hz frame, against 88.33 ms.

| View | 1. From the area integral: hardware / compute path | 2. Warm frame: hardware edges / one ray | Compute path | Edge pixels |
|---|---|---|---|---|
| Textured scene | 3.52% / 3.52% (**1.000×**, at most 1.05×) PASS | 3.31 ms / 2.58 ms (**1.29×**, at most 1.5×) PASS | 4.74 ms (1.84×) | 1.0% |
| Field, 150 m up | 10.50% / 10.51% (**0.999×**) PASS | 11.08 ms / 3.66 ms (**3.03×**) FAIL | 88.33 ms (24.15×) | 51.1% |

Reported: one ray is 3.88% and 21.33% from the area integral (silhouette
rays: 0.91× and 0.49×, as gate 3). The hardware image is 0.063% and 0.776%
from the compute path's. Hardware and software camera rays alone differ by
0.84% on the field (development), so that difference is the camera's, not
the edge rays'.

Apple M4 (fanless MacBook Air), 1280×720, cache settled, release build, run
once at full size on 2026-11-07 (commit `0d05247`; binary
`experiments/room3d/src/bin/m7_gate4.rs`, first committed in `35dc774`,
criterion 1 amended before any full-size run in `9cb81e5`, below).

## Method

`CacheOptions::edge_rays` with hardware camera rays (not with movers):

- **`hw_edges`**, after the camera's hardware visibility, finds edge pixels
  with gate 3's test. Each traces its four quarter-rays with ray queries
  (front faces only, back-face hits traced again as the camera's are) and
  is listed with its four hits. The list is counted in the stats' edge
  counter.
- **`vc_shade_edges`** shades only the listed pixels: each hit from the
  cache, with textures for half a pixel, averaged, as gate 3's compute path
  does. The main shading pass skips the edge test and the four-sample loop.
- **Without hardware rays** (or with movers), gate 3's compute path is
  unchanged.
- **Memory:** the list holds 64 bytes per pixel (59 MB at 720p), and only
  exists with `edge_rays` on.

## Disclosures

Criterion 1 changed after the binary's first commit, before any full-size
run. It was "within 0.1% of the compute path's image". In development
(`edges_dev`, 640×360, the field with 100,000 boulders), hardware and
software camera rays differed by 0.84% with one ray per pixel and no edge
rays, so that bound measured the camera rays.

Development numbers:

| Development run | Result |
|---|---|
| Textured scene | 1.21× one ray's frame (compute path 2.20×), 0.058% from the compute path |
| Field | 2.55× (compute path 16.0×) |
| Field, a quarter on the pixel's own object reusing the pixel's color | 2.49×, image 20% off; not kept |

## Why the field is still slow

From 150 m up, half the pixels are edges, and each shades four quarters
from the cache: about three times the main shading pass's work. Reusing
the pixel's color barely helped, because most quarters there hit another
object. Fewer quarters (two diagonal rays), or a cap on edge pixels per
frame, would trade quality for time; either is for a new gate.
