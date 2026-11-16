# M15 gate 1: anti-aliasing on the default path, in motion

**PASS** on all four criteria in both views.

| View | 1. Added to one ray's frame (at most 3.3 ms) | 2. Edges: error / one ray's (at most 0.5×) | 3. Elsewhere (at most 1×) | 4. Motion: change off the reference's / one ray's (at most 0.5×) |
|---|---|---|---|---|
| Textured scene with foliage | 1.26 ms (2.80 → 4.05 ms, 1.449×) PASS | 7.42% / 23.75% (**0.312×**) PASS | 3.129% / 3.307% (0.946×) PASS | 6.97% / 23.16% (**0.301×**) PASS |
| Field standing (a million boulders) | 2.77 ms (2.58 → 5.35 ms, 2.071×) PASS | 20.54% / 54.28% (**0.378×**) PASS | 3.548% / 4.925% (0.720×) PASS | 23.76% / 56.84% (**0.418×**) PASS |

Reported, not gated:

| View | M7 quarters: frame, edges, motion | Whole image: one ray / anti-aliased / M7 | Change over the reference's: one ray / anti-aliased / M7 |
|---|---|---|---|
| Textured scene | 1.499×, 0.392×, 0.442× | 4.682% / 3.269% / 3.482% | 2.16 / 1.15 / 1.33 |
| Field standing | 1.719×, 0.397×, 0.399× | 12.091% / 5.448% / 6.077% | 2.95 / 1.55 / 1.52 |

| View | Edge pixels listed | Stage 0 samples a pixel | Samples traced (a pixel of the image) | Shaded | Done after stage 0 |
|---|---|---|---|---|---|
| Textured scene | 32,328 (3.51%) | 4 | 193,864 (0.210) | 40,350 (20.8%) | 16,190 |
| Field standing | 71,828 (7.79%) | 4 | 447,388 (0.485) | 130,678 (29.2%) | 31,809 |

Both views took stage 1 (the inner four samples) for the pixels whose
first four disagreed: within the budget of a ray a pixel.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `d8da985`; binary
`experiments/room3d/src/bin/m15_gate1.rs`, stub committed in `9a96c5e`;
output `docs/results/m15-gate1-run.txt`). Both views were timed before any
reference was made, after three minutes idle; the one-ray frames (2.80
and 2.58 ms) match development (2.78–2.84 and 2.53–2.57 ms), the
control.

## Method

`CacheOptions::anti_aliasing` (DESIGN §3.4, "Anti-aliasing on the default
path"), with hardware camera rays:

- `hw_aa_list` lists edge pixels: a neighbour (left, right, up, down) sees
  another object, light or the sky, or a point more than half a pixel
  footprint off the pixel's own triangle's plane. A workgroup's pixels
  are listed together.
- `hw_aa` (stage 0) traces the outer four samples of a fixed pattern of
  eight (two squares turned against the pixel grid, each sample on its own
  row and column of 16), one invocation per sample. A sample on the
  pixel's own surface takes the pixel's colour; any other becomes a record
  to shade. Pixels whose four samples did not all see their own surface
  trace the inner four (`hw_aa_again`, stage 1) if the frame's total fits
  the ray budget (a ray a pixel; a choice taken up within 90% of it and
  given up past it).
- Samples that hit a back face are traced again in a pass of their own
  (`hw_aa_retrace`); `vc_aa_shade` shades the records from the cache;
  `hw_aa_scatter` writes each pixel's average of its own ray and its
  samples.

The reference is the cached image at 16×16 sub-pixel offsets, each settled,
textures read for 1/16 of a pixel. Geometric edge pixels are those whose 256
reference samples see more than one object or light, or one object at
distances more than 5% apart. Errors are relative RMSE over the image's mean
over the pixels named. Motion: the camera yaws a quarter pixel a frame for
six frames, each frame settled with its own reference; over the pixels that
are edges in any frame, the RMS of each frame-to-frame change's difference
from the reference's.

## Disclosures

- The ray budget's default changed from half a ray a pixel to one before
  the run, with the hysteresis, after development mode showed the field at
  half a ray falling to two samples a pixel at 640×360 (0.54× one ray's
  error at edges, failing criterion 2 there). Both changes are in the
  commit before the run (`d8da985`); development mode then passed.
- The field's frame cost (2.071×, 2.77 ms added) is near the bar; most of
  it is the horizon band, where boulders are smaller than a pixel and every
  pixel is an edge. That band is the sub-pixel path's territory (M13).
- On the field M7's quarter rays, which shade every sample, are slightly
  better in motion (0.399× against 0.418×) and cost less (1.719×); the new
  path is better at edges and elsewhere, and on the textured scene on
  every measure. Each costs a different thing: the new path traces up to
  eight samples and shades only those off the pixel's own surface.
- The error left elsewhere on the textured scene is mostly texture detail
  inside surfaces (grout and mortar lines; M7 G2's filtering), not edges.
- Development numbers in DESIGN §3.4 were measured with the field at
  100,000 boulders; the gate used a million.
