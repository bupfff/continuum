# M4 gate 12: pages requested three levels finer

**PASS.** A pixel showing a coarser page than it wants now asks for the page
up to three levels finer instead of one. While flying over the
million-boulder field the image is 0.82× as far from settled (20.8% against
25.4%, relative RMSE, at a fixed bake budget), views warm in 0.70–0.76× the
frames with identical warm images, and flights cost the same.

This gate also gives Landmark 2's first measure of **the error while
flying**: each check compares the flying image with a renderer settled at
the same camera.

| Criterion | Three levels | One level | Bar | Result |
|---|---|---|---|---|
| 1. Flight mean (three each, interleaved) | 9.44, 9.70, 9.53 ms (9.55) | 9.25, 9.41, 9.60 ms (9.42) | at most 1.05× (1.014×) | PASS |
| 2. Error while flying, fixed budget: mean (largest) | **20.77%** (26.12%) | 25.36% (27.36%) | at most 0.9× (0.819×) | PASS |
| 2. Reported, default options | 21.00% (25.99%) | 25.93% (28.91%) | | |
| 3. Frames until warm, standing | 223 | 296 | at most 1.0× | PASS |
| 3. 10 m up | 477 | 628 | | PASS |
| 3. 40 m up | 482 | 639 | | PASS |
| 3. 150 m up | 308 | 438 | | PASS |
| 3. Warm images apart | 0.0000% at all four | | at most 0.01% | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `c39715f`; implementation and binary `6014854`,
before the run), after three minutes idle. M9's flight (600 frames from
standing height to 150 m and pitching down 45°) over the field with sun and
sky (sky through rays); criterion 2 checks every 60th frame against a
renderer settled at that camera, at a fixed budget of 11,000 units a frame
(about the flights' average) so it does not depend on the machine's heat;
criterion 3 at M5 gate 13's viewpoints with the same budget.

## Method

`CacheOptions::refine_levels` (3 by default; it was 1): in the shading pass
a pixel walks from its wanted level to coarser ones until it finds a page,
and requests the page `refine_levels` finer than the one it found (or its
wanted one, if nearer). Pages keep arriving coarse to fine, in a third of
the rounds (each a bake and a readback two frames late), without baking the
levels in between: warm-up baked 25% fewer sky samples and sun texels in
development.

**Disclosed (before the run):** one to four levels were compared in
development at the fixed budget (25.37, 21.77, 20.77, 20.76% mean; four's
largest error rose to 28.85%), and the development mode's single flight pair
failed criterion 1 from a fast first flight; three pairs gave 1.007× (all
in the binary).

## The error while flying, what it is

- **About a fifth to a quarter of the image while flying** at this speed,
  14.6% even at a tenth of it (development, one level). The flying image
  shows the ground's boulder shadows smeared into blotches (its coarse
  levels' texels are metres wide) and boulders unshadowed (pages missing or
  with a guessed sky); the settled image shows fine shadows.
- **What drives it:** about 16,000–20,000 distinct pages are wanted a frame
  and 800–1,500 baked; under 3% of sampled pixels find their wanted level.
  The field's ~100,000 visible boulders each need pages of their own, so a
  view needs hundreds of thousands of pages: a scale problem that Landmark
  3's prefiltered appearance (far boulders shaded from the hierarchy, not
  per-boulder pages) addresses at the root.
- Guessed skies are not the cause: baking each page's sky with it (sky
  later off) raised the error slightly (26.75% against 24.22%) while
  fallback fell to 1.5%.

## Caveats

- One flight on one scene; the error depends on speed and on how many
  distinct objects a view holds.
- "Bounded" is not yet met: the error is measured, not certified.
