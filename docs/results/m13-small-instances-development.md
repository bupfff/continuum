# M13 development: small instances' lighting on the cache-lit path

Development, not a gate (2026-11-14, thirteenth session, after gate 14).
This is DESIGN §3.4 "Into the renderer", part C, whose design asked for
small instances' error against per-sample lighting to be measured first.
Binary: `m13_small_dev`, at full size (a million boulders, 1280×720).
- **Scene:** gate 13's: the field's sun, its sky, no bounce light
  (`CacheOptions::indirect` is off by default).
- **Views:** gate 13's two (standing at 1.7 m; 150 m up, 45° down) and
  its reported view (20 m up, 10° down).
- **The reference:** 16×16 samples a pixel through the renderer's own
  one-ray shading, cache settled.

## Where the error lies

State word 28 = 2 makes the pixels' pass write the share of the pixel
small entries cover. Pixels are split into four classes by that share:
none (under 5%), some (5–50%), most (50–95%), all (95% or more). Before
any change:

| View | Whole | most: share of our squared error, mean / reference | all: share, mean / reference |
|---|---|---|---|
| Standing | 0.437× | 25.4%, 0.989 | 14.4%, 1.122 (2,717 px) |
| 150 m up | 0.414× | 35.1%, 0.941 | 40.0%, 0.915 |
| 20 m up | 0.428× | 8.4%, 0.972 | 63.9%, 1.037 |

From above and at the horizon, most of the error is in pixels of small
instances and pages. Small instances were lit by:
- the first light, with one visibility an object;
- a constant half of the sky (`G12_SKY_OPEN`).

Coverage pages for far groups baked the old model's ambient (a fifth of the
light's irradiance), not the sky.

## Levers tried (whole-image error, ours / one ray's)

| Change | Standing | 150 m up | 20 m up |
|---|---|---|---|
| Before | 0.437× | 0.414× | 0.428× |
| Sky by the normal for small instances only | — | 0.388× | — |
| + pages and splats lit by the sky by the normal | 0.395× | 0.388× | 0.351× |
| + visibility as a line in the height (5 rays) | 0.399× | 0.449× | 0.363× |
| + visibility weighted toward the eye (5 rays) | 0.418× | 0.398× | 0.411× |
| 13 rays a boulder, flat | 0.382× | 0.396× | 0.339× |
| 13 rays a boulder, weighted toward the eye | 0.385× | 0.426× | 0.342× |

**Kept:**
- **Sky by the normal:** a small instance's sample takes the sky's
  irradiance on a point facing its normal over open ground, π L (1 + n·up)
  / 2. Before, it was half the sky whatever the normal.
- **Pages and splats:** the variant lit by the cache bakes them with the
  same sky term, in place of the model's ambient.
- **Records:** they carry the world's up in the object's frame, 6 bits a
  coordinate, beside the light's direction at 10 bits (was 16).
- **Shadow rays:** five per boulder, weighted by the light's cosine as
  before. The points' results are cached as bits per object.

**Not kept:**
- **Visibility as a line in the height:** fitted from five yes/no rays a
  boulder, it is noisy, and worse from 150 m.
- **Weighting toward the eye:** it removes the mean's bias from 150 m
  (0.942 → 0.998) but raises the error elsewhere.
- **13 rays:** about the same as five, at 2.6× the rays.

What is left in those pixels is mostly shadow varying inside a boulder (a
3 px boulder half in shadow), which one value an object cannot hold.
Their mean stays 4–6% dark from 150 m and 20 m (classes "most" and
"all"), as gate 12 found.

## Not done (part C's rest)

- **Lights after the first:** they do not light small instances. With
  gate 14's four lights the far field takes the sun alone.
- **The sky's occlusion by neighbours:** not in the term.
- **Bounce light:** not taken by small instances (off by default in the
  cache).
- **The coverage pages' colour:** still the model's (a hard shadow a hit,
  the sky as above), not the cache's.

Cost not measured separately. The term adds a dot product a sample and a
second decoded direction a record; the shadow rays are cached per object
as before.
