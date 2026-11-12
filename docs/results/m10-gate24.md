# M10 gate 24: the cache settles with bounce light (M10 gate 23 re-run)

**FAIL on accuracy (19.71% from Cycles against 5%), PASS on liveness: the
core bug is fixed.** With bounce light on, both renderers now settle at
the first view and at all 16 sub-pixel offsets (gate 23's never settled
in 50,000 frames). Emissive faces as lights (a parked, experimental
feature) remain less accurate than emission through bounce light.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 1. Liveness, faces as lights: frames to settle, first view then each offset | 3,873; 869, 559, 762, 479, 725, 534, 216, 395, 843, 473, 611, 558, 837, 588, 422, 522 | each within 20,000 | PASS |
| 1. Liveness, emission through bounce | 613; 132, 140, 200, 169, 179, 174, 175, 167, 138, 227, 199, 200, 153, 156, 193, 157 | each within 20,000 | PASS |
| 2. Faces as lights against Cycles, over surfaces (4×4 supersampled) | 19.71% (through bounce 7.46%: 2.64×) | at most 5% and 0.5× | **FAIL** |

Means over surfaces: faces as lights −2.82%, through bounce −0.94%. 202
pixels set apart. Pages resident when settled: 7,161 and 7,168.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-12 (stub `259357b`; the fix and the binary `8ce4054`,
before the run). Cycles: gate 11's render of the same job (4,096 samples
per pixel, one bounce), reused from its work directory.

## The bug and the fix

Diagnosed after gate 23 with `emitters_dev` and `CONTINUUM_TRACE_BAKES`
(disclosed): in the 3,000 frames after the view first settled and moved
by a sub-pixel offset, 16,307 bounce retries and 1,745 page bakes, one
page retried 552 times, and the same secondary pages (light at bounce
hits) baked again every 125–190 frames.

A page's bounce samples complete only when every one of their hits finds
a resident page at once. Pages off screen, read by no pixel, count as
unused and were evicted after 64 frames, while the bake budget (at its
floor near the panel's lights) baked only a few missing ones a frame: the
early ones aged out before the last arrived, so the waiting page retried
forever. A liveness bug in the certified cache, not specific to emissive
faces.

Fix (`vcache.rs`, `pinned`): pages baked for bounce hits stay resident
while any page's bounce samples wait (at most a quarter of the pool, the
latest kept); once none wait they age out as before. The bounce bake pass
already binds Metal's 29 buffers, so the pin is kept on the CPU rather
than by marking slot use on the GPU. Emission through bounce settled to
gate 12's image (7.46% from Cycles, identical), so the fix changes when
the cache converges, not what it converges to.

## Why the faces as lights miss

From the settled first view at one sample per pixel against Cycles (a
diagnostic after the run, `emitters_dev dump`): 8.6% over surfaces, 77% of
it on the panel's faces, the brightest surfaces kept. There our image
shows exactly the lights' radiance (3, 1.86, 0.9); Cycles shows the faces'
emission plus the light they reflect, (3.10, 1.95, 0.98). The panel's top
edge also differs by about a row. Supersampled, the gate measured 19.71%,
more than one sample per pixel: unexplained, and pointing at how light
rectangles are shown at sub-pixel offsets. Not pursued: the feature is
parked (DESIGN §3.8).
