# M5 gate 9: sky warm-up, the sky later

**PASS** at all four viewpoints. With the sky later, the image comes within
5% of its warm self 10–20× sooner, the error while warming falls to
0.11–0.17× the baseline's, views warm in 0.61–0.65× the frames, and the warm
image is identical.

| Viewpoint | 1. Same result | 2. Frames until warm | 3. Mean error while warming | Within 5% at frame | Fallback (reported) | Sky samples until warm |
|---|---|---|---|---|---|---|
| standing | 0.0000% | 1,165 against 1,806 (**0.65×**) | **1.16%** against 6.97% (**0.17×**) | 120 against 1,200 | 13.5% against 0.0% | 1,089,765 against 1,720,418 |
| 10 m up | 0.0000% | 2,495 against 3,890 (**0.64×**) | **1.57%** against 11.90% (**0.13×**) | 300 against 4,060 | 19.4% against 0.2% | 2,373,579 against 3,859,469 |
| 40 m up | 0.0000% | 2,584 against 4,007 (**0.64×**) | **1.79%** against 14.45% (**0.12×**) | 300 against 4,220 | 25.2% against 0.6% | 2,467,610 against 3,992,313 |
| 150 m up | 0.0000% | 1,726 against 2,827 (**0.61×**) | **1.93%** against 18.05% (**0.11×**) | 200 against 3,000 | 21.6% against 1.1% | 1,665,094 against 2,855,442 |

Bounds: 1. at most 0.1%; 2. at most 1.2×; 3. at most 0.5×. Errors are
relative RMSE against the baseline's warm image, read back every 20th frame
up to the baseline's frames until warm. The baseline's second run (the one
compared in criterion 3) warmed in 2,014, 4,338, 4,457 and 3,138 frames: run
to run, warm-up varies by about 10%.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `5ba3f73`; binary
`experiments/room3d/src/bin/m5_gate9.rs`, criteria pre-registered in
`233f7a7`). Frame counts use a fixed bake budget (11,000 units a frame,
`frame_target_ms` 0), so they do not depend on the machine's heat.

## Method

`CacheOptions::sky_later`, on by default since this gate:

- A page the pixels ask for is baked with its lights only. Until its sky is
  baked, pixels show the open sky over their own normal, the guess M9 gate 3
  shows on moved pages. Sun shadows are exact from the first bake.
- From the next frame the page waits in the queue of skies re-baked alone
  (M9 gate 3's), oldest first. Each frame bakes from that queue with what its
  budget leaves after new pages.
- A page with sky costs about 11× one without: a sky sample counts as ten
  sun texels. So pages arrive about 11× faster, and the sky follows.
- A page evicted before its sky comes up never bakes it. These are mostly
  coarse pages that finer ones replaced while the view warmed. That is why
  warm-up takes fewer frames and 0.62–0.63× the sky samples, with the same
  warm image.
- Ignored with bounce light, movers or moving lights. Bounce samples keep
  whatever coarser pages they find when they are baked, so another bake
  order settles to a different image (about 2% on the library's test scene,
  found in development). Movers and moving lights use the same sky guess for
  their own re-bakes.

Changed after the binary's first commit, before the full run: skies re-baked
alone now stay within the frame's budget. Before, a frame could overshoot by
one page's sky, which would have favoured the sky later in this comparison.
A development run (100,000 boulders, standing) passed every criterion.

**Fallback** counts pixels with no page *or* a guessed sky, so the two
columns are not like for like. The baseline shows almost no fallback. Its
pixels find coarse pages quickly, but those pages are dark where they should
be lit and lit where they should be shadowed, which criterion 3 measures.

Test: `sky_later_settles_to_the_same_image` (continuum-gltf GPU tests).

Prior art: progressive refinement, coarse to fine, and shading with an
approximation until the exact value arrives, are standard in light caches
and texture streaming. No novelty is claimed.

## Changed since

M9 gate 5 gave queued skies half of each frame's budget before new pages
(in flight they starved). M5 gate 9's development check with that change
(100,000 boulders, standing): error while warming 0.24× the baseline's
(0.17× before), warm in 0.73× the frames (0.65×), the same warm image. The
full-size numbers above are with new pages first.

## Not measured

- Flights. The still views above are the case M9 gate 3 found weakest
  (21–22% fallback with sky). A flight keeps asking for new pages, which come
  first, so skies can wait longer while moving.
- Wall time with the time-based budget. A page without sky is cheaper per
  frame, and the bake model fits cost per unit of budget, so this should
  carry over; it is not shown here.
