# M13 gate 1: sub-pixel geometry through prefiltered pixel cones (coverage only)

**FAIL.** Speed fails in both views by about 100×; accuracy passes from
150 m (0.38× one ray's error) and fails standing, mostly from a flaw in
the measure (below).

| View | Coverage error, ours | One ray | Ratio (at most 0.5×) | Frame (at most 16.7 ms) |
|---|---|---|---|---|
| Standing | 51.56% | 9.97% | 5.17×: FAIL | 2,002 ms: FAIL |
| 150 m up | 22.81% | 60.14% | **0.379×: PASS** | 1,207 ms: FAIL |

Mean coverage: standing, reference 0.2741, ours 0.2951, one ray 0.2746;
150 m up, 0.2821, 0.2861, 0.2823.

Work per pixel (a separate kernel with work counting):

| View | Threshold | Object nodes | Triangle nodes (max) | Counting frame | Coverage error |
|---|---|---|---|---|---|
| Standing | 8 bins (a pixel) | 53.7 | 83.6 (5,759) | 2,011 ms | 51.56% |
| Standing | 1 bin (M7 G1's) | 52.7 | 581.1 (263,385) | 44,597 ms | 51.60% |
| 150 m up | 8 bins | 73.0 | 75.3 (551) | 1,185 ms | 22.81% |
| 150 m up | 1 bin | 70.6 | 361.9 (4,274) | 9,598 ms | 14.77% |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criteria committed as `038fcf2`; the
binary `experiments/room3d/src/bin/m13_gate1.rs` and the kernel's profile
stages 10–11 are committed with this result). A game had been running
until shortly before; a control just before the run (`vcolor_dev`, the
textured scene's warm frame) read 3.12–3.14 ms against 3.02 ms earlier.

## Method

Each pixel's cone (its square from the eye, 8×8 bins) walks the object
and triangle hierarchies with the light cones' coverage walk
(`scene_coverage`), prefiltered at 8 bins: nodes projecting within a pixel
splat their prefiltered coverage, objects combine as independent
occluders. The ground (object 0) is left out. The reference counts, per
pixel, the share of 16×16 rays through its square that hit a boulder.

## What the numbers say

- **Speed: the per-node test is the cost, not the node count.** About 150
  node visits per pixel, far fewer than M7 G1's hundreds to thousands, but
  each projects the node's box, takes its hull and rasterizes it into the
  8×8 bins (a 68-value buffer per pixel): about 9 ns per node visit across
  the GPU. That cost is affordable in the light cache's bakes, which are
  amortized over frames; per pixel per frame it is about 100× too slow.
- **Standing accuracy: the measure was flawed.** The boulders are partly
  buried. The reference's rays hit the ground first, so buried parts do not
  count; the cone walk left the ground out, so they did. The error map
  shows crescents of extra coverage along the ground's gaps, not a shift
  (ours shifted by one or two pixels compares no better: 46.5 against
  47.4–54.9 in 8-bit RMS). From 150 m, buried parts lie under the
  boulders themselves, and the method passes. A fair measure leaves the
  ground out of both, or lets it occlude in both.
- **Prefiltering at the pixel trades accuracy for work as intended** from
  150 m: 8 bins does 0.2× the triangle work of 1 bin at 22.81% against
  14.77%.

## Next (M13 gate 2)

A per-node test cheap enough for every pixel: the node's projected box
overlapping the pixel's square, times its prefiltered opacity, as a
scalar (no bins), composited front to back; resolved surfaces (larger than
a pixel) found by the camera's ray as now. The reference and ours treat the
ground the same way.
