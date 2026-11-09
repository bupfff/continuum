# M13 gate 3: sub-pixel geometry through directional coverage maps (coverage only)

**FAIL on speed; accuracy passes in both views** (for the first time in
M13): about a quarter of one ray's error.

| View | Coverage error, ours | One ray | Ratio (at most 0.5×) | Frame (at most 10 ms) |
|---|---|---|---|---|
| Standing | 2.29% | 8.69% | **0.264×: PASS** | 706.99 ms: FAIL |
| 150 m up | 13.39% | 57.99% | **0.231×: PASS** | 178.46 ms: FAIL |

Mean coverage: standing, reference 0.2951, ours 0.2953, one ray 0.2956;
150 m up, 0.2909, 0.2883, 0.2910. Per pixel: standing 51.0 object nodes
(at most 7,409) and 51.0 triangle nodes (at most 2,996); 150 m up, 70.3
(at most 131) and 19.0 (at most 548). The maps: built in 0.05 s, 0.3 MB
(one geometry: 256 directions × 341 texels as f32).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-09 (stub with the criteria committed as `d9d7fb8`; the
method in `97b5af7`, before the run). The field without its ground, for
the reference and the test alike. Development runs before it (`M13_DEV=1`,
10,000 boulders, disclosed): standing 230 ms, 150 m up 103 ms.

## Method

`coverage_maps.rs` builds, per geometry, its opacity seen from 256
directions (an octahedral grid) over its bounding sphere at 16×16 texels,
with mip levels. Each pixel keeps 4×4 bins. The object walk is gate 2's;
object nodes within a pixel splat per bin; instances at most 8 pixels
across read their maps (the four nearest directions, blended, at the level
matching a bin) per bin; larger instances walk their triangles per bin.

## What the numbers say

- **Accuracy is solved for coverage.** Gate 2's horizon row (one number per
  pixel saturating) is gone: maps know where within its sphere a boulder
  is, and bins keep where within the pixel. Off the horizon the error was
  already low; now both views pass with margin.
- **Speed is the one blocker, 18–71× over.** Two causes are visible:
  - *Long tails:* a pixel the far boulders only partly cover (the horizon)
    never fills its bins, so it walks everything behind (at most 7,409
    object nodes), and its SIMD group waits for it. Gate 2's saturated
    scalar stopped early, which is why it was faster standing. A node that
    overlaps only full bins could be skipped.
  - *Heavy map reads:* each read is 16 bins × 4 directions × 4 texels, with
    the level's offset recomputed per texel; and instances over 8 pixels
    still walk their triangles per pixel (19–51 triangle nodes per pixel).
- **What remains to try:** skipping nodes over full bins; a lighter read
  (the nearest direction, offsets hoisted); one walk shared by a tile of
  pixels (coherent loads, the cost gates 2 and 3 found in scattered
  per-pixel walks); hardware rays for surfaces larger than a few pixels.

## Where the time goes now (development, after the gate)

`M13_DEV=1` (10,000 boulders; numbers not the gate's), after three changes
that leave the result alone or nearly (nodes over full bins skipped; far
object nodes no taller than a bin splatted; map addressing hoisted), with
parts switched off by branches never taken:

| Variant | Standing | 150 m up |
|---|---|---|
| All parts | 208 ms | 100 ms |
| Without walks inside instances over 8 px | 25.2 ms | 26.0 ms |
| Without those or the map reads (the object walk) | 16.6 ms | 13.0 ms |

- Walking the triangles of instances larger than 8 pixels, per pixel, is
  75–180 ms of it. These are the near, resolved surfaces, where one ray per
  pixel is right except at their edges.
- The map reads cost about 10–13 ms, and the object walk alone 13–17 ms:
  even without the near field, about 25 ms against a 10 ms target.

So a 10 ms coverage pass needs both: the near field resolved cheaply (a
grid of hardware rays at pixel corners: a pixel whose corners all hit the
same large object is covered, and only edge pixels walk), and the far
field's walk shared by a tile of pixels (coherent loads) or the quality
dial trading resolution for time.
