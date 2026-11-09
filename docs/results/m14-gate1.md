# M14 gate 1: unique geometry streamed by need (camera levels)

**FAIL on frame rate in one flight of three (1.124× the fully resident
renderer against 1.10×); scale, settled coverage and coverage in flight
pass.** A scene whose finest geometry is 6.35× the budget flies at about
4 ms a frame on the fanless laptop, never over budget, with coverage
within a tenth of one ray's error of the fully resident image, using
about 3.4× less GPU memory than fully resident.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Finest levels against the budget | 6.35× (16,647,679 / 2,621,440 triangles) | | | at least 5× | PASS |
| 1. Frames over budget | 0 of 8,920 (at most 2,589,348 resident) | | | none | PASS |
| 2. Streamed mean frame | 4.14 ms | 4.41 ms | 3.97 ms | at most 16.7 ms | PASS |
| 2. Streamed 99th percentile | 7.50 ms | 7.87 ms | 7.02 ms | at most 33.3 ms | PASS |
| 2. Streamed against resident | 1.074× | 1.085× | **1.124×** | at most 1.10× | **FAIL** |
| 3. Settled coverage error / one ray's | 0.075× | 0.096× | 0.062× | at most 0.25× | PASS |
| 4. Pixels whose object differs in flight | 0.013% | 0.029% | 0.019% | at most 1% | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-09 (stub with the criteria committed as `d09ad21`, the
method and binary in `73469e2`, before the run). The machine was warm from
development runs; the speed criterion compares two renderers in
interleaved rounds, which thermal drift affects alike.

## Details

- **Scene:** 256 unique procedural rocks (`room3d::rocks`), each its own
  geometry, 6 m apart on a 96 m square of a 2 km ground, one sun. Four
  levels per rock, band-limited (one more octave of displacement per
  level): the lighting level (32 × 16 segments, about 1,000 triangles),
  then 4,096, 16,384 and 65,536 triangles. Mean errors from the finest
  95.2, 32.2 and 10.3 mm.
- **Frame rate**, three rounds per flight (resident/streamed mean ms):
  flight 0 3.95/4.19, 3.79/4.11, 3.83/4.13; flight 1 4.24/4.41, 3.96/4.42,
  3.99/4.41; flight 2 3.58/3.98, 3.51/3.97, 3.51/3.97. Worst frames:
  resident 9.2–11.4 ms, streamed 31.7–32.2 ms. Cache fallback 0.01–0.02%
  in both.
- **Streaming in flight** (three rounds of 600 frames): flight 0 707 loads
  (13.4 million triangles) and 529 evictions; flight 1 1,252 loads
  (20.4 million) and 972; flight 2 873 loads (19.5 million) and 608.
  Streaming's CPU time 0.16–0.19 ms a frame on average, 2.5–4.1 ms at
  worst.
- **Memory (GPU allocations, Metal's `currentAllocatedSize`):** the
  streamed renderer 984 MB at creation (pool, lighting levels, cache), the
  fully resident one 4,329 MB (the same cache and lighting levels, every
  finest level and its structure); both grew 289 MB more over the flights,
  mostly the streamed one's level structures. The streamed renderer keeps
  every level in CPU memory (2,702 MB) as the stand-in for a disk.
- **Settled coverage:** 0.0239%, 0.0378% and 0.0157% against one ray's
  0.3172%, 0.3921% and 0.2521%; worst sampled flight frame 0.031–0.105%.
  Settled colour difference (4×4 supersampled, reported): 1.42%, 1.59%,
  1.31%.

## Method

1. Each geometry may carry finer camera levels (`InstancedScene::
   geometry_levels`: `Levels`, each `Level` with its error from the
   finest). The geometry itself stays the lighting level: always resident,
   what mask cones, the cache's bakes and bounce light see, and the
   camera's stand-in.
2. Camera rays (hardware ray queries) see each geometry at its finest
   resident level: its top-level instances point at that level's
   bottom-level structure, and the level's first triangle sits in the
   object's transform's spare word, so a hit names a triangle in a pool
   after the lighting level's triangles in every per-triangle buffer the
   shading reads.
3. After the visibility pass a feedback pass (one pixel of each 2×2) turns
   the hit distance into the coarsest level whose error is at most half a
   pixel and keeps the highest per geometry. Two frames later the CPU loads
   the most needed levels (at most 2¹⁷ triangles a frame) into a buddy
   allocated pool of 2.5 × 2²⁰ triangles, evicting the least recently seen,
   then the most over-detailed; structures build straight from the pool's
   triangle buffer (fast-build for streamed levels).
4. The fully resident reference loads every finest level at start
   (fast-trace) through the same code.

**Disclosed, changed before the run after development runs** (`M14_DEV=1`,
36 rocks; `stream_dev`, which sizes the working set): the flights turn so
their last views look across the field (they ended looking away, which
made criterion 3 empty); each flight is timed in three interleaved rounds
with both renderers alive (development showed resident-then-streamed
ordering inflating the ratio); the rocks' octaves fall as frequency^-1.8
instead of ^-1.2 (at -1.2 a view wanted up to 4.9 million triangles, more
than a budget a fifth of the scene; at -1.8, 1.9 million); streamed
structures build for speed of building (a third of a load's cost in
development); the feedback samples one pixel in four; structures build
from the pool instead of a copy. Criteria unchanged.

## What the numbers say

- **Streaming works as designed:** the image settles to within a tenth of
  one ray's coverage error of fully resident, misses almost nothing in
  flight (0.03% of pixels at most on average), and the budget held on
  every frame, with 3.4× less GPU memory.
- **Its cost is loads, and there are far too many:** about half a load a
  frame, 4.5–6.8 million triangles per 600-frame flight, a third of the
  scene, though a view never wants more than 1.9 million. Each load costs
  about a millisecond (upload, structure build, top-level rebuild), and
  load frames reach 32 ms at worst. Causes visible in the policy: a rock
  approaching loads each level in turn (up to three loads); an
  over-detailed rock evicted to make room drops to its lighting level and
  reloads its wanted level later; an upgrade allocates the new level
  before freeing the old.
- **Positional error does not bound shading error:** rocks at a level
  within half a pixel of the finest still shade with smoother normals,
  1.3–1.6% colour difference settled. Level choice should also bound the
  normal's error, or levels should carry prefiltered normals (M13's
  appearance work).

Next (a new gate, M14 gate 2): fewer, cheaper loads. Skip levels a
geometry will pass through within frames (load the level wanted a short
time ahead), downgrade in place rather than drop to the lighting level,
free before allocate where possible, and spread structure builds; then
streaming the lighting level by cone footprint.
