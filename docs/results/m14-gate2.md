# M14 gate 2: fewer, cheaper loads, and detail that follows the zoom

**FAIL on frame rate in three flights of four (1.110×, 1.111× and 1.113×
the fully resident renderer against 1.10×; 1.091× in the fourth); scale,
settled coverage and coverage in flight pass, the zoom included.**
Streaming cost 0.33–0.42 ms a frame, against frames of 3.6–3.8 ms, which
this run measured faster than gate 1's did. The zoom works as intended:
narrowing the view from 50° to 1.5° onto a rock 103 m away cut the
working set from 1.55 to 0.52 million triangles and settled to exactly the
fully resident image.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Zoom | Bar | Result |
|---|---|---|---|---|---|---|
| 1. Finest levels against the budget | 6.35× | | | | at least 5× | PASS |
| 1. Frames over budget | 0 of 12,237 (at most 2,600,704 resident) | | | | none | PASS |
| 2. Streamed mean frame | 3.92 ms | 4.26 ms | 3.96 ms | 3.98 ms | at most 16.7 ms | PASS |
| 2. Streamed 99th percentile | 7.15 ms | 7.56 ms | 7.17 ms | 7.16 ms | at most 33.3 ms | PASS |
| 2. Streamed against resident | 1.091× | **1.110×** | **1.111×** | **1.113×** | at most 1.10× | **FAIL** |
| 3. Settled coverage error / one ray's | 0.072× | 0.087× | 0.075× | 0.000× | at most 0.25× | PASS |
| 4. Pixels whose object differs in flight | 0.016% | 0.028% | 0.019% | 0.011% | at most 1% | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-12 (stub `31aadc5`; method and binary `d36654c`, before
the run). Gate 1's scene, budget, flights and reference, plus the zoom.

## Details

- **Frame rate, rounds** (resident/streamed mean ms): flight 0 3.52/3.90,
  3.65/3.97, 3.60/3.88; flight 1 3.84/4.28, 3.84/4.24, 3.85/4.27; flight 2
  3.48/3.94, 3.62/4.02, 3.59/3.91; zoom 3.56/3.96, 3.56/3.94, 3.60/4.03.
  Worst frames: resident 8.0–11.5 ms, streamed 29.0–34.3 ms. Streaming's
  CPU time 0.07–0.10 ms a frame on average, but 10.3 and 14.1 ms at worst
  in two flights (new since gate 1; not diagnosed).
- **Loads over three rounds:** flight 0 646 (13.8 million triangles; 263
  upgrades, 27 reloads within 120 frames), flight 1 872 (17.8 million; 328,
  13), flight 2 653 (18.0 million; 311, 73), zoom 566 (17.5 million; 460,
  11). Evictions 694, 923, 708, 982. Per round that is 4.6, 5.9, 6.0
  million triangles for gate 1's flights, against gate 1's 4.5, 6.8, 6.5.
- **The zoom:** standing 10 m outside the field, toward a rock 103 m away,
  the field of view from 50° to 1.5° over 600 frames. Working set 1,551,957
  triangles at 50°, 520,231 at 1.5°. At 1.5° a pixel there is 3.75 mm; the
  cache's finest texel (5 mm, at 2 pixels a texel) still resolves it.
- **Memory (GPU allocations):** streamed renderer 984 MB at creation, fully
  resident 4,597 MB (pages round each level up to 4,096 triangles), both
  grew 362 MB over the flights; the streamed renderer's levels in CPU
  memory, the stand-in for a disk, 2,702 MB.
- **Settled colour difference** (reported): 1.39%, 1.36%, 1.48%, 0.00%,
  as gate 1's: coarser levels shade with smoother normals.

## Method

Gate 1's (camera levels by need; `docs/results/m14-gate1.md`) with, from
the diagnosis after it:

1. A pool of fixed pages of 4,096 triangles. A level's bottom-level
   structure has one geometry per page; a hit's geometry index names the
   page through a page table after the objects' ranges (`cam_tri`).
   Gate 1's buddy allocator had forced most of its evictions
   (fragmentation).
2. Uploads through a reused staging belt.
3. The feedback samples one pixel in sixteen (streaming's fixed cost on a
   still view: 1.009× the resident renderer's).
4. A chain of resident levels per geometry (camera rays see the finest),
   and room made in this order: a shown level more detailed than wanted
   when a kept coarser one meets the want; any level unseen for 30 frames;
   coarser levels hidden under finer ones; a shown over-detailed level
   without one, replaced at once by the wanted level; recently seen levels.

**Disclosed:** the diagnosis and the sizing of these changes ran on gate
1's scene and flights with `stream_dev` (no resident renderer, no timing):
loads per flight 6.4–7.7 → 6.2–7.1 million triangles, levels loaded again
within 600 frames 0.4–2.8 → 0.3–1.9 million. A load measured 45 → 29 ns a
triangle on the 36-rock scene (structure build 22 ns of it, about 45
million triangles a second on the M4; no fixed cost per load). The first
three changes preceded the stub; the chains and the order came after it,
before the run. Development runs (`M14_DEV=1`, 36 rocks) gave 1.089–1.101×.

## What the numbers say

- **Correctness and the zoom pass with room:** coverage within a twelfth
  of one ray's error settled, under 0.03% of pixels off in flight, the zoom
  exact at 1.5°, never over budget.
- **The speed bar was missed by more than development predicted.** Loads
  fell 3–12% per round and cost less per triangle, yet the streamed
  renderer's absolute overhead (0.33–0.42 ms) matched gate 1's (0.28–0.44
  ms); the resident renderer ran 4–7% faster than in gate 1, tightening the
  ratio. About 0.1 ms a frame is not explained by the measured load cost
  (5.9 million triangles × 29 ns over 600 frames is 0.29 ms); the new CPU
  spikes and the evictions (up to 982 a flight) are candidates, untested.
- **Most of the cost is inherent to flying past unique detail:** each rock
  within about 16 m of the path needs its 65,536-triangle level built once,
  at about 45 million triangles a second. Against a 60 fps frame (16.7 ms)
  the measured overhead is 2.0–2.5%; against these unthrottled 3.6–3.8 ms
  frames it is 9–11%.

Next, if speed is pursued in a new gate: a streaming time budget (the
quality contract, §3.3: loads limited to a share of the frame, the error
from deferred loads reported), and the CPU spikes found and fixed.
