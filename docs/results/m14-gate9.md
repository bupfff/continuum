# M14 gate 9: the frame budget with a sky and bounce light at scale

**FAIL** (frame budget; run incomplete). A cap on the cache's bake budget
kept by the measured frame period lowered gate 8's flights from 14.1–15.9
ms to 13.2–13.3 ms on average, but the 99th percentile stayed at 19–24 ms.
The run was stopped from outside (the session's usage limit) during the
certification part's last flight; what it measured is below.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. (b) Mean frame | 13.28 ms | 13.15 ms | 13.22 ms | at most 16.7 ms | PASS |
| 1. (b) 99th percentile | **19.08 ms** | **19.34 ms** | **24.11 ms** | at most 16.7 ms | FAIL |
| 1. (b) Worst frame | 21.75 ms | 22.17 ms | **106.16 ms** | at most 33.3 ms | FAIL |
| 1. (b) Frames over 16.7 ms (reported) | 108 | 127 | 114 | | |
| 2. (a) Sun / sky pages compared in flight, differing | 83,453 / 83,453, 0 | 66,635 / 66,635, 0 | not run | none differing | (see below) |
| 2. (a) Settled pages compared, differing | 21,114 sun (**1**, worst 1.0) / 21,112 sky (0) | 26,611 / 26,611, 0 | not run | none differing | FAIL as measured |
| 3. (a) Settled image's mean difference | 0.080% | 0.101% | not run | at most 1% | |

Apple M4 (fanless MacBook Air), 1280×720, release build, run on
2026-11-13 (stub `ba527a3`; implementation and binary `bd3b134`, before
the run), after the field's build (157 s) and two minutes idle.

## Details

- **The cap held the average, not the tail:** the bake budget averaged
  8,947–13,253 units, at its cap; 86–114 pages baked a frame. About 6% of
  frames still passed 16.7 ms: the cap follows the measured period a frame
  or two late, while each frame's streaming work varies (45–60 lighting
  loads a frame; lighting levels produced 84,000–193,000 a flight, several
  times gate 7's, many for bounce samples).
- **The differing page:** the settled comparison counts every page
  resident here (shown or not) against those shown in the reference. The
  streamed renderer had not settled (8,000 frames), so pages could still
  be waiting, hidden; whether the differing page was shown was not
  recorded. Gate binaries from here compare only pages shown here, as the
  criterion says.
- **Development** (`M14_DEV=1`, 2,304 rocks): 8.9–9.4 ms mean, 99th
  percentile 14.1–15.2 ms; the larger field's streaming load is what the
  cap does not absorb.

## Method

`CacheOptions::frame_budget_ms` (15 ms by default): while a frame's period
(measured on the CPU between frames) exceeds it, a cap on the bake budget
shrinks in proportion; under 85% of it, the cap grows 5% a frame, never
past the GPU model's budget. The model's swing for its fit now goes only
down (100% and 80%).

## What the numbers say

- The bake is no longer what holds the frames up on average; the tail
  comes from frame-to-frame variance in streaming work (and the GPU work it
  starts) under a sky and bounce. Next: measure a slow frame's parts at
  this scale (as gate 7 did for the sun) before changing more.
