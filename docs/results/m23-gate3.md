# M23 gate 3: bounce hits lit by rays, over three motions — PASS

**PASS.** With every one-bounce hit lit by rays (`CacheOptions::hit_rays`)
the field settles (never within 60 s → 10.97 s, again 4.40 s after its
walk), and so does Sponza (never → 34.80 s, again 16.52 s). Against
Cycles every scene is within its bar, most closer: the field 1.23% →
1.13%, the room 6.93% → 6.65%, Sponza 25.33% → 25.08%; the field from
above 0.34 points worse (bar 0.5). Settled frames 0.79–1.00× (Sponza
0.870×, the field 0.793×: nothing left to bake), moving frames
0.979–1.042×.

Candidate as gate 2's (`docs/results/m23-gate2.md`, failed on Fox's
moving frames, 1.138×, a measure that moved by up to 1.2× between runs),
with the ray-lit hits' shadow rays opaque to the first hit; speed measured
over three passes of each motion (`scorecard::speed_pair_passes`). Both
sides had pending hit requests, retry tickets and shared sample texels
(`docs/results/m23-settling.md`).

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-27 (stub and candidate `beb73f6`). 11.1 minutes. Output:
`experiments/room3d/out/m23_gate3.out`.

## Criteria (fixed before the run)

| Criterion | Result | Bar | |
|---|---|---|---|
| 1. The field settles | 10.97 s; again 4.40 s; without a page in motion 0.02% against 0.02% | within 60 s; again within 30 s; +0.1 points | PASS |
| 2. Against Cycles | see below | each within 0.2 points (Sponza 0.5; from above 0.5) | PASS |
| 3. Speed (side by side, three motions) | see below | settled ≤ 1.05×, moving ≤ 1.10×, settling ≤ 1.25× | PASS |

| Scene | Against Cycles: without | With | Settled at (frame) |
|---|---|---|---|
| sponza | 25.33% | **25.08%** | 147, 114 |
| field | 1.23% | **1.13%** | never, 139 |
| room | 6.93% | **6.65%** | 41, 41 |
| movers | 5.37% | 5.31% | 48, 45 |
| helmet | 4.46% | 4.47% | 29, 58 |
| fox | 2.17% | 2.11% | 99, 92 |
| brainstem | 3.28% | 3.27% | 87, 96 |
| field from above | 11.83% | 12.17% | never, 365 |

| Scene | Settled frame (ratio) | Moving frame (ratio) | Settled at: without, with (ratio) | Again after the motion |
|---|---|---|---|---|
| sponza | 33.22, 28.89 ms (0.870×) | 35.05, 35.93 ms (1.025×) | never, **34.80 s** | never, 16.52 s |
| field | 9.89, 7.85 ms (0.793×) | 10.66, 10.64 ms (0.999×) | never, **10.97 s** | never, 4.40 s |
| room | 5.79, 5.78 ms (0.999×) | 6.32, 6.29 ms (0.996×) | 2.10 s, 2.23 s (1.06×) | 0.06 s, 0.06 s |
| movers | 5.54, 5.54 ms (1.000×) | 22.53, 22.06 ms (0.979×) | 2.03 s, 1.58 s (0.78×) | 1.05 s, 0.97 s |
| helmet | 5.03, 5.03 ms (1.000×) | 5.31, 5.44 ms (1.024×) | 0.95 s, 0.67 s (0.71×) | 0.07 s, 0.08 s |
| fox | 5.82, 5.80 ms (0.998×) | 9.91, 10.33 ms (1.042×) | 1.59 s, 1.26 s (0.79×) | 2.16 s, 3.11 s |
| brainstem | 7.31, 7.29 ms (0.998×) | 34.95, 34.22 ms (0.979×) | 4.45 s, 3.60 s (0.81×) | 8.44 s, 6.97 s |

## Caveats

- Fox settles again after its animation in 3.11 s against 2.16 s
  (1.44×; settling after the motion is not a criterion), as in gate 2
  (3.13 against 2.15 s): its bounce samples, re-baked by the ledger while
  it moves, now trace four rays a hit.
- The helmet settles at frame 58 against 29 in the quality half (at a
  fixed budget), while its warm-up is faster in the speed half (0.67 s
  against 0.95 s).
- The field from above stays 0.34 points worse, in a saturated strip at
  the ground's far edge where both are under Cycles.
- The viewer's field (no sky, so two bounces) is not measured here: there
  the camera samples' hits still need pages, now holding ray-lit samples
  of their own.
- Ray-lit hits need hardware ray queries; without them (lavapipe, WARP)
  bounce light reads pages as before, with pending hits and retry tickets.
