# M23 gate 2: bounce hits lit by rays — FAIL (speed: Fox moving 1.138×)

**FAIL** on criterion 3 alone: Fox's moving frames took 1.138× those
without the candidate (9.13 → 10.39 ms; bar 1.10×). Everything else
passed: the field settles (never → 11.48 s, again 4.80 s after its walk),
Sponza now settles within the minute (never → 44.45 s), every scene within
its bar against Cycles and most closer (the field 1.23% → 1.13%, the room
6.93% → 6.65%, Sponza 25.33% → 25.04%), settled frames 0.79–1.02×.

Candidate (`CacheOptions::hit_rays`): every hit of a one-bounce sample (the
last level's, with more bounces) reads no page; its direct light is the
unshadowed light times one hardware shadow ray toward each light (movers
seen too), its sky one cosine-distributed ray. Both sides had pending hit
requests, retry tickets and shared sample texels
(`docs/results/m23-settling.md`).

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-27 (stub and candidate `4776b2d`; the sky fix after rehearsal 1,
`3d321ef`, before the run). 10.2 minutes. Output:
`experiments/room3d/out/m23_gate2.out`.

## Criteria (fixed before the run)

| Criterion | Result | Bar | |
|---|---|---|---|
| 1. The field settles | 11.48 s; again 4.80 s; without a page in motion 0.05% against 0.05% | within 60 s; again within 30 s; +0.1 points | PASS |
| 2. Against Cycles | see below | each within 0.2 points (Sponza 0.5; from above 0.5) | PASS |
| 3. Speed (side by side) | Fox moving **1.138×** | settled ≤ 1.05×, moving ≤ 1.10×, settling ≤ 1.25× | **FAIL** |

| Scene | Against Cycles: without | With | Settled at (frame) |
|---|---|---|---|
| sponza | 25.33% | **25.04%** | 146, 114 |
| field | 1.23% | **1.13%** | never, 139 |
| room | 6.93% | **6.65%** | 41, 41 |
| movers | 5.38% | 5.30% | 48, 45 |
| helmet | 4.46% | 4.47% | 29, 58 |
| fox | 2.17% | 2.11% | 99, 92 |
| brainstem | 3.28% | 3.27% | 87, 96 |
| field from above | 11.83% | 12.17% | never, 365 |

| Scene | Settled frame (ratio) | Moving frame (ratio) | Settled at: without, with | Again after the motion |
|---|---|---|---|---|
| sponza | 32.87, 28.63 ms (0.871×) | 37.58, 34.22 ms (0.910×) | never, **44.45 s** | never, 18.97 s |
| field | 9.96, 7.86 ms (0.789×) | 10.70, 10.66 ms (0.997×) | never, **11.48 s** | never, 4.80 s |
| room | 5.60, 5.63 ms (1.006×) | 8.20, 8.13 ms (0.990×) | 2.16 s, 1.52 s | 0.07 s, 0.04 s |
| movers | 5.54, 5.54 ms (1.001×) | 22.26, 21.56 ms (0.968×) | 2.01 s, 1.57 s | 0.85 s, 0.80 s |
| helmet | 5.03, 5.04 ms (1.001×) | 7.15, 7.08 ms (0.991×) | 0.92 s, 0.66 s | 0.11 s, 0.14 s |
| fox | 5.83, 5.95 ms (1.020×) | 9.13, 10.39 ms (**1.138×**) | 1.57 s, 1.27 s | 2.15 s, 3.13 s |
| brainstem | 7.28, 7.30 ms (1.003×) | 34.25, 34.02 ms (0.994×) | 5.28 s, 2.92 s | 7.25 s, 5.53 s |

## Caveats

- Development's side-by-side runs had Fox moving at 0.998× (and 1.004×
  before the movers' variant); the gate's one run is its number. The
  likely cost: with movers each ray-lit hit traces four queries (shadow
  and sky, static and movers) where a page read traced none, while the
  bake budget still charges a bounce sample its page-read cost, and
  Fox's moving frames re-bake bounce samples around it (the ledger).
- Sponza without the candidate did not settle within the gate's 60 s
  (47.15 s in development with a 120 s cap).
- The field from above stays 0.34 points worse (within its bar): a
  saturated strip at the ground's far edge, both under Cycles
  (`docs/results/m23-settling.md`).
