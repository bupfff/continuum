# M24 gate 1: two rays a bounce bin by default — PASS

**PASS.** Two rays a bin (each bin's diagonal quarters,
`CacheOptions::bounce_rays` 2) bring Sponza 1.49 points closer to Cycles
(25.09% → 23.60%), the movers 5.30% → 5.04%, the room 6.66% → 6.52%, Fox
2.11% → 2.06%, the rest within 0.01 points; settled frames 1.000–1.003×,
moving 1.000–1.051×, settling 0.84–1.13× (the field 7.53 → 8.50 s,
Sponza 36.38 → 31.29 s).

Why: a recess sees the sunlit courtyard through a few of a bounce
sample's 192 bins, so one ray a bin aliases differently from sample to
sample, and each page rebuilding its own nine samples shows it as
page-sized blotches; with ray-lit hits each hit's one sky ray adds noise
(`docs/results/m23-settling.md`, "Sponza's blotches").

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-28 (stub and candidate `069f726`). 7.7 minutes. Output:
`experiments/room3d/out/m24_gate1.out`.

## Criteria (fixed before the run)

| Criterion | Result | Bar | |
|---|---|---|---|
| 1. Sponza | 25.09% → 23.60% (−1.49 points) | at least 1.0 point closer | PASS |
| 2. Elsewhere against Cycles | see below | each within 0.2 points; from above 0.5 | PASS |
| 3. Speed (side by side, three motions) | see below | settled ≤ 1.05×, moving ≤ 1.10×, settling ≤ 1.5×; the field within 60 s | PASS |

| Scene | Against Cycles: one ray | Two | Settled at (frame) |
|---|---|---|---|
| sponza | 25.09% | **23.60%** | 111, 111 |
| field | 1.13% | 1.12% | 139, 139 |
| room | 6.66% | 6.52% | 41, 41 |
| movers | 5.30% | 5.04% | 47, 45 |
| helmet | 4.47% | 4.47% | 58, 58 |
| fox | 2.11% | 2.06% | 92, 92 |
| brainstem | 3.27% | 3.28% | 96, 96 |
| field from above | 12.17% | 12.18% | 365, 365 |

| Scene | Settled frame (ratio) | Moving frame (ratio) | Settled at: one, two (ratio) | Again after the motion |
|---|---|---|---|---|
| sponza | 22.32, 22.39 ms (1.003×) | 29.64, 31.14 ms (1.051×) | 36.38 s, 31.29 s (0.86×) | 15.86 s, 15.82 s |
| field | 6.56, 6.56 ms (1.000×) | 10.60, 10.79 ms (1.018×) | 7.53 s, 8.50 s (1.13×) | 1.62 s, 2.22 s |
| room | 5.23, 5.23 ms (1.000×) | 5.29, 5.29 ms (1.000×) | 2.03 s, 2.17 s (1.07×) | 0.10 s, 0.05 s |
| movers | 5.18, 5.17 ms (1.000×) | 20.19, 20.81 ms (1.031×) | 1.86 s, 1.84 s (0.99×) | 0.76 s, 0.95 s |
| helmet | 4.67, 4.68 ms (1.001×) | 4.68, 4.68 ms (1.001×) | 0.80 s, 0.68 s (0.85×) | 0.18 s, 0.07 s |
| fox | 5.03, 5.03 ms (1.000×) | 8.83, 9.06 ms (1.026×) | 1.25 s, 1.19 s (0.95×) | 1.86 s, 1.89 s |
| brainstem | 5.91, 5.91 ms (1.000×) | 33.29, 34.05 ms (1.023×) | 3.51 s, 3.55 s (1.01×) | 8.25 s, 7.30 s |

## Caveats

- Sponza's settling with two rays was faster than with one in every run
  (development 26.71 against 37.38 s): not explained; its time to settle
  varies by run (31–47 s with one ray this session).
- Four rays a bin gain 0.2 points more on Sponza (23.43%) at 1.16–1.74×
  the settling; not taken.
- The blotches are smaller, not gone (Sponza's recesses).
