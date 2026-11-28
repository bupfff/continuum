# M25 gate 1: coarse flat texels average their area — PASS

**PASS.** Flying up 40 m over the field, the ground no longer shows dark
patches while its wanted pages bake: on arrival 0.52% of pixels at most
half as bright as the same view settled, against 4.81%. Flat texels now
average their area over a chart's top ten mips (`CacheOptions::area_flat`,
`area_mips` 10; M22 gate 10 averaged only the top three mips' texels that
curve). Against Cycles the room improves (6.52% → 6.12%) and the field
from above (12.18% → 11.67%); the field's start view loses 0.14 points
(1.12% → 1.26%) and Sponza 0.16 (23.60% → 23.76%), within their bars;
frames 0.946–1.037×, settling 0.75–1.13×.

Why: coarse ground texels' sun visibility, traced as cones through the
prefiltered hierarchy over many small boulders, came out dark in places;
flights show coarser pages while the wanted ones bake. Hardware rays from
16 points over a texel see the shadows across it
(`docs/results/m23-settling.md`, "Flying up: the dark band").

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-28 (stub and candidate `340944f`). 10.6 minutes. Output:
`experiments/room3d/out/m25_gate1.out`.

## Criteria (fixed before the run)

| Criterion | Result | Bar | |
|---|---|---|---|
| 1. Flying up 40 m: on arrival at most half as bright as settled | 4.81% → 0.52% | at most 1.5%, at most half A's | PASS |
| 2. Against Cycles | see below | each within 0.2 points (Sponza 0.5; from above 0.5) | PASS |
| 3. Speed (side by side, three motions) | see below | settled ≤ 1.05×, moving ≤ 1.10×, settling ≤ 1.25× | PASS |

| Scene | Against Cycles: before | After | Settled at (frame) |
|---|---|---|---|
| sponza | 23.60% | 23.76% | 114, 114 |
| field | 1.12% | 1.26% | 139, 139 |
| room | 6.52% | **6.12%** | 41, 41 |
| movers | 5.04% | 5.04% | 45, 45 |
| helmet | 4.47% | 4.45% | 58, 58 |
| fox | 2.06% | 2.06% | 92, 92 |
| brainstem | 3.28% | 3.28% | 96, 96 |
| field from above | 12.18% | **11.67%** | 365, 365 |

| Scene | Settled frame (ratio) | Moving frame (ratio) | Settled at: before, after (ratio) |
|---|---|---|---|
| sponza | 23.40, 23.21 ms (0.992×) | 32.45, 31.89 ms (0.983×) | 29.38 s, 33.11 s (1.13×) |
| field | 6.68, 6.93 ms (1.037×) | 10.40, 10.37 ms (0.997×) | 9.35 s, 9.48 s (1.01×) |
| room | 5.48, 5.33 ms (0.974×) | 5.66, 5.65 ms (0.998×) | 2.06 s, 1.95 s (0.95×) |
| movers | 5.53, 5.24 ms (0.946×) | 22.00, 22.24 ms (1.011×) | 2.03 s, 1.53 s (0.75×) |
| helmet | 4.76, 4.89 ms (1.029×) | 5.05, 5.08 ms (1.006×) | 0.89 s, 0.76 s (0.85×) |
| fox | 5.47, 5.47 ms (1.000×) | 9.53, 9.69 ms (1.017×) | 1.35 s, 1.23 s (0.91×) |
| brainstem | 6.32, 6.09 ms (0.964×) | 34.01, 34.82 ms (1.024×) | 3.57 s, 3.33 s (0.93×) |

## Caveats

- The rise still takes about 2,800 frames to settle again (the wanted
  pages bake at about 30 a frame under the cap), and 2.3% of pixels show
  twice as bright on arrival (far boulders with no page yet, unshadowed).
- The field's start view and Sponza move slightly away from Cycles.
- Settled frames vary by ±5% this run on both sides (heat); the settled
  frame does not depend on the bake.
