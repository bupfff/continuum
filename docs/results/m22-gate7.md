# M22 gate 7: gate 6's candidate, speed measured side by side

Date: 2026-11-22. **Failed** on criterion 2 (settling: Sponza 2.02× against
2×); 1, 3 and 4 passed.

Candidate as in gate 6 (`docs/results/m22-gate6.md`): two exact bounces
with the orders past them estimated (`bounces` 2, `bounce_rest`), against
one bounce. Speed measured with `room3d::scorecard::speed_pair`: both
renderers alive, frames alternated (warm-up and settling counting each
one's own frames' time). Binary `m22_gate7`; output
`experiments/room3d/out/m22_gate7/gate_run.txt`; criteria fixed in the stub
(43e3ce2).

| Scene | Settled frame: one, two+rest (ratio) | Moving frame (ratio) | Settled at: one, two+rest (ratio) |
|---|---|---|---|
| sponza | 28.12, 28.28 ms (1.006×) | 34.70, 33.41 ms (0.963×) | 28.74 s, 58.03 s (**2.02×**) |
| field | 10.26, 10.44 ms (1.017×) | 9.89, 9.75 ms (0.985×) | never within 120 s, either |
| room | 5.22, 5.29 ms (1.014×) | 8.53, 8.35 ms (0.978×) | 1.79 s, 1.79 s (1.00×) |
| movers | 5.12, 5.19 ms (1.014×) | 22.13, 21.19 ms (0.958×) | 1.78 s, 1.74 s (0.98×) |
| helmet | 4.64, 4.71 ms (1.015×) | 7.14, 7.84 ms (1.099×) | 1.02 s, 1.16 s (1.13×) |
| fox | 5.56, 5.58 ms (1.003×) | 9.07, 9.19 ms (1.013×) | 1.53 s, 1.81 s (1.19×) |
| brainstem | 6.59, 6.61 ms (1.002×) | 31.54, 31.58 ms (1.001×) | 7.85 s, 9.31 s (1.19×) |

Against Cycles (fixed bake budget, as gate 6): Sponza 29.94% → 31.32% (bar
31.94%), field 1.68% → 1.31%, **room 46.55% → 7.68%**, **movers 44.46% →
5.99%**, helmet 4.46% → 4.46%, Fox 2.15% → 1.21%, BrainStem 3.22% → 3.11%.

1. Speed: **PASS** (settled 1.002–1.017×, moving 0.958–1.099×).
2. Settling: **FAIL** (Sponza 2.02×; every other scene 0.98–1.19×).
3. Indoors: **PASS**.
4. Elsewhere: **PASS**.

Rehearsal (room and Sponza): Sponza settled 34.33 → 52.79 s (1.54×). Across
gate 6, its rehearsal and this gate, Sponza's two-bounce settling took
1.54–2.02× one bounce's: a real cost, near the bar, varying with how fast
one bounce settled (28.7–34.3 s).

## Reading it

- Side by side, the candidate costs nothing a frame (gate 6's 1.184× and
  1.225× were the machine's heat): the speed question is answered.
- What is left is Sponza's warm-up: its second level of bounce samples (the
  pages its bounce rays hit, across the building) takes half again to twice
  as long to settle, at a budget held near its floor by 22–28 ms frames.

## Levers left (DESIGN §3.8)

- The second level's samples cheaper or fewer: the tail test also for hits
  without samples (estimating their bounce share from their closure-weighted
  albedo), or their samples at a coarser mip.
- Sponza's frame (22–28 ms against the 14 ms target) keeps the bake budget
  at its floor; a faster frame (anti-aliasing's passes are about 13 ms of
  it) settles both faster.
- Or ship the gain for indoor scenes as an option (`bounces` 2 with
  `bounce_rest`) and keep one bounce the default.
