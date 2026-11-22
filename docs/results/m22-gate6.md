# M22 gate 6: two bounces with the rest estimated, by default

Date: 2026-11-22. **Failed** on criterion 1 (speed); 2–4 passed.

## What was tested

The candidate defaults, against the one bounce of today's: two exact
bounces (`CacheOptions::bounces` 2) with the orders past them estimated
(`CacheOptions::bounce_rest`): the pages a bounce sample's rays hit hold
one-bounce samples of their own, each scaled by 1 / (1 − ρ), ρ its
closure-weighted albedo (progressive radiosity's ambient term per sample,
Cohen et al. 1988; GTAO's multi-bounce fit, Jimenez et al. 2016, is a
screen-space relative). Hits whose samples hold a bounce ask for no more
where bounce light is under `bounce_tail` (0.15) of their light.

Measured as the basics scorecard measures (`room3d::scorecard`), its seven
scenes, per scene one bounce then the candidate; speed first, warm-up
capped at 120 s. Binary: `m22_gate6`; output
`experiments/room3d/out/m22_gate6/gate_run.txt`. Criteria and the
development numbers were fixed in the stub (commit 37a03aa) before the run.

## Results

| Scene | Settled frame: one, two+rest (ratio) | Moving frame (ratio) | Settled at: one, two+rest (ratio) |
|---|---|---|---|
| sponza | 22.22, 26.31 ms (**1.184×**) | 30.43, 30.71 ms (1.009×) | 30.39 s, 55.36 s (1.82×) |
| field | 8.52, 10.43 ms (**1.225×**) | 9.72, 9.74 ms (1.002×) | never within 120 s, either |
| room | 5.22, 5.53 ms (1.059×) | 8.62, 8.93 ms (1.037×) | 1.96 s, 2.04 s (1.04×) |
| movers | 5.13, 5.14 ms (1.002×) | 21.09, 20.65 ms (0.979×) | 1.99 s, 2.23 s (1.12×) |
| helmet | 4.76, 4.91 ms (1.032×) | 7.13, 7.94 ms (1.114×) | 1.01 s, 1.30 s (1.29×) |
| fox | 5.47, 5.44 ms (0.995×) | 8.87, 8.90 ms (1.004×) | 1.54 s, 1.94 s (1.26×) |
| brainstem | 6.51, 5.86 ms (0.900×) | 31.60, 31.23 ms (0.988×) | 7.83 s, 9.40 s (1.20×) |

| Scene against Cycles (clamped relative RMSE) | One bounce | Two + rest | Bar |
|---|---|---|---|
| sponza | 29.94% | 31.32% | at most 31.94% |
| field | 1.67% | 1.32% | at most 1.87% |
| room | 46.55% | **7.68%** | at most 10% |
| movers | 44.46% | **5.99%** | at most 10% |
| helmet | 4.46% | 4.46% | at most 4.66% |
| fox | 2.15% | 1.21% | at most 2.35% |
| brainstem | 3.22% | 3.11% | at most 3.42% |

1. Speed (settled frame at most 1.10×, moving at most 1.15×): **FAIL**
   (Sponza 1.184×, the field 1.225×).
2. Settling (within 2×): **PASS** (1.04–1.82×).
3. Indoors (at most 10%): **PASS** (7.68%, 5.99%).
4. Elsewhere: **PASS**.

The rehearsal (room and Sponza, same measures) passed: Sponza's settled
frame 22.20 → 22.22 ms (1.001×), settled 29.69 → 51.17 s.

## Reading it

- **The field** never settles with either (its bounce light keeps
  re-baking), so its "settled frame" is a frame still baking under the time
  budget; frames below the 14 ms target let the budget take more work, and
  two bounces leave more to do. The criterion compared two unsettled states.
- **Sponza**: 1.001× in the rehearsal and 1.184× here, both measured in the
  60 frames after 32 quiet ones. Either work resumes after "settled" with
  two bounces (retries, the bounce ledger's drain every eighth still frame),
  or it is the machine's state; not separated.
- Accuracy is the gain this was for: indoors 6–8× closer to Cycles, Fox 1.8×,
  no scene worse beyond its bar.

## Levers left (for a new gate)

- Find whether Sponza's frames after settling carry bounce work with two
  bounces (the cache's counters over a few hundred still frames).
- Speed criteria that compare like with like: settled frames after a longer
  quiet run, and for scenes that never settle, frames at a fixed bake budget.
