# M22 gate 8: bounce light past the first by default where no sky lights the scene

Date: 2026-11-22. **Passed.**

The candidate: `CacheOptions::bounces` 0 by default, meaning two bounces
with the orders past them estimated (`bounce_rest`: the last level's
samples times 1 / (1 − ρ), ρ their closure-weighted albedo; progressive
radiosity's ambient term per sample, Cohen et al. 1988) where no sky or
environment lights the scene, and one bounce under a sky. `bounce_rest`
acts only below the first of several levels. Gates 6 and 7 had the same
candidate everywhere; it cost nothing a frame but Sponza settled 1.54–2.02×
slower (gate 7 failed at 2.02×). Two levers tried after gate 7 (reading the
second level two mips coarser: room 16.91%; the deepest level one mip
coarser: 10.83%) lost accuracy without settling Sponza faster.

Measured as gate 7 (`room3d::scorecard::speed_pair`: both renderers alive,
frames alternated, warm-up capped at 120 s), against one bounce. Binary
`m22_gate8`; output `experiments/room3d/out/m22_gate8/gate_run.txt`;
criteria fixed in the stub (9c2ea0e). Also in since gate 7, on both sides:
anti-aliasing's edge shading pass traces no mirror ray on dielectrics.

| Scene | Settled frame: one, default (ratio) | Moving frame (ratio) | Settled at: one, default (ratio) |
|---|---|---|---|
| sponza | 22.34, 22.37 ms (1.002×) | 28.61, 28.57 ms (0.998×) | 27.58 s, 28.44 s (1.03×) |
| field | 9.43, 10.33 ms (1.096×) | 9.90, 9.66 ms (0.976×) | never within 120 s, either |
| room | 5.37, 5.22 ms (0.971×) | 8.90, 8.67 ms (0.974×) | 2.01 s, 1.70 s (0.85×) |
| movers | 5.07, 5.09 ms (1.003×) | 20.55, 20.51 ms (0.998×) | 1.96 s, 1.65 s (0.84×) |
| helmet | 4.62, 4.61 ms (0.999×) | 6.80, 6.85 ms (1.006×) | 1.00 s, 0.84 s (0.84×) |
| fox | 5.00, 5.06 ms (1.012×) | 8.07, 8.08 ms (1.001×) | 1.44 s, 1.34 s (0.93×) |
| brainstem | 5.84, 6.25 ms (1.070×) | 32.08, 31.29 ms (0.976×) | 7.51 s, 7.67 s (1.02×) |

| Scene against Cycles (clamped relative RMSE) | One bounce | Default | Bar |
|---|---|---|---|
| sponza | 29.94% | 29.94% | at most 31.94% |
| field | 1.68% | 1.68% | at most 1.88% |
| room | 46.55% | **7.68%** | at most 10% |
| movers | 44.46% | **5.99%** | at most 10% |
| helmet | 4.46% | 4.46% | at most 4.66% |
| fox | 2.15% | 2.15% | at most 2.35% |
| brainstem | 3.22% | 3.22% | at most 3.42% |

1. Speed: **PASS** (settled 0.971–1.096×, moving 0.974–1.006×).
2. Settling: **PASS** (0.84–1.03×).
3. Indoors: **PASS** (7.68%, 5.99%).
4. Elsewhere: **PASS** (the sky-lit scenes run one bounce on both sides).

Rehearsal: Sponza 1.002× / 1.12×, the room 7.68%.

Notes:
- Under a sky the defaults are unchanged; the field's 1.096× is one
  configuration against itself, still baking (it never settles).
- With two bounces two renderers settle to images that differ in their
  last bits (pages become resident in the GPU's order); four GPU tests that
  compare renderers bit for bit (anti-aliasing, fog, depth of field, motion
  blur, whose scenes have no sky) now set `bounces: 1`.
- Sponza and the field under a sky would gain from more bounces in
  accuracy (the field 1.68% → 1.31%) but settle slower; left for later.

## After the gate: outdoors under a sun alone (2026-11-22)

The rule as gated gave two bounces to any scene without a sky or
environment, so also to one lit by a sun alone (the viewer's "Sun"
lighting). Sponza under the sun alone, development (`gltf_dev`, a fixed
budget against Cycles at 4 bounces, 4,096 samples): one bounce 69.61%,
two with the rest 70.28% from Cycles; settling (time budget) 24.2 s against
42.9 s (1.78×), frames equal. So `Renderer` now gives two bounces only
where local lights alone light the scene: no sky, environment, or light
more than 100 scene radii away. Every scene measured in the gate keeps
its configuration (the room and the movers are lit by a panel inside;
the others have a sky); Sponza under a sun alone is back to one bounce
(settled at frame 711, 69,908 pages, as before).
