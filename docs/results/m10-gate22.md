# M10 gate 22: bounce light with more than three lights (gate 21 re-run)

**FAIL on speed, narrowly (1.167× against 1.15×); accuracy and re-coloring
pass.** Kept: scenes with more than three lights now have bounce light.
The speed miss is bounce light's own cost on this night scene, the same
with three lights (below).

| Criterion | Result | Verdict |
|---|---|---|
| 1. Median over five interleaved rounds of the warm frame, bounce on to off, at most 1.15× | **1.167×** | FAIL |
| 2. Over surfaces: within 5% of Cycles, and at most 0.5× the error with bounce off | **3.07%**; bounce off 17.78% (**0.17×**); 659 pixels apart | PASS |
| 3. The fourth light re-colored, then settled: within 1% of a fresh renderer | **0.407%** (the warm image is 344.82% from it); settled in 106 frames after re-coloring, the fresh one in 847 | PASS |

| Round | Bounce off | On | Ratio |
|---|---|---|---|
| 0 | 2.84 ms | 3.30 ms | 1.163× |
| 1 | 2.83 ms | 3.30 ms | 1.167× |
| 2 | 2.84 ms | 3.31 ms | 1.167× |
| 3 | 2.86 ms | 3.33 ms | 1.167× |
| 4 | 2.84 ms | 3.30 ms | 1.162× |

Means over surfaces: ours +0.21%, bounce off −5.23% from Cycles'. Settled
in 839 (bounce on) and 697 (off) frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (binary `experiments/room3d/src/bin/m10_gate22.rs`,
committed in `81a7606` before the run). Cycles: gate 21's reference,
reused (one bounce, 4,096 samples per pixel, 94 s on Metal).

## Method

Gate 21's (`m10-gate21.md` for its crash): lights past the third add their
bounce into the emission channel, which holds absolute radiance, at their
radiance when baked; the first three keep their own channels and re-color
free. Re-coloring a later light clears the bounce samples; shading asks for
them again (a page shown without samples requests them) and the cache
re-bakes them. Fix since gate 21: the bounce samples' buffer has the
copy-destination usage that clearing needs.

## The speed miss (development, after the gate)

`bounce_dev` measured bounce light's warm-frame cost on the same scene
with all five lights and with the first three (bounce light's old limit),
three alternated rounds each:

| Lights | Bounce off | On | Ratio |
|---|---|---|---|
| 5 | 2.83–2.85 ms | 3.30 ms | 1.158–1.166× |
| 3 | 2.69–2.93 ms | 3.13–3.17 ms | 1.081–1.167× (1.165× and 1.167× in the two steady rounds) |

Warm frames do no bounce baking, and the new path adds no work to them:
bounce light costs about 0.45 ms a warm frame here (reading 3×3 samples ×
five channels per pixel), against about 0.2 ms (1.07×) on M10 gate 1's day
scene, where the bound came from. Making bounce light's shading cheaper is
a separate item.
