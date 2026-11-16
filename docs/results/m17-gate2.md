# M17 gate 2: auto-exposure at a quarter of the pixels

**PASS** on all three criteria.

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Cost: auto-exposure's added warm frame (day view) | **0.078 ms** (2.827–2.876 against 2.750–2.786 ms, five rounds) | at most 0.15 ms | PASS |
| 2. PBR Neutral against Blender's view, day (exposure 0.5006) | mean **0.444/255**, 99th percentile **0.91/255** (largest 1.9/255) | 1/255, 3/255 | PASS |
| 2. Same, night with five lamps (exposure 0.5659) | mean **0.319/255**, 99th percentile **1.21/255** (largest 4.9/255) | 1/255, 3/255 | PASS |
| 3. Adaptation after the sun drops to a sixteenth | no overshoot (0.000%); 0.431% off at 2.5 s (exposure 0.5006 → 0.9881 over 1,184 frames) | 0.5%, 2% | PASS |

Reported: our AgX approximation against Blender 5.2's AgX: mean 11.48/255
(day) and 6.64/255 (night).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `522d5f7`; binary
`experiments/room3d/src/bin/m17_gate2.rs`, stub committed in the commit
before; output `docs/results/m17-gate2-run.txt`), after three minutes with
no game or indexing running and one more minute idle.

## What changed from gate 1

Gate 1 failed on cost only (0.168 ms). The meter's histogram now takes
every other pixel in each direction (a quarter of the reads and workgroup
atomics); the CPU's rule in the gate and in the GPU test
(`auto_exposure_shows_the_middle_at_the_key`) takes the same pixels and
matches the display's exposure to 1e-3. Tone map and adaptation are
unchanged.
