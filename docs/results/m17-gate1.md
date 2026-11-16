# M17 gate 1: the display's tone map and auto-exposure

**FAIL on criterion 1 (cost); criteria 2 and 3 pass.**

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Cost: auto-exposure's added warm frame (day view) | **0.168 ms** (2.862–2.917 against 2.695–2.738 ms, five rounds) | at most 0.15 ms | FAIL |
| 2. PBR Neutral against Blender's view, day (exposure 0.5002) | mean **0.423/255**, 99th percentile **0.92/255** (largest 1.9/255) | 1/255, 3/255 | PASS |
| 2. Same, night with five lamps (exposure 0.5654) | mean **0.319/255**, 99th percentile **1.21/255** (largest 4.9/255) | 1/255, 3/255 | PASS |
| 3. Adaptation after the sun drops to a sixteenth | no overshoot (0.000%); 0.432% off at 2.5 s (exposure 0.5002 → 0.9890 over 1,172 frames) | 0.5%, 2% | PASS |

Reported: our AgX approximation against Blender 5.2's AgX: mean 11.50/255
(day) and 6.63/255 (night), 99th percentiles 15.6 and 16.8/255 (it
approximates the original AgX, not Blender's).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `51752ee`; binary
`experiments/room3d/src/bin/m17_gate1.rs`, stub committed in `46ea75d`;
output `docs/results/m17-gate1-run.txt`), after three minutes with no game
or indexing running and one more minute idle.

## Method

DESIGN §3.4, "Display": the display pass maps linear light with Khronos
PBR Neutral to an `Rgba8UnormSrgb` target; each view's exposure is the
meter's rule computed on the CPU, and Blender 5.2 applies its own view
transforms to the same linear image times that exposure
(`cycles/view_transform.py`). The meter: a histogram of every pixel's
log2 luminance (64 half stops, workgroup atomics), a one-invocation
resolve, an exponential adaptation (0.5 s).

## Why it is slow

In development at 640×360 the meter cost 0.025 ms; at 1280×720 it reads
four times the pixels (15 MB of colours) and adds each to a shared
workgroup histogram, where a smooth image's pixels contend for the same few
bins. The exposure's statistic needs far fewer samples.

## Next

Gate 2: the histogram takes every other pixel in each direction (a quarter
of the reads and atomics).
