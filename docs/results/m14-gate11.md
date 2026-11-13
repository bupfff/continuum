# M14 gate 11: the frame-period cap times the renderer's own frames

**PASS.** The cap on the bake budget (M14 gate 9) now measures how long the
renderer's frame call takes, waiting for the GPU included, instead of the
gap between calls. Reading the image back every 20th frame no longer slows
a view's warm-up (0.96–1.05× an undisturbed one, against up to 4.5× in
development before), and gate 10's flights still hold the frame budget.

| Criterion | Flight 0 / view 1 | Flight 1 / view 2 | Flight 2 / view 3 | View 4 | Bar | Result |
|---|---|---|---|---|---|---|
| 1. Mean frame | 9.15 ms | 9.90 ms | 9.41 ms | | at most 16.7 ms | PASS |
| 1. 99th percentile | 15.02 ms | 15.32 ms | 14.42 ms | | at most 16.7 ms | PASS |
| 1. Worst frame | 16.98 ms | 18.06 ms | 17.61 ms | | at most 33.3 ms | PASS |
| 1. Frames over 16.7 ms (reported) | 1 | 1 | 2 | | | |
| 2. Frames to warm, reading back every 20th / undisturbed | standing 179 / 171 (1.05×) | 10 m 359 / 357 (1.01×) | 40 m 405 / 415 (0.98×) | 150 m 264 / 274 (0.96×) | at most 1.25× | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `879547e`; implementation and binary `a246390`,
before the run). Criterion 1 is gate 10's timed field (36,864 unique rocks,
camera and lighting levels on demand, a uniform sky and bounce light,
default cache options), built in 142 s, then two minutes idle; gate 10 gave
means 9.2–10.2 ms and 99th percentiles 14.5–15.0 ms. Criterion 2 is M5
gate 13's field and viewpoints (sun and sky through rays, default options).

## Method

`VirtualCache::end_frame`: the renderer times its frame call (from the
camera's upload to the submission and its readbacks started) and the next
frame's budget adapts to that, both the frame-time model's fallback without
timestamps and the frame-period cap. In a GPU-bound frame the call waits for
the readback of two frames before, so it still measures the GPU's pace;
what the caller does between calls (a readback, an engine's own work, a
wait for the display) no longer shrinks the bake budget.

**Disclosed (before the run):** development runs with `sky_dev` (412 and
364 frames reading back against 421 and 374 undisturbed) and the
development mode (flights 8.7–9.0 ms on 2,304 rocks; warm-ups 1.02–1.08×),
listed in the binary.

## Caveats

- The budget no longer leaves room for the caller's own work: an engine
  that spends 8 ms of its own per frame gets the same bake budget as one
  that spends none. The frame-time model's GPU target (`frame_target_ms`)
  still bounds the GPU's share.
- The bake budget in these flights averaged 1,340–4,182 units, as in gate
  10 (1,068–3,486): the flights are GPU-bound, so the call's time and the
  period between calls agree there.

## What the numbers say

- Measuring the renderer mid-flight (as the error while flying needs) no
  longer changes what it does.
