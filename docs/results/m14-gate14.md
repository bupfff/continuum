# M14 gate 14: several bounces with streamed lighting

**FAIL on criterion 3, at one view, with a disclosed confound.** Eight
bounces of light on 36,864 streamed rocks with a sky fly within 60 fps, and
direct light stays certified; but at flight 1's last view the image after
8,000 frames was 1.038% from the all-resident renderer's against a 1% bar,
neither renderer having settled there. During criteria 2–3 the machine was
not left alone (see Caveats).

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. (b) Mean frame | 9.30 ms | 9.99 ms | 9.64 ms | at most 16.7 ms | PASS |
| 1. (b) 99th percentile | 15.28 ms | 14.97 ms | 14.81 ms | at most 16.7 ms | PASS |
| 1. (b) Worst frame | 17.98 ms | 18.30 ms | 19.33 ms | at most 33.3 ms | PASS |
| 2. (a) Sun / sky pages in flight, differing | 68,362, 0 | 79,168, 0 | 54,740, 0 | none (1e-5) | PASS |
| 2. (a) Settled shown pages, differing | 12,515, 0 | 1,533, 0 | 6,098, 0 | none | PASS |
| 2. Pages compared in all | 222,416 sun and 222,416 sky | | | at least 100,000 and 30,000 | PASS |
| 3. (a) Final image's mean difference | 0.027% | **1.038%** | 0.583% | at most 1% | **FAIL** |
| Reported: one bounce's final image from 8 bounces' | 0.199% | 0.305% | 0.297% | | |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `f7e8b36`; binary `5735888`, before the run), after
the field's build (139 s) and two minutes idle. Gate 10's scenes and
criteria with `CacheOptions::bounces` 8.

## Details

- Flights (b): bake budget 1,158–4,894 units on average (at its cap),
  11.6–48.0 pages a frame; lighting levels 38,240–88,497 loads and
  37,712–130,853 produced a flight (gate 10 with one bounce: 44,528–81,404
  and 35,695–122,528). Frames over 16.7 ms: 6, 5, 2.
- Settling (a): at all three last views neither renderer settled within
  8,000 frames (gate 10, one bounce: flights 0 and 2 settled in 8,000 and
  6,026 frames streamed, 5,592 and 4,504 resident; flight 1 neither). So
  criterion 3 compared images still converging.
- Outdoors, light past the first bounce changes these views by 0.2–0.3%,
  less than the streamed and resident renderers' difference before they
  settle.

**Disclosed (before the run):** the development mode (2,304 and 576 rocks)
passed all three (flights 7.7–9.5 ms, 240,108 pages identical, final images
0.047–0.061% apart).

## Caveats

- **The machine was not left alone for criteria 2–3.** After criterion 1
  (timed) I ran another gate's development runs (GPU and CPU work) while
  criteria 2–3 ran. The cache's bake budget follows GPU time by default
  (`frame_target_ms` 14), so that load shrank both renderers' budgets and
  slowed their convergence in frames; criterion 3 compares the images at a
  frame cap before either settled. Gate 10's flight 1, with one bounce and
  the machine free, was 0.577%. Criteria 1 and 2 do not depend on it
  (criterion 1 ran before; criterion 2 is an equality check of baked pages).
- Per the gate workflow this is not re-run. A new gate would run criterion 3
  at a fixed bake budget with the machine free (as M14 G13 did for G12).
