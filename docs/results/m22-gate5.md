# M22 gate 5: the time budget steered by the frame period's median

**PASS** on all three criteria. M14 gate 11's flights hold the frame
budget and its readback warm-up claim holds again (it failed in gate 4's
control); the night scene settles in 0.661× gate 4's frames.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 1. M14 G11's flights: mean / p99 / worst | 10.11 / 14.66 / 19.32 ms; 10.18 / 14.65 / 17.24 ms; 10.10 / 14.53 / 19.50 ms (4, 2, 6 frames over 16.7 ms) | 16.7 / 16.7 / 33.3 ms | PASS |
| 2. M14 G11's warm-up, reading back every 20th / undisturbed | standing 139 / 128 (**1.09×**); 10 m 279 / 265 (**1.05×**); 40 m 299 / 290 (**1.03×**); 150 m 184 / 180 (**1.02×**) | 1.25× | PASS |
| 3. The night scene through `Renderer` (bounce on), frames to settle | **1,038** (8.6 s) against gate 4's 1,570 (**0.661×**) | 0.8× | PASS |

Reported: without bounce the night scene settles in 641 frames (gate 4:
898); the day scene in 283 frames with bounce, 125 without (2.26×); gate
4's new view (materials, far side) in 730 with, 245 without (**2.98×**,
gate 4's bar was 3×). Against M14 gate 11's own run (2026-11-13), means are
0.2–0.9 ms higher (9.15–9.90 ms then), 99th percentiles 14.5–14.7 ms
against 14.4–15.3, and undisturbed views warm in 128–290 frames against
171–415.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once on
2026-11-20 (stub and code `7c96b2c`): M14 gate 11's binary first (the
timed criteria, `docs/results/m22-gate5-m14.txt`), then
`experiments/room3d/src/bin/m22_gate5.rs` (`docs/results/m22-gate5-run.txt`).
Rehearsed before: M14 gate 11's development mode (both criteria pass) and
criterion 3 (1,071 frames).

## Method

DESIGN §3.4, "The time budget steered by the frame period's median": the
budget follows the median of the last three frame periods (a caller
waiting every frame still counts, a lone readback does not), and where
frames with the bake at its floor already overrun the frame budget or
target, frames may take a quarter more than those
(`CacheOptions::floor_overrun`).

## Disclosures

- Bars set with development known (readback warm-up 1.06–1.09× on M14
  gate 11's development scenes; the night scene 1,018–1,096 frames).
- Criterion 3 compares with gate 4's recorded run, not an interleaved one
  in this run; frame counts depend on the time budget, which depends on
  the machine's state.
- The floor's overrun changed little with the median filter in
  development; most of the gain is the filter.
