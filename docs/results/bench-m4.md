# Benchmark: Apple M4 baseline

The benchmark (`experiments/room3d/src/bin/bench.rs`, `docs/BENCHMARK.md`)
on the dev machine, as the reference for other GPUs.

Apple M4 (fanless MacBook Air), macOS, Metal; ray queries, timestamps and
subgroups available. 1280×720, release build, 2026-11-07, commit after
`44fb92f`. The machine was warm: it had run gates for hours, though
nothing else ran during the benchmark. On this machine timings move by 2×
or more with heat (CLAUDE.md), so treat them as sustained, not best-case.

| Part | Result |
|---|---|
| 1. Field flight, sun (3 runs) | mean 9.59–9.68 ms, p99 15.70–16.68 ms, fallback 1.6% |
| 1. Field flight, sun and sky | mean 13.23–13.32 ms, p99 18.95–19.46 ms, fallback 82.9% (mostly a guessed sky, below) |
| 1. Field flight, three colored lights | mean 14.74–14.95 ms, p99 28.93–31.78 ms, fallback 1.8% |
| 2. Warm frame, sun and sky: standing / 10 m / 40 m / 150 m | 2.84 / 3.47 / 3.71 / 3.61 ms (warm after 851 / 3,179 / 7,973 / 5,040 frames) |
| 3. Sky warm-up, standing, budget 11,000 | 1,294 frames (12.2 s) with the sky later; 1,824 (18.9 s) without |
| 4. Textured scene, materials and bounce light | warm frame 4.86 ms (settled in 589 frames) |
| 5. Silhouette rays, field from 150 m: one ray / four / two quarters | 3.63 / 11.15 / 8.39 ms (3.07× and 2.31×; 51.1% edge pixels) |

**Fallback with sun and sky.** With the sky later (on by default since M5
gate 9), a pixel whose page's sky has not been baked yet shows the open sky
over its normal, with exact shadows, and counts as fallback. In this flight
most pixels do. M5 gate 5's flights, before the sky later, showed 4.5%
fallback, but lit those pixels from coarser pages. M9 gate 5's development
flights found the same error against the settled image either way
(`m9-gate5.md`).
