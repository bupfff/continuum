# M4 gate 11: baking to a frame-time target

Date: 2026-10-24. M4 gate 10 took the page manager's walk off the CPU. Its
flight was then bound by two things:

- **GPU work per frame.** A fixed 8,000-cone bake does not fit a frame-time
  target on a hot machine.
- **Eviction rounds.** Every 32 frames, up to 32,768 pages were freed at once.

## What changed

- **The bake budget follows the frame time.** This is the frame budget of
  the quality contract (DESIGN §3.3). Each frame measures its period P and
  its wait for the GPU, against a target T = 14 ms:
  - if the CPU waited more than 1 ms and P exceeds T, the budget scales by
    T / P;
  - if P is under T, it grows by 10%;
  - otherwise it holds.

  It stays between 1,000 and 20,000 cones and starts at 8,000. These are now
  the defaults: `CacheOptions::frame_target_ms` = 14, `bake_budget` = 20,000
  (the maximum).
- **Eviction is spread over frames.** Pages unused for 64 frames are queued,
  least recently used first, when slot use arrives. At most 2,048 are freed
  per frame.

## Pass criterion

Fixed before the first run, in e6eed5d. It is the same as gate 10's:

1. **Same images.** Warm, within 0.2 points of gate 8's relative RMSE against
   per-pixel exact cones.
2. **Speed.** Warm, at most 16.7 ms per frame.
3. **Flight.** Gate 8's flight from an empty cache, three runs. In every run:
   - mean frame at most 16.7 ms;
   - 99th-percentile frame at most 33.3 ms;
   - mean fallback at most 10%.

**Verdict: PASS.**

| Criterion | Result |
|---|---|
| Same images | identical to gate 8 |
| Warm frames (hardware rays) | 6.5–8.0 ms |
| Flight mean | 14.6–15.0 ms |
| Flight 99th percentile | 25.9–32.1 ms |
| Flight fallback | 1.7% |

## Results

Apple M4 (MacBook Air), 999,986 boulders, 1280×720. Reproduce with `cargo run
--release -p room3d --bin m4_gate11`.

| Rays | Viewpoint | Frames until warm | Warm frame | RMSE vs per-pixel exact (gate 8) |
|---|---|---|---|---|
| compute | standing | 1,295 | 20.23 ms | 14.49% (14.49%) |
| compute | 10 m up | 2,402 | 38.21 ms | 25.86% (25.86%) |
| compute | 40 m up | 2,279 | 45.34 ms | 30.88% (30.88%) |
| compute | 150 m up | 1,266 | 38.18 ms | 38.80% (38.80%) |
| hardware | standing | 102 | 8.03 ms | 14.49% (14.49%) |
| hardware | 10 m up | 382 | 6.46 ms | 25.87% (25.87%) |
| hardware | 40 m up | 515 | 7.76 ms | 30.88% (30.88%) |
| hardware | 150 m up | 251 | 7.52 ms | 38.80% (38.80%) |

- **Compute rays.** Compute frames cannot reach the 14 ms target, so their
  budget sits at the 1,000-cone floor and warm-up takes 1,266–2,402 frames.
  The controller behaves as designed. Hardware rays are the faster mode and
  decide the gate.

Flights (hardware rays, 600 frames from an empty cache):

| Run | Mean frame | 99th percentile | Worst (frame) | Fallback | CPU planning mean (max) | Waiting for the GPU | Samples at their wanted mip | Bake budget mean (range) |
|---|---|---|---|---|---|---|---|---|
| 1 | 14.59 ms | 30.15 ms | 62.04 ms (513) | 1.71% | 2.82 ms (53.78) | 9.62 ms | 0.6% | 2,415 (1,000–20,000) |
| 2 | 15.01 ms | 32.07 ms | 84.36 ms (2) | 1.70% | 2.61 ms (16.13) | 10.21 ms | 0.6% | 2,140 (1,000–20,000) |
| 3 | 14.75 ms | 25.94 ms | 70.88 ms (2) | 1.69% | 2.03 ms (33.44) | 11.16 ms | 0.6% | 2,319 (1,000–20,000) |

Flight 1 over 100-frame stretches:

| Frames | Frame | Waiting for the GPU | Bake budget |
|---|---|---|---|
| 0–99 | 12.9 ms | 9.1 ms | 8,668 |
| 100–199 | 15.2 ms | 10.4 ms | 1,062 |
| 200–299 | 13.5 ms | 8.8 ms | 1,211 |
| 300–399 | 15.4 ms | 10.6 ms | 1,105 |
| 400–499 | 16.0 ms | 10.7 ms | 1,240 |
| 500–599 | 14.3 ms | 8.1 ms | 1,202 |

## What this means, with the caveat that matters

- **Flying over a million boulders at 720p now holds about 67 fps on a
  fanless Air:**
  - 14.6–15.0 ms mean frames;
  - 99th-percentile frames under 33 ms;
  - with exact soft sun shadows once the view settles.

  Warm views take 6.5–8 ms.
- **But the flight passes by baking very little while moving.** After the
  first 100 frames, the budget sits near its 1,000-cone floor. On this hot
  machine, rendering alone takes 12–15 ms of the 14 ms target.
  - Only 0.6% of sampled pixels had a page of their wanted mip during the
    flight. Most pixels are lit from coarser pages: shadows are softer and
    blurrier while flying.
  - Pixels with no page at all stay rare (1.7%), because coarse pages come
    first.
  - Lighting sharpens once the camera slows or stops (100–515 frames to
    fully warm from empty). This is the same trade texture streaming makes.

  The criterion did not measure sharpness in motion. A later gate should,
  against the area-integrated image.
- **Warm frames.** They were slower than in gate 10 (6.5–8.0 vs 3.6–6.6 ms),
  partly from the machine's heat and partly from the controller raising the
  bake budget while the view refines.
- **What would raise sharpness in motion:**
  - cheaper rendering, leaving more of the frame for baking (anti-aliased
    primary cones will change this cost, roadmap item 4);
  - cheaper cones per texel (bake throughput is about 0.6–1 µs per cone);
  - a higher frame target on faster GPUs. The controller adapts to whatever
    the machine allows.

This completes roadmap item 1 (the view-dependent lighting cache) for the
static case. Movers in the instanced path, and a sky, are later work.
