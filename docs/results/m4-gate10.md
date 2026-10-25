# M4 gate 10: smooth flight through the lighting cache

Date: 2026-10-24. M4 gate 9 rendered warm views of the million-boulder field
in 2.9–4.7 ms at 720p. Its flight was bound by the CPU page manager: each
frame it re-derived a backlog of about 50,000 wanted pages by walking from up
to 32,000 request tiles to coarser pages.

## What changed

The cache's pages, mips, baking and shading are unchanged.

- **Progressive requests from the shader.** The shader already walks from the
  wanted mip to the finest resident one. A sample pixel now requests the page
  one mip finer than the one it found, or the coarsest page when none is
  resident.
  - Priority: pixels with no page first, then the largest gap between found
    and wanted mip.
  - The CPU no longer walks.
- **Requests deduplicated on the GPU.** A per-frame set of 32-bit key
  fingerprints (atomic compare-exchange) feeds a compacted list, so the CPU
  sees each page once per frame.
- **Page table deletion without tombstones** (Knuth's Algorithm R for linear
  probing), so the table never needs rebuilding. A randomized test checks it
  against the shader's lookup rule.
- **Eviction capped** at 32,768 pages per round, least recently used first.

The last two were added after development runs on a 20,000-boulder field,
before the first official run. There, a table rebuild had caused an 86 ms
spike.

## Pass criterion

Fixed before the first run, in cfdbd8d:

1. **Same images.** Warm, each viewpoint is within 0.2 points of gate 8's
   relative RMSE.
2. **Speed.** Warm, each viewpoint takes at most 16.7 ms per frame.
3. **Flight.** Gate 8's flight from an empty cache, three runs in a row. In
   every run:
   - the mean frame is at most 16.7 ms;
   - the 99th-percentile frame is at most 33.3 ms;
   - fallback is at most 10%.

**Verdict: FAIL, on the flight.** Same images and warm speed pass. All three
flights miss:

| Measure | Limit | Result |
|---|---|---|
| Mean frame | 16.7 ms | 17.6–17.9 ms |
| 99th percentile | 33.3 ms | 38.3–38.6 ms |
| Fallback | 10% | 1.6% (pass) |

## Results

Apple M4 (MacBook Air), 999,986 boulders, 1280×720, budget 8,000 cones per
frame. Reproduce with `cargo run --release -p room3d --bin m4_gate10`.

| Rays | Viewpoint | Frames until warm | Warm frame | RMSE vs per-pixel exact (gate 8) |
|---|---|---|---|---|
| compute | standing | 180 | 17.59 ms | 14.49% (14.49%) |
| compute | 10 m up | 304 | 37.26 ms | 25.86% (25.86%) |
| compute | 40 m up | 292 | 48.66 ms | 30.88% (30.88%) |
| compute | 150 m up | 170 | 41.56 ms | 38.80% (38.80%) |
| hardware | standing | 180 | 3.64 ms | 14.49% (14.49%) |
| hardware | 10 m up | 304 | 5.00 ms | 25.87% (25.87%) |
| hardware | 40 m up | 292 | 6.61 ms | 30.88% (30.88%) |
| hardware | 150 m up | 170 | 6.57 ms | 38.80% (38.80%) |

Flights (hardware rays, 600 frames from an empty cache):

| Run | Mean frame | 99th percentile | Worst (frame) | Fallback | CPU planning mean (max) | Waiting for the GPU, mean | Samples at their wanted mip |
|---|---|---|---|---|---|---|---|
| 1 | 17.69 ms | 38.64 ms | 81.03 ms (2) | 1.60% | 1.92 ms (26.28) | 14.41 ms | 47.2% |
| 2 | 17.63 ms | 38.30 ms | 78.74 ms (225) | 1.60% | 1.92 ms (49.76) | 14.39 ms | 47.3% |
| 3 | 17.89 ms | 38.46 ms | 61.55 ms (2) | 1.60% | 1.81 ms (24.51) | 14.81 ms | 47.2% |

Per frame of flight 1, averaged over each 100 frames:

| Frames | Frame | Waiting for the GPU | Texels baked | Pages requested |
|---|---|---|---|---|
| 0–99 | 12.6 ms | 10.5 ms | 7,487 | 10,826 |
| 100–199 | 21.9 ms | 17.6 ms | 7,983 | 21,762 |
| 200–299 | 19.3 ms | 15.0 ms | 7,960 | 23,882 |
| 300–399 | 22.4 ms | 18.6 ms | 7,960 | 23,712 |
| 400–499 | 16.4 ms | 13.3 ms | 7,983 | 15,756 |
| 500–599 | 13.4 ms | 11.5 ms | 4,453 | 1,790 |

## What this means

- **The CPU problem is solved.** Planning fell from 11.7 ms per frame (gate
  9's flight) to 1.8–1.9 ms. The CPU now waits for the GPU for 14.4–14.8 ms of
  each frame.
- **The flight is now bound by GPU work per frame.** Rendering plus baking up
  to 8,000 cones takes 11.5–18.6 ms on average over 100-frame stretches. A
  fixed cone budget cannot hold a frame-time target, since the cost per cone
  varies with what the cones see and with the machine's heat.
- **Thermal state.** This run was hot. Warm frames for identical work were
  about 25% slower than in gate 9's run: compute at 150 m up 41.6 ms vs 29.6
  ms; hardware 6.6 ms vs 4.0 ms. Gate 9's 12.8 ms flight (CPU-bound) and this
  17.7 ms flight (GPU-bound) are not directly comparable. Each says which side
  was the bottleneck.
- **Progressive refinement trades sharpness during flight for CPU time.**
  47% of sample pixels found their wanted mip during the flight; the rest
  used coarser pages.

## Next

Bake by time, not by count: the frame budget of the quality contract (§3.3).
Each frame's bake budget would follow the measured GPU frame time, so busy
frames shrink their bake and coarse pages keep fallback low. Fallback here was
1.6% against the 10% allowance, which leaves room to bake less per frame.
This is the remaining step for smooth flight, and its own gate.
