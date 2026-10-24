# M4 gate 6: hardware ray queries for camera rays

Date: 2026-10-24. Flying above the boulder field in the viewer was unplayable:
from 150 m up, a 1080p frame took about 500 ms. Based on M4 gates 2–4, I
expected camera rays to be the bottleneck. The M4's ray-tracing hardware is
reachable through wgpu's experimental ray queries, so camera rays now go
through a hardware visibility pass where supported:

- one bottom-level structure per geometry, one top-level instance per object
  with its transform and index;
- it writes the same visibility buffer as the compute pass, so shading is
  unchanged;
- the compute hierarchy remains the fallback.

Pass criterion, fixed before the first run (committed in 3694c9a before
implementing): 1,000,000 boulders, per-pixel prefiltered cones, best of 5.

1. **Agreement.** At 1920×1080 from 150 m up, within 0.5% relative RMSE of
   the compute-ray render, and at least 99.5% of pixels within 1e-3 of peak.
2. **Speed.** At 1920×1080 from 150 m up, at least 4× faster than compute rays.
3. **Playable.** At 1280×720, every viewpoint (standing; 10, 40 and 150 m up)
   in at most 16.7 ms.

**Verdict: FAIL on all three.** 1.31% RMSE (99.77% of pixels close), 0.9× the
speed, and 67–270 ms per 720p frame.

## Results

Apple M4 (MacBook Air, on battery). Reproduce with
`cargo run --release -p room3d --bin m4_gate6`.

| 1280×720 viewpoint | Hardware rays | Compute rays |
|---|---|---|
| standing | 66.6 ms | 55.6 ms |
| 10 m up | 139.1 ms | 144.8 ms |
| 40 m up | 249.0 ms | 224.1 ms |
| 150 m up | 270.5 ms | 265.6 ms |

At 150 m up and 1920×1080: compute rays 464.4 ms, hardware rays 504.2 ms.

## Why it failed: the bottleneck was never the rays

The earlier diagnostics that blamed camera rays (M4 gates 2 and 4) were wrong.
To time "rays only" they used the light with *both* edges reversed, which
leaves its emitting side unchanged. The light was never dark, so the "rays
only" frames were full frames. Redone correctly (one edge reversed, so the
light faces away and no cone runs):

| 1280×720, 1,000,000 boulders | Camera rays | Sun cones |
|---|---|---|
| standing, compute rays | 7.9 ms | 41.0 ms |
| standing, hardware rays | 21.9 ms | 41.5 ms |
| 150 m up, compute rays | 21.5 ms | 230.6 ms |
| 150 m up, hardware rays | 23.9 ms | 230.8 ms |

- **The per-pixel sun cones dominate.** Camera rays are cheap after
  nearest-first traversal (gate 3).
- **Hardware rays are not faster than compute rays here.** Part of that is the
  extra pass the split adds.
- **Prefiltering costs more than it saves from above:** 190.5 ms exact against
  254 ms prefiltered. It projects every visited node's box.

Further variations, all from 150 m up at 720p with exact cones:

| Variation | Sun-cone time |
|---|---|
| 0.5° sun | 163.9 ms |
| 0.05° sun | 169.0 ms |
| 5° sun | 320.7 ms |
| 10,000 boulders | 94.0 ms |

A cone's cost barely depends on what is inside it: it is a fixed cost per
cone. The prime suspect is the bin weights. Every cone computes 64 bins × 4
edge integrals (256 `atan2`) even when nothing blocks it; M2 gate 3's profile
flagged the same loop.

## What remains useful

- Hardware ray queries work through wgpu on Metal, and agree with compute rays
  to 1.3% (the differences are triangle-edge and grazing pixels). The path
  stays available (`set_hardware_rays`; `MaskConeKernel::has_hardware_rays`).
- The corrected diagnostic names the real target: a cone's fixed cost, and
  beyond that, not running a cone per pixel per frame at all. That is the
  lighting cache's job.
