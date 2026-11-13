# M4 gate 13: the error while flying, reported each frame

**PASS.** With `CacheOptions::error_probes`, the renderer reports each frame
an estimate of its image's relative RMSE and an upper bound on it
(`CacheStats::error_estimate`, `error_bound`). On M9's flight over the
million-boulder field the bound held the error measured against settled
renderers in every check, and it costs 2% of the flight.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 1. Cost: mean frame with probes, three pairs interleaved | 9.81 ms against 9.59 ms (**1.023×**) | at most 1.05× | PASS |
| 2. Calibration: bound at least the measured error | **10 of 10** checks | at least 9 of 10 | PASS |
| 3. Tightness: mean estimate over mean measured | 25.25% against 20.78% (**1.215×**) | 0.8–1.3× | PASS |

| Frame | Measured (against settled) | Estimate | Bound | Settled reference's own estimate | Probes on surfaces |
|---|---|---|---|---|---|
| 60 | 19.00% | 23.10% | 28.45% | 15.19% | 293 |
| 120 | 26.12% | 30.20% | 35.25% | 15.55% | 332 |
| 180 | 25.97% | 25.98% | 29.42% | 14.25% | 373 |
| 240 | 25.24% | 27.45% | 31.27% | 16.90% | 411 |
| 300 | 25.20% | 26.93% | 30.14% | 18.64% | 461 |
| 360 | 23.38% | 24.52% | 27.76% | 22.71% | 502 |
| 420 | 22.70% | 25.94% | 29.62% | 16.37% | 512 |
| 480 | 15.98% | 22.18% | 25.08% | 18.36% | 512 |
| 540 | 13.87% | 23.19% | 25.78% | 20.79% | 512 |
| 600 | 10.34% | 22.99% | 26.17% | 21.75% | 512 |

Flights for criterion 1 (default options): without probes 9.57, 9.60, 9.61
ms; with 9.94, 9.58, 9.91 ms. Camera passes on the GPU 3.82 → 4.84 ms: the
probe passes take about 1.0 ms of GPU time; this flight is not GPU-bound,
so the frame grows by about 0.2 ms.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `f382061`; implementation and binary `89fab5a`,
before the run), after three minutes idle. M9's flight (600 frames from
standing height to 150 m, pitching down 45°) with sun and sky (sky through
rays), 512 probes; criteria 2–3 at gate 12's fixed budget (11,000 units a
frame), every 60th frame against a renderer settled at that camera (16
quiet frames).

## Method

- **Probes:** 512 pixels a frame on an R2 pattern moved each frame.
- **Exact shares** (`hw_vc_error_shares`, a workgroup per probe): at the
  pixel's own surface point, each light's share is the visible part of its
  irradiance over an 8×8 grid of points on it (one hardware ray each); the
  sky's is the bake's (the projected solid angle of the hemicube's 192 bins
  whose ray meets nothing, three a lane).
- **Shading** (`vc_error_probes`): the shading's page lookup and its light
  and sky terms were factored out of `vc_hit_color` (`vc_find_shown`,
  `vc_light_terms`); a probe's exact color is the shown color less the
  difference those terms make with the page's shares and with the exact
  ones (everything else in the color is the same either way).
- **Estimate:** over the probes, `sqrt(3·mean d) / mean s` (`d`: squared
  difference summed over channels, `s`: exact value summed over channels;
  as `relative_rmse`), and a one-sided 95% bound adding 1.645 standard
  errors of `d`'s mean. Probes on the sky or a light count with no error.

**Disclosed (before the run):** the stub computed the shares with the
bake's light cone and the sky's rays one probe an invocation; in
development that cost 6.6 ms of GPU time a frame (1.37×), latency-bound
(serial cone walks and 192 serial rays with 512 invocations). Hardware rays
spread over a workgroup gave estimates within 0.1 point of the cone version.
Development (`M4_DEV=1`, 100,000 boulders, 200 frames): 1.035×, 10 of 10,
1.199×.

## What the numbers say

- **The report measures error against lighting exact at each pixel**, so it
  includes the settled cache's own error: 14–23% here (each settled
  reference's own estimate), the blur of shadows finer than its texels
  (pixels want texels about two pixels across). That matches earlier
  measurements of the cache against per-pixel cones (M4 G8, M5 G7). The
  estimate tracks the error while flying where that dominates (frames
  120–420: 0.99–1.18×) and sits near the settled floor as the flight slows
  into its last view (frame 600: 22.99% against 10.34% from settled).
- **Bounded, in the statistical sense:** a 95% bound from 512 probes held
  in every check; it is not a certificate. A certified bound per pixel would
  need pages whose texels hold area means (DESIGN §3.3).
- **Not probed:** bounce light (both shadings take the page's), movers
  (their own structure), and blended objects' partial shadows (the rays are
  opaque).

## Caveats

- One flight on one scene. Off by default (about 1 ms of GPU time a frame
  at 512 probes, which matters where frames are GPU-bound, as M14's are).
