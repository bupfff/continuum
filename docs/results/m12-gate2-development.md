# M12 gate 2: deforming geometry within the frame's budget (development only)

**Not run.** Development at full size failed criterion 1 (cost) by a wide
margin. The rest works.
- **Exact again once stopped:** the cache is exact after the blob stops
  (no texel differing).
- **Exact mode:** with `deforming_budget` 0 it stays exact.
- **Error while deforming:** small around the blob.

Decided by the user (2026-11-15): lighting near a deforming object may lag
its pose within the frame's budget, its shape current every frame, the lag
reported (`CacheOptions::deforming_budget`, 512 texel–light cones a
frame; 0 keeps M12 G1's exactness).

## Method (as kept)

- **One light at a time catches up** with the latest pose, a slice of its
  shadows a frame; the lights take turns.
  - The cache is told, for that light only, of the move since the light
    last caught up: per light the box then and now (`vcache::Moved`).
  - A slice is the space between two planes across the longest axis of
    the light's shadow region at the catch-up's start. The first and last
    slices are open-ended, so every texel lies in one.
  - The object's own texels lie inside its box, hence inside the shadow
    region, so they re-bake texel by texel with the slices, not as whole
    pages at once.
- **Slices:** a catch-up takes enough slices to fit the budget, from the
  re-bakes measured two frames on. At most 16 frames of lag over all
  lights (`DEFORMING_MAX_LAG`), so at most five slices with three lights.
- **The next catch-up** starts from the union of the object's boxes over
  the last one's slices, the poses its texels were baked against.
- **Catch-up after stopping:** a light lags only while its last update
  predates the latest pose, and lagging lights catch up after the object
  stops.
- **Reported:** `InstancedRenderer::deforming_lag`.
- **GPU test:** `deforming_geometry_catches_up`. After a pose and catch-up
  the image is within 1.2e-9 relative RMSE of a renderer built with that
  pose; the rest pose is 0.40 from it.

## Development results

| Run | Deforming / still (mean) | 99th percentile | Error around the blob | Caught up | Exact mode |
|---|---|---|---|---|---|
| 10,000 boulders, budget 1,024, light turns | 1.33× | 39.0 ms (still 41.0) | ≤ 1.0% (3-radius box) | lag never ended (fixed: lag counted against poses) | exact |
| 10,000 boulders, budget 512, light turns | 1.24× | 40.8 ms (still 46.2) | 2.6–4.5% (1.5-radius box) | exact | exact |
| 10,000 boulders, slices | 1.21× | 37.2 ms (still 38.3) | 0.04–4.7% | exact | exact |
| Full size, blob 14 m, light turns | 1.70× | 84.1 ms (still 43.9) | 2.5–4.8% | exact | exact |
| Full size, blob 14 m, slices | 1.95× | 75.7 ms (still 42.1) | 0.15–2.86% | exact | exact |
| Full size, slices, own texels sliced too | 24.3 ms mean (one run) | 74.4 ms | — | — | — |
| Full size, slices, 48-frame lag cap | 20.7 ms mean (1.6×) | 71.5 ms | — | — | — |
| Full size, the cache never told (geometry alone) | 14.6 ms (still 12.8) | 42.0 ms | — | — | — |

M12 G1's exact mode at full size: 108–114 ms means (9.6–10.1×).

**What is left:**
- **Catch-ups are large.** One light's catch-up re-bakes tens of thousands
  of texel–light cones, mostly fine ground pages near the camera in the
  lamp's shadow region of a 2 m object (and boulders there). Five slices
  still put thousands of cones in a frame.
- **Spikes:** a 99th percentile of 71–84 ms remains. It does not follow
  the counted re-bakes, which arrive two frames late in the statistics;
  the cause is not yet established.
- **The geometry's own cost is small:** about 1.7 ms of the frame (refit,
  hardware structure, uploads).

**Levers not yet tried:**
- receivers' lighting a level coarser while an object deforms, sharpening
  when it stops;
- a deforming budget in time rather than cones;
- profiling the spikes on the GPU (Instruments).
