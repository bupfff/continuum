# M7 gate 3: anti-aliasing at silhouettes only

**FAIL on both criteria.** Four rays at edge pixels halve the error on the
boulder field from 150 m up, but cost 13× there. On the textured scene
edges are rare and the gain is small.

| View | Error, edge rays | Error, one ray | Ratio (at most 0.75×) | Warm frame | Ratio (at most 1.5×) | Edge pixels |
|---|---|---|---|---|---|---|
| Textured scene | 3.53% | 3.88% | 0.91× FAIL | 4.15 ms against 2.50 ms | 1.66× FAIL | 1.0% |
| Field, 150 m up | 10.51% | 21.32% | **0.49×** PASS | 83.10 ms against 6.28 ms | 13.24× FAIL | 51.1% |

Apple M4, 1280×720, cache settled, 2026-11-07, run once (binary
`experiments/room3d/src/bin/m7_gate3.rs`, criteria in `e0f4643`). The
reference is the cached image at 16×16 sub-pixel offsets, each settled,
with textures for 1/16 of a pixel.

## Method

`CacheOptions::edge_rays` (off by default). A pixel is an edge if one of
its four neighbours sees another object, or a depth 5% away. An edge pixel
averages four compute rays through its 2×2 quarters, each shaded from the
cache with textures read for half a pixel.

## What it shows

- **On dense geometry, silhouettes are the error.** From 150 m, half of
  the field's pixels are edges, and resolving them halves the error from
  the area integral.
- **But compute rays through a million objects are slow.** Four per edge
  pixel cost 13×. The camera's own rays use hardware ray queries, which
  the shading pass cannot use.
- **On the textured scene** the remaining error is inside surfaces
  (shading and texture detail), not at its 1% of edges. The 1.66× is
  surprising for so few edge pixels; it may be the edge test's reads of
  the visibility buffer, unconfirmed.

## What next

Silhouette rays through hardware ray queries (a separate pass over a list
of edge pixels), and a cap on edge rays per frame. That is for a new gate.
