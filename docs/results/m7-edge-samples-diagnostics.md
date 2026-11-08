# Silhouette rays: two quarters against four (diagnostics, not a gate)

Date: 2026-11-07, after M7 gate 4 (which failed on the field's speed:
3.03× one ray's frame against at most 1.5×).

## Where the field's cost goes

Camera passes on the field from 150 m up with four quarters (`edges_dev
field profile`: 640×360, 100,000 boulders; each pass submitted and waited
for alone, medians of 30):

| Pass | Time |
|---|---|
| Hardware visibility | 0.76 ms |
| `hw_edges` | 1.61 ms |
| Main shading | 0.88 ms |
| `vc_shade_edges` | 1.75 ms |

Tracing and shading the quarters cost about the same. About half the
pixels are edges there, so every extra sample costs about one frame's
worth.

## Two quarters

`CacheOptions::edge_samples` = 2: one diagonal pair of quarters,
alternating from pixel to pixel, averaged with the pixel's own ray.

`edges_dev quality`: development at 640×360. The reference is the cached
image at 16×16 sub-pixel offsets, each settled, as M7 gates 3–4.

| View | One ray | Four quarters | Two quarters |
|---|---|---|---|
| Field, 150 m up (100,000 boulders) | 29.10% (1.65 ms) | 11.80% (**0.41×**), 2.62× the frame | 16.26% (**0.56×**), 2.00× |
| Textured scene | 5.43% (1.05 ms) | 4.85% (0.89×), 1.23× | 4.93% (0.91×), 1.17× |

Two quarters sit between one ray and four in both cost and quality. None
reaches 1.5× on the field.

At full size (M7 gate 4), four quarters took the field's frame to
11.08 ms, inside a 60 Hz frame on the fanless Air, at 0.49× one ray's
error.

Kept as an option, not gated: no claim is made beyond these development
numbers. Silhouette rays stay off by default.
