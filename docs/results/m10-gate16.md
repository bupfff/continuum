# M10 gate 16: alpha blending

**FAIL on speed, narrowly (1.33× against 1.3×); accuracy passes.** Blending
is kept: it makes glass and blinds see-through, 4.71% from Cycles.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 5% of Cycles; over the pixels blending changes, at most 0.5× the error of the panes opaque | **4.71%** (opaque 57.31%); over 206,646 changed pixels **5.55%** against 81.83% (**0.07×**) | PASS |
| 2. Warm frame at most 1.3× the panes opaque | 4.27 ms against 3.20 ms (**1.33×**) | FAIL |

Means: ours +0.98%, opaque +18.43% from Cycles'. Settled in 538 and 537
frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (commit `ddc18b1`; binary
`experiments/room3d/src/bin/m10_gate16.rs`, criteria in `f66a381`).
Cycles: one bounce, 4,096 samples per pixel, 62 s on Metal.

## Method

`InstancedScene::geometry_blend`: an opacity factor, times the base color
texture's alpha.

- **Camera.** With hardware camera rays, the pass after visibility traces
  on through blended surfaces, front faces only, up to four layers. Each
  layer weighs its opacity times what the layers before let through; the
  fourth takes what is left.
- **Compositing.** The cache's shading of the extra-samples list (M7 gate
  4's silhouette pass, generalized) composites the layers front to back. A
  record is five words: up to four samples (object, triangle, distance,
  weight), then the pixel, its own sample's weight, the count, and each
  sample's quarter. A pixel is blended or a silhouette, not both.
- **Shadows and sky.** A blended object covers cones by its opacity
  factor, the texture's alpha averaged out. Its triangles rasterize with
  that weight, its hierarchy's coverage samples are scaled by it, and the
  factor rides in the transform's fourth row.
- **Not covered.** Bounce and reflection rays see blended surfaces opaque.
  Without hardware rays, or with movers, blended surfaces fall back to the
  old behavior (textured: cut at 0.5; untextured: opaque).

Scene (`Textured::with_glass`): M6's textured scene with a blue glass pane
(opacity 0.35) and a banded blind (alpha 0.15 and 0.85), double-sided as
glTF's import makes them. Cycles renders the panes single-sided: it shades
both sides of a face, and coincident faces would attenuate twice.

## The cost

About 22% of the pixels see a pane: 1.07 ms more a frame, from tracing and
shading their layers. A cheaper path is for a new gate:

- skip the extra pass's per-pixel test where no blended object is in view;
- shade layers at a coarser texture level;
- keep the pass coherent by listing pixels in screen order.
