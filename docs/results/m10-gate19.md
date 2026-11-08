# M10 gate 19: alpha blending's speed

**PASS.** Blending now costs 1.23× the frame with the panes opaque (gate
16: 1.33×), at gate 16's accuracy.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Median over five interleaved rounds of the warm-frame ratio, blended to panes opaque, at most 1.3× | **1.226×** | PASS |
| 2. Within 5% of Cycles; over the pixels blending changes, at most 0.5× the error of the panes opaque | **4.70%** (opaque 57.31%); over 206,780 changed pixels **5.50%** against 81.44% (**0.07×**) | PASS |

| Round | Panes opaque | Blended | Ratio |
|---|---|---|---|
| 0 | 3.25 ms | 3.98 ms | 1.222× |
| 1 | 3.21 ms | 3.98 ms | 1.239× |
| 2 | 3.24 ms | 3.97 ms | 1.223× |
| 3 | 3.24 ms | 3.98 ms | 1.226× |
| 4 | 3.23 ms | 3.99 ms | 1.235× |

Means: ours +0.94%, opaque +18.42% from Cycles'. Settled in 536 (blended)
and 560 (opaque) frames. Gate 16 measured 4.71% and 5.55% over 206,646
changed pixels; the small differences come from the settled caches, not
from the samples, which are the same.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criteria committed as `b0a6a2b`; the
binary `experiments/room3d/src/bin/m10_gate19.rs` is committed with this
result). Warm machine after development runs; no stray processes. Cycles:
gate 16's scene re-rendered (its job gained an empty vertex-colors list),
one bounce, 4,096 samples per pixel, 61 s on Metal.

## Method

Development runs before the gate (`blend_dev`, each camera pass submitted
alone, disclosed in the binary's doc comment) found the cost in two
full-screen passes that most pixels leave at once: `hw_edges` and
`vc_shade_edges` cost about 0.25 ms each in the warm frame even with their
work removed. The layer traces cost about 0.13 ms; a per-workgroup append
to the list changed nothing; CPU encoding was 0.12 against 0.09 ms.

- **Layers in the visibility pass.** `hw_visibility_blend` finds the
  camera hit (as `hw_trace`) and, where it is blended, traces the layers
  and lists the pixel. Without silhouette rays, the `hw_edges` pass is
  gone; with them, it lists silhouettes only and skips blended pixels.
- **Shading only the list.** `vc_args_edges`, one thread, writes
  `vc_shade_edges`'s dispatch (a workgroup per 64 records) into the
  dispatch arguments' spare words, and `vc_shade_edges` runs indirectly.

After the change (`blend_dev`): visibility 1.76 ms alone (from 1.37 ms),
shading 1.83 ms, the one-thread pass 0.27 ms alone (mostly submission),
listed shading 0.72 ms (from 0.90 ms).

## Silhouette rays

`edges_dev` (M7 gate 4's scene at 640×360), before and after: images
identical (hardware against compute 0.0578% both, 4,571 edge pixels both);
hardware edges 1.31 ms before, 1.34 ms after (1.19× and 1.23× one ray): the
extra one-thread pass costs about 0.03 ms where the list is small.
