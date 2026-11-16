# M18 gate 1: fog with light shafts

**PASS** on both criteria.

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Cost: fog's added warm frame, view (a) | **0.802 ms** (5.007–5.139 against 4.205–4.241 ms, five rounds) | at most 1.0 ms | PASS |
| 2. (a) An even layer (0.05 per metre, top 6 m) | **9.58%** from Cycles; without fog 44.29% (**0.216×**) | at most 12%, 0.35× | PASS |
| 2. (b) Falling with height (0.08 at the ground, 0.3 per metre), from beside the brick block | **5.51%** from Cycles; without fog 37.69% (**0.146×**) | at most 12%, 0.35× | PASS |

Means against Cycles: (a) ours +1.43%, without fog +25.71%; (b) ours
−1.20%, without −9.59%. Every view settled. Cycles took 417 s and 463 s
(Metal, 4,096 samples; volumes are slow there).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-16 (commit `03dacbd`; binary
`experiments/room3d/src/bin/m18_gate1.rs`, stub committed in `c7ee11b`;
output `docs/results/m18-gate1-run.txt`), after three minutes with no
game or indexing running and one more minute idle.

## Method

DESIGN §3.4, "Fog with light shafts": `CacheOptions::fog`. The first
light's visibility in the air from a world-space grid around the eye (128
× 128 × 32 cells to the top, snapped to its cells), traced by hardware rays
and kept while the eye stays in its cells and no light moves; per frame,
froxel columns (8 × 8 pixels, 64 slices) integrated front to back (each
slice's optical depth exact, the light gathered at the middle of its part
in the layer); each pixel's colour times what the fog lets through to its
hit, plus the light gathered; surfaces lit through the fog. The textured
scene with M10 gate 16's panes as thin glass, sun and sky; ours at 4×4
sub-pixel offsets against Cycles' box of Volume Scatter and Volume
Absorption (single scattering).

## Disclosures

- The bars were set with development numbers known (0.218× and 0.151×,
  0.80 ms).
- What is left is mostly the panes' shadows (Cycles tints them; the cache
  holds one channel per light) and the sky's horizon.
- The cost is a still camera's: the visibility grid is kept; when the eye
  crosses a cell or a light moves it is traced again, about 0.3 ms more
  that frame.
- Not measured: fog in reflections and through glass layers, fog lit by
  surfaces, shadows of other lights in the fog, movers (fog needs a
  static scene).
