# M13 gate 5 in development: how the implementation got to its run

Not a gate result: the state of gate 5's implementation (profile stage 17,
`hw_m13_tiles` in `hw_visibility.wgsl`) at the end of the tenth session,
with development numbers at full size (`M13_DEV=1M`: the million-boulder
field, the passes alone, no reference). The gate's criteria are in
`m13_gate5.rs` (at most 25 ms from 150 m and 50 ms standing; accuracy at
most 0.25× and 0.15× one ray's error).

## What is built

- **A, exact culling by the boxes' faces** (`frustum_misses_box`, with
  `set_rect_frustum` for pixels and tiles).
- **B, bin rays only near large objects:** a pixel traces its 16 bin rays
  only if a corner of its 3×3 block hits a large object. From 150 m 13.2% of
  pixels trace them (gate 4: 96%), standing 4.7%; about 6 ms of the 150 m
  frame.
- **E, the maps through a texture:** `coverage_texture` (an R8 texture
  array, a layer per geometry and direction, mips the map's levels) and
  `instance_map_tex` (one hardware-filtered sample per bin and direction).
- **C, one walk per 8×8 tile:** the first invocation walks the object
  hierarchy against the tile's frustum, listing entries (object nodes to
  splat, instances to read) in chunks of 128; each pixel reads those over
  its open bins; the walk skips nodes over pixels' full bins only (each
  pixel's open bins' rectangle in workgroup memory) and ends when every
  pixel is full.

Coverage matches gate 4's (mean 0.2953 standing, 0.2932 from 150 m).

## Development numbers (1,000,000 boulders, 1280×720)

| Version | Standing | 150 m up |
|---|---|---|
| Gate 4's path (control) | 122.7 ms | 104.4 ms |
| Tile walk, first version (one lane walks) | 202.4 ms | 38.6 ms |
| … without bin rays | 205.8 | 32.9 |
| … without reading entries | 42.1 | 17.1 |
| … without the walk | 2.9 | 2.9 |
| Cooperative walk (64 lanes share a stack, 24 KB of workgroup memory) | 528.4 | 240.8 |
| One lane walks, pixels' open bins pruning nodes and instances | **116.4** | **42.0** |

In the last version: standing, entries read a pixel 1.4 (at most 816),
node visits a tile 131 (at most 8,471); from 150 m 1.3 entries (at most 8),
88 visits a tile (at most 165).

## What the numbers say

- **One lane walking for 64 wastes the other 63.** A tile's walk is ~88
  visits from 150 m, but it holds 64 lanes while one works, so it costs
  about as many lane-slots as 64 per-pixel walks of 67 visits: 14 ms
  against gate 4's ~27. The cooperative walk was worse (occupancy from its
  workgroup memory, and four barriers a round on walks that are narrow at
  the top of the tree).
- **Pruning by open bins is what made standing cheaper** (202 → 116 ms):
  without it horizon pixels read up to 2,191 instance maps.
- **Next, by this reading:** a separate pass with one thread per tile
  (14,400 threads all busy) walking to the tile's cut and listing subtree
  roots (or entries) in global memory, then the pixels' pass walking the
  listed subtrees front to back with open-bin pruning (gate 4's walk, from
  lower roots); the maps' read with the nearest direction only (a quarter
  of the samples and set-up), if accuracy allows. The horizon's long
  walks stay for gate 6's coverage pages.

## Eleventh session: two passes, cheaper map reads (development, full size)

Same scene and views (`M13_DEV=1M`, the passes alone). Accuracy in
development was measured on other views than the gate's (`M13_ACC=1`:
standing turned 20°, and 120 m up at 40° turned −20°), so the gate's views
were not tuned on. Roblox ran during the first rows (the control read
126.5 / 94.7 ms early on, 185.9 / 129.7 ms at its worst); the rows from the
cut-size test on ran with the machine otherwise idle.

| Version | Standing | 150 m up |
|---|---|---|
| Gate 4's path (control, `M13_STAGE=16`) | 126.5 ms | 94.7 ms |
| Roots pass (one invocation a tile, subtrees ≤ 8 px) + packet walks below | 85.9 | 28.7 |
| … walks each on its own (no packets) | 116.8 | 32.7 |
| + full bins skipped in map reads, two directions (quiet machine) | 51–53 | 25.5 |
| + set-up linearized about the centre ray | 49.8 | 22.6 |
| + open bins as a bit mask (walk tests, map reads' bins) | 42.6 | 22.8 |
| + bin rays in their own pass | **38.8** | **20.4** |
| … walks each on its own (no packets) | 42.5 | 20.3 |

Accuracy on the development views (relative RMSE against 16×16 rays, as a
multiple of one ray's): four directions 0.199× standing, 0.131× from 120 m;
the nearest direction alone 0.199× and 0.147× (40 / 18 ms, too near the
0.15× bar); two directions along one axis 0.199× and 0.133–0.134× (kept).
Linearizing, the bit masks and the separate pass left the coverage as it was.

Where the time went from 150 m before the last two rows (skipping parts):
corner rays and roots 3.1 ms, bin rays 6.3, walk 3.5, map reads 9.2 (their
texture samples about 2 of it). The cut size (4, 8, 16 pixels) changed
little; 8 kept.
