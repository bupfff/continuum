# M13 gate 5 in development (not run): where the implementation stands

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
