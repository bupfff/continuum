# M13 gate 5: exact culling, bin rays near large objects, tile roots, lighter map reads (coverage only)

**PASS.** Gate 4's accuracy, held within the stated loss, at 3.2× (standing)
and 5.2× (150 m up) less time a frame.

| View | Coverage error, ours | One ray | Ratio | Frame |
|---|---|---|---|---|
| Standing | 1.90% | 8.69% | **0.219× (at most 0.25×): PASS** | **38.25 ms (at most 50): PASS** |
| 150 m up | 7.81% | 57.99% | **0.135× (at most 0.15×): PASS** | **20.35 ms (at most 25): PASS** |

Gate 4 on the same views: 0.218× and 0.132×, 122.3 and 135.0 ms (its
path with the packet walk since: 122.0 and 105.8 ms, the control below).
Mean coverage: standing, reference 0.2951, ours 0.2953, one ray 0.2956;
150 m up, 0.2909, 0.2924, 0.2910.

Per view: pixels that walked nothing 80.4% standing, 16.1% from 150 m
(see below); pixels tracing bin rays 4.7% and 13.2% (gate 4: every unresolved
pixel); subtree roots a tile 3.8 (at most 63) and 6.9 (at most 23); object
nodes a walking pixel 34.1 (at most 3,275) and 7.9 (at most 23).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub with the criteria committed in the tenth session;
the method in `7da54df`, before the run). Speed timed first (median of 10
frames, each waited for). Machine checked free of other work; control just
before the run, gate 4's path at full size (`M13_DEV=1M M13_STAGE=16`):
122.0 ms standing, 105.8 ms from 150 m, against 122.7 and 104.4 when that
path was profiled (`docs/results/m13-far-field-profile.md`).

(The binary prints that share as "resolved by their corners", but it
counts every pixel that visited no node: resolved by its corners, filled by
its bin rays, or in a tile with no roots, such as the sky's. Gate 4's 28.7%
and 4.3% counted corner-resolved pixels only, so the shares are not
comparable.)

## Method (passes after gate 4's corner rays)

1. **Roots** (one invocation per 8×8 tile): walks the object hierarchy
   against the tile's frustum, nearest first, with the exact box-face test
   (A), and lists the roots of subtrees at most 8 pixels across, at most 63
   a tile (past that, nodes are listed as they stand).
2. **Bins** (B, one invocation per pixel): a pixel whose four corners hit
   one large object is covered; another traces its 16 bin rays only if one
   of the 16 corners of its 3×3 block hits a large object. Writes the
   covered bins as a mask.
3. **Walks** (C, a workgroup per tile): each pixel walks its tile's listed
   subtrees in order as a packet (gate 4's walk from lower roots), until its
   bins are full; nodes over full bins only are skipped (bit masks of open
   bins). Instances read their maps (E) through an 8-bit texture array,
   two directions blended along one axis of the directions' grid, each
   bin's map coordinates affine in its screen position about the
   instance's centre ray.

## Disclosed

- Developed at full size on the gate's scene; speed was developed on the
  gate's views, accuracy only on other views (turned 20°; 120 m up at 40°),
  where it read 0.199× and 0.134×
  (`docs/results/m13-gate5-development.md`).
- Changed from the stub's method before the run: C as two passes instead
  of one walk per tile in workgroup memory; B in its own pass; E with two
  directions instead of four (the stub's doc comment lists the changes).
- Roblox ran during the first development rows; the run and its control
  had the machine to themselves.

## What the numbers say

- **Where the time went:** a single invocation walking for 64 pixels held
  the others idle (116 / 42 ms); listing subtree roots in a separate pass
  and walking below them per pixel fixed that (86 / 29). The rest came from
  the maps' reads: skipping full bins (standing's horizon pixels stop
  early), set-up linearized about the centre ray, bit masks of open bins,
  and the bin rays moved out of the walk's kernel (no ray-query state, more
  occupancy).
- **From 150 m** about 3 ms is corner rays and roots, 6 bin rays, 3.5 the
  walk and 8 the maps' reads. Gate 6's bar is 20 ms in both views, so this
  view needs more than pages (the design expects it to use none).
- **Standing** is now mostly the horizon: pixels walking up to 3,275
  nodes. That is gate 6's coverage pages.
