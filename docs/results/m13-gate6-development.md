# M13 gate 6 in development: coverage pages, and speed for 150 m up

Development at full size (`M13_DEV=1M`) before gate 6's run, on gate 5's
pass. Speed on the gate's views; accuracy only on other views
(`M13_ACC=1`: standing turned 20°, 120 m up at 40° turned −20°).

## Speed for 150 m up first (no pages there)

Gate 6's bar is 20 ms in both views, and from 150 m no node above the
instances fits a page, so that view needed speed of its own:

| Step | Standing | 150 m up |
|---|---|---|
| Gate 5 as run | 38.25 ms | 20.35 ms |
| Bin rays compacted into a list, one invocation a ray (indirect dispatch) | 37.0 | 18.0 |
| Each object's world bounding sphere for map reads (16 MB) | 35.9 | 17.6 |
| Roots' projected boxes stored; the walk takes a root's box from the list | 36.4 | 16.5 |

Skipping parts from 150 m after the first row: passes before the walk 5.3
ms, the walk 4.7, the maps' per-instance tests 2.9, their set-up and samples
5.3. A pixel skipping roots by their stored boxes saved nothing alone (in a
packet one overlapping lane walks the root for all).

## Coverage pages

- **Where pages are found.** First only in the roots pass: no page was ever
  asked for, since horizon tiles fill their 63 roots with near nodes and
  list the far ones as they stand, too large for a page. Pages are now
  also found in the pixels' walks (the highest fitting node with enough
  instances; nodes below it want none).
- **Bakes.** Any-hit queries (every candidate triangle back to the shader)
  took about 0.4 ms a page; opaque queries stepping past other instances
  were slower still (one page about 220 ms of latency), because traversal
  from the eye meets everything a grazing ray passes before the node's box.
  Rays from the box's entry: 512 pages in about 50 ms. Allocating pages in
  the arguments pass and baking each with eight workgroups (a texel a lane):
  about 0.06 ms a page in bulk, 0.087 ms a page at 20 a frame (1.75 ms a
  frame; 32 a frame cost 2.75 ms).
- **Smallest node with a page** (standing, settled):

  | Instances at least | Pages | Frames to settle | Frame |
  |---|---|---|---|
  | 32 | 2,628 | 11 | 21.9 ms |
  | 16 | 4,443 | 14 | 19.5 |
  | 8 | 10,224 | 25 | **15.7** |
  | 4 | 14,020 | 33 | 14.9 |

  Eight kept. With it, on the development views: 0.197× one ray's error
  standing (0.199× without pages), 0.134× from 120 m (no pages), 16.3 and
  16.1 ms.
- **The walk** (`M13_WALK=1`: 3 s along the turned view, 32 bakes a frame
  then): checks at 0.182×, 0.179×, 0.180× one ray's error (bar 0.3×);
  largest shift read 0.139 bin (bound 0.25). Pages lag demand while
  walking (about 3,700 asked a frame against 32 baked), so frames ran about
  35 ms; unbaked nodes are walked, so accuracy does not depend on it.
