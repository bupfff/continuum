# M13 gate 8: the coverage pass in 10 ms (coverage only)

**PASS.** The settled coverage pass runs in about 9 ms in both views, and
its accuracy improved on gate 6's.

| View | Settled frame | Coverage error, ours | One ray | Ratio |
|---|---|---|---|---|
| Standing | **9.01 ms (at most 10): PASS** | 1.38% | 8.69% | **0.159× (at most 0.25×): PASS** |
| 150 m up | **9.10 ms (at most 10): PASS** | 7.41% | 57.99% | **0.128× (at most 0.15×): PASS** |

Reported: 20 m up looking 10° down (gate 7's flight view), settled 11.77
ms (4,285 pages). Gate 6 on the same views: 16.12 and 17.11 ms, 0.212× and
0.135×; gate 4 (before the far field): 122–135 ms. Standing settled in 10
frames at 512 bakes a frame (2,828 pages of 2,048 texels); from 150 m in 2
(10 pages). Mean coverage: standing, reference 0.2951, ours 0.2952, one
ray 0.2956; 150 m up, 0.2909, 0.2905, 0.2910.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `7366b0d`; the method in `caa0634`, before the
run). Speed timed first (median of 10 settled frames, each waited for).
Machine free of other work; control just before, gate 4's path at full
size: 120.0 ms standing, 102.9 ms from 150 m (122.0 and 105.8 at gate 5).

## Method (on gate 7's pass, profile stage 18)

A frame runs nine passes:
1. **Once-a-frame pass**, one invocation per page of the pool, per node of
   the object hierarchy and per object:
   - **Page records:** validity from the eye, refresh requests, and the
     homography from screen to texels.
   - **Node projections:** each node's box projected to screen, unless it
     straddles the eye's plane.
   - **Instance records:** 32 bytes for each small instance in view (its
     square, mip level, nearest-direction layer, and affine map
     coordinates), from static per-object data (sphere, rotation
     quaternion, first layer).
2. **Roots:** per tile, as gate 5. It now reads stored projections, and
   records whether the tile may hold a large instance (each node's
   largest instance at its nearest depth).
3. **Corner rays:** only in tiles that may hold a large instance. They are
   plain opaque queries, and the instance records say whether a hit is
   large.
4. **Bins, arguments, bin rays:** a pixel traces bin rays when one of its
   own corners hits a large object.
5. **Pixels:**
   - Each tile lists its leaf roots' instances once in workgroup memory.
   - Pixels read that list.
   - Pixels walk the other roots with stored projections.
   - Map reads take one sample per bin from the nearest of 32×32
     directions, skipping bins outside the map's rectangle of non-zero
     texels.
   - Pages hold 2,048 texels (16,384 pages, 32 MB).
6. **Page arguments and bakes:** as gate 7.

## Disclosed

- Developed at full size on the gate's scene: speed on the gate's views,
  accuracy only on other views (0.183× and 0.127× there)
  (`docs/results/m13-gate8-development.md`). The changes from the stub's
  method are listed in the binary's doc comment.
- The coverage maps now have 32×32 directions (1,024 layers a geometry; the
  device allows 2,048, so a scene with more than two mapped geometries
  needs another layout).
- Memory added: 32 bytes an object for records and 32 for static data
  (64 MB for the million boulders), 16 bytes a node for projections (19
  MB), 4 bytes a node for largest radii; the page pool doubled in bytes
  (32 MB of texels).

## What the numbers say

- **Landmark 3's speed bar is met for coverage:** about 9 ms in both
  views (gate 4: 122–135). What made it: work moved out of the pixels'
  pass into once-a-frame passes; nearest-direction maps at twice the
  angular resolution (one sample, not two, and more accurate); pages
  large enough for the horizon's mid-distance nodes; a tile reading its
  leaves' instances once.
- **Accuracy improved** with the bin-ray rule on the pixel's own corners
  and the finer directions: 0.159× standing (from 0.212×).
- **Left:** colour (appearance) in the same structures; the 20 m flight
  view at 11.8 ms; memory per object to bound by the screen.
