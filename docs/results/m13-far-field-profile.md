# M13's coverage pass at a million boulders: where the time goes (development)

Not a gate: a profile to ground the far-field design (DESIGN §3.4,
§3.8 item 1). M13 gate 4's path (profile stage 16, with the packet walk of
`a6799f6`) on the gate's scene and views: the million-boulder field
without its ground, standing at 1.7 m and 150 m up looking 45° down,
1280×720, vertical field of view 50°. Apple M4 (fanless MacBook Air),
release build, median of 10 frames each waited for, 2026-11-13. Parts
were switched off by temporary branches (not committed). Switching a part
off changes the others' work (pixels whose bins no longer fill walk
further), so the shares are approximate.

| Variant | Standing | 150 m up |
|---|---|---|
| Gate 4's path (control) | 122.67 ms | 104.43 ms |
| Without the bins' hardware rays | 107.58 ms | 53.51 ms |
| Without those or the maps' reads (the walk) | 53.52 ms | 29.08 ms |
| Corner rays only | 2.21 ms | 2.06 ms |

The control repeats gate 4's standing frame (122.32 ms in the gate), so the
machine was fit to measure. From 150 m it is 104 ms against the gate's 135
ms: the packet walk, committed after the gate.

Shares, roughly:
- **150 m up:** the bins' rays about 51 ms (16 per pixel, 96% of pixels),
  the maps' reads about 24 ms, the walk about 27 ms (67.2 object nodes a
  pixel, at most 131, evenly spread: 42–101 by row). No node above the
  instances is small enough to stop at: a boulder is 2–5 pixels across and
  a leaf of two boulders 5–11.
- **Standing:** the bins' rays about 15 ms, the maps' reads about 54 ms
  and the walk about 51 ms. Object nodes by row: 33–46 a pixel in the sky
  (rows 0–320), 2,438 at the horizon row (360), 0–49 below it. The horizon
  band (rows 330–390) holds 17% of node visits and 32.7% counted as SIMD
  groups wait (each 8×4 group at its slowest pixel).

**The sky's visits are waste, removed exactly.** Pixels above the horizon
descended every box straddling the eye's plane: the side planes' test
cannot reject them, since behind the eye a side plane also bounds the
mirrored frustum. Testing the box's faces against the pixel frustum (the
frustum lies wholly on the eye's side of a face when every edge direction
points away from it) left one visit a pixel in the sky: standing, 36.0 → 18.6
object nodes a pixel with the same coverage (mean 0.2953, the same visits
in every row below the sky), 122.7 → 113.4 ms; from 150 m no change (67.2,
105.0 ms). After it the horizon band holds 52.5% of the SIMD-weighted
visits.

What it means for the design:
- From 150 m the cost is the bins' rays, the maps' reads and a walk from
  the root to the instances in every pixel. Hierarchy nodes holding
  coverage do not help there: they would sit above the instances.
- Standing, the horizon is the far field: a pixel there sees boulders from
  about 540 m to the field's edge, each about a pixel across, and walks
  thousands of nodes and reads their maps. That is what nodes holding
  coverage are for.
