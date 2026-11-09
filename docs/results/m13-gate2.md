# M13 gate 2: sub-pixel geometry, a scalar per-node test (coverage only)

**FAIL.** Speed fails in both views (about 26× over); accuracy passes from
150 m and misses standing (0.59× against 0.5×), from one row of pixels at
the horizon.

| View | Coverage error, ours | One ray | Ratio (at most 0.5×) | Frame (at most 10 ms) |
|---|---|---|---|---|
| Standing | 5.14% | 8.69% | 0.592×: FAIL | 257.44 ms: FAIL |
| 150 m up | 21.83% | 57.99% | **0.377×: PASS** | 281.24 ms: FAIL |

Mean coverage: standing, reference 0.2951, ours 0.2954, one ray 0.2956;
150 m up, 0.2909, 0.2837, 0.2910. Per pixel: standing 46.5 object nodes
(at most 1,352) and 56.4 triangle nodes (at most 2,129); 150 m up, 69.7
(at most 131) and 73.2 (at most 510).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criteria committed as `13111e2`; method
changes before the run, disclosed in the binary's doc comment, in
`a697df6`). The field without its ground, for the reference and the test
alike.

## Method

Each pixel walks the object and triangle hierarchies testing boxes in
screen space (corners through one camera transform per object, culled
against the pixel frustum's planes, nearer child first): a box within a
pixel adds its projected box's overlap with the pixel times its
prefiltered opacity, as one number; larger boxes are opened; triangles add
the exact area they cover. Within an object, contributions add; objects
combine as independent occluders.

## What the numbers say

- **Accuracy is close, except at the horizon.** By rows of the standing
  image, the horizon band (rows 330–370) holds 84% of our squared error,
  and there ours equals one ray (1.00×); everywhere else ours is 0.26–0.29×
  one ray's error. The error is one row, the pixels the far boulders' top
  edge crosses (row 360): the reference covers 0.632 of them, ours 1.000.
  Thousands of distant boulders per pixel, each spread evenly over its box
  and combined as independent occluders, saturate the pixel, though all
  sit in its lower part. That is correlation within the pixel, which the
  8×8 bins of gate 1 kept and one number per pixel loses. A few sub-pixel
  bins (4×4) would keep it.
- **Speed: about 2 ns per node visit across the GPU,** about 100–140 visits
  per pixel. Per lane that is thousands of cycles a visit for a few hundred
  operations: the walk is latency- and occupancy-bound, not arithmetic.
  The pass runs inside the pixel-cone entry point, which also holds M7's
  fragment walk (its 68-value buffers), so it likely runs at that entry's
  low occupancy. Next to try: its own entry point; then sharing the walk
  across a tile of pixels.

## Where the time goes (development, after the gate)

`M13_DEV=1` (10,000 boulders; numbers not the gate's), the walk in its own
entry point (`m13_coverage`, profile stage 14, `M13_OWN_ENTRY=1`), with
parts switched off by a branch never taken (so the code, and its register
cost, stays):

| Variant | Standing | 150 m up |
|---|---|---|
| As in the gate, inside the pixel-cone entry | 190 ms | 114 ms |
| Its own entry point | 192 ms | 113 ms |
| Object hierarchy only (objects never entered) | **14.3 ms** | **10.7 ms** |
| No exact triangle clipping | 211 ms | 110 ms |

- **Occupancy was not it:** its own entry point changes nothing.
- **The object walk is cheap:** about 0.3 ns a node visit across the GPU,
  about 30–47 visits per pixel. Neighboring pixels walk the same nodes, so
  loads are shared.
- **Walking each instance's triangle hierarchy is the cost:** about 3 ns a
  visit, ten times an object node's, for 35–50 visits per pixel, with or
  without the exact triangle clipping. Neighboring pixels enter different
  boulders and walk different paths, so their loads scatter.

So the next gate should keep the object walk and avoid per-pixel walks
inside small instances: give each geometry a small directional coverage
map (its opacity seen from a set of directions, a few texels across), read
for instances a few pixels across or less. That also keeps correlation
within a pixel, which one number per node loses (the horizon row).
