# M13 gate 4: near surfaces resolved by hardware rays (coverage only)

**FAIL on speed; accuracy passes in both views, better than gate 3.**
Hardware rays removed the triangle walks (standing 707 → 122 ms; 150 m up
178 → 135 ms), but the far field still costs more than the 50 ms bar.

| View | Coverage error, ours | One ray | Ratio (at most 0.5×) | Frame (at most 50 ms) |
|---|---|---|---|---|
| Standing | 1.90% | 8.69% | **0.218×: PASS** | 122.32 ms: FAIL |
| 150 m up | 7.63% | 57.99% | **0.132×: PASS** | 135.02 ms: FAIL |

Mean coverage: standing, reference 0.2951, ours 0.2953, one ray 0.2956;
150 m up, 0.2909, 0.2932, 0.2910. Pixels resolved by their corners:
standing 28.7%, 150 m up 4.3%. Per pixel: standing 36.0 object nodes (at
most 3,241), 150 m up 67.2 (at most 131); no triangle nodes in either.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-09 (stub with the criteria committed as `b5422a1`; the
method in `23549ad`, before the run). The field without its ground, for
the reference and the test alike.

## Method

1. A hardware ray through each pixel's top-left corner records the object
   it hits and whether that object is large (more than 8 pixels across, or
   without a coverage map).
2. A pixel whose four corners hit the same large object is wholly covered.
   Every other pixel casts a hardware ray through each of its 4×4 bins'
   centres (a bin whose ray meets a large object is covered), then walks
   the object hierarchy for small instances (read from their coverage maps)
   and far object nodes, leaving large instances out.

**Disclosed:** a development run (`M13_DEV=1`, 10,000 boulders) of the first
version, which walked large instances' triangles in unresolved pixels,
took 162 ms standing and 82 ms from 150 m; the bins' rays replaced those
walks before the run (development: 25.6 ms and 55 ms).

## What the numbers say

- **Accuracy keeps improving:** bins resolved by rays are exact where gate
  3 splatted triangle nodes; from 150 m the error is 0.13× one ray's.
- **The near field is solved for speed:** no triangle walks remain.
- **The far field is what is left:** standing, a pixel at the horizon
  still walks thousands of object nodes (its bins never fill, correctly),
  and its SIMD group waits for it; from 150 m every pixel walks about 67
  object nodes and reads several maps. At 10,000 boulders the same passes
  take 26–55 ms; the million-boulder field multiplies the far field's
  work. Next: one walk shared by a tile of pixels (coherent loads, and the
  horizon's long walk paid once per tile), or a coarser far field through
  the quality dial.
