# M9 gate 1: moving objects in the view-dependent lighting cache

**PASS on all three criteria.** 65 objects moving every frame among a
million static boulders, with three lights, through the cache that
`Renderer` uses:

- **No stale light:** 30 checks over 300 frames, up to 137,260 resident
  pages (44.5 million values) each, and not one value differed from a
  from-scratch re-bake.
- **Work:** re-baking followed the change: 0.84% of the pages read per
  frame.
- **Speed:** flights cost 1.06–1.09× the same flights with the movers
  still, 15.76–16.14 ms mean.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `07ad668`; binary
`experiments/room3d/src/bin/m9_gate1.rs`, criteria pre-registered in
`59259a5`).

## Method

The paradigm's second rule (apply what changed exactly, where that is
cheap) had run only in the lightmap path (M2 gates 5–6). Now it runs in the
view cache.

- **Scene structure.** Instances marked movable leave the static object
  hierarchy. The root's second child is a subtree over them, rebuilt on
  the CPU each frame something moves. Its nodes carry no prefiltered
  coverage, so cones descend to the movers' own triangle hierarchies,
  which move rigidly with them. Hardware camera rays query a second, small
  top-level acceleration structure of the movers, rebuilt each frame.
- **Exact dependency test.** A texel's channel for a light depends on a
  mover only if the mover's box meets the hull of the texel's cone apex
  and the light's bounding box: a light cone never leaves that hull.
  - Each page keeps the object-space box of its texels' apexes.
  - Each frame something moves, a GPU pass tests every resident page
    against every moved object's old and new boxes. Per mover and light, a
    box around its shadow region, clipped to the scene, culls first.
  - A page on a moved object is stale whole. A page whose box may be
    touched is tested texel by texel, at each texel's apex, light by
    light.
- **Same-frame re-bake.** Stale texels of pages read in the frame before
  are re-baked in the same frame, before the camera passes, for only the
  lights they depend on. They are appended on the GPU to the frame's bake,
  with indirect dispatches.
- **Never stale.** Other pages with a stale texel are marked invalid at
  once (shading treats them as missing) and evicted when the CPU reads the
  list back.

`RendererOptions::movable` and `Renderer::set_transform` expose it. The GPU
test `movers_leave_no_stale_light` (`continuum-gltf/tests/gpu.rs`) checks
the same property on a small scene, and that the settled image equals,
bit for bit, a cache built for the movers' final places.

## Scene

- The million-boulder field with M5 gate 6's three colored lights, no sky,
  hardware camera rays, cache defaults, prefilter 1 bin.
- 64 boulders move: the ones nearest 64 points spaced evenly from 15 m to
  450 m ahead of the camera's start. Each circles a center 1.5 m from its
  start once every 240 frames, turning 1/60 radian per frame.
- A 4 × 2 × 1.6 m box (a new geometry) drives along the flight, 6 m to
  its left, from 20 m ahead, at 0.7 m per frame.
- The flight is M5 gate 8's path: 600 frames from standing height to
  150 m up, pitching down to 45°, from an empty cache.

## Results

**Control** (static, sun only, standing viewpoint): warm frame 1.85 ms
after 112 frames (at most 3.3 ms): fit.

**Criterion 3, speed** (measured first): flights alternating, movers still
and moving.

| Flight | Mean | 99th percentile | Worst | Fallback | Bake budget | Pages re-baked per frame | Pages read per frame |
|---|---|---|---|---|---|---|---|
| 1, still | 14.80 ms | 28.59 ms | 49.40 ms | 1.76% | 2,712 | | |
| 1, moving | **15.76 ms** | 25.45 ms | 56.72 ms | 3.51% | 1,426 | 453.4 | 63,433 |
| 2, still | 14.77 ms | 28.40 ms | 52.79 ms | 1.78% | 2,893 | | |
| 2, moving | **16.14 ms** | 26.04 ms | 56.13 ms | 3.31% | 1,501 | 456.6 | 65,099 |
| 3, still | 14.84 ms | 27.65 ms | 57.11 ms | 1.76% | 2,731 | | |
| 3, moving | **15.87 ms** | 26.12 ms | 44.45 ms | 3.01% | 1,522 | 453.4 | 67,596 |

The control flights average 14.81 ms. The criterion is that each flight
with movers has a mean of at most 16.7 ms and at most 1.25× that
(18.51 ms), a 99th percentile of at most 33.3 ms, and fallback of at most
10%. The flights with movers took 1.06×, 1.09× and 1.07× the control, so
this is a **PASS**.

Stale pages that could not be re-baked averaged 3.9–4.1 per frame, marked
invalid. No stale page read in the frame before missed the bake.

**Criterion 2, work**: pages re-baked because something moved averaged
0.87%, 0.84% and 0.81% of the pages read in the frame before, 0.84% over
the three flights. The criterion is at most 15%, so this is a **PASS**. The
alternative that ignores dependency regions re-bakes all of them.

**Criterion 1, no stale light**: over the first 300 frames with movers,
after every 10th frame, every value of every page shading may read was
re-baked from scratch and compared bit for bit.

| Check (frame) | Pages | Values | Differing |
|---|---|---|---|
| 10 | 9,386 | 3,041,064 | 0 |
| 100 | 62,542 | 20,263,608 | 0 |
| 200 | 111,204 | 36,030,096 | 0 |
| 300 | 137,260 | 44,472,240 | 0 |

All 30 checks found zero differing values, so this is a **PASS**. Pages
checked, by frame: 10: 9,386; 20: 12,641; 30: 15,888; 40: 19,137; 50:
22,401; 60: 29,749; 70: 39,314; 80: 48,718; 90: 57,507; 100: 62,542; 110:
70,636; 120: 77,171; 130: 81,188; 140: 85,920; 150: 90,886; 160: 96,475;
170: 99,409; 180: 104,272; 190: 109,772; 200: 111,204; 210: 116,174; 220:
120,993; 230: 121,479; 240: 125,027; 250: 128,512; 260: 127,990; 270:
131,305; 280: 134,622; 290: 135,889; 300: 137,260 (81 texels × 4 channels
each).

## Changes before the full-size run

These are disclosed in the binary's doc comment.

- **Re-baked pages** are those read in the frame before. The stub's method
  said "the last two frames", its criterion 2 "the frame before"; every
  pixel marks the page it reads.
- **Texel-level tests.** The first implementation tested whole pages. A
  development run showed the coarsest ground page re-baked every frame:
  7×7 texels over the whole 2.1 km field, whose box contains every mover.
  Its long grazing cones cost about 13 ms a frame. Candidate pages are now
  tested at each texel's apex, and only stale texel-light pairs re-bake.
- **The movers.** The pre-registered ones (the 64 boulders nearest a point
  15 m ahead, and a box at 0.1 m per frame) left the view within the
  first 30 frames, so the timed criteria would have measured an almost
  static view. They were spread along the flight to keep movers in view
  throughout. That makes the test harder, not easier.

## Development runs (disclosed)

At 640×360, with `M9_DEV=1` or the profiling binary `m9_dev`:

| Run | Still | Moving | Notes |
|---|---|---|---|
| First development run | 13.01 ms | 24.52 ms | 1.64% of pages re-baked |
| Before the fix below, 1280×720 (`m9_dev`) | 12.94 ms | 25.32 ms | |
| After the fix below | 12.71 ms | 16.76 / 25.44 ms | about 2 pages re-baked per frame |
| After texel-level tests, 1280×720 (`m9_dev`) | 12.66 ms | 14.04 ms | |
| Gate, original movers | 14.08 ms | 15.21 ms | movers out of view |
| Gate, final movers | 13.47 ms | 16.06 ms | 0.89% of pages re-baked |

A bug found on the way: the fill pass marked every baked page as read, so
a re-baked page counted as visible in the next frame. It was then re-baked
every frame its mover moved, even unseen. Now only pages the CPU planned
are marked.

Two negative controls showed the check is sensitive. With the
shadow-region test disabled, the GPU test found 1,977 stale values in its
first check. With the texel test disabled, it found 1,977 again.

## Caveats

- **No sky with movers in the cache yet.** Sky cones see the whole
  hemisphere, so exact invalidation would re-bake almost every page. Their
  bounded deferral is M9 gate 2, the error ledger's job.
- **Rigid motion only.** Movers keep their scale. Lights stay put; their
  colors still change for free.
- **More fallback with movers.** It rose from about 1.8% to 3.0–3.5%.
  Re-baking comes out of the same frame-time budget, which roughly halved.
- **GPU timestamps around the bake are unreliable** in frames that also
  rebuild the movers' acceleration structure, which explains the lower
  "camera" figures there. The gate's timings are CPU frame times.
- **Black boulders** appear mid-flight in both the still and the moving
  flights, at the same places. The suspected cause, not yet checked, is a
  boulder's coarsest page, whose single texel lies below the ground. This
  is a static-cache issue, unrelated to movers.
- Apple M4 only, as every timing so far.

## Prior art

Dependency regions with exact incremental updates are the paradigm's
second rule (`docs/DESIGN.md` §3.1). The closest prior work is incremental
radiosity with line-space hierarchies (Drettakis & Sillion 1997) and the
render cache (Walter et al. 1999). No claim of novelty is made for the
invalidation test itself.
