# Sky texels in the view cache: where the time goes (diagnostics, not a gate)

Date: 2026-10-25. M5 gate 4 found sky texels about 100× costlier than sun
texels and listed a time-based bake floor as the next step. Before building
it, these measurements asked why a sky texel costs so much, and what can be
done inside the paradigm. Everything here is a development measurement on the
million-boulder field at gate 8's viewpoints (Apple M4 MacBook Air, plugged
in, release builds). The tools are `bake_cost`, `packet_check` and
`sky_spacing_dev` in `experiments/room3d/src/bin/`.

## The machine was fit this time

- **Control:** gate 11's sun-only warm-up at the standing viewpoint
  (`warm_check`) took a **2.25 ms** warm frame, after 130 frames. Gate 11 had
  8.03 ms after 102 frames, and gate 4's run 79.8 ms.
- **The same per-pixel sky render as before gate 4** (`sky_prefilter_sweep`,
  unchanged code, 320×180, prefilter ×8): **6.8 µs per pixel**, against 24.5 µs
  measured before gate 4. That run was on a machine about 3.6× slower,
  consistent with gate 4's control.
- No stray processes (`ps`), on AC power.

## 1. A time-based floor alone cannot fit a sky bake in a frame

Bake time against texels baked, from an empty cache at the standing viewpoint,
fixed budget, each frame profiled (first 12 frames after 4 warm-up frames):

| Texels per frame | Sun: bake per frame | Sky: bake per frame | Sky: per texel |
|---|---|---|---|
| 250 | 1.25 ms | 42.2 ms | 169 µs |
| 1,000 | 2.12 ms | 72.8 ms | 73.5 µs |
| 4,000 (3,486) | 2.68 ms | 142 ms | 40.9 µs |
| 16,000 (10,611) | 3.95 ms | 368 ms | 34.7 µs |

- The sky bake is **latency-bound**: 250 texels take 42 ms. No budget,
  counted in texels or in time, keeps a frame that bakes any sky near 14 ms.
- These are coarse pages (the first frames of a warm-up).

## 2. Divergence costs about 10×

- **The test.** All 32 invocations of each SIMD group bake the same texel.
  Every walk is then identical.
- **The result.** The 10,611-texel sky bake drops from 368 ms to **36 ms**,
  and 1,000 texels from 72.8 to 11.3 ms.
- **Why.** Neighbouring texels' walks through the hierarchies differ, and a
  SIMD group runs them one after another.

## 3. Packet traversal: identical results, 25% faster over a warm-up

- **What it is.** The invocations of a subgroup walk each hierarchy together:
  - a node is visited when any of them needs it;
  - each acts only on the nodes its own walk would visit, in the same order.

  Without subgroups (`KernelOptions::without_packets`, or a device without
  them), each invocation is its own packet.
- **Results are bit-identical** with and without packets, per pixel and
  through a warm cache (`packet_check`).
  - A cache still warming differs from run to run, packets or not. When
    several pixels request a page in one frame, whichever request lands
    first sets its priority, and so the order in which pages arrive.
  - An earlier version of this check compared caches after 80 frames, which
    happened to be warm before sky samples became separate jobs. Corrected
    the same day.
- **The union is large.** A packet visits 2.6× the object nodes and 12–22×
  the triangle-hierarchy nodes its invocations need. Neighbouring texels'
  near fields barely overlap.

| Measure | Solo | Packets |
|---|---|---|
| Sky bake, 1,000 texels per frame (first frames) | 67 ms | 97 ms |
| Sky bake, 10,611 texels per frame (first frames) | 305 ms | 257 ms |
| Whole sky warm-up at 20,000 texels per frame | 26.7 s | **20.2 s** |
| Per-pixel sun and sky, 320×180 | 0.40 s | **0.30 s** |

Packets are on by default where the device has subgroups. Over a whole
warm-up, and for per-pixel cones (neighbouring pixels are coherent), they
pay.

## 4. The horizon cull: exact for the integral, about 10% less work

- **The rule.** Geometry wholly below the receiver's tangent plane can only
  block directions below its horizon, which carry no light to it. Sky cones
  now skip it (`set_sky_horizon_cull`, on by default).
- **Accuracy.** Per pixel, the image moves by **0.002%** (relative RMSE).
  Before the cull, below-horizon geometry slightly darkened bins that the
  horizon crosses.
- **Work.** Per sky texel, object nodes fall from 712 to 648, triangle-hierarchy
  nodes from 808 to 647 and triangles tested from 302 to 265. Time falls by
  about 10%.

## 5. One job per sky face, face by face

- **The layout.** Each sky sample is now six bake jobs, one per cube face,
  listed face by face, so neighbouring invocations trace the same face from
  nearby texels. The fill pass sums the faces.
- **Whole sky warm-up, standing, sky sample cost 1** (so batches match the
  old layout):

  | Layout | Time |
  |---|---|
  | One invocation per texel, six faces in a row (packets) | 20.2 s |
  | One per face, texel by texel | 28.7 s |
  | **One per face, face by face** | **17.4 s** |
  | The same, 32 invocations per workgroup | 17.5 s |

## 6. Where a sky texel's time goes

Whole sky warm-up (17.4 s), by profiling stage:

| Stage | Cumulative | Share |
|---|---|---|
| No cones | 0.11 s | — |
| Setup and weights | 0.14 s | 1% |
| Object hierarchy and object splats | 3.54 s | 20% |
| Triangle-hierarchy descents | 12.0 s | 49% |
| Rasterization | 17.4 s | 31% |

- **Counts per sky texel** (`KernelOptions::count_work`): 5.9 faces traced,
  634 object nodes (67 splatted), 44 meshes entered, 698 triangle-hierarchy
  nodes (83 splatted), 305 triangles tested (45 rasterized).
- **The own object dominates.** Of the 698 triangle-hierarchy nodes, **590
  (85%) belong to the object the texel lies on**. Its hierarchy is cut by
  every face's near plane at the apex, so each face walks down to the leaves
  around the apex.
- **Skipping the own object** (wrong results, timing only) cuts the whole
  warm-up from 17.4 s to **6.6 s** (2.6×).

## 7. Fewer sky samples cost accuracy where texels are coarse

Sky every 1, 2, 4 or 8 texels of a page, interpolated bilinearly. Measured at
640×360 (a development size), warm cache against the per-pixel sky at 4×4
per pixel (the area integral):

| Viewpoint | Per pixel, 1 sample | Every texel | Every 2nd | Every 4th | Every 8th |
|---|---|---|---|---|---|
| standing | 2.64% | 4.30% | 4.93% | 5.88% | 7.76% |
| 10 m up | 7.82% | 11.93% | 16.43% | 19.20% | 22.05% |
| 40 m up | 17.52% | 21.15% | 34.44% | 36.93% | 38.79% |
| 150 m up | 18.01% | 22.49% | 33.94% | 34.43% | 34.74% |

- **A fixed stride wrecks the views from above.** There, whole boulders are a
  few texels. The cache now takes a spacing in meters instead
  (`CacheOptions::sky_spacing`, default 0: every texel). The stride is the
  largest whose spacing fits.
- **The spacing barely applies here.** At 640×360, spacings up to 10 cm
  barely apply: almost every texel is already coarser than that. The
  standing view saves 18% of samples at 10 cm, for 0.16 points.

## 8. The budget in cost units keeps frames on target, slowly

- **The budget.** It now counts sun texels or their worth: a sky sample
  counts `CacheOptions::sky_sample_cost` (default 50, from the measured
  ratio). The first page of a frame always fits.
- **The result.** With sun and sky, the standing view warms in 12,924 frames
  at **13.2 ms** each on average (`warm_check --sky`). Frames hold the 14 ms
  target, but warm-up takes **171 s**.
- **Why so slow.** The 20,000-unit maximum allows about 400 sky samples per
  frame, and the view needs 1.5 M.

## 9. A caution about bit-exactness across builds

- **Why results drift.** wgpu compiles Metal shaders with fast math, so
  moving code between functions or kernels changes float rounding.
- **Today's drift.** The sun-only cache moved by 0.02% (relative RMSE)
  between the previous build and this one, with no change in logic. The sky
  channel moved by 0.006% with the cull off.
- **Within one build**, packets against solo walks are bit-identical.

## What this means

- **The sky is expensive because of its near field, and above all its own
  object.** The far field is cheap thanks to object-level prefiltering.
- **The next lever is the own object.** A rigid instance's self-occlusion of
  a uniform sky is the same for every instance in object space. Coverage masks
  per face and bin, baked once per geometry with faces in the object's frame,
  would replace 85% of the triangle-hierarchy work. Texels would then walk
  only the other objects and combine per bin, as today. The new approximation
  would be the masks' spatial interpolation. Measured ceiling: 2.6×.
- **Packets and the horizon cull are kept.** Packets give results identical
  to solo walks; the cull is exact for the irradiance integral.

## After M5 gate 6: sharing the far field does not pay

Date: 2026-10-25, after gates 5 and 6.

- **What was tried.** The plan for sky warm-up was to walk distant objects
  once per page and face, from the page's middle, and each sky sample's near
  objects alone.
  - **The split.** An object is near when its hierarchy leaf's box meets a
    sphere k page half-diagonals across. Node splats are allowed only for
    nodes wholly on one side.
  - **Combining.** Near and far coverage combine per bin as independent
    occluders.
  - **Where it is.** It was built and measured, then removed. The patch is
    not in the repository.
- **Results.** Development runs at 640×360, sky only, whole warm-ups at a
  fixed budget, against the same cache without the split:

  | Viewpoint | No split | k = 8 | k = 16 | k = 32 |
  |---|---|---|---|---|
  | standing | 3.65 s | 4.97 s (0.19%) | 6.76 s (0.16%) | 9.44 s (0.13%) |
  | 10 m up | 8.53 s | 12.85 s (0.35%) | 17.09 s (0.24%) | 22.51 s (0.14%) |
  | 40 m up | 10.32 s | 17.99 s (0.32%) | 23.45 s (0.22%) | 30.88 s (0.12%) |
  | 150 m up | 7.04 s | 14.91 s (0.24%) | 19.82 s (0.16%) | 27.06 s (0.08%) |

  The bake times are GPU time. The percentages are the image's distance from
  the cache without the split.
- **Accurate, but 1.4–3.8× slower.**
  - **Why.** With object-level prefiltering, distant objects already cost
    little: groups of them splat as single nodes. The cost is near each
    texel, which the near walk still pays.
  - **What the split adds.** The far pass walks every face without a horizon
    cull, and nodes straddling the sphere cannot splat on either side.

## Oversized objects under the root

- **The change.** The object hierarchy now gives objects whose bounds exceed
  64 times the median diagonal a subtree of their own under the root (here,
  the 2 km ground among 1 m boulders). Before, every node above the ground
  took its bounds.
- **Whole sky warm-up, sky sample cost 1, cache bakes walking alone:**

  | Viewpoint | Before | After |
  |---|---|---|
  | standing | 9.71 s | 7.79 s (−20%) |
  | 150 m up | 21.20 s | 18.32 s (−14%) |

- **Work per sky texel barely changed** (632 object nodes against 633), so
  the gain is in memory locality, not in nodes visited.
- **Images.** Sun images are bit-identical. Packets and solo walks still
  agree bit for bit.

## Where a sky face's time goes now

- **Counts per sky texel** (5.9 faces): about 630 object nodes (60
  splatted), 38 meshes, 105–113 triangle-hierarchy nodes, 21–28 triangles
  tested.
- **Per face, about 107 object nodes.** These are the descent of about 20
  levels toward the apex with their siblings, plus the splatted far field.
- **Remaining levers:**
  - a wider hierarchy, to halve the dependent steps;
  - faces turned with the receiver's normal (one face and four half faces);
  - packets for large bakes. Solo walks win below about 10,000 units per
    frame, packets above: 5.7 s against 9.7 s for a whole standing warm-up at
    20,000.

## After M6 gate 1: why sparser sky samples do not pay yet

Date: 2026-10-25. Every texel is a sky sample (`sky_spacing` 0), and
warm-up is slow (gate 5). These measurements asked how many samples a
view really needs. Development runs on the million-boulder field, sun
and sky, cache defaults.

- **An oracle** (`sky_density_oracle`). It warms the cache at the standing
  viewpoint (1280×720), reads every resident page's sky channel back
  (`InstancedRenderer::read_cache_pages`), and finds each page's coarsest
  stride (8, 4, 2 or 1) whose bilinear interpolation, as the fill pass does
  it, stays within a tolerance of every texel. That bounds what any
  criterion choosing strides per page could save.

  | Tolerance (of the mean sky texel) | Samples, every texel | Samples, oracle | Saving | Pages at stride 8 / 4 / 2 / 1 |
  |---|---|---|---|---|
  | 0.5% | 947,912 | 846,727 | 1.1× | 2,191 / 175 / 258 / 15,114 |
  | 1% | 947,912 | 785,881 | 1.2× | 2,757 / 311 / 395 / 14,275 |
  | 2% | 947,912 | 731,065 | 1.3× | 3,438 / 378 / 357 / 13,565 |

  - Stride 2's error is large on most pages: its median is 52% of the
    mean sky texel, and 99% of pages stay under 143%.
  - Without own-object masks the oracle is the same (median 54%), so the
    masks are not the cause.
- **The cause is the normal.** Sky irradiance on the bumpy boulders follows
  the shading normal, which turns at the scale of their 2 cm triangles.
  - Dumped pages show smooth gradients with steps of 5–10% between
    neighbouring texels.
  - Ground pages under a boulder's edge also jump where the boulder meets
    the ground. This is a true discontinuity, inside the boulder.
- **Sharing occlusion as a vector fails.** A covered bin blocks
  `n · v` of the light, where `v` is the bin's vector irradiance, and the
  derivative with respect to `n` has no horizon term. So texels stored the
  face-summed occlusion vector `O`, and shading evaluated `π − n · O` with
  each pixel's normal.
  - Sky-only cache error at 1280×720 went from 4.11% to 4.43% standing and
    from 9.70% to 12.92% at 10 m up (against per-pixel cones with the same
    prefiltering).
  - **Why.** The model is first order in the normal. Under a half-space
    occluder (the ground), a pixel whose normal is β from the texel's is off
    by (π/2)(1 − cos β): 9% at 30°, 30% at 60°. From above, texels are
    coarse and normals across them differ that much.
  - The exact form under a half-space is linear in `n` with a free constant
    term (order 1 in spherical harmonics). It needs visibility below the
    receiver's horizon, which the walks do not compute.
- **A ratio to the open sky over flat ground** (`π(1 + n_z)/2`) was mixed:
  3.69% standing (better) and 10.95% at 10 m up (worse). It is also
  specific to scenes on a ground plane. Dropped.
- **What this means.**
  - Fewer sky samples will not pay until texels carry the sky's dependence
    on the normal beyond first order.
  - Lights are different. A ratio of shadowed to unshadowed light, with the
    unshadowed light evaluated per pixel, removes most of their normal
    error (M5 gate 7).
