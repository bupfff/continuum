# M5 gate 5: sky light through the view cache at speed

Date: 2026-10-25. Gate 4 put sky light in the view-dependent cache, but sky
texels cost about 100× sun texels, the bake budget's 1,000-texel floor made
frames slow, and its timings were confounded by the machine. The
diagnostics before this gate (`m5-sky-bake-diagnostics.md`) found why sky
texels cost so much. This gate measures the cache with the fixes.

## What was built

- **Sky jobs per face, and a horizon cull.** Each sky sample is six bake
  jobs, one per cube face, listed face by face. Sky cones skip geometry
  wholly below the receiver's tangent plane, which is exact for the
  irradiance integral.
- **Own-object sky masks** (`CacheOptions::self_spacing`, default 1 cm).
  - **The fact they use.** A rigid instance's coverage of its own sky, seen
    through cube faces in its own frame, is the same for every instance of
    its geometry.
  - **The bake.** It is baked once per geometry, on a grid over each chart,
    as 64 bins of 8-bit coverage per face.
  - **Use.** Sky texels on such objects turn their faces with the object,
    walk only the other objects, and combine the interpolated masks per bin
    as one more independent occluder, exactly as meshes combine.
  - **Skipped geometry.** A chart whose grid would exceed 1,024 texels walks
    itself (here, the 2 km ground).
  - **Why it matters.** The own object was 85% of a sky texel's
    triangle-hierarchy work. A whole sky warm-up at the standing viewpoint
    went from 17.4 s to 5.7 s of GPU time (development, a fixed budget).
- **A bake budget that follows GPU time.**
  - **Timing.** Where the device has timestamp queries, the bake passes and
    the camera passes are timed on the GPU.
  - **The fit.** Bake time is fit as a latency plus a cost per unit of
    budget, with exponential forgetting. Alternate frames bake 20% more and
    less, so the fit can tell the two apart.
  - **The budget** is what fits the 14 ms target after the camera passes. A
    sky sample counts as 10 sun texels.
  - **Metal timestamps.** An end-of-pass timestamp followed only by copies
    can land too late on Metal (it read the previous frame's value), and a
    render pass's start is not ordered after compute passes. The camera span
    therefore ends at the start of an empty compute pass.
- **The cache's bakes walk alone.**
  - **Why.** Packet traversal (subgroups walking hierarchies together) made
    small sky bakes slower. In one late-flight frame, a packet visited
    916,000 triangle-hierarchy nodes where its invocations needed 30,000.
  - **Per-pixel cones keep packets.** There they are 25% faster.
- **`set_sky_object_frame`** turns per-pixel sky faces with the hit object.
  It exists for criterion 1 and for the comparison with Cycles below.

Prior art: precomputed self-visibility for rigid objects is precomputed
radiance transfer (Sloan, Kautz & Snyder 2002). Per-object occlusion fields
combined at run time for dynamic scenes are shadow fields (Zhou et al. 2005)
and ambient occlusion fields (Kontkanen & Laine 2005). Here the masks are
binned cube-face coverage combined per bin with walked coverage. No novelty
is claimed. Budgets driven by GPU timers are standard in engines (dynamic
resolution).

## Pass criterion

Fixed before the first run, in a6e85eb. Control first: gate 11's sun-only
warm frame at the standing viewpoint at most 8.03 ms, or the run is void.

1. Masks are faithful: at each of gate 8's viewpoints, the warm cached
   sky-only image with masks within 0.5% relative RMSE of the warm cache
   without them (texels walking their own objects through faces in the
   object's frame).
2. The sun channel is unchanged: warm cached sun-only images within 0.2
   points of gate 8's relative RMSE against per-pixel exact cones.
3. Speed with sun and sky: warm, each viewpoint at most 16.7 ms per frame;
   gate 8's flight from an empty cache, three runs, each with mean at most
   16.7 ms, 99th percentile at most 33.3 ms and fallback at most 10%.
4. Warm-up with sun and sky: at each viewpoint, frames until warm at most 10
   times the sun-only cache's.

Gate 4's criterion 1 (per-pixel sky within 3% of Cycles) is reported, not
gated. Development runs before this gate showed that turning the sky's cube
of faces moves its error against Cycles between 1.9% and 4.2% (below).

**Verdict: FAIL on criterion 4.** Criteria 1–3 pass.

## Results

Apple M4 (MacBook Air), plugged in, 999,986 boulders, 1280×720, hardware
rays, sky prefilter ×8. Reproduce with `cargo run --release -p room3d --bin
m5_gate5`.

- **Control:** sun-only warm frame at the standing viewpoint **2.25 ms**
  (gate 11: 8.03 ms), after 112 frames. The machine was fit.
- **Own-object masks:** precompute 463 ms, 21.0 MB.

| Viewpoint | 1. Sky: masks vs walk | 2. Sun vs per-pixel exact (gate 8) | Cached vs per-pixel sky | 3. Warm sun and sky |
|---|---|---|---|---|
| standing | 0.063% | 14.49% (14.49%) | 3.25% | 2.24 ms |
| 10 m up | 0.100% | 25.87% (25.87%) | 9.12% | 2.47 ms |
| 40 m up | 0.147% | 30.88% (30.88%) | 18.87% | 2.56 ms |
| 150 m up | 0.102% | 38.80% (38.80%) | 20.32% | 2.33 ms |

Frames until warm (criterion 4):

| Viewpoint | Sun only | Sky only, masks | Sky only, walking own object | Sun and sky | Ratio to sun only |
|---|---|---|---|---|---|
| standing | 102 | 1,413 | 13,580 | 1,365 | **13.4×** |
| 10 m up | 132 | 7,677 | 31,737 | 7,337 | **55.6×** |
| 40 m up | 128 | 12,937 | 35,399 | 12,217 | **95.4×** |
| 150 m up | 82 | 7,042 | 19,922 | 6,945 | **84.7×** |

Flights with sun and sky (600 frames from an empty cache):

| Run | Mean | 99th percentile | Worst | Fallback | Samples at wanted mip | Bake budget mean | Sky samples | GPU bake | GPU camera |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 14.84 ms | 24.67 ms | 31.19 ms | 4.44% | 0.0% | 11,306 | 653,466 | 10.80 ms | 3.09 ms |
| 2 | 14.82 ms | 24.28 ms | 36.99 ms | 4.45% | 0.0% | 11,317 | 654,432 | 11.14 ms | 3.21 ms |
| 3 | 14.78 ms | 23.96 ms | 37.37 ms | 4.47% | 0.0% | 11,276 | 651,579 | 11.27 ms | 3.26 ms |

Reported: sky only, 4×4 per pixel (640×360, sun 1 km) against Cycles (1,024
samples per pixel): **1.87%** with world-axis faces, **3.68%** with faces in
object frames.

## The sky's error depends on the cube's orientation

Development runs before the gate (`sky_frame_check` and two temporary
variants), sky only, 4×4 per pixel against Cycles:

| Faces | Standing, ×8 | Standing, ×4 | Standing, ×2 | 40 m up, ×8 |
|---|---|---|---|---|
| World axes | 1.87% | 1.45% | 1.23% | 7.21% |
| One fixed frame, turned (one instance's rotation) | 2.81% | 2.64% | 2.55% | 7.32% |
| World axes turned 45° about z | 4.20% | — | — | — |
| Each object's frame | 3.68% | 3.49% | 3.38% | 10.19% |
| Each object's frame, turned 45° | 3.37% | — | — | — |

- **Binned sky coverage has an orientation-dependent error of a few
  percent.** It barely shrinks with finer prefiltering (×8 to ×2), so it
  comes from the 11° bins and how coverage within them is weighted, not
  from splatting.
- **World axes are a lucky orientation on this field**, whose boulders lie
  on an axis-aligned grid. Gates 3 and 4 measured that best case (1.15–1.87%).
- **Every variant is darker than Cycles:** −1.1% (world axes) and −1.2%
  (object frames) at the standing viewpoint, −6.3% and −7.5% at 40 m.
- **A candidate fix:** coverage is applied per bin as an area fraction times
  the bin's average cosine weight. Accumulating first moments of covered
  area per bin would weight it correctly to first order. Untested.

## What this means

- **Sky light now runs in the cache at frame rate.** On the fanless Air, three
  flights over a million boulders with sun and sky average 14.8 ms, with
  99th-percentile frames under 25 ms and 4.5% of pixels without a page.
  Warm frames take 2.2–2.6 ms. Gate 4's run on a slow machine took 242–267 ms
  per frame.
- **Own-object masks are faithful (0.06–0.15%) and make sky warm-up
  2.8–9.6× faster** than walking the own object.
- **But sky light converges slowly.**
  - **Frame counts.** The standing view needs 1,365 frames (about 19 s at
    14 ms), and the views from above 6,900–12,200 (100–170 s). With the sun
    alone, it is 82–132 frames.
  - **During the flights,** no sample reached its wanted mip, against 0.6%
    with the sun alone in gate 11: lighting stays at coarse pages.
  - **What remains expensive.** A sky sample still costs about 10 sun
    texels' worth of GPU time. The work left is the object hierarchy and the
    neighbours' triangle hierarchies.
  - **From above it is worse.** In the last 100 frames of development
    flights, a single page on the ground took about 13 ms of GPU time: its
    texels' faces look across the whole field.
- **The cache is no blurrier with sky:** cached against per-pixel sky is
  3.3–20.3%, as gate 4's 2.4–20.5%.
- **What would fix warm-up.** Sky light from distant geometry changes slowly
  across a page.
  - **Sharing the far field.** Walk it once per page from the page's center
    and combine it with each texel's own near-field walk. Objects would be
    split by distance, not cut by a far plane, which made walks costlier in
    the diagnostics.
  - **Expected gain.** It would amortize most of the object-hierarchy walk
    over 81 texels.
  - **Status.** These are wide cones in cascades, the plan for indirect
    light (M10); the change goes into a new gate.
