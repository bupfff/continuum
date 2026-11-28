# M23: why the field never settled, and bounce hits lit by rays (development, 2026-11-27)

Development after M23 gate 1 (parked unrun), not a gate. Numbers from
`viewer_repro` (the viewer's field: 1470×923, the sun, no sky, so two
bounces; `VR_FRAME_BUDGET=0`: no frame cap, the bake at its 20,000-unit
ceiling) and the scorecard's field (1280×720, sun and sky, one bounce,
`Renderer` defaults, so the 11 ms frame cap). `CONTINUUM_TRACE_BAKES`
traces (stderr) and `SC_TRACE=<n>` (new: the cache's counters while the
scorecard settles).

## Diagnosis

At rest, after 1,000 frames, every frame still baked 250–500 pages, all
pages asked for by bounce rays' hits (secondary requests), alternating
with a stream of bounce retries (277 pages a frame, nearly every sample
incomplete). Three causes, each enough to keep it from settling:

1. **Dropped requests.** A frame's secondary requests the bake budget
   could not fit were dropped; only the sample's retry asked again. A
   retry batch of 277 pages asked for ~8,000 pages, of which the next
   frame baked ~195. The retry queue was ~100,000 pages long (served
   FIFO at the budget's rate: a page baked at frame 578 retried at frame
   1,050), so each retry found most of its hits still missing, asked
   again, and waited its turn again.
2. **Two bounces' working set.** With no sky the viewer's field takes two
   bounces: the camera samples' hit pages carry samples of their own,
   whose hits need pages too. 211,000 distinct first-level hit pages
   baked in 1,500 frames, the second level unbounded; resident pages
   passed `EVICT_ABOVE` (131,072), so pages were evicted and asked for
   again.
3. **Distant boulders.** 28,000 boulders a few pixels across each had a
   page at their chart's top mip with nine bounce samples, all nine at
   the chart's single texel, whose rays asked for their neighbours'
   pages.

On the scorecard's field (one bounce) the same: after its camera pages,
~100,000 hit pages (distinct boulders' top-mip pages, the ground at mips
6–7, then near boulders at mips 2–6), which the frame cap's budget (about
1,500 units, 15–20 pages a frame) cannot bake within a minute.

## Changes

- **Pending hits** (`hit_pending`): secondary requests wait in a FIFO
  (deduplicated, at most 2^18) instead of being dropped, baked after the
  pixels' pages and the eligible retries.
- **Retry tickets:** a retry from an incomplete sample waits until the
  pages its hits asked for (in the same readback) have baked; retries of
  pages shown without samples, or waiting for lighting levels, do not
  wait. (A first version made every retry wait for the oldest pending hit:
  pixels' requests for samples then waited behind the backlog and the
  scene never went quiet; fixed.)
- **Shared sample texels baked once:** on a page at a chart's edge several
  of the 3×3 samples clamp to one texel (all nine on a one-texel chart);
  the first is baked and stored for all, and the bake budget counts only
  distinct ones.
- Options, off by default, hardware ray queries without movers:
  - `last_rays`: with more than one bounce, the last level's hits lit by
    rays (as M23's `far_rays`: the unshadowed light times one shadow ray
    toward each light, the sky one cosine-distributed ray, no page).
  - `far_samples`: a page at its chart's top mip bakes its samples as the
    last level, every hit lit by rays.
  - **`hit_rays`**: every hit of a one-bounce sample lit by rays. Bounce
    samples read no page, so they are complete at their first bake and
    ask for nothing.
  - `bounce_coarser`: the hits' mip offset (2 as before).
- Bug found on the way: a ray that missed everything reached `far_gather`
  through the new branch (no mesh), whose emission read garbage: NaN in
  every far page's emission channel, the whole frame black. Metal's fast
  math folds `x != x` to false, so finding NaN in a shader takes a test of
  the exponent bits.

## Measurements

`viewer_repro`, uncapped (two bounces):

| | Requests and incomplete samples at rest | Pages baked a frame at rest |
|---|---|---|
| Before | bursts every ~250 frames: ~2,400 samples, nearly all incomplete, 4,000–10,000 requests | 250–500, never stops |
| Pending hits + tickets | ~97% of samples incomplete (the second level) | ~190 |
| + `last_rays` | 0 from frame 400 (a burst at 2,000–2,100, resident ~190,000) | ~150 through frame 3,800 (a backlog of ~110,000+ first-level pages) |
| + `far_samples` | the same | the same (first-level keys baked by frame 1,500: 162,000 against 211,000) |

The scorecard's field (one bounce, the 11 ms cap), warm-up to settled
(cap 60 s), then against Cycles (2560×1440, 120,000 texels a frame, at
most 600 frames):

| | Settled | Settles again after the walk | Against Cycles |
|---|---|---|---|
| Before (recorded 2026-11-22) | not within 60 s | not within 30 s | 1.79% (1.29% with area texels) |
| Pending hits + tickets | not within 60 s (requests 0 from 7.6 s; ~100,000 hit pages left) | not within 30 s | 2.22%, not settled in 600 frames |
| + hits 3 mips coarser | not within 60 s | | |
| + `far_rays`, `far_samples` | not within 60 s | | |
| + `hit_rays` | **7.86 s (777 frames)** | **2.35 s** | **1.16%**, settled at frame 139 |

With pending hits and tickets alone the page path converges (nothing
incomplete, nothing asked) but slowly: samples wait for their hits, so at
600 frames the field is darker than before (2.22% against 1.79%). With
`hit_rays` nothing waits: the settled frame drops to 6.65 ms (no bake).

Side by side (`pair_dev`, `PAIR_OPTION=hit_rays`, before the movers'
variant; then movers, Fox and BrainStem again with it): settled frame,
moving frame, settled at, against Cycles, `hit_rays` off / on.

| Scene | Settled frame | Moving | Settled at | Against Cycles |
|---|---|---|---|---|
| sponza | 22.53 / 22.56 ms | 30.87 / 31.46 ms | 47.15 / **26.14 s** | 25.33% / 25.05% |
| field | 10.55 / **7.17 ms** | 12.55 / 10.84 ms | never / **9.25 s** | 2.25% / **1.16%** |
| room | 5.29 / 5.34 ms | 8.29 / 8.59 ms | 2.10 / 1.41 s | 6.93% / 6.65% |
| movers | 5.21 / 5.20 ms | 19.91 / 19.76 ms | 1.99 / 1.42 s | 5.38% / 5.30% |
| helmet | 4.79 / 4.75 ms | 6.98 / 6.86 ms | 0.95 / 0.66 s | 4.46% / 4.46% |
| fox | 5.11 / 5.10 ms | 9.22 / 9.20 ms | 1.47 / 1.17 s | 2.17% / 2.13% |
| brainstem | 5.96 / 5.95 ms | 34.92 / 36.59 ms | 5.13 / 3.81 s | 3.28% / 3.27% |

Then: retries also go `EVICT_AGE` frames after asking (behind a long
backlog a retry with some hits beats none), and `far_gather_movers` (shadow
and sky rays that see movers) so the ray-lit paths run with movers too.

A test caught a race the pending hits exposed
(`movers_under_a_sky_stay_within_the_ledger`, deterministic on Metal; it
failed "now and then" on lavapipe before): a mover's own page whose sky was
due to bake alone in the frame the mover moved again. The stale pass
re-bakes the page for the move, marking its sky a guess after the sky's
bake cleared it, and does not list it again (listed once per sky bake), so
the guess stayed. Secondary pages used to take that frame's budget first
and push the sky later. Fix: a sky baked alone on an object that moves in
the same frame is due again `SKY_SETTLE` frames later.

## Rehearsal 1 (`M23G2_REHEARSE=1`, 2026-11-27): missed, fixed

Field settles (7.37 s; again 2.35 s), the movers within their bars, but
the field from above 11.84% → 12.44% from Cycles (bar +0.5). Against a
one-bounce Cycles render (`altitude_dev`, new `AD_CYCLES_BOUNCES=1`; mean
0.0334 against four bounces' 0.0337) the same gap (10.81% → 11.42%), so not
bounce depth. The loss sat in a strip at the ground's far edge: `far_gather`
(M23's, reused) counted its sky ray only above the horizon, where the sky's
own bake and Cycles count any ray that escapes, and past the ground's edge
rays escape below it. Counted there too: 12.17% (one bounce: 11.16%); the
rest of the gap is in the same saturated far strip, both under Cycles.
Nearer bands with `hit_rays` are closer to Cycles than without (band RMSE
0.0026–0.0035 against 0.0047–0.0049). Without `hit_rays`, the retries'
`EVICT_AGE` timeout brought the field back to 1.22% (2.22% with tickets
alone).

## After gate 2 (failed on Fox moving 1.138×)

Repeated side by side, one pass of the motion: Fox 1.010×, 0.989×;
BrainStem 1.009×, then 1.195× (its side without the candidate 34.18, then
29.10 ms). Three passes (`speed_pair_passes`, new; `pair_dev
PAIR_PASSES=3`): Fox 0.969×, 0.994×; BrainStem 0.963×, 0.989×; movers
1.006×, 1.030×. The ray-lit hits' shadow rays now go opaque and end at the
first hit (`hw_segment_open`): `hw_segment_open_but` with nothing to skip
visited every candidate along a ray across the field for the same answer.
Gate 3: the same criteria, three passes.

## After gate 3 (passed): on by default; the viewer's field

`hit_rays` on by default. The viewer's field (no sky, so two bounces:
camera samples' hits still read pages, which hold ray-lit samples of
their own) then showed a starvation the retry timeout brought: cheap
one-texel pages' retries, timed out, took the whole budget every frame
ahead of the pending hits they waited for (~2,500 retries a frame, all
incomplete, resident pages flat at 75,571 for 2,000 frames). Now retries
whose hits arrived go first, then the pending hits, then timed-out retries
with what is left. `viewer_repro` after (4,000 frames):

| | Requests, incomplete | Pages baked a frame at 3,950 | Resident |
|---|---|---|---|
| Uncapped | 0, 0 from frame 300 | 221 | ~190,000 (evicting) |
| The frame cap (11 ms) | 0, 0 from frame 1,000 | 76 | 179,000 |
| Uncapped, `far_samples` | 0, 0 (a burst at 1,000) | 167 | ~190,000 |
| Capped, `far_samples` | 0 from 1,000 | 26 | 163,000 |

Not settled in 4,000 frames either way: two bounces' first-level pages
(each with nine samples) outnumber what stays resident under
`EVICT_ABOVE` (131,072). Levers left: one bounce outdoors under a sun
alone (`Renderer`'s rule counts a sun as distant only past 100 scene
radii; the field's is at 1 km), a higher `EVICT_ABOVE` (the pool holds
524,288), first-level pages' samples shared across a chart.

## The viewer's field (2026-11-28)

The viewer's field defaults to its sun and sky (one bounce), but the
viewer turned bounce light off there because it never settled.
`viewer_repro VR_SKY=1` (new; 1470×923, bloom, anti-aliasing, the 11 ms
cap): settled at frame 1,406; after turning to the sun and back, 0.00% of
the ground darker than settled. Bounce light now on by default in the
viewer's field too.

Tried for the sun alone (two bounces): `one_level` (every sample the last
level, the orders past one bounce estimated by `bounce_rest`; off,
experimental): the closed room 6.66% → 20.24% from Cycles, the movers
5.30% → 16.96%; the series cannot stand in for a real second bounce
indoors. The sun-only field against Cycles (`pair_dev SC_ONLY=field_sun`,
new) is not measurable at the scorecard's samples: its mean is 0.0003
(the sky gives nearly all of the sky-lit field's 0.03), and Cycles' noise
gives 150–163% either way.

## Sponza's blotches (2026-11-28, development; code kept as a patch)

Sponza's remaining gap is mostly low-frequency (62% of the error survives a
33-pixel blur), ours about 6% darker; not bounce depth (Cycles at 1 and 4
bounces give the same error, 41.46% and 41.00% in `gltf_dev`). The visible
part: recessed walls (the lion wall, the upper gallery) blotchy, in
page-sized cells. Bisected: bounce light; with `hit_rays` the hits' one
sky ray a hit makes it worse, a checker of pages (sky-only clay: pages
smooth, rays checkered), since each hit's sky is a binary estimate (~14%
noise a sample where hits see a fifth of the sky) and each page rebuilds
its nine samples. Not the hash (PCG3D: the same). Tried, in
`experiments/room3d/out/patches/bounce-refine-and-sky-rays.patch` (not
committed): four stratified sky rays a hit (about half the noise, still
noisier than pages); one stratum a hit across a sample's 192 hits (worse:
banded); `bounce_refine` (four rays only in bins at a bright edge: a
little smoother, far short of four rays a bin). Four rays a bin
everywhere: Sponza 25.04% → 23.43% from Cycles, but settling 1.16–1.74×.
Next: the hits' sky from a resident page where there is one (no request),
rays only where none is; then measure four rays a bin against it.

Then (2026-11-28): hits' sky from resident pages where any (no request),
one ray where none: Sponza 25.04% → 25.76% from Cycles (worse; the upper
gallery's hits land on pages nobody asked for, and coarse pages' sky);
dropped. Two rays a bin (the diagonal quarters) side by side over three
motions (`pair_dev PAIR_A=1 PAIR_B=2 PAIR_PASSES=3`): Sponza 25.09% →
23.61%, the movers 5.30% → 5.04%, the room 6.66% → 6.52%, Fox 2.11% →
2.06%, the rest equal; settled frames 1.00×, moving 0.97–1.06×; settled
at 0.71–1.23× (the field 7.20 → 8.48 s, Sponza 37.38 → 26.71 s). M24 gate
1 stub: two rays a bin by default. Kept: ray-lit hits trace no sky ray
without a sky.

## The field under the sun alone (2026-11-28)

A higher `EVICT_ABOVE` does not settle it: at 0.6 and 0.9 of the pool
(313,000 and 471,000 pages resident) `viewer_repro` uncapped still baked
150–200 first-level pages a frame after 4,000 frames: two bounces' hit
pages on a million boulders outnumber the pool. `Renderer`'s rule now
counts as a sun (one bounce outdoors) a rect light outside the scene's box
and under 2° across from its centre, as well as one past 100 scene radii:
the field's sun (1 km up, the field 2 km across) qualifies; the
showcase room's skylight does not (the room 6.52%, the movers 5.04% from
Cycles, unchanged). The viewer's field under its sun alone then settles at
frame 989 (about 11 s at the 11 ms cap, 1470×923); after turning to the
sun and back the ground comes back 2% brighter than its settled start
(0.1408 against 0.1379), none darker. One bounce there leaves shadows lit
by one bounce of sunlight only (no sky); not compared with Cycles (at the
scorecard's samples Cycles' noise dominates a sun-only field).

`gltf_dev` now takes the cache's bounces and rays a bin from
`CacheOptions`'s defaults (it fixed one ray, so it read Sponza at 25.07%
where the scorecard reads 23.60%).

Sponza's remaining gap, by surface (`gltf_dev` at the scorecard's end
view, `GD_IDS`): the red drape over the arch (3.1% of pixels) 19.8% of the
squared error, ours 0.62 of Cycles and more saturated (blue half, red
two-thirds); not its normal map (0.618 without); in clay it is no worse
than its neighbours, so its material response, unexplained (it is rough,
0.98, dielectric). The arches 16.7% (0.96), the spandrels 12.1% (0.92),
the floor 11.3% (1.05).

## Flying up: the dark band (2026-11-28, diagnosis)

`viewer_repro VR_RISE=<m>` (new): from the settled start the eye rises
that far over `VR_RISE_FRAMES` (240) frames; on arrival the image is
compared with the same view settled. The viewer's field (sun, sky, bounce,
1470×923, the 11 ms cap):

| Rise | Most pixels without a page | On arrival: darker than half, brighter than twice | Settles again |
|---|---|---|---|
| 40 m in 240 frames | 75,926 (5.6%) | 4.81%, 2.78% | 2,953 frames |
| 40 m in 2,000 frames | 7,443 | 4.78%, 2.86% | 2,164 frames |
| 150 m in 240 frames | 198,274 (14.6%) | 0.71%, 2.06% | not in 4,000 |

The band is the user's: dark patches of ground between boulders in the
middle distance and near the camera. On arrival only 26% of sampled pixels
show their wanted mip (a slow rise the same): the ground shows pages
coarser than wanted, and those are dark in patches. Not the pixels without
any page (those are the far boulders, shown unshadowed and brighter); not
bounce light (the same without it); not a finer page standing in (tried:
no change). Settled at 256 pixels a texel, sun alone, no bounce, the same
patches appear: the near band 14% darker than at 2 pixels a texel (0.1097
against 0.1281), the whole lower half within 2.5%. So coarse ground texels'
sun visibility, traced as cones through the prefiltered hierarchy over
many small boulders, comes out dark in places, and flights show those
pages while the wanted ones bake (about 30 pages a frame at the cap).
Area texels (16 hardware rays) cover only a chart's top three mips and
curved texels; forced over the top six and flat texels too, no change, so
the dark pages are coarser than wanted but finer than that. Levers: area
visibility for texels by size (larger than the occluders), or shadow rays
for coarse texels; faster arrival of the wanted pages (distant boulders
each take a page).

Confirmed: area visibility (16 hardware rays) on every texel removes the
patches (settled at 256 pixels a texel, sun alone: the near band 0.1179
against 0.1097 with cones, 0.1281 at 2 pixels a texel). New options
`area_mips` (a chart's top mips that average; 3) and `area_flat` (flat
texels too). The 40 m rise, on arrival at most half as bright: 4.81%
before; flat over the top 10 mips 0.52%; over all mips 0.45% (the start
slower to settle: 2,289 frames against 1,737). Side by side (top 10, flat;
three motions): Sponza 23.61% → 23.89%, the field 1.12% → 1.26%, the room
6.52% → 6.12%, the field from above 12.18% → 11.67%, the rest equal;
frames 0.996–1.023×, settling 0.89–1.17×. M25 gate 1 stub.
