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
