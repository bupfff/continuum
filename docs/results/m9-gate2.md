# M9 gate 2: a moving light through the error ledger

**FAIL on one criterion of four (speed, on fallback alone).** A sun drifting
at a 24-minute day's speed (flatland gate 7's "realistic" speed) over the
million-boulder field:

| Criterion | Result | Verdict |
|---|---|---|
| 1. Soundness: no page beyond its certified bound | 0 violations in 30 checks; no page shown differed from a re-bake at all | PASS |
| 2. Quality: no texel more than ε from a re-bake | 0 pages | PASS |
| 3. Work | 0.69% of pages read were re-baked for the sun (at most 25%) | PASS |
| 4. Speed | mean and 99th percentile passed; fallback 33.3–34.2% (at most 10%) | **FAIL** |

Where it fails: pages off screen that need a re-bake are set aside, and the
fast flight brings them into view a frame before they can be re-baked.

What the ledger certified is exactness, not deferral within ε. Its ε went
into snapping estimator noise. Penumbra texels re-baked whenever the light
moved, so no texel was ever kept with a non-zero bound.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `92461f5`; binary
`experiments/room3d/src/bin/m9_gate2.rs`, criteria pre-registered in
`98f70ce`).

## Method

Texels store each light's shadowed share of its unshadowed light (ratios,
M5 gate 7); shading multiplies by the unshadowed light where the light is
now, so only a ratio can be stale. For a light marked as moving
(`RendererOptions::moving_lights`, `Renderer::set_light`):

- **A guard cone.** Each texel also traces a cone around the light's
  center, three times its angular radius.
  - **Clear:** nothing meets it. The ratio stays exactly 1 while the
    light, seen from the texel, stays inside the guard.
  - **Fixed:** the texel faces away by at least twice the light's radius.
    It cannot change while the light's center stays below its horizon.
  - **Mixed:** anything else. It re-bakes, alone and without a guard cone,
    whenever the light moves. Its bound (the band the light's edge sweeps,
    over the unshadowed light, plus ε for snapping) always exceeds ε then.
- **Exact ends.** A cone that nothing covers stores 1, one wholly covered
  0. A moving light's ratios within ε of 0 or 1 are stored as 0 or 1.
- **Per-texel frames and a page's slack.** Each page keeps, per moving
  light, its version (the frame it was baked whole) and each texel's frame
  and state. It also keeps its slack: the smallest guard of its clear and
  fixed texels, less the light's shift since the version. The light's poses
  are kept in a ring of 1,024 frames.
  - The light's shift seen from a page bounds every point's angular motion:
    twice the arcsine of half the largest displacement over the distance to
    the page's box.
  - Past a hashed share (a half to all) of its slack, a page re-bakes
    whole, with fresh guards.
- **Every frame**, for pages read in the frame before, the ledger appends
  what needs re-baking to the frame's bake. Other such pages are set aside:
  kept but shown as missing, and re-baked the frame after a pixel wants
  them.

`verify_ledger` re-bakes every page shading may read from scratch, compares
each page's largest change with the bound certified for it, and restores
the cache. The GPU test `a_drifting_sun_stays_within_the_ledger` checks the
same on a small scene.

## Scene

- The million-boulder field with M5 gate 6's three colored lights, no sky,
  hardware camera rays, cache defaults, prefilter 1 bin.
- The sun turns 1/240° per frame about the vertical axis through the
  field's center: 0.25° per second at 60 frames per second. It stands about
  2.2 km from that axis, so seen from the field it shifts about 1.6e-4
  radian per frame, its guard of twice its 0.35° radius lasting about 80
  frames.
- The flight is M5 gate 8's path: 600 frames from standing height to
  150 m up, from an empty cache.

## Results

**Control**: static, sun-only warm frame 1.84 ms after 112 frames (at most
3.3 ms): fit.

**Criterion 4, speed** (measured first): flights alternating, sun still and
drifting.

| Flight | Mean | 99th percentile | Worst | Fallback | Bake budget |
|---|---|---|---|---|---|
| 1, still | 15.70 ms | 26.94 ms | 47.32 ms | 1.82% | 2,628 |
| 1, drifting | 14.47 ms | 29.08 ms | 56.90 ms | **33.84%** | 2,078 |
| 2, still | 15.57 ms | 26.15 ms | 57.17 ms | 1.83% | 2,469 |
| 2, drifting | 14.45 ms | 29.74 ms | 57.43 ms | **33.30%** | 2,081 |
| 3, still | 15.66 ms | 28.40 ms | 56.89 ms | 1.82% | 2,483 |
| 3, drifting | 14.58 ms | 28.54 ms | 53.66 ms | **34.20%** | 1,843 |

The still flights average 15.64 ms. Each drifting flight's mean is under
16.7 ms and under 1.25× that (19.56 ms), and its 99th percentile under
33.3 ms. But fallback exceeds 10% in all three, so this is a **FAIL**.

The drifting flights are faster than the still ones because a pixel shown
with fallback costs less than a cached one. Even with the sun still, these
renderers trace guard cones for it.

**The ledger's work per frame** in the drifting flights:

| Flight | Pages read | Passed whole | With mixed texels | Re-baked (texels) | Set aside |
|---|---|---|---|---|---|
| 1 | 72,743 | 42,759 | 209 | 214.3 (14,685) | 29,919 |
| 2 | 72,083 | 42,646 | 201 | 207.5 (14,509) | 29,374 |
| 3 | 70,995 | 41,903 | 168 | 174.3 (12,860) | 29,051 |

**Criterion 3, work**: pages re-baked for the sun averaged 0.78%, 0.70% and
0.60% of the pages read in the frame before, 0.69% over the three flights.
The criterion is at most 25%, so this is a **PASS**; exact invalidation
re-bakes all of them.

But about 29,000–30,000 pages a frame were set aside, which is what the
fallback shows. That count is larger than the guard's lifetime alone
explains: about 1/80 of the resident pages would be a few thousand a frame.
Its cause is not yet established. One suspicion is that the ledger also
walks slots the CPU has freed, whose page records stay behind.

**Criteria 1 and 2, soundness and quality**: over the first 300 frames,
after every 10th, every page shading may read was re-baked from scratch.
Checks covered 10,548 pages at frame 10, rising to 57,409 at frame 300. In
all 30 checks:

- no page changed beyond its bound (criterion 1, **PASS**);
- no page changed beyond ε (criterion 2, **PASS**);
- in fact no page changed at all, and the largest certified bound was 0.

**Reported, the lamp circling** (0.5 m per second, a 1.5 m panel 6 m up):

- flight mean 28.10 ms, 99th percentile 71.07 ms, fallback 63.57%;
- 0.54% of pages read re-baked, about 28,500 set aside a frame;
- 10 checks found no page changed.

## What the result says

1. **Sound, and in fact exact.** The ledger never showed a stale value. The
   values it kept were exactly a re-bake's: clear texels hold 1 within
   their guard, fixed texels stay away, mixed texels re-bake.
2. **The ε margin went into snapping, not deferral.** Mixed texels' bounds
   at this speed are about 16% a frame for the sun, about 5× their true
   change of about 3.4%. With ε added for snapping, they always exceed ε.
   So no texel was kept with a non-zero bound. Bounded deferral (the
   paradigm's ε at work) would need slower lights, a tighter bound, or a
   larger ε.
3. **Clear texels carry the savings.** Most pages passed whole each frame
   (about 59% of the pages read), and only 0.7% re-baked.
4. **Set-aside pages are the cost.** Off-screen pages that outlive their
   slack are shown missing until wanted, and in a fast flight a third of
   the pixels land on them.

## Changes before the full-size run

These are disclosed in the binary's doc comment. They followed the
development test on the small scene (`a_drifting_sun_stays_within_the_ledger`)
and development runs at 640×360.

- **No dark state.** A page wholly covered in its guard, even by one mesh
  alone and with the light's own cone wholly covered, still let a re-bake
  0.37° later see 1% of the light. Shadowed texels count as mixed.
- **Snapping, exact ends, and a horizon margin** (clear only with the
  light's center at least twice the guard, and 0.1 radian, above the
  horizon). Estimates of grazing or nearly covered cones wavered by up to
  6% as bins realigned.
- **A page's slack, partial re-bakes without guard cones, staggered whole
  re-bakes.** Checking every texel's shift cost about 5 ms a frame.
- **Fixed texels only well below the horizon.** A sliver of margin made a
  page's slack a sliver.
- **Set-aside pages kept**, not evicted: evicted pages came back as whole
  planned pages with every light, and fallback rose to about 30%. A cap at
  the frame's budget did the same (32%); the cap is now eight times it, a
  safety limit.

## Development runs (disclosed)

At 640×360, one flight each (`M9_DEV=1`):

| Version | Still | Drifting | Fallback (drifting) | Pages re-baked |
|---|---|---|---|---|
| First | 14.05 ms | 17.57 ms | 13.17% | 6.22% |
| Per-texel shift checks | 14.15 ms | 19.19 ms | 13.12% | 6.99% |
| Slack per page, mixed re-baked alone | 14.79 ms | 14.47 ms | 11.28% | 6.61% |
| Ledger capped at the budget | 13.76 ms | 14.54 ms | 32.36% | 1.32% |
| Staggered whole re-bakes | 13.95 ms | 13.73 ms | 29.80% | 3.03% |
| Pages set aside, kept | 14.46 ms | 14.12 ms | 33.13% | 1.91% |
| Fixed texels well below the horizon (final) | 14.31 ms | 14.29 ms | 31.72% | 1.96% |

Every development run's checks found no page beyond its bound or beyond ε.
The small-scene test failed its checks before the changes above: first
with 61 pages beyond their bounds, then 47, 32, 6, 5, 1.

## Caveats

- **Not with movers, nor a sky, in the cache** (both refused at
  construction). Spots cannot move.
- The ledger's records take 100 MB per moving light at the default 2^19
  pages.
- The sun's speed is flatland gate 7's "realistic" one, a 24-minute day:
  60× a real sun. At a real sun's speed, guards last about 80 seconds.
- Apple M4 only.

## What next

Each item below is for a new gate.

- **Fallback:** re-bake set-aside pages before they come into view (for
  example, within spare budget, nearest the camera first), and find why so
  many are set aside each frame.
- **Deferral within ε:** a tighter bound for mixed texels (the window's
  true perimeter instead of its circumscribed circle, and no doubled
  term). Then slower lights could keep penumbra texels for several frames.
- **A dark state that holds:** find why a guard wholly covered by one mesh
  still let a moved light through.

## Prior art

Error-bounded reuse with a ledger is the paradigm's third rule
(`docs/DESIGN.md` §3.1). Flatland gate 7 tested it in 2D. The closest
prior art is listed there; no novelty claim is made for the guard-cone
test.
