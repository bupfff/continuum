# M10 gate 1: one bounce of light in the view-dependent cache

**FAIL on one criterion of three.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Accuracy against Cycles with one bounce | **1.79%**, against 7.09% without bounce light | PASS |
| 2. Speed | warm frame **1.07×**; settled in 1.21× the frames | PASS |
| 3. Re-coloring bakes nothing (30 frames) | **794 pages baked** | FAIL |

Criterion 3 failed on the gate's procedure, not on re-coloring. Re-coloring
on a settled view baked nothing in 200 frames in a development check. The
794 pages came from cache activity that resumed after the supersampling the
gate ran just before, which the settle test had not caught. Its cause is not
yet established.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `922dc55`; binary
`experiments/room3d/src/bin/m10_gate1.rs`, criteria pre-registered in
`2bac765`).

## Method

`CacheOptions::indirect` turns it on.

- **Samples per page.** Each page holds a 3×3 grid of bounce samples, at
  its texels 0, 4 and 8 along each axis. Shading reads them bilinearly.
  Each sample is stored per light and for the sky, RGB, per unit radiance
  (half floats), so re-coloring a light stays free.
- **A hemicube of rays.** A sample traces one ray through the center of
  each of 192 bins of a hemicube around its normal (M2 gate 6's layout).
  Each bin is weighted by its projected solid angle, computed in the shader.
- **Light at the hits from the cache.** A hit's light is its albedo
  (textured, at the bin's footprint) over π times the direct light the
  cache holds there:
  - each light's ratio times its unshadowed light, plus the sky's channel;
  - read at the mip matched to the bin's footprint, or the finest coarser
    one resident.
  - A back face contributes nothing.
- **Missing hits.** A hit with no page resident requests one. A page whose
  samples missed more than 1% of the hemisphere re-bakes them 4 frames
  later, once the requested pages can have arrived.
- **Limits.** At most three lights. Not with movers or moving lights yet:
  a bounce depends on everything, which is the ledger's job.

## Scene

The textured scene of M6 (`Textured::new()`: sun and sky), its camera,
1280×720, cache defaults.

## Results

**Criterion 2, speed** (measured first):

| | Without bounce light | With |
|---|---|---|
| Warm frame (median of 30, each waited for) | 2.23 ms | 2.39 ms (1.07×) |
| Frames until settled (32 quiet frames) | 530 | 640 (1.21×) |

The criterion is at most 1.25× and at most 3×, so this is a **PASS**.
While settling, 124,128 samples were baked, and 7,048 met missing hits and
were re-baked.

**Criterion 1, accuracy**: the settled images were supersampled 4×4, each
sub-pixel offset's view settled first. The reference is instanced Cycles
with one diffuse bounce, at 4,096 samples per pixel (54 s on Metal).

| Image | Against Cycles, one bounce | Mean against Cycles |
|---|---|---|
| With bounce light | **1.79%** | +0.15% |
| Without | 7.09% | −2.43% |

The criterion is at most 5%, and at most half the error without bounce
light. Ours is 0.25× that, so this is a **PASS**.

- Bounce light adds 2.63% to the image's mean.
- Control: the image without bounce light is 1.57% from Cycles without
  bounces. M6 gate 1's 0.43% was per pixel at 16×16 samples; this one is
  the cache at 4×4.

**Criterion 3, re-coloring**: after supersampling, the view was settled
again (32 quiet frames), the sun re-colored, and 30 frames rendered. They
baked 794 pages, so this is a **FAIL**.

The development check behind the opening note, run after the gate (at
1280×720, `M10_DEV=1 M10_TRACE=1`):

| Sequence | Frames with anything baked, of the next 200 |
|---|---|
| Settle, re-color | 0 |
| Settle, nothing | 0 |
| Supersample, settle, nothing | yes: requests resume (13, then 31, 24, ...), with evictions and samples re-baked |

So re-coloring bakes nothing, as designed. After supersampling, "settled"
did not hold, because requests resumed after the 32 quiet frames. Its cause
is not yet established. One suspicion is pages used only at the sub-pixel
offsets, or only by bounce hits, which are evicted and then requested
again.

## Caveats

- One scene, a sunny outdoor one where bounce light is a few percent. A
  scene lit mostly by bounce light (an interior) is untested.
- The bounce samples' 3×3 grid per page smooths contact shadows of
  indirect light at fine mips; at coarse mips a page spans meters.
- The light at hits comes from the cache, so it shares its blur at coarse
  mips. Hits on surfaces never seen are requested and baked for them.
- Bounce samples take 151 MB at the default 2^19 pages.
- Apple M4 only.

## What next

Each item below is for a new gate.

- Find why the cache does not stay settled after the view has moved
  (sub-pixel offsets here), and test re-coloring again.
- Bounce light with movers and moving lights, through the ledger.
- An interior scene, where most light is indirect.

## Prior art

One bounce gathered from a cache of direct light is irradiance caching
(Ward et al. 1988) and radiance caching. This gate makes no novelty claim.
