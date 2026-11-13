# M10 gate 25: more than one bounce

**PASS.** The cache now carries light past its first bounce
(`CacheOptions::bounces`). In the showcase's closed gallery, lit by one
skylight panel, 8 bounces bring the image from 45.4% to 6.5% from Cycles at
12 bounces; the textured scene outdoors improves slightly; views settle in
tens of frames.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 1. Room, 8 bounces, against Cycles at 12 | **6.54%** against one bounce's 45.36% (0.144×) | at most 10% and 0.25× | PASS |
| 2. Settling, 8 bounces | every offset within 80 frames | each within 2,000 | PASS |
| 3. Textured scene by day, against Cycles at 12 | 8 bounces 2.11%, one bounce 2.36% | at most +0.1 point | PASS |

| Room | 1 bounce | 2 | 4 | 8 | Cycles, 12 |
|---|---|---|---|---|---|
| Mean radiance | 0.1856 | 0.2421 | 0.2968 | 0.3032 | 0.3182 |
| From Cycles at 12 bounces | 45.36% | 26.62% | 8.37% | 6.54% | |
| Frames to settle from empty | 31 | 31 | 42 | 80 | |
| Bounce samples to settle | 27,243 | 30,672 | 64,503 | 157,068 | |

Textured scene: from empty in 48 frames with 24,759 bounce samples (one
bounce) and 103 with 184,788 (eight).

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-13 (stub `6064d42`; implementation and binary `df8c5db`, before the
run). 640×360, the cache settled at 4×4 sub-pixel offsets (32 quiet frames
each) against instanced Cycles at 4,096 samples per pixel and 12 bounces.
The room is the showcase's static surfaces as an instanced scene, no sky.

## Method

- A bounce sample (192 rays) adds, at each hit, the bounce light of the
  page there (its 3×3 samples, bilinearly, per channel) to the direct light
  it already read, times the hit's albedo over π.
- Each sample records the bounces it holds (in the half-float its slot kept
  for "baked"): one more than the fewest its hits held, or 1 with one
  bounce.
- A page's samples should hold `bounces` if pixels asked for it, one fewer
  per bounce away if bounce rays did (request priority bits 24–27, carried
  into the bounce jobs), down to pages with no samples. A hit holding too few
  is asked for with samples (resident pages are scheduled at once); the
  sample's share whose hits held too few counts as missing, so the page
  retries (at most 4·bounces + 8 times per allocation, so a page whose hits
  never gain bounces cannot keep the view from settling).
- `bounces` 1 (the default) is the old behavior exactly.

**Disclosed (before the run):** the room's numbers in development and the
development mode's results are listed in the binary; the textured scene
with several bounces was first run in the development mode.

## Caveats

- **At least, not exactly, `bounces`:** a sample gathering hits that hold
  more than it needs holds more itself, so the room's 4 bounces (0.297) sit
  near Cycles' 4 (0.283) plus a fifth bounce. The setting is a floor.
- **Still 4.7% dark at 8 bounces** (0.303 against 0.318): the one-bounce
  estimate is 1.3% dark against Cycles' one bounce, and each bounce reads
  hits through 3×3 samples per page and coarser pages, compounding.
- **The result depends a little on the bake order** (about 1% between runs
  with 4 bounces in development).
- **Several bounces are opt-in.** Outdoors they cost 7× the bounce samples
  to settle for 0.25 point, and with streamed lighting levels and movers they
  are untried; the default stays 1.
- A page first asked for by bounce rays and later seen directly keeps the
  bounces it was asked for (one fewer), unless never baked.

## What the numbers say

- **Interiors now have their light:** a closed room goes from 45% to 6.5% of
  Cycles' fully bounced image, settling in 80 frames.
