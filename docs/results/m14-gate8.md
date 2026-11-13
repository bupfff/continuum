# M14 gate 8: a sky and bounce light with streamed lighting levels

**FAIL** (on the frame budget only). Lighting levels now stream with a sky
and bounce light. Direct light stays certified: 436,004 sun pages and
436,002 sky pages compared against a renderer with every lighting level
resident, none differing. Bounce light converges (settled images within
0.07–0.52% of the reference's). But on 36,864 rocks with a sky and bounce
the flights' 99th percentile is 20–25 ms, against 16.7 ms.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. (a) Sun pages compared in flight / differing | 76,236 / 0 | 198,884 / 0 | 101,190 / 0 | none differing (1e-5) | PASS |
| 1. (a) Sky pages compared in flight / differing | 76,236 / 0 | 198,884 / 0 | 101,190 / 0 | none differing | PASS |
| 1. (a) Settled sun / sky pages compared, differing | 20,478 / 20,478, 0 | 20,525 / 20,525, 0 | 18,691 / 18,689, 0 | none differing | PASS |
| 1. Pages compared in all | 436,004 sun, 436,002 sky | | | at least 100,000 and 30,000 | PASS |
| 2. (a) Settled image's mean difference | 0.065% | 0.520% | 0.087% | at most 1% | PASS |
| 3. (b) Mean frame | 14.05 ms | 15.34 ms | 15.87 ms | at most 16.7 ms | PASS |
| 3. (b) 99th percentile | **20.46 ms** | **21.73 ms** | **25.13 ms** | at most 16.7 ms | FAIL |
| 3. (b) Worst frame | 26.88 ms | 25.40 ms | 31.87 ms | at most 33.3 ms | PASS |
| 4. (b) GPU memory per added rock | 20.0 KB, 0.096× a resident lighting level (207.1 KB) | | | at most 1/6 | PASS |
| 4. (b) CPU heap per added rock | 17.3 KB, 0.059× (291.7 KB) | | | at most 1/6 | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `2eb3d70`; implementation `183905f`; binary
`b4aa8d5`, before the run). Field (b) was built in 206 s and the machine
idled two minutes before the timed flights.

## Details

- **Scenes:** (a) 96 × 96 = 9,216 unique rocks, lighting levels on demand
  into slots for an eighth of them (1,105,920 triangles), beside the same
  scene with every lighting level resident; (b) gate 7's 36,864 rocks and
  options. Both: a uniform sky (radiance 0.05; the sun's irradiance is
  about 0.9), bounce light, the cache's frame target 12 ms.
- **Frames (b):** 256, 540 and 693 frames of 1,800 over 16.7 ms. The
  cache bakes until a frame's GPU time reaches its target (±20% on
  alternate frames), and with a sky and bounce under streaming there is
  always baking left, so frames sit near the target plus the CPU's
  streaming work: 14–16 ms on average. Lighting loads ran at 46–58 a
  frame, production 46–109 a frame.
- **Settling (a):** the all-resident reference settled its last views in
  2,174–3,194 frames; the streamed renderer did not within 8,000 (bounce
  samples keep retrying as rays meet absent levels), yet its settled image
  was within 0.5% of the reference's.
- **Memory:** a rock's record grew from 17.8 to 20.0 KB of GPU memory with
  stubs of four hierarchy levels.

## Method

1. The sky's bake marks a page that meets an absent level, as the sun's
   does. Every absent object a page meets is listed once a frame (a hash
   bitset), so a page needing several levels learns them in one round.
2. **Deeper stubs:** a streamed geometry keeps the top four levels of its
   hierarchy (boxes and prefiltered coverage), not its root alone. Wide
   cones stop within them as a full walk would; a walk marks only where it
   would descend below the frontier. At a still view on 256 rocks this
   took the sun from partly to fully converged.
3. **Bounce light:** a ray meeting an absent level counts its bin as
   missing (as an absent page at a hit already did) and the sample retries,
   the level wanted after direct light's needs; the page stays shown.
   Direct light stays certified; bounce converges.
4. Skies baked alone and bounce retries run only once their levels are
   resident (pins held by pages that cannot progress deadlocked the slots
   in development); a bounce retry no longer clears its page's marks (it
   had shown pages baked against absent levels).

**Disclosed (before the run):** the mechanism was developed before the
stub, with `stream_dev` `PAGES` (every comparison exact); the gate set the
cache's frame target to 12 ms (default 14), warmed the timed flights'
first views for at most 1,200 frames, and settled (a)'s first views for at
most 3,000; the sky is certified on (a), not against gate 7's region
reference, because it sees rocks out to the horizon.

## What the numbers say

- **Streamed lighting now covers the sun, a sky and bounce light**, direct
  light exact, bounce within half a percent once settled.
- **Left: the frame budget with a sky and bounce.** The cache's bake budget
  targets GPU time alone and swings ±20% around it; with work always left,
  frames sit at the target plus the CPU's streaming. A budget aimed at the
  whole frame (CPU included, never above 16.7 ms) is the fix to gate next.
  Also left: bounce under streaming not settling fully (rays to far
  surfaces need more levels than are resident at once).
