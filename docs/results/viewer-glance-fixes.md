# Viewer: tearing, the exposure flash and pages lost to a glance (2026-11-22)

The user, flying the viewer (the showcase and the million-boulder field,
2940×1846, defaults): parts of the screen sheared apart while the camera
moved, in both scenes; in the field, looking up toward the sun made
everything brighter, and looking back down was "a flash bang"; and after
looking back down, boulders showed dark blocks (their screenshot: page-sized
dark patches on boulders' sides).

Reproduced without a window by `viewer_repro` (the field as the viewer
builds it: a million boulders, the sun, no sky, bounce light,
anti-aliasing, bloom, PBR Neutral; 1470×923): settle at the viewer's start
(600 frames), turn toward the sun over 90 frames (117° up), hold 30, turn
back over 90, stay 240. Each frame waited for and read back, so frames come
slower than in the viewer (the exposure's adaptation per frame is faster).

## Tearing: vsync was off by default

The viewer configured the surface `AutoNoVsync` unless `--vsync`: on Metal
frames are shown as they finish, mid-scan, and a moving view tears. Both
renderers showed it, as the cause is the presentation. Now vsync is on by
default (`--no-vsync` or V turns it off; `--frames` timing runs keep it
off unless `--vsync`). Not measured (no window here); the fix is the
present mode.

## The flash: the exposure meter metered the empty background

Auto-exposure (on by default) brings the trimmed mean of the frame's log
luminance to middle grey. With no sky the background is black, so looking
up filled the metered half with black, and the exposure climbed to its
bound (65,536×). Back down, the ground showed overexposed until the
exposure came back (0.5 s time constant).

Lower half of the image (the ground), mean of the displayed pixels made
linear:

| | Settled at the start | First frame back | 30 frames later |
|---|---|---|---|
| Before | 0.184 | 0.977 (5.3×) | 0.441 |
| Black not metered | 0.103 | 0.0003 (black) | 0.0117 |
| Black not metered, adaptation by the share metered (the fix) | 0.122 | 0.109 | 0.100 |

Not metering black alone left only the sun's disc metered while looking at
it: the exposure fell 17 stops and the view came back black. The fix
(`meter.wgsl`): black pixels are not metered, and the adaptation's rate is
scaled by the share of the frame metered, up to a quarter (a frame of only
black holds the exposure).

## Dark blocks: pages evicted after a second unused

The cache evicted every page unused for 64 frames, with the pool nearly
empty. A second looking at the sky dropped the ground's pages; back down,
their direct light returned within frames, but their bounce light waited
on the field's starved bake (below), so boulders showed dark blocks for
seconds. Now unused pages are evicted by age only once a quarter of the
pool (131,072 pages) is resident; a full pool still evicts the least
recently used at once.

Fixed exposure, the reference the frame just before the turn; pixels at
most half as bright as there:

| Frames after the return | 1 | 6 | 21 | 51 | 91 | 141 | 240 |
|---|---|---|---|---|---|---|---|
| Before (64-frame age) | 6.77% | 6.77% | 6.77% | 6.77% | 6.77% | 6.77% | 6.77% |
| Pixels with no page (before) | 4,871 | 4,349 | 3,972 | 3,829 | 2,462 | 1,098 | 956 |
| After | 0.07% | 0.07% | 0.07% | 0.06% | 0.17% | 0.60% | 0.60% |
| Pixels with no page (after) | 939 | 0 | 0 | 0 | 0 | 0 | 0 |

Before, the resident pages fell from 40,104 to 22,216 during the hold
(evicted by age, counted outside `CacheStats::evicted`); after, they only
grew (31,025 to 73,227 by the run's end). The pool's memory is allocated
either way. The passes that scan every slot (movers' stale pass, the
bounce ledger) run to the highest slot used, which can now be higher; not
measured.

## Seen, not fixed

- **The field's bake starves.** At the start view the cache took ~26,000
  page requests a frame and baked 13–160 pages; the bake budget fell from
  12,040 to 1,247 units over 500 frames with frames at ~10 ms (the model
  path; the target 14 ms), every bounce sample incomplete. This is the
  field's bounce light never settling (DESIGN §3.8, basics item 2).
- **Boulders without sunlight, once.** In one of seven runs the settled
  start view showed mid-distance boulders flat dark brown, unshaded and
  casting no shadow (their direct light zero), right again after the turn
  re-baked them. Not reproduced since; cause unknown.

Basics scorecard after the fixes, against Cycles (`SC_SPEED=0`): every
scene unchanged (Sponza 29.94%, the field 1.68%, the room 7.68%, the
movers 5.99%, the helmet 4.46%, Fox 2.15%, BrainStem 3.22%). Speed not
rerun yet.

## The field's starved bake, located (development)

`viewer_repro`'s settle (1,200 frames at the start view, fixed exposure).
This device's GPU timestamps never fit the bake model here
(`gpu_bake_ms` 0 every frame), so the frame period's cap (M14 gate 9)
steers alone: it shrinks in proportion while a frame takes over 11 ms and
grows only below 85% of it (9.35 ms). Frames near 10 ms with occasional
spikes ratchet it down: 14,635 units at frame 100, 1,621 at 300, 1,251
at 1,100 (8–20 pages baked a frame). Without the cap (`frame_budget_ms`
0) the budget stays at its 20,000 maximum (230–330 pages a frame), but
the field still does not settle in 1,200 frames: its bounce samples keep
asking for distant boulders' pages (187,000 resident by frame 1,100,
from 48,000 with the cap). Two things, then: the cap's dead band (a
change to M14 gates 9–10's design, for a gate), and the field's far,
small hits (the cone milestone, M23).

## Looking up under a sky (the user, the same night)

With the rebuilt app (the default lighting is now sun and sky), looking
up still brightened everything, less than before. The field's sky is
dimmer than its sunlit ground (its irradiance a quarter of the sun's), so
a frame mostly sky raised the exposure; the first fix had removed only
black pixels. The meter now reads each pixel's hit (with the cache): a
light seen is not metered, and the sky goes in a histogram of its own,
counted at most as much as the surfaces, so a view half sky is metered as
before and looking up cannot let the sky take over. Metering surfaces
alone held the exposure too, but darkened the usual view by about 0.8 of
a stop (the brighter half of the ground at middle grey); lowering the
lower percentile to 0.25 or 0.1 gave back only part of it.

The viewer's field, sun and sky, 2560×1440, 200 frames at each tilt from
the start view (`VIEWER_PITCH`), mean of the displayed image made linear:

| Tilt | Before: bottom tenth, top tenth (sky) | After |
|---|---|---|
| 0° | 0.1678, 0.0222 | 0.1678, 0.0222 |
| 12° | 0.3837, 0.0708 | 0.1639, 0.0233 |
| 18° | 0.6147, 0.1209 | 0.1871, 0.0215 |
| 24° | 0.3227, 0.1730 | 0.0478, 0.0155 |

At 24° the bottom tenth is mostly sky and far ground. The glTF tests pass
(auto-exposure's included: its view has no sky).
