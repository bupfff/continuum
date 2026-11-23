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
