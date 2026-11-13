# M13 gate 7: coverage pages that keep up in flight (coverage only)

**PASS** on all four criteria, criterion 1 narrowly.

| Criterion | Measured | Bar | |
|---|---|---|---|
| 1. Flight: 95th percentile of frame over the settled frame at its place | 1.191 | at most 1.2 | **PASS** |
| 2. 56 bakes a frame, the page passes' own time with and without | 1.39 ms (0.48 → 1.87) | at most 2 ms | **PASS** |
| 3. Flight accuracy at 10 checks | 0.153×–0.158× one ray's error | at most 0.3× | **PASS** |
| 4. Gate 6's walk from a cold pool: median frame | 16.50 ms | at most 20.11 (1.25 × 16.09) | **PASS** |

The flight: 20 m up, 10° down along the standing view's heading, 10 m/s for
10 s (600 frames), from a settled start (28 frames to settle, 13,805
pages). Bakes 50.9 a frame (at most 56), requests 152.8 a frame (at most
863), pages read about 13,600; largest shift read 0.2500 bin (the bound);
frames 26.42 ms median, 36.95 ms worst. Settled frames along it, every
second (median of 10 each): 21.61, 23.49, 25.44, 25.70, 29.32, 25.57,
27.59, 23.04, 24.23, 25.54, 25.53 ms.

| Flight frame | 0 | 60 | 120 | 180 | 240 | 300 | 360 | 420 | 480 | 540 |
|---|---|---|---|---|---|---|---|---|---|---|
| Frame (ms) | 21.88 | 23.25 | 24.43 | 25.00 | 25.37 | 26.42 | 26.82 | 27.26 | 28.37 | 30.20 |
| Requests | 0 | 11 | 37 | 55 | 596 | 588 | 38 | 27 | 24 | 23 |
| Baked | 0 | 14 | 56 | 56 | 56 | 56 | 56 | 56 | 56 | 56 |

Accuracy at the checks (frames 60 to 546): 0.157, 0.154, 0.153, 0.158,
0.155, 0.156, 0.156, 0.155, 0.153, 0.154× one ray's error (one ray about
20%). The walk from a cold pool baked 20.9 pages a frame and read 10,237 at
its end; its worst frame (the first) 102.81 ms.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `059cae2`; criteria 1–2 amended and the method in
`cbbde1e`, both before any run). Machine free of other work; control just
before, gate 4's path at full size: 119.3 ms standing, 104.8 ms from 150 m.

## Method (on gate 6's pages)

- **Bakes** one invocation per ray (128 workgroups a page, 4 texels and 64
  rays each, shares counted in workgroup memory): about 0.02–0.03 ms a page
  against gate 6's 0.087.
- **Leads:** a page's bake eye moves along the eye's last step by half the
  steps after which its contents would shift a quarter of a bin, so it is
  valid at once and about twice as long along the path. Validity is gate
  6's test from the current eye.
- **Refreshes:** a page read past a fifth of a bin is asked again on a
  second list, baked with what the frame's requests leave of the budget.
- **Arguments pass** as one workgroup of 256 (a single invocation took
  about 2 ms to drop thousands of requests).

## Disclosed

- Criteria 1 and 2 were amended before any run (the binary's doc comment):
  the stub compared each frame with the settled frame at the start (the
  view itself costs more along the path) and timed bakes by whole cold
  frames of about 80 ms (too noisy to resolve 2 ms).
- The settled references are noisy: the flight's start measured 21.61 ms
  as the first sample and 25.09 ms when settled again just before the
  flight. Criterion 1's 1.191 is within that noise of the bar.
- Developed at full size on paths turned 20° from the gate's (3–10 s).
  There the flight's 95th percentile read 1.148 and 1.214 on two runs
  before refreshes; with them 1.126–1.136 at 64 bakes a frame, 1.154 at
  56, 1.167 at 48 (which lagged: up to 270 requests a frame). The bakes'
  cost read 1.66–1.85 ms at 64, 1.66 at 56, 1.23 at 48. GPU timestamps
  around the page passes read about 0.1 ms for 64 bakes, implausibly low,
  and were not used.

## What the numbers say

- **Pages now keep up with a 10 m/s flight most of the time:** requests
  stay under the budget except a wave at 4–5 s (about 600 a frame for
  about a second, pages read dipping from 13,700 to 13,100), and frames
  stay near the settled frame at their place.
- **The flight's view costs more than gate 6's:** settled frames 21.6–29.3
  ms at 20 m up, against 16 ms standing; the 10 ms bar needs speed in the
  pass itself, not only in the pages.
- **Next for motion:** waves come from many pages nearing their bound
  together; refreshing earlier when the budget is idle, or ordering
  refreshes by how soon each page expires, would spread them.
