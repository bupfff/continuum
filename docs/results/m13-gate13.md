# M13 gate 13: the cache-lit sub-pixel path at the cost of gate 11's

**PASS** on both criteria, standing narrowly (inside development's
run-to-run spread). The run started by accident; see "Disclosed".

| View | Cache-lit / gate 11's path | Rounds' ratios | Lit | Gate 11's | Colour, ours / one ray |
|---|---|---|---|---|---|
| Standing | **1.045× (at most 1.05×): PASS** | 1.069, 1.045, 1.037, 1.067, 1.042 | 17.07 ms | 16.11 ms | **0.436× (3.93% / 9.02%): PASS** |
| 150 m up | **1.014× (at most 1.05×): PASS** | 1.016, 1.014, 1.009, 1.014, 1.011 | 17.87 ms | 17.62 ms | **0.414× (8.23% / 19.88%): PASS** |

Lit and gate 11's times are medians over the five rounds. Gate 12 on the
same views, same machine state (control 119.99/105.17 ms): 17.80 and 18.77
ms lit, 1.11–1.13× gate 11's path timed after it; colour 0.435× and
0.414×.

Reported: the 20 m view (10° down) settled in 1,229 frames at 28.05 ms
(gate 12: 28.16). The renderer's own path (one ray a pixel, the cache):
2.66 ms standing (settled in 138 frames), 3.43 ms from 150 m (180). The
cache-lit path settled in 670 and 353 frames, gate 11's in 10 and 2. Mean
colour: standing, reference (0.1142, 0.1143, 0.1112), ours (0.1137, 0.1141,
0.1109); 150 m up, reference (0.1751, 0.1680, 0.1467), ours (0.1718,
0.1648, 0.1436), 1.9% dark (as gate 12).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-14. The run's lines are in `experiments/room3d/out/
m13_gate13.txt` (not in the repository); its terminal output was the same
lines.

## Method (on gate 12's)

- **The corners' records write their own words:** the pass that lights
  the corners' records from the cache writes their colours and the cache's
  shares into the corners' words (its 29th buffer); the other records'
  targets need two buffers, so they keep a scatter pass. Records are plain
  words, their count kept apart.
- **Quarter samples from the corners' shares:** a resolved pixel's quarter
  sample is lit from its corners' cached shares (bilinearly, with its own
  point, normal and albedo) where the corners showing its object agree on
  the light's share within 0.02; elsewhere (shadow edges) it is lit in full
  by the cache. Standing, 324,216 quarter records fell to 51,508.
- **The rect light's unclipped case unrolled** (the renderer's own shading,
  every path): with the whole light above the point's horizon, its four
  edges summed in order, each corner's direction normalized once, without
  arrays indexed at run time. The same sums in the same order; the GPU tests
  pass (the golden image included).
- **The measure:** both paths settled in one process, then five rounds of
  20 frames to warm up and the median of 21, lit then gate 11's.

## Disclosed

- **The run started by accident.** A command meant to add and then
  exercise a dry run on 10,000 boulders ran the run mode at full size: the
  script that would have added the switch failed its first check, so the
  binary built without it and, with no development variable set, ran the
  gate. The binary was the working tree as intended for the run: the stub
  (`e992c2e`, criterion 2 at 1.05× with rounds of 11 frames) plus the
  amendment written before the run (20 frames to warm up, the median of 21,
  after development found the GPU's clock takes about nine frames to step
  after a switch between renderers), the method above and the disclosure of
  development numbers in the doc comment. These were committed after the run
  (`714cf50`), unchanged. The gate is not re-run.
- **No process check or control before the run.** After it: gate 4's path
  at full size 119.74 ms standing and 104.92 ms from 150 m (gate 12's
  control before its run: 119.99 and 105.17); the window server at about 32%
  of a core with external displays, Claude's renderer about 50%,
  Notification Center 17%. Criterion 2 is a ratio taken in one process to
  divide this out; the absolute times are this state's.
- Developed at full size on the gate's scene, speed on the gate's views
  (ratios 1.04–1.07× standing, 1.00–1.04× from 150 m between development
  runs), accuracy on other views (0.422× and 0.426×). Standing's 1.045× is
  within that spread of the bar: the pass is narrow.

## What the numbers say

- **The renderer's lighting costs about 1–5% over the toy model's** in the
  sub-pixel path: the light's area, the sky and the cache's shadows at 0.41–
  0.44× one ray's error, for 1.014–1.045× gate 11's frame (gate 12:
  1.11–1.13×). What it still costs: lighting the corners' records by the
  cache (about 0.6 ms standing against the model's 0.3), the lean lighting
  of quarter samples and bins (about 0.4 ms), the other records (about
  0.25 ms).
- **The absolute frame is not at 60 fps in this machine state:** 17.07 and
  17.87 ms (gate 11's path itself 16.11 and 17.62; 13.85 and 14.69 in its
  own run, in a cooler state). The absolute bar wants a run in that state,
  or a cheaper path.
- **Next:** the path as a `RendererOptions` field, small instances lit from
  the cache (still the model's), and colour in motion (gate 11's after-run
  development: the sun cache misses where the view moves).
