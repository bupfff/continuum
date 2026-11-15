# M13 gate 12: the sub-pixel path in the renderer, lit by its cache

**FAIL**: colour passes in both views; speed fails in both, by 1.1 and
2.1 ms.

| View | Colour error, ours | One ray | Ratio | Settled frame |
|---|---|---|---|---|
| Standing | 3.93% | 9.02% | **0.435× (at most 0.5×): PASS** | **17.80 ms (at most 16.7): FAIL** |
| 150 m up | 8.23% | 19.88% | **0.414× (at most 0.5×): PASS** | **18.77 ms (at most 16.7): FAIL** |

The reference and one ray are the renderer's own shading (the lighting
cache: the light's area, the sky, bounce light), not gate 11's model, so
the errors are not comparable with gate 11's (0.335× and 0.386×, 13.85 and
14.69 ms under the model).

Reported:

| View | Settled in | The renderer's own path (one ray a pixel, the cache) |
|---|---|---|
| Standing | 671 frames | 2.74 ms, settled in 138 frames |
| 150 m up | 353 frames | 3.94 ms, settled in 180 frames |
| 20 m up, 10° down | 1,229 frames, settled frame 28.16 ms | — |

Mean colour: standing, reference (0.1142, 0.1143, 0.1112), ours (0.1137,
0.1141, 0.1109); 150 m up, reference (0.1751, 0.1680, 0.1467), ours
(0.1718, 0.1648, 0.1436), 1.9% dark.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-14 (stub and method in `4224065`, before the run). Speed
timed first. The run's output was saved as it ran
(`experiments/room3d/out/`, not in the repository); its lines are the
numbers above.

## The machine (disclosed)

Not free of other work: external displays were connected through
DisplayLink (the window server at about 40% of a core) and Spotlight was
indexing (about 31%). The control just before, gate 4's path at full size,
read 119.99 ms standing and 105.17 ms from 150 m: within the range gates
5–8 ran at (118.9–122.0 and 102.9–105.9), which the project's rule accepts,
but 8% and 16% above gates 9–11's (109.5–111.3 and 91.0–91.1). Gate 11's
speed was measured in the cooler state; this gate's in the slower one. The
numbers are not rescaled.

## Method (on gate 11's pass, profile stage 20)

- **The renderer takes the path** with its lighting cache, pixel cones
  and profile stage 20 (`InstancedRenderer`; not yet a `RendererOptions`
  field). The passes that shade come from a variant of the hardware-ray
  module whose binding lists come from naga's analysis of the source.
- **Large hits lit by the cache in passes of their own.** Corners, the
  centres of pixels whose background needs one, samples of pixels the
  corners resolved, and bins of objects no corner shows record their hits;
  a lighting pass shades the records with the cache's own shading and a
  scatter pass writes the colours back (the device's 29 buffers a stage do
  not hold the cache's shading beside a pass's own). A corner also keeps
  its cached shares (the light's shadowed share and the sky's value).
- **Bins lit from their corners' shares:** a bin of an object its corners
  show is lit from their shares, bilinearly, with its own point, normal and
  albedo, without a cache lookup.
- **Small instances** keep a model: direct light from the maps' normals,
  with the sun's visibility per instance as the cosine-weighted share of
  five points on its bounding sphere's sunlit side; the sky's radiance over
  half the hemisphere (a sphere resting on a plane). Coverage pages bake
  colour by the model too.
- **The cache bakes with a fixed budget** on this path (its frame target
  would shrink the bakes to their least, since the camera passes alone take
  the whole frame); settled frames bake nothing.

## Disclosed

- Developed at full size on the gate's scene, on other views (standing
  turned 20°, 120 m up at 40° turned −20°) for both speed and accuracy:
  0.420× and 0.426×, 16.0–16.6 ms
  (`docs/results/m13-gate12-development.md`). The stub said before the run
  that criterion 2 was within a few percent of its bar.
- The development timings late in that stretch were taken in the slower
  machine state too (controls 113.1/105.4 ms).

## What the numbers say

- **The sub-pixel path is in the renderer's lighting at colour's bar:**
  0.41–0.44× one ray's error against the renderer's own supersampled
  shading, with the light's area, the sky and bounce light from the cache.
- **It costs about 15 ms beyond the renderer's own path** (2.7–3.9 ms): the
  frame is anti-aliasing. Speed misses by 7% standing and 12% from 150 m in
  a machine state whose control ran 8% and 16% slow; whether it would pass
  in gate 11's state is not measured, and the gate is not re-run.
- **The machine and the method, separated after the run** (development,
  not the gate): gate 11's path (the model's lighting) on the same views,
  same build, timed right after this run: 15.82 ms standing and 16.96 ms
  from 150 m, against 13.85 and 14.69 in gate 11's run (1.14× and 1.15×;
  20 m: 23.49 against 20.91). In the same state the cache's lighting costs
  1.98 ms standing and 1.81 ms from 150 m over the model's (1.13× and
  1.11×), and 4.67 ms at 20 m.
- **The 20 m view is slow** (28.2 ms; gate 11 under the model: 20.9).
- **Settling is slow:** 353–1,229 frames with a fixed bake budget, against
  138–180 for the renderer's own path; a renderer would need its bake
  budget to count the camera passes' cost.
- **Next** (a new gate): the cache's lighting at most the model's cost
  plus a little (a bin's page found once per pixel, its texel read per bin;
  fewer records lit in full), small instances lit from the cache, the 20 m
  view's cost; and the speed criterion stated against gate 11's path timed
  in the same run as well as the absolute bar, so the machine's state is
  measured rather than assumed.
