# M13 gate 10: colour in the coverage pass (appearance, second)

**FAIL**: colour passes in both views; speed fails in both, as development
predicted.

| View | Colour error, ours | One ray | Ratio | Settled frame |
|---|---|---|---|---|
| Standing | 2.28% | 5.96% | **0.383× (at most 0.5×): PASS** | **16.05 ms (at most 15): FAIL** |
| 150 m up | 8.40% | 18.05% | **0.465× (at most 0.5×): PASS** | **18.03 ms (at most 15): FAIL** |

Gate 9 on the same views: 0.652× and 0.547×, 26.50 and 22.05 ms.

Reported: 20 m up looking 10° down, settled 23.68 ms (gate 9: 28.24). Pixels
shaded bin by bin: 105,345 standing, 88,391 from 150 m. Standing settled in 9
frames (2,828 pages), from 150 m in 2 (10). Mean colour: standing, reference
(0.1897, 0.1828, 0.1678), ours (0.1897, 0.1828, 0.1677); 150 m up, reference
(0.2212, 0.2090, 0.1782), ours (0.2225, 0.2102, 0.1794), 0.6% bright (gate 9:
2.4%).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-14 (stub and method in `d52d42e`, before the run). Speed
timed first. Machine free of other work; control just before, gate 4's path
at full size: 111.3 ms standing, 91.1 ms from 150 m (gate 9: 109.5 and
91.1). Output in `experiments/room3d/out/m13_gate10_run.log`.

## Method (on gate 9's pass, profile stage 20)

- **Bins shaded one by one where the corners do not suffice.** A pixel
  tracing bin rays shades each bin's own large hit; the sun's visibility is
  its corners' where those showing the object agree, else a shadow ray. A
  pixel whose corners resolved one large object, where they disagree on the
  sun or differ in colour by more than a tenth of the brightest, shades four
  samples (one per 2×2 bins) from just before its corners' nearest hit.
- **Backgrounds from the corners.** Every corner shades what lies behind
  small instances (its large hit, else its background's hit, with a shadow
  ray). A pixel whose corners agree on the background takes it from them;
  its distance per bin is 1/t bilinear over the corners (exact for a plane).
  Background objects never make a tile trace corner rays.
- **Memory and passes.** Colour words are arrays per field; what bins leave
  is coloured at the end of the pixels pass, corners read as vectors.

## Where the time goes

A real frame per pass (Instruments, Metal System Trace, development build
just before the run; `docs/results/m13-gate10-development.md`):

| Pass | Standing | 150 m up | Coverage alone (gate 8) |
|---|---|---|---|
| Once-a-frame (records, sun visibility per instance) | 2.5 | 1.2 | ~2 / 1.5 |
| Roots | 0.4 | 0.3 | |
| Corners (shading, a shadow ray each) | 2.4 | 4.4 | 1.3 / 1.7 |
| Bins (lists, backgrounds, entries) | 1.6 | 1.5 | 0.8 |
| Bin rays (shading their own hits) | 1.8 | 2.0 | 1.2 |
| Shading pass (pixels the corners resolved) | ~0.9 | ~0.6 | — |
| Pixels (walk, maps, colour) | ~5.8 | ~7.3 | 3.9 |

The ray passes run with the RT unit 95–99% active. From 150 m the corners'
ground shading costs about 2.7 ms (1.5 of it shadow rays) and the pixels
pass takes 7 million map samples, of which the buried test costs about
1.6 ms and shading about 0.7.

## Disclosed

- Developed at full size on the gate's scene: speed on the gate's views
  (16.1 and 18.0 ms), accuracy only on other views (0.38× and 0.46×). The
  stub's doc comment stated before the run that criterion 2 was expected to
  fail.
- The ground's shadows stay bilinear across a pixel: shading them per bin
  gave 0.40× and 0.39× on the development views but cost about 8 ms from
  150 m.

## What the numbers say

- **Colour is solved at gate 9's bar** on these views: 0.383× and 0.465× one
  ray's error, with the same model and reference. Shading each bin where the
  corners disagree did most of it; backgrounds from the corners kept it
  cheap.
- **Speed is 1–3 ms over.** Colour adds about 7 ms standing and 9 ms from
  150 m to coverage's 9. Small changes will not close 150 m: its cost is the
  ground's shading (a shadow ray per corner) and per-sample work on 7
  million map samples.
- **Next:** a structural lever for the background's sun visibility (cached
  like coverage pages: exact while valid, baked at bins, and better than
  corners at the ground's shadow edges), and cheaper map samples (DESIGN
  §3.4, "Appearance, third").
