# M13 gate 10 in development: colour, second

Development at full size (`M13_DEV=1M`) on gate 9's pass before gate 10's
run. Speed on the gate's views; accuracy only on other views (`M13_ACC=1`:
standing turned 20°, 120 m up at 40° turned −20°), as one ray's multiple.

| Step | Accuracy (standing / 120 m) | Settled frame (standing / 150 m) |
|---|---|---|
| Gate 9 as run (gate views) | 0.652× / 0.547× | 26.5 / 22.1 ms |
| Gate 9 + background structure of its own (development) | 0.64× / 0.58× | 23.2 / 29.0 (with first per-bin shading) |
| Bins shaded one by one where corners do not suffice (per pixel, 16 rays) | 0.370× / 0.386× | shading pass 13.6 / 40.4 ms alone |
| Same, a list pass then one invocation a bin | 0.403× / 0.404× | |
| Colour words as field arrays in a plain buffer; backgrounds from corners | 0.464× / 0.470× (ground shadows bilinear) | 18.8 / 20.4 |
| Entries made in the bins pass; bin rays shade their own hits | 0.360× / 0.451× | 17.4 / 18.5 (traced) |
| Resolve in the pixels pass; 4 samples a resolved pixel | 0.383× / 0.463× | 16.1 / 18.0 |

Lessons:
- **Array-of-structs colour words cost ~1 ms a pass** (partial cache lines);
  fields as arrays halved the bins and list passes. Private arrays indexed in
  loops (corners) were slow; vectors fixed the resolve (2.45 → 1.55 ms).
- **Ground shadows per bin** (shadow rays at bins where the background's
  corners disagree) give 0.40×/0.39× but cost ~8 ms from 150 m; left
  bilinear (`G10_BG_SHADOWS`).
- **Pixels their corners resolved** with spread colours matter (0.50× →
  0.36× standing); 16 primary rays each cost 2 ms (about 72,000 pixels),
  4 cost 0.9.
- Real-frame pass times come from Instruments (`xctrace` Metal System
  Trace, `metal-gpu-intervals`); ray passes run with the RT unit 95–99%
  active.
- **Left at 150 m:** corners 4.4 ms (ground shading with a shadow ray per
  corner, 1.5 ms of it), pixels 7.4 ms (7 million map samples; the buried
  test ~1.6 ms, shading ~0.7). 15 ms there needs a structural lever (sun
  visibility of the background cached like coverage pages, or cheaper
  samples).
