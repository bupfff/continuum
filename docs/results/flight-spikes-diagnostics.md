# Where the flights' slowest frames come from (diagnostics, not a gate)

Date: 2026-11-07. The M4 benchmark's flight with three colored lights
(`docs/results/bench-m4.md`) averages 14.9 ms but has a 99th percentile of
29–31 ms and worst frames of 47–83 ms. `spikes_dev` (M5 gate 6's lights,
M5 gate 8's path, 1280×720, two flights) looked at what the slowest 5% of
frames share.

## A frame waits for the GPU work of the frame two before

A frame's wall time is mostly waiting for the GPU to finish the frame two
before, whose requests its plan reads. Pairing each frame's wall time with
that earlier frame's record (comparing like frames misattributes it):

| Frames | Wall | GPU bake (two before) | Camera passes | Pages baked | Budget |
|---|---|---|---|---|---|
| Slowest 5% | 27–28 ms | **18.5–19.2 ms** | 4.1 ms | 488–506 | 7,742–8,422 |
| The rest | 14.1–14.2 ms | 7.1–7.4 ms | 3.2–3.4 ms | 718–727 | 8,932–9,142 |

- **The cost per unit triples in bursts.** The slow frames follow bakes of
  similar size and budget that cost about 2.6× as much. They cluster around
  frames 373–397 (about 95 m up, looking down 28°), where single bakes take
  24–28 ms. Example: frame 395 baked 342 pages in 25.2 ms.
- **Planning and eviction are not the cause.** They stay under 2 ms.

## A faster-reacting budget does not help

Tried in development and reverted:

- the cost per unit raised at once when a bake overran its prediction by
  25%;
- the budget's alternating ±20% (which lets the fit tell latency from
  cost) cut to ±10%.

The 99th percentile stayed at 28.8–28.9 ms. The worst frame rose to
54–64 ms.

The budget learns from GPU timings that arrive two frames late, and the
bursts last only a few frames, so measured corrections come too late.

## What would

- Predicting a page's cost before baking it: from its mip, or from the
  light cones' grazing angle over dense geometry.
- Splitting a frame's bake when it would overrun.

Either is a design change for a new gate. The 99th percentile (29–31 ms)
already meets M5 gate 6's bound of 33.3 ms.
