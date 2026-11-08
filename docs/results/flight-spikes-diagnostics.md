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

## Not explained by any count planned or measured

Later development runs fitted each frame's GPU bake time on the first
flight and tested the fit on the second (`spikes_dev`; per-frame CSV in
`out/spikes_run*.csv`):

| Model | Test R² |
|---|---|
| One cost per budget unit | 0.05 |
| One cost per page | 0.05 |
| A cost per mip level (`CacheStats::spent_by_mip`, mips 6 to 13+) | −0.43 |
| The kernel's work counters (`KernelOptions::count_work`: object nodes, meshes, triangle-hierarchy nodes, triangles) | 0.37, residuals up to 13 ms |

- **Slow bakes cost about 5× per page** (1.04 ms against 0.21 ms). They
  lean toward fine (7–8) and very coarse (11–13) mips, but with fewer pages
  and units: the budget had already been cut after the frames before.
- **They are tied to content, not chance.** They fall on the same frames
  (about 373–397) in both flights.
- **Most of their cost is something the counters do not see.** Hierarchy
  work explains only a third of the variation. Candidates: memory access
  patterns, or the bake's other passes (table updates, fill).
- **Which pass:** `FrameProfile::bake_passes` (each bake pass submitted
  and waited for alone; `spikes_dev passes`) shows the light bake (cones
  toward the lights) takes nearly all of it. Across the flight it averaged
  26.19 ms against 0.24 ms for table updates and 0.36 ms for the fill.
  Profiled frames are not timed, so the budget stays at its maximum.
- **The slowest bakes:**

  | Frames | Light bake | Pages |
  |---|---|---|
  | 400–406 | 50–61 ms | 336–414 |
  | 583–587 | 52–53 ms | 118–124 |

  The cost per unit of work varies 2–3× with content.
- **A likely cause, not established: divergence.** On coarse pages,
  neighbouring texels lie metres apart, so the 32 invocations of a SIMD
  group walk different paths, and the group costs its slowest. Summed work
  counters cannot see that. M5's sky diagnostics measured the same effect
  at about 10× (`m5-sky-bake-diagnostics.md`).
- **To test it:** time pages of one mip at a time, or bake coarse pages
  with packet walks (M5 gate 5 found packets faster above about 10,000
  units a frame). A per-mip cost weight in the budget would follow if mip
  explains it.

`CacheStats::spent_by_mip` is kept as a diagnostic.

## What would

- Predicting a page's cost before baking it: from its mip, or from the
  light cones' grazing angle over dense geometry.
- Splitting a frame's bake when it would overrun.

Either is a design change for a new gate. The 99th percentile (29–31 ms)
already meets M5 gate 6's bound of 33.3 ms.
