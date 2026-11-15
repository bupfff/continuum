# M13 gate 13 in development: the cache-lit sub-pixel path at the cost of gate 11's

Development (not the gate) on the million-boulder field with its light and
sky, 1280×720. Speed on the gate's views (gate 12's: standing, 150 m up),
accuracy on other views. Machine: the fanless M4 with external displays
connected (the window server at about 30–40% of a core); timings move by
about ±0.5 ms from frame to frame, and single passes show occasional
stalls of about 2 ms in both paths alike.

## The measure

`m13_gate13`'s `M13_DEV=1M M13_SPEED=1`: the cache-lit path and gate 11's
path (the same passes lit by its model, no cache, no sky) settled in one
process, then three interleaved rounds of 11 frames each (lit, then
model); the ratio of the rounds' medians. Rounds agree within about 0.03×;
absolute times drift by more between runs.

Per-pass GPU times come from Instruments (Metal System Trace) with a
compute pass per dispatch (`CONTINUUM_M13_PASSES=1`; the renderer otherwise
runs them in one pass), medians over the settled frames of both paths in
one trace.

## Where gate 12's cost went (standing, per-pass medians, traced)

| Pass | Gate 11's path | Cache-lit (gate 12) | Difference |
|---|---|---|---|
| Corners | 2.02 | 1.73 | −0.29 (their shading deferred) |
| Lighting the corners' records | — | 0.68 | +0.68 |
| Their scatter to the corners | — | 0.16 | +0.16 |
| Bin rays | 1.71 | 1.89 | +0.18 (bins lit from corners' shares) |
| Shading rays (resolved pixels' quarter samples) | 0.51 | 0.66 | +0.15 |
| Lighting the other records | — | 0.61 | +0.61 |
| Their scatter | — | 0.20 | +0.20 |
| Frame (sum of passes) | 15.30 | 17.21 | +1.91 |

Records a frame: standing 842,329 (459,520 corners; then 324,216 quarter
samples of resolved pixels, 30,718 bins of objects no corner shows, 27,875
regions); from 150 m 1,047,332 (921,600 corners). The cache's CPU planning
and waiting are under 0.05 ms a frame.

## Levers

| Change | Standing | 150 m up |
|---|---|---|
| Gate 12 as run | 1.089–1.104× | 1.025× |
| The corners' lighting writes their words itself (the pass's 29th buffer; the others' targets need two, so they keep a scatter) | — | — |
| Quarter samples lit from the corners' shares where those agree on the light within 0.02 (else in full): 324,216 → 51,508 full records standing | 1.078–1.097× | 1.033× |
| The rect light's unclipped case unrolled, without arrays indexed at run time (exact: the same sums in the same order) | 1.057× | 1.010× |
| The corners' words read once in the lean lighting; each corner direction normalized once in the rect light | 1.067× | 1.007× |

(Rows are separate runs; standing's lit frames are bimodal, about 16.6 or
17.2 ms, which moves the ratio by about 0.03.)

Per pass after the rect light's fast path (standing, traced): lighting the
corners 0.62 ms, quarter samples lit lean 0.98 against the model's 0.51,
the other records 0.21 + 0.05; frame 16.56 against 15.68 ms.

- Deferring the quarter samples instead (with the fast rect light) costs
  more: 0.66 + 0.53 + 0.20 ms against 0.98 + 0.21 + 0.05.
- The rect light's exact term costs about 0.22 ms of the lean lighting
  (replacing it by a cosine, timing only).
- The workspace's GPU tests pass with the rect light's fast path (the
  golden image included).

## Before the run (continued)

- **Records as plain words** (their count in a buffer of its own, so the
  lighting passes read them through the caches): no measurable change
  (1.068× / 1.005×).
- **The measure's rounds.** In order, the lit path's frames after the
  model's ran 17.2–17.4 ms for about eight frames, then one at 19.2–19.3,
  then 16.4–16.8 ms; after a 0.5 s pause they started at 26 ms and fell
  over twenty frames. The GPU's clock takes about nine frames (some 150 ms)
  to step after the load changes, so rounds of 11 frames measured the
  switch. Rounds became 20 frames to warm up, then the median of 21
  (amended in the stub before the run). With them: 1.064× / 1.042×, then
  1.041× / 1.004× and 1.069× / 1.024× in later runs.
- **Alternating frame by frame** measures something else: lit 18.74 ms,
  gate 11's 14.51 ms standing (1.254×), as one renderer's trailing GPU work
  lands in the other's frame. Not used.
- **Colour with the lean quarter samples** (other views): 0.422× and
  0.426× (gate 12's development: 0.420× and 0.426×).
- Machine during this stretch: the disk nearly full (traces in the
  scratchpad filled it; removed), memory 80% free, no thermal warning
  recorded; the window server at 30–40% of a core.
