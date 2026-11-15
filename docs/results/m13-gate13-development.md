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
