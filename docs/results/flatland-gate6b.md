# Flatland gate 6b: exact incremental indirect light with cone G-buffers

Date: 2026-10-24 (overnight run). Follow-up to gate 6, which found that bounds
on how much indirect light can change were about 3× too loose.

Question: can the changes in that scene be applied exactly and cheaply instead
of bounded?

Pass criterion, fixed before the first run: (1) the incremental output equals a
full recompute with the same estimator at every receiver and frame, within float
rounding (1e-5); (2) the incremental path is at least 4× faster than the full
recompute at the same thread count.

**Verdict: PASS.** Bit-for-bit exact (maximum difference 0) and 7.4× faster.

## Method: cone G-buffers

- Each gather cone caches, per bin (64 per cone), the nearest static patch along
  the bin's center direction and its depth: 384 bytes per cone, 50 MB for the
  131,072 cones here. Built once in 28 ms.
- **Black movers** are composited on top by depth: project the mover into the
  bins and compare depths. Only cones whose frustum the mover's swept hull can
  reach are updated.
- **Radiance changes**: when a patch's radiance changes, only cones whose
  G-buffer actually contains that patch are re-summed. This is an exact
  dependency, not a frustum test.
- Light intensity changes stay free (per-light caches, applied at composition).

The baseline full recompute rebuilds every cone's nearest-surface layer and
gathers, which is what a renderer without caches does each frame. Same scene as
gate 6: 128×128 receivers, 8 cones of 45° each, 64 patches, 240 frames, 10
threads for both paths. Reproduce with `cargo run --release -p flatland --bin gate6b`.

## Results

| | ms/frame |
|---|---|
| Full recompute | 14.74 |
| Incremental (G-buffer reuse, exact updates) | 1.98 (7.4× faster) |

- Cones updated per frame: 29.7% for mover sweeps, 9.0% for radiance changes;
  61.3% untouched.
- Exactness: maximum difference from the full recompute 0.0 in every receiver
  and frame.
- For comparison, gate 6's best bounded ledger was 1.7× faster at ε = 1% and
  carried error up to ε; its oracle with exact charges was 5.3×.

## What this shows

1. **Where exact updates are cheap, they beat bounds.** For black movers and
   radiance changes over static geometry, a per-cone G-buffer makes every
   update exact and the work proportional to what changed.
2. **The incremental principle holds; the mechanism shifts.** "Never ask again
   until the answer could have changed" is here realized by exact dependency
   tracking (which cones a mover's sweep reaches, which cones show a patch)
   rather than by error bounds. The ledger stays the right tool where exact
   updates are expensive: a moving light (every cone toward it shifts), slow
   global changes, and frame-budget pressure.

## Caveats

- **Accuracy:** the binned estimator samples one direction per bin and is 2.5%
  (relative RMSE) off the exact sweep, worse than continuous masks are for
  visibility (0.35%). Options: more bins, or partial coverage per bin made
  depth-correct.
- **Memory:** 384 bytes per cone on the CPU. A GPU layout needs compression
  (16-bit depth, or depth only where a mover can reach).
- **Static geometry assumed.** If the reflective surfaces themselves move, their
  G-buffer entries go stale; that case needs the ledger, or a rebuild of the
  affected cones.
- 2D, CPU, one bounce.
