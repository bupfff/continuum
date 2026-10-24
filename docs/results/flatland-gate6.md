# Flatland gate 6: one-bounce indirect light with propagating charges

Date: 2026-10-23 (overnight run). Question: does the ledger pay off for indirect
light, where a change anywhere (a shadow sweeping a wall) must propagate to
every cached gather that sees it?

Pass criterion, fixed before the first run: zero soundness violations at every ε
and frame, and at ε = 1% of mean indirect fluence at most half the cone
recomputes of ε = 0.

**Verdict: FAIL.** Three charging schemes were tried, all with zero
violations, and none reached the target (76.1%, 68.1%, 66.9%). A diagnostic
"oracle" ledger, charged with the true per-frame change, reaches 27.4%. The
ledger itself can deliver for indirect light; my implementable charges are
about 3× too loose.

## Setup

- The gate 1 room. Static segments are split into 64 reflective patches
  (albedo 0.7, both sides); movers are black. The door, walker and fan move as
  before, except the walker runs at y = 0.33 and the door sits 0.003 off its
  hinge, so no mover touches a static surface. Crossings are detected anyway;
  2 frame steps had one (the door tip meeting the right wall) and were charged
  conservatively.
- Lights A and B as in the typical scenario. B switches on at frame 60 and
  flickers from frame 150. Patch radiance is kept per light, so intensity
  changes are free and exact.
- 128×128 receivers in 8×8 tiles; each gathers indirect light through 8 cones
  of 45°. The gather is an exact 2D angular sweep (nearest surface in every
  direction), validated against 20,000 rays per cone. A binned gather with
  partial coverage depends on processing order when several surfaces share a
  bin, which would break the charge bounds; a depth-correct mask gather is
  left as a 3D design question.
- Patch radiance is recomputed exactly every frame (0.07 ms). Cached gathers
  carry per-light ledgers per (cone, tile), charged by:
  - **Geometry**: a black mover changes the nearest surface only where its own
    coverage changes, priced by a radiance bound.
  - **Propagation**: a patch whose radiance changed charges every cached gather
    that can see it, by its change times the angle it can subtend.
- Reference: a full gather of every cone every frame. 240 frames, 10 threads
  for both. Reproduce with `cargo run --release -p flatland --bin gate6`.

## Results

Full recompute: 11.6 ms/frame. On average 12.4 patches change radiance per frame.

| Charging | Cones recomputed, ε = 0 | Cones recomputed, ε = 1% | Ratio (target ≤ 50%) | Bound use at 1% | Speedup at 1% | Speedup at 4% |
|---|---|---|---|---|---|---|
| Attempt 1: (gained + lost) × brightest patch anywhere | 40.3% | 30.7% | 76.1% | 6.8% | 1.6× | 1.9× |
| Attempt 2: (gained + lost) × brightest patch visible through the sweep | 40.1% | 27.3% | 68.1% | 5.9% | 1.7× | 2.3× |
| Attempt 3: max(gained, lost) × the same (movers are black) | 40.1% | 26.9% | 66.9% | 5.7% | 1.7× | 2.5× |
| Diagnostic: oracle (true change; not implementable) | 34.9% | 9.6% | 27.4% | 28.2% | 5.3× | 10.0× |

Zero violations in every configuration, every ε (0, 1%, 4% and a 16% stress
value) and every frame. When light B switched on, the indirect light was within
ε in that same frame (max error 0.00055 vs ε 0.00057). There is no fade-in.

About 5–8% of all charges came from propagation; the rest came from geometry.

## What this shows

1. **The ledger idea holds for indirect light; the charges are the problem.**
   With exact charges it would skip 90% of gathers at ε = 1%, a 5.3× speedup.
   The best implementable charges skip 73%, a 1.7× speedup. Bound use (actual
   error divided by the certified bound) is about 6%, against 28% for the oracle.
2. **The radiance-pricing ideas barely helped** (76% → 67%). Pricing by the
   brightest visible patch, or by max(gained, lost) instead of the sum, trims
   the bound, but the remaining slack is elsewhere. Likely sources: one charge
   per tile set by its worst receiver, and the brightest-patch price, where the
   oracle charges the true radiance behind the mover.
3. **Propagation works and is cheap.** Shadows sweeping across patches charged
   the right gathers (zero violations), at 5–8% of total charge.
4. **Dependency regions are larger for gathers** (40% recomputed at ε = 0,
   versus 12% for direct light in gate 3), because every wide gather cone that
   can see a mover is affected.

## What to try next

- **Cone G-buffers (most promising).** Cache, per gather cone, which static
  surface is nearest in each bin, and its depth. A black mover's effect then
  becomes exact and cheap: project it into the bins and compare depths. A patch
  radiance change becomes exact too: the bins that show it, times its change.
  That turns most of this gate's charges into exact incremental updates (the
  oracle's regime) at the cost of 64 small entries per cone.
- **Per-receiver charges** instead of one per tile.
- **Decide what ε is relative to.** Here it is 1% of mean indirect light, which
  is strict. Perceptually, the threshold is about 1% of total brightness
  (direct plus indirect); in this room that is roughly 5% of the indirect mean.
  This is a quality-contract decision, not a reason to change this gate's verdict.
