# Flatland gate 5: magnitude deferral with wide cones

Date: 2026-10-23. Question: once visibility is continuous, does the ledger's
magnitude half (deferring changes too small to matter) pay off for wide cones?

Pass criterion, fixed before the first run: with continuous masks at ε = 1%,
zero soundness violations and at most half the cone evaluations of ε = 0.

**Verdict: attempt 1 failed (59.5%); attempt 2 passed (34.0%)** with the same
criterion, after changing how ledgers are charged. Both attempts are reported.

## What changed to make this testable

- **Continuous visibility** (`continuum::CoverageMask`): partially covered bins
  keep the largest fraction any single occluder covers. Visibility becomes
  continuous and 1-Lipschitz per occluder (checked by a randomized test), so a
  ledger can charge a moving occluder only what its edges sweep, with no
  one-bin floor. It is also more accurate: gate 1 error drops from 0.69% to
  0.09% RMSE and the banding is gone, for 35% more cost per cone (220 vs 163
  ns, still 0.07× the cost of 64 rays).
- **Sky scenario**: the gate 1 room and the same door, walker and fan, lit only
  by a sky gathered through 8 cones of 45° per receiver. The sky brightens and
  dims slowly (free, since intensity is applied at composition).
- **Per-tile budget policy**: per tile, recompute the cones with the largest
  weighted balances until the rest fit in ε.
- **Projected charges** (attempt 2): instead of a geometric worst case for the
  whole tile, project the mover's old and new positions into each receiver's
  cone and charge the larger of the coverage it gained and lost (for binary
  masks, ignoring bins the statics already block). Two projections per
  receiver, versus a full cone recompute.
- Shared engine: gates 2–3 and gate 5 now run on `flatland::incremental`. The
  gates 2–3 numbers reproduce exactly.

## Results

128×128 receivers, 8×8 tiles, 180 frames, 10 threads for both the reference and
the incremental renderer. Reproduce with `cargo run --release -p flatland --bin gate5`.

Accuracy at frame 0 against the exact answer: binary masks 0.516% RMSE,
continuous masks 0.351%.

| Configuration | Work at ε = 1% vs ε = 0 | Cones recomputed at ε = 1% | ms/frame at ε = 1% | Speedup vs full recompute |
|---|---|---|---|---|
| Binary, geometric bounds | 92.5% | 26.5% | 0.52 | 7.8× |
| Continuous, geometric bounds (attempt 1) | 59.5% | 18.0% | 1.88 | 3.1× |
| Binary, projected charges | 53.3% | 8.6% | 0.64 | 6.3× |
| **Continuous, projected charges (attempt 2)** | **34.0%** | **10.2%** | **1.42** | **4.1×** |

Soundness: 0 violations in every configuration, at every ε (0%, 1%, 4%, and a
16% stress value) and every frame. At ε = 1% the attempt 2 ledger deferred work
on 97.1% of receiver-frames, and the largest error was 0.99 ε. At ε = 4% it
recomputed only 4.5% of cones (6.7× faster than a full recompute).

## What this shows

1. **Magnitude deferral works once visibility is continuous and charges are
   tight.** On top of what dependency regions already skip, the ledger cut
   cone evaluations by a further 2.9× at ε = 1% (29.8% → 10.2% of a full
   recompute). With binary masks the same idea managed only 1.1×.
2. **Dependency regions matter even for wide cones.** At ε = 0 they skip about
   70%, because each mover only reaches the 1–2 sectors pointing at it. My
   earlier expectation that they would save little was wrong.
3. **Charge tightness is the lever.** Geometric bounds used 2.7% of their
   certified bound; projected charges used 6.2%. Doubling tightness roughly
   halved the work.

## Caveats

- **Wall-clock: binary masks still win here.** Binary recomputes reuse cached
  static masks, so a recompute only projects the 3 movers. Continuous recomputes
  project all 44 occluders, because a static coverage cache would cost 264
  bytes per cone. A continuous static cache is the obvious next step and should
  make attempt 2 the fastest configuration as well.
- **2D cost ratios.** A projected charge costs two projections per receiver; in
  2D a full recompute is only about 44. In 3D a recompute means a hierarchy
  traversal, so the ratio should favor projected charges more, but that has to
  be measured in M1.
- **Uniform radiance.** The sky has the same radiance in every direction, so
  coverage gained and lost cancel and charging the larger of the two is valid.
  With direction-dependent radiance (real one-bounce GI) the charge must be
  gained plus lost, weighted by the radiance difference, which is up to 2× looser.
- **Bounds are still loose.** Actual error used about 6% of the certified bound.
- 2D, CPU, a single sky term; the charging dependency chain of one-bounce GI
  (a surface's light changes, so everything that sees it is charged) is not yet tested.

## Update: continuous static cache and occlusion-aware charges

Same scenario, same criterion. Two changes:

- Continuous recomputes now start from cached static coverage (264 bytes per
  cone) and project only the movers, as binary recomputes already did.
- Projected charges skip bins the statics already block fully
  (`CoverageMask::open_length`), so a mover hidden behind static geometry
  costs nothing. This is the occlusion-aware invalidation from the design.

| Configuration | Work at ε = 1% vs ε = 0 | Cones recomputed at ε = 1% | ms/frame at ε = 1% | Speedup vs full recompute |
|---|---|---|---|---|
| Binary, geometric bounds | 92.5% | 26.5% | 0.50 | 8.0× |
| Continuous, geometric bounds | 59.5% | 18.0% | 0.64 | 9.1× |
| Binary, projected charges | 53.3% | 8.6% | 0.63 | 6.4× |
| Continuous, projected charges | **25.4%** | 6.3% | 0.84 | 7.0× |

Full recompute: 5.9 ms/frame (continuous) and 4.0 ms/frame (binary); a full
recompute that only reuses static coverage, with no ledger: 1.36 and 0.54 ms.
Zero violations everywhere, and cached-static recomputes match the reference
bit for bit (asserted every frame).

Takeaways:

- **Continuous masks now win on wall-clock too** (9.1× vs 8.0× for binary), on
  top of being more accurate.
- **Occlusion-aware charges tightened the bound** (bound use 6.2% → 10.9%) and
  cut the ledger's work from 34.0% to 25.4% of ε = 0.
- **Work vs time trade-off:** projected charges recompute 3× fewer cones than
  geometric bounds, but in 2D a recompute is now so cheap (statics cached, 3
  movers) that charging costs more than it saves. Which wins in 3D depends on
  the recompute-to-projection cost ratio there; both stay in the engine.
