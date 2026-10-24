# Flatland gate 7: a moving light (the ledger's own job)

Date: 2026-10-23 (overnight run). After gate 6b showed that exact updates beat
bounds when they are cheap, this gate tests the case where they are not. When
a light drifts (a sun on a day-night cycle), every cone toward it changes a
little every frame, so exact invalidation must recompute 100% of cones every
frame. Only error bounds can skip that work.

Pass criterion, fixed before the first run: at the realistic drift speed (1×,
a 24-minute day: about 0.25° of sun motion per second) and ε = 1%, zero
soundness violations and at most 25% of cones recomputed. Faster speeds are
informational.

**Verdict: FAIL, narrowly.** 25.1% of cones recomputed against a target of 25%,
with zero violations. That is still 4× less work than exact invalidation and
2.7× faster than a full recompute, with certified error.

## Setup

- 256×256 receivers in 8×8 tiles; the gate 1 room with the door, walker and
  fan moving; one light drifting sideways. 240 frames, 10 threads for both
  paths. Visibility uses the exact 2D estimator, so no bin-alignment effects:
  a mask's bins move with the cone, which would add an estimator term to the
  bounds.
- Charges per tile, by the triangle inequality, light first (movers fixed),
  then movers (light at its new position):
  - **Light motion**: directions enter or leave a receiver's cone only as the
    light's endpoints move. So the visible angle changes by at most the angles
    the two endpoint paths subtend from the tile. If an occluder lies between
    the old and new light, the whole cone is charged.
  - **Movers**: the geometric region/motion bounds from gates 2–3.
- Reproduce with `cargo run --release -p flatland --bin gate7`.

## Results

| Drift speed | ε | Violations | Max error / ε | Cones recomputed | Speedup vs full recompute |
|---|---|---|---|---|---|
| 1× | 0% | 0 | — | 100.0% | 0.9× |
| 1× | **1%** | 0 | 0.87 | **25.1%** | **2.7×** |
| 1× | 4% | 0 | 0.88 | 15.9% | 3.5× |
| 4× | 1% | 0 | 0.69 | 41.5% | 2.0× |
| 4× | 4% | 0 | 0.87 | 21.2% | 3.1× |
| 16× | 1% | 0 | 0.54 | 78.0% | 1.2× |
| 16× | 4% | 0 | 0.84 | 38.1% | 2.2× |
| 64× | 1% | 0 | 0.00 | 100.0% | 0.9× |
| 64× | 4% | 0 | 0.69 | 71.7% | 1.3× |

Full recompute: about 2.7–3.0 ms/frame.

## What this shows

1. **The ledger does the job exact invalidation cannot.** With a slowly moving
   sun, every cone changes every frame; the ledger recomputes a quarter of them
   (at ε = 1%) and certifies the rest are within ε. Engines today re-render
   shadows every frame or blend old frames, which shows up as lag.
2. **The benefit fades with speed, as it should.** A 64× faster sun (a
   22-second day) changes lighting by more than ε every frame, and the ledger
   correctly recomputes everything. Its overhead then is about 10% (0.9×).
3. **Worst-case overhead**: at ε = 0 the incremental path costs 1.1× a full
   recompute (charging plus bookkeeping), the same parity seen in gate 4's
   preview.

## Why it missed, and the likely fix

The light-motion charge is the same for every receiver in a tile and ignores
occlusion: a receiver whose view of the light's edges is blocked by static
geometry (everything under the wall, for example) cannot see the edges move,
yet it is charged as if it could. An occlusion-aware light charge (skip edges
hidden by statics, using the static coverage cache) should help most in
shadowed regions. It was not tried, to avoid tuning against a pre-set target
after a near miss.

## Caveats

- 2D, CPU, exact visibility. With mask estimators the cone's bins shift with
  the light, so a bound must add a term for bin realignment.
- Recomputes include the movers' share (gate 3 needed about 6–16% for movers
  alone with two lights), so light motion alone costs less than 25.1%.
