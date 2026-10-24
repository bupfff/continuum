# Flatland gates 2 and 3: the error ledger

Date: 2026-10-23. Questions: is the ledger's certified bound sound (gate 2), and
how much lighting work does it skip in a typical animated scene (gate 3)?

**Verdicts: both pass in 2D, with an important qualification.** At realistic ε
all of the skipping comes from dependency regions (knowing exactly which tiles a
change can reach); the magnitude half of the ledger never deferred anything for
direct shadows. Details and why below.

## Setup

- 256×256 receivers in 8×8 tiles, 2 segment lights, 240 frames at 60 fps,
  CPU, single thread, release build. Reproduce with
  `cargo run --release -p flatland --bin ledger`.
- Caches hold mask-cone visibility per (light, receiver). Light intensity is
  applied when composing, so intensity changes (switching, flicker) are exact
  and free; only geometry motion is charged.
- Each frame, every moved occluder's swept hull charges every (light, tile)
  whose cones it could reach, with the smaller of two worst-case bounds (swept
  region angle, or endpoint-path angles), plus bin quantization. A (light, tile)
  is recomputed when radiance × balance exceeds its share of ε.
- Reference: a full recompute of every cone against every occluder, every frame.
  Gate 2 compares every receiver, every frame, at every ε.
- Scenarios:
  - **Typical:** the gate 1 room plus a swinging door, a person walking at
    about 1.5 m/s, and a spinning fan. Light B switches on at frame 60 and
    flickers from frame 150.
  - **Worst case:** every one of the 41 occluders drifts every frame.
- ε is a fraction of mean fluence at frame 0. 16% and 64% are deliberately
  unusable "stress" values whose job is to force deferral, so the magnitude
  bounds are actually exercised.

## Typical scenario

Full recompute: 16.66 ms/frame. Full recompute reusing static masks, no ledger: 0.99 ms/frame.

| ε | Gate 2 violations | Deferred receiver-frames | Max error / ε | Bound use | Skipped | Incremental ms/frame | Speedup vs full |
|---|---|---|---|---|---|---|---|
| 0% | 0 | 0.0% | — | — | 87.7% | 0.34 | 48.8× |
| 1% | 0 | 0.0% | 0.00 | — | 87.7% | 0.33 | 50.5× |
| 4% | 0 | 0.1% | 0.24 | 0.2% | 87.8% | 0.33 | 50.4× |
| 16% (stress) | 0 | 6.8% | 0.29 | 2.2% | 90.6% | 0.27 | 61.3× |
| 64% (stress) | 0 | 20.6% | 0.30 | 2.6% | 95.7% | 0.16 | 105.5× |

Cones recomputed per frame at ε = 1%, by phase: movers only 5.7%, the frame
light B switches on 26.2%, movers with two lights 15.9%, movers plus flicker
12.9%. At the switch-on frame the output matched a full recompute exactly
(max error 0): instant, exact response, no fade-in.

## Worst case

Full recompute: 15.33 ms/frame.

| ε | Gate 2 violations | Deferred receiver-frames | Skipped | Incremental ms/frame | Speedup vs full |
|---|---|---|---|---|---|
| 0% | 0 | 0.0% | 24.2% | 13.36 | 1.1× |
| 1% | 0 | 0.0% | 24.2% | 13.36 | 1.1× |
| 4% | 0 | 0.6% | 24.5% | 13.30 | 1.2× |
| 16% (stress) | 0 | 19.2% | 33.7% | 11.99 | 1.3× |
| 64% (stress) | 0 | 53.5% | 54.4% | 8.74 | 1.8× |

## What this shows

1. **Gate 2 passes, and the test is not vacuous.** Zero violations across both
   scenarios, five ε values, and 240 frames of every receiver. At the stress ε
   values the ledger deferred work on up to 53.5% of receiver-frames, and the
   certified bound held every time. Actual error never exceeded 0.38 ε.
2. **Gate 3 passes: 87.7% of cone evaluations skipped at ε = 1%, with output
   identical to a full recompute.** The orange tiles in
   `out/ledger_typical.png` are exactly the movers' shadow paths.
3. **Where the speedup comes from (honest decomposition):** reusing static masks
   accounts for 16.8× (16.66 → 0.99 ms), which is a known kind of optimization
   (engines cache static shadows too). The ledger's dependency regions add a
   further 3.0× (0.99 → 0.33 ms) with exact output. Against a smart
   conventional renderer that already caches static work, the ledger's edge
   here is about 3×, not 50×.
4. **Gate 4 preview: no penalty in the worst case.** With every occluder moving,
   the incremental path was still 1.1× faster than a full recompute, because
   charging is cheap and tiles with a clear view of the light are still skipped.

## Findings that change the plan

- **Binary masks block magnitude deferral.** A mover that clips a single bin
  center flips a whole bin (1/64 of the light), so every charge carries at least
  one bin of quantization. That always exceeded realistic ε, so ε had no effect
  between 0% and 1%. Making visibility continuous (analytic coverage for
  partially covered edge bins) would remove that floor. It is the same fix
  already planned for gate 1's banding.
- **Direct shadows are the wrong place to look for magnitude gains.** A moving
  object's own shadow genuinely changes by more than a just-noticeable amount
  each frame, so recomputing it is correct. The magnitude half of the ledger
  should pay off in wide, low-frequency queries: GI cones, where a moving object
  is a sliver of the cone, and slow global changes like a moving sun. That needs
  a flatland GI test.
- **Bounds are sound but loose.** Actual error was 0.2–8% of the certified bound
  (the "bound use" column). Main sources: one worst-case charge per tile applied
  to all 64 receivers, the bin terms, and triangle-inequality accumulation over
  oscillating motion. Tighter bounds mean more deferral at the same ε.

## Caveats

- 2D, CPU, direct lighting only, lights static (moving lights are not handled yet).
- Receivers are world-space points, so camera motion is not modeled. In 3D,
  world-space lighting caches are unaffected by the camera, but view-dependent
  work is not.
- Timings vary run to run by about 10% on the fanless M4 Air.
