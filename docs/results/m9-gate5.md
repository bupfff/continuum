# M9 gate 5: the sky later, with movers (development only)

**Not run at full size.** The development version failed criterion 2 by a
wide margin: 1.02× the baseline's error while moving, against at most
0.5×. Soundness and cost passed there. The code is kept: it is sound, costs
the same, and keeps skies from starving in long flights.

Binary: `experiments/room3d/src/bin/m9_gate5.rs` (criteria committed in
`6fa0e0e`). The implementation came before the binary; only the GPU tests
had run on it. Development runs used `M9_DEV=1`: 640×360, 200-frame
flights, one flight each, quality at frames 50, 100 and 200.

## Method

- **A guessed sky stays a guess with movers.** A page planned with its sky
  later gets a sky ledger saying "never baked" (stride 0) until its sky is
  baked alone, which writes a fresh ledger.
- **The stale pass skips such pages' sky.** It does not charge them (they
  show no sky). If a mover may change their light, they are marked invalid
  and baked anew rather than re-baked in place: an in-place light re-bake
  would clear the guess's mark and show a sky never baked.
- **Skies get half the budget first** (added after the first development
  run). Queued skies are listed up to half of each frame's budget before
  new pages, then with whatever new pages leave. In the first run, new
  pages took the whole budget in flight, and skies were rarely baked (84%
  of pixels showed a guess or nothing).
- **Still off** with moving lights and with bounce light.

## Development results

| Run | Cost (mean frame) | Quality while moving (mean error from settled) | Fallback | Soundness (6 checks) |
|---|---|---|---|---|
| First (new pages first) | 10.77 ms against 18.65 ms (0.58×) | 28.74% against 28.10% (1.02×) | 83.9% against 35.2% | 0 violations |
| With half the budget for skies | 14.63 ms against 14.45 ms (1.01×) | 28.59% against 28.11% (1.02×) | 45.4% against 36.9% | 0 violations |

- **In flight, both caches are 25–30% from the settled image.** They lag
  in which pages are resident, coarse pages included, and the open-sky
  guess is no closer than the baseline's coarser pages. The sky later
  orders the work; it does not make a flight converge.
- **Fallback** counts pixels with a guessed sky, so it rises.
- **Static warm-up with half the budget for skies** (M5 gate 9's
  development check, 100,000 boulders, standing): error while warming 0.24×
  the baseline's (0.17× with new pages first), warm in 0.73× the frames
  (0.65×), the same warm image. M5 gate 9's full-size numbers were measured
  with new pages first.

## What this means for movers under a sky

M9 gate 3's criteria count guessed skies as fallback, so the sky later
does not unblock them. The flights' error comes from streaming, not from
the sky's charges.
