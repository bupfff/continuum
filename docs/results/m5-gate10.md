# M5 gate 10: a guessed sky from coarser pages (development only)

**Not run at full size; the option is removed.** Development runs failed
criterion 1 by a wide margin: 1.00× the baseline's error in flight,
against at most 0.8×.

Binary: `experiments/room3d/src/bin/m5_gate10.rs` (criteria committed in
`ce4a9f1`). Development runs used `M5_DEV=1`: 640×360, 100,000 boulders, a
200-frame flight, quality at frames 50, 100 and 200.

## What was tried

With the sky later (M5 gate 9), a page whose sky is not baked yet shows the
open sky over the pixel's normal. In the M4 benchmark's sun-and-sky flight,
82.9% of pixels count as fallback, mostly such guesses.

**First version.** A guessed page showed the sky of the finest coarser
resident page whose sky was baked.

| Measure | Coarser pages' sky | Open sky |
|---|---|---|
| Flight, mean error from settled | 24.04% | 24.00% (1.00×) |
| Warming, mean error | 2.68% | 2.69% |
| Flight cost | 1.02× | — |
| Fallback | 83.4% | 83.4% |

Coarse pages with a baked sky were rarely there. Pages unused for 64 frames
are evicted each frame, taking their queued skies with them, and coarse
pages fall out of use as soon as finer ones arrive.

**Second version.** Coarser pages met were also kept as used, and queued
skies were baked coarsest first.

| Measure | Coarser pages' sky | Open sky |
|---|---|---|
| Flight, mean error from settled | 23.98% | 23.93% (1.00×) |
| Warming, mean error | 3.07% | 3.14% (0.98×) |
| Frames until warm | 609 | 483 |

Coarsest-first ordering made both warm-ups worse than the first version's
(about 3.1% against 2.7%).

## What it shows

In flight, the open-sky guess is not what keeps the image from its settled
self. Both versions sit at the same 24–28%. That is the cache's known
softness in motion (M4 G11: pages stream in coarse first while the view
moves), not the sky. The high fallback with the sky later is mostly
cosmetic: M9 gate 5's development flights also found the same error with
and without the sky later.

Reverted: the option, the coarsest-first ordering, and keeping coarser
pages alive.
