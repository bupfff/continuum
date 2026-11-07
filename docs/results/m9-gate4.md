# M9 gate 4: the ledger without its work limit, at a real sun's speed

**FAIL on criterion 2 (speed); criterion 1 (soundness) passes.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Soundness and quality at gate 2's speed (30 checks) | 0 pages beyond their bound, 0 beyond ε | PASS |
| 2. Speed at a real sun's speed | means **24.65–25.29 ms** against 15.78 ms still (1.6×; at most 16.7 ms and 1.25×); fallback **4.85–5.04%** (at most 10%); 99th percentile 43–44 ms (at most 33.3) | FAIL |

Apple M4, 1280×720, 2026-11-07, run once (binary
`experiments/room3d/src/bin/m9_gate4.rs`, criteria in its first commit).

## What changed since gate 2

Gate 2's fallback (33–34%) came from the ledger's work limit (eight times
the bake budget), which bound and set visible pages aside. Now the limit is
the buffer's capacity, and fallback fell to about 5%. Development runs at
640×360 are disclosed in the binary: no limit (11.4% fallback, 1.28×), and a
guard of five radii (not kept).

## Why it is slow

Pages with texels near shadow edges ("mixed" texels: about 3,100 pages a
frame) re-baked those texels whenever the light moved at all. That was
about 51,000–54,000 texel jobs a frame, even at a real sun's speed of 0.25°
a minute.

- Their bound includes ε, for snapping, so it always exceeds ε once the
  light moves.
- The band term itself is about 7.5× looser than the true change: the
  circumscribed circle's perimeter instead of the window's, and a doubled
  ratio term.
- At a real sun's speed, a tight bound would keep a penumbra texel for
  about 8 frames.

Reported, at gate 2's speed: mean 27.15 ms, fallback 5.22%, 76,872 texel
jobs a frame.

## What next

A tight bound for mixed texels: the window's true perimeter, no doubled
term, and snapping's risk counted only for ratios near 0 or 1. That is for
a new gate.
