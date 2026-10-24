# M4 gate 4: a visibility buffer

Date: 2026-10-24. M4 gate 3 left a high floor: 72 ms at 1080p for 1,000
boulders, mostly camera rays.

**The hypothesis.** One large shader did both the ray cast and the mask cones,
and the cone state (an accumulator, coverage bins, clipping polygons) costs
registers, so few rays run at once.

**The change.** A lean visibility pass finds each pixel's nearest hit and
writes it to a buffer, and a second pass shades from it. Both shade with the
same functions, and the shading pass rebuilds the identical hit, normal
included.

Pass criterion, fixed before the first run (committed in 5226aa6 before
implementing): all at 1920×1080, best of 5, split vs single pass in the same
session.

1. **Same output.** On the 1,000,000-boulder field (per-pixel prefiltered
   cones) and on the showcase (cached render, direct and indirect light),
   every pixel of the split render is within 1e-5 of peak of the single pass.
2. The 1,000,000-boulder frame is at least 1.3× faster split.
3. The showcase's cached render is no more than 1.05× slower split.

**Verdict: FAIL on (2) and (3).** The output is identical, but the split is
slower: 0.88× on the field and 1.11× on the showcase.

## Results

Apple M4 (MacBook Air, on battery). Reproduce with
`cargo run --release -p room3d --bin m4_gate4`.

| | Single pass | Split | |
|---|---|---|---|
| 1,000,000 boulders | 103.0 ms | 116.4 ms | 0.88× faster |
| Showcase cached render | 5.88 ms | 6.51 ms | 1.11× slower |

Max difference between split and single pass: **0.0** on both.

## Why it failed

A diagnostic timed rays alone, using a light that faces away so no cone runs:

| Boulders | Order of passes | Full frame | Rays only |
|---|---|---|---|
| 1,000 | single | 74.8 ms | 82.9 ms |
| 1,000 | split | 100.4 ms | 126.2 ms |
| 1,000,000 | single | 112.1 ms | 106.8 ms |
| 1,000,000 | split | 111.7 ms | 120.6 ms |

Timings move by about ±10% between back-to-back runs on this fanless laptop
on battery (here "rays only" was sometimes slower than the full frame). The
pattern still holds: a visibility pass that does nothing but cast rays is no
faster than the single kernel, and the split adds a pass and a buffer round
trip. So the hypothesis was wrong. Register pressure from the cone code is not
what limits the rays; the traversal itself is.

The likeliest cause is the traversal stack: a 48-entry per-thread array
indexed at runtime. Apple GPUs tend to keep such arrays in slow memory (M2
gate 3 saw a similar slowdown from per-thread arrays). Workgroup memory is the
usual fix, and is the next gate.

## Decision

The split stays in the code (`set_visibility_buffer`, and the live renderer
honors it), but **single pass is the default**, since it is faster today. A
view-dependent lighting cache will want the visibility buffer back, to know
which surfaces are seen.
