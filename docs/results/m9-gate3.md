# M9 gate 3: moving objects under a sky, through a ledger

**FAIL on criteria 2 (work) and 3 (speed); criterion 1 (soundness) passes.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Soundness (30 checks over 300 frames) | 0 light values differing; 0 pages' sky beyond their ledger or ε | PASS |
| 2. Work: sky re-baked ≤ 25% of pages charged | **99.73%** | FAIL |
| 3. Speed: moving means ≤ 20 ms and ≤ 1.25× still; p99 ≤ 40 ms; fallback ≤ 10% | means **53.0–58.3 ms** (still 14.66 ms: 3.6–4.0×); p99 60.7–65.4 ms; fallback 30.2–30.7% | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `db9dcf2`; binary
`experiments/room3d/src/bin/m9_gate3.rs`, criteria pre-registered in
`e7e65a8`). Control: gate 1's sun-only warm frame 2.11 ms (at most 3.3),
fit. The machine was warm from development runs and tests just before.

## Method

The cache used to refuse movers with a sky. Now:

- **Light channels** stay exact, through gate 1's path.
- **Sky charges.** Each frame something moves, each resident page is
  charged a bound on how much its sky can have changed since its bake, per
  unit sky radiance. Per mover moved since the bake, that is the solid
  angle of the mover's bounding sphere at the bake, plus its solid angle
  now, both seen from the nearest point of the page's box (π if inside).
  It telescopes: a page's sky can only have changed where a mover covered
  it then or covers it now. Movers beyond the 96th are charged both spheres
  every frame.
- **ε.** The threshold is ε × max(the page's smallest sky value, 0.05π).
  - A page read in the frame before re-bakes its sky in the same frame once
    its charge passes the threshold. The stale pass appends it, with its
    sky samples, to the frame's bake.
  - Any other page is marked invalid.
- **Pages on moved objects** keep their exact light channels. Their sky
  shows the cache's guess, flagged as fallback, until the CPU re-bakes it
  30 frames later.

Changed before the full-size run (listed in the binary):

- The charges telescope. Summing each frame's old and new spheres had pages
  near movers passing ε within frames.
- Same-frame sky re-bakes replaced listing at half of ε for the CPU, after
  a 640×360 development run showed 67% fallback.
- A race in the fill was fixed: a sky-only re-bake gave up when it saw the
  mark that a light re-bake sets until its own fill clears it. The GPU test
  `movers_under_a_sky_stay_within_the_ledger` found it.

A second 640×360 development run gave 86% of pages re-baked and 37%
fallback, so this result was expected.

## Results

| Flight | Mean | p99 | Fallback | GPU bake | Sky re-baked / charged per frame |
|---|---|---|---|---|---|
| 1 still | 14.66 ms | 17.22 ms | 22.33% | 8.87 ms | — |
| 1 moving | 58.27 ms | 65.38 ms | 30.19% | 41.62 ms | 1,497.5 / 1,497 |
| 2 still | 14.66 ms | 18.38 ms | 21.67% | 8.35 ms | — |
| 2 moving | 53.01 ms | 60.72 ms | 30.67% | 34.03 ms | 1,359.9 / 1,359 |
| 3 still | 14.65 ms | 17.01 ms | 21.22% | 7.15 ms | — |
| 3 moving | 55.90 ms | 63.14 ms | 30.34% | 36.28 ms | 1,449.5 / 1,449 |

- Pages charged are those read in the frame before. Almost every one passed
  ε in every frame something moved and re-baked its sky: the "ledger"
  deferred nothing.
- Light channels re-baked 5.4–5.6% of the pages read (gate 1's path), with
  about 23 pages a frame marked invalid. Another 1,500–1,700 off-screen
  pages per frame were invalidated for their sky.
- Soundness passed, but for the sky it passed vacuously: every page shown
  had a charge of zero (re-baked that frame), and no page shown had a sky
  that changed at all.
- Even still, the flight with sky has 21–22% fallback (against 10%): the
  sky makes streaming slow (bake budget about 1,700, against about 9,000 at
  640×360). That is M5 gate 5's slow sky convergence, not the movers.

## Why it failed

The bound is per page and per sphere. In development checks, readable
pages' real sky changes were around 1e-6, with charges around 5e-3 (about
5,000× loose). Two things make it loose:

- **Whole spheres.** A boulder circling 1.5 m mostly covers what it covered
  before. The bound charges its whole silhouette twice, and ignores the
  cosine to the surface: a boulder seen near the horizon from the ground
  blocks a few percent of its solid angle's irradiance.
- **Whole pages.** A coarse page's box spans tens of meters and contains a
  mover, so it is charged π, even though only texels near the mover see it.

## What next

Each item below is for a new gate.

- Texel-level sky charges, as gate 1 tests light texel by texel: each sky
  sample's apex against the movers, with the cosine to its normal and the
  horizon, re-baking only the samples past ε.
- Faster sky streaming (M5 gate 5's slow convergence), which also limits
  the still flights here.

## Prior art

Bounding the change of occlusion by a mover's solid angle is standard in
conservative visibility and shadow caching. This gate makes no novelty
claim.
