# M9 gate 6: a moving sun at real speed, its shadows' edges held within a certified bound

**PASS.** A sun moving at a real sun's speed over the million-boulder field
now costs what a still one does: three flights averaged 15.16–15.70 ms
against 15.43 ms with the sun still (0.98–1.02×; gate 4: 1.6×). The
ledger re-bakes about 2,000 texels a frame instead of 51,000: shadow edges
("mixed" texels) hold their value until a certified bound on their change
reaches ε, which at this speed takes several frames. Every one of 886,506
pages checked stayed within its bound and within ε.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 1. Mean frame, real sun (three flights) | 15.16, 15.51, 15.70 ms; still 15.25–15.66 ms (mean 15.43) | at most 16.7 ms and 1.25× still (19.29 ms) | PASS |
| 1. 99th percentile, real sun | 27.48, 28.29, 28.14 ms | at most 33.3 ms | PASS |
| 1. Fallback, real sun | 5.94%, 5.21%, 5.78% (still 2.38–2.47%) | at most 10% | PASS |
| 2. Soundness at a real sun's speed (10 checks) | 353,205 pages: 0 beyond their bound, 0 beyond ε; largest change 1.29e-3, largest bound 9.98e-3 | none | PASS |
| 2. Soundness at 8× a real sun (10 checks) | 299,876 pages: 0, 0 | none | PASS |
| 2. Soundness at gate 2's speed, 60× (10 checks) | 233,425 pages: 0, 0 | none | PASS |
| 3. At rest, against the sun fixed there | 0.054% (relative RMSE) | at most 1% | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `f6490cb`; implementation and binary `9b49c1b`,
before the run), after 90 s idle. Control: the sun alone warm in 2.18 ms
(gate 4: 2.15 ms; at most 3.3 ms), so the machine was fit to measure.

## Details

- **Per moving frame (means over a flight):** of 46,138–50,741 pages read,
  3,393–3,694 held mixed texels; 2,366–2,545 of those held within their
  bound; the ledger re-baked 1,837–2,125 texels on 1,450–1,610 pages (gate
  4: 51,463 texels on 3,425 pages). Most re-baked pages re-bake one or two
  texels near the light's horizon, which still re-bake whenever it moves.
- **Bake budget:** 2,499–2,601 units a frame moving against 6,042–6,350
  still. The frame-time model subtracts the ledger's appended work, which
  is costly per texel (shadow edges and near-horizon texels walk far), so
  new pages bake more slowly and fallback rises from about 2.4% to 5.2–5.9%
  (gate 4: 4.9–5.0%).
- **Reported, gate 2's speed (60× a real sun):** mean 17.00 ms, 99th
  percentile 21.38 ms, fallback 9.39%; 25,724 texels re-baked a frame (gate
  4: 27.15 ms, 76,872). At that speed the bound passes ε within a frame, so
  almost no mixed texel holds (150 of 1,978 pages).
- **Settling (criterion 3):** the moving renderer's last view settled in
  2,741 frames, the fixed one's in 1,914.

## Method

1. **A lattice for each page's cones.** A moving rect light's cones keep
   their bins on a lattice fixed in the frame of the light's pose at the
   page's version (its last whole bake): forward toward the light's center
   then, right along its first edge, bins a seventh of that footprint (plus
   3%) on a side. The window is the lattice's 8×8 bins holding the light's
   footprint now; each bin counts by the footprint's area in it times its
   projected solid angle per area. A bin's coverage then does not depend on
   where the light is (bins outside the footprint start covered, so walks
   still stop once the footprint is), and the ratio, an average of bins'
   openness, moves only as the footprint's weighting does.
2. **The bound.** Every point of the light moves at most δ seen from the
   page (as the ledger already measured), at most sec²ρ δ in the window's
   plane within the guard's reach ρ; the footprint A' then lies within that
   distance of A, so each footprint loses or gains at most P δ' + π δ'²
   (Steiner; P the perimeter). With the bins' weights spread by at most κ,
   the two weightings differ in total variation by at most 1.5 κ (P δ' + π
   δ'²) / Ω (Ω a lower bound on the footprint's solid angle), and so does
   the ratio. A page's mixed texels re-bake once that bound, since they were
   baked, passes a hashed share (a half to all) of ε; no snapping.
3. **What keeps it valid:** every texel the light reaches keeps a slack
   (the guard's reach less the light's radius and a bin's diagonal) so its
   bins stay inside the guard and inside the region κ was taken over; held
   texels re-bake if their footprint might outgrow the lattice's window; a
   texel whose weights spread more than κ = 8 (the light near its horizon)
   is marked to re-bake whenever the light moves and bakes on the light's
   own cone.
4. **Pages set aside** off screen are checked texel by texel when wanted
   again, instead of re-baking whole (their records were kept). This alone
   took the 60× flight from 76,872 to about 26,000 re-baked texels.
5. `verify_ledger` re-bakes each page on its own lattice (the ledger's
   records kept for the re-bake).

**Disclosed (before the run):** the gate used gate 4's options, the cache's
defaults without the frame-period cap added since (M14 gate 9), chosen
after a development run with the cap gave 11.83% fallback. Three
development runs at 640×360 are listed in the binary's doc comment; the
second found set-aside pages re-baked whole, near-horizon texels on the
lattice's larger window, and the growth check evaluated for every page
(about 4 ms a moving frame), all changed before the third, which passed
(1.08×, fallback 8.56%).

## Caveats

- **The bound is exercised at a real sun's speed only.** At 8× and 60×,
  mixed texels' bound passes ε within a frame, so they re-bake every frame
  (bound 0); those checks test the rest of the ledger.
- **How tight:** the largest change seen (1.29e-3) was about an eighth of
  the largest bound (9.98e-3). Steiner's P δ is about π/2 to 3× the
  footprint's true change for a translation, and 1.5 covers areas that
  differ; a tighter bound would hold texels longer.
- **Certified against a re-bake on the page's own lattice.** A whole
  re-bake re-centres the lattice, and the value can move then by the
  estimator's discretization (a few thousandths, as cones re-centred on the
  light always did); that is a fresh bake, not staleness.
- **Rect lights only** (suns here are rects); moving sphere lights keep
  gate 4's behavior. The cone's far plane follows the light, so geometry
  within the light's own travel of the light is outside the bound (none in
  this scene).
- **Fallback roughly doubles** while the sun moves (5.2–5.9% against
  2.4%), from the ledger's appended work coming out of the bake budget.

## What the numbers say

- **Landmark 2's moving sun at real speed is done on this scene:** frame
  times as if the sun were still, every shown page within a certified
  bound, at rest within 0.05% of a renderer built with the sun there.
- **Closest prior art:** stable cascaded shadow maps snap the shadow map's
  projection to whole texels so edges do not shimmer as the view moves
  (Valient 2008); the lattice does the same for a cone's bins as the light
  moves, which is what lets a bound on the light's motion bound the cached
  value. Hedged until a literature check.
