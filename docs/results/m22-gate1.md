# M22 gate 1: bounce light with movers, through the ledger

**FAIL** on criterion 4 (cost) in both flights and on criterion 1 at one
check of the new flight; criteria 2 and 3 pass in both.

| Criterion | (a) six boulders, 0.02 m a frame (developed) | (b) twelve boulders, 0.04 m a frame (new) |
|---|---|---|
| 1. Pages over ε against a renderer without re-bakes, frames 30/60/90/120 (bar 0.7×) | 260/494 (**0.526×**), 306/586 (**0.522×**), 289/587 (**0.492×**), 300/657 (**0.457×**): PASS | 463/575 (**0.805×**, FAIL), 484/793 (**0.610×**), 551/860 (**0.641×**), 440/930 (**0.473×**): FAIL |
| 2. The image from its bounce re-baked in place (bar 0.6×) | 0.795% against 2.175% (**0.365×**): PASS | 1.239% against 2.414% (**0.513×**): PASS |
| 3. Stopped and settled: charges drained; pages over ε against the scene settled without motion | largest charge 0; **123** of 2,404 against 158 of 2,501: PASS | largest charge 0; **25** of 2,450 against 169 of 2,485: PASS |
| 4. Moving frames' median against bounce light off (bar 1.2×) | 44.02 ms against 20.35 ms (**2.163×**): FAIL | 69.36 ms against 33.43 ms (**2.075×**): FAIL |

No page over ε was missed by the ledger (its charge below ε) at any check.
Every renderer settled. Largest bounce values 2.2–2.7e-5; the largest
change while moving 9.4e-6–2.5e-5.

Apple M4 (fanless MacBook Air), 640×360, release build, run once at full
size on 2026-11-16 (commit `c2cc78f`; binary
`experiments/room3d/src/bin/m22_gate1.rs`, stub committed in `81935ad`;
output `docs/results/m22-gate1-run.txt`), after three minutes with no
game or indexing running and one more minute idle.

## Method

DESIGN §3.4, "Bounce light with movers": pages holding bounce samples are
charged, each frame something moves, the share of their view the moves can
change; past ε (0.01) their bounce samples alone re-bake (listings
expiring after 16 frames); at rest a drain re-bakes what is charged above
ε / 100 and what was baked while something moved. M6's textured scene, sun
only, bounce light on, a fixed bake budget. `verify_bounce` re-bakes every
resident page's bounce samples in place and counts pages that changed by
more than ε of the largest bounce value, restoring the cache;
`rebake_bounce` leaves them re-baked for the image comparison.

## What failed

- **Cost.** The ledger's re-bakes double the moving frame: bounce samples
  (nine points of 192 rays a page) for about 200 pages a frame. The
  development figure (1.10×) compared a fixed-budget run with bounce
  against a time-budgeted run without it, taken separately: a flawed
  comparison, found by the gate's interleaved timing.
- **The new flight's first check** (0.805×): twelve boulders twice as fast
  charge more pages than the budget re-bakes in the first 30 frames; later
  checks are within the bar.

## Disclosures

- The bars were set with development numbers known; flight (b) was not
  used in development, but the development mode's run (320×180, 30
  frames) showed its first checks at 0.83–0.86× before the run.
- The largest charges reported while moving (up to 3,000) are on the
  moved boulders' own pages, which the stale pass re-bakes every frame:
  their fill can race the bounce retry's in the same frame and keep the
  old charge. Their bounce samples do re-bake (no page over ε was missed).
- Sun only: movers under a sky cost 331 ms a frame at 640×360 on this
  scene (M9 gate 3's sky re-bakes), a gap of its own.
- At rest both the ledger's renderer and a scene that never moved keep
  pages over ε: bounce samples baked against coarse hits are not baked
  again when finer hits arrive (the cache's own; item 5's "quicker to
  settle").
