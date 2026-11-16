# M20 gate 2: depth of field, with seen lights and highlights fixed

**PASS** on all four criteria, in every view.

| Criterion | (a) focused on the brick block | (b) focused at 5 m | (c) night, focused at 3 m | (d) metallic boulders, focused at 3 m (new) |
|---|---|---|---|---|
| 1. Cost at 1280×720 (bar) | **+1.789 ms** (2.2) PASS | **+1.901 ms** (2.2) PASS | **+2.488 ms** (2.8) PASS | +1.824 ms (reported) |
| 2. From the sampled lens: ours / the pinhole (bar 0.2×) | 1.89% / 14.11% (**0.134×**) PASS | 1.95% / 15.10% (**0.129×**) PASS | 121.12% / 4818.91% (**0.025×**) PASS | 4.81% / 101.54% (**0.047×**) PASS |
| 3. From Cycles: ours / the sampled lens (bar 1.12×) | 7.09% / 6.92% (**1.025×**) PASS | 6.90% / 6.56% (**1.053×**) PASS | 50.75% / 107.60% (**0.472×**) PASS | 6.66% / 6.65% (**1.001×**) PASS |
| 4. The mean of ours against the sampled lens's (bar 1%) | **+0.000%** PASS | **−0.036%** PASS | **−0.497%** PASS | **−0.137%** PASS |

Means against Cycles (ours, the sampled lens, the pinhole): (a) −2.40%,
−2.40%, −2.41%; (b) −2.44%, −2.40%, −2.40%; (c) −3.33%, −2.85%, −2.80%;
(d) −0.86%, −0.72%, −0.70%. The pinholes against Cycles: 15.40%, 16.09%,
4676.59%, 100.67%. Highlights at 1280×720: 160, 160, 3,347 and 970, no
tile over its list. Every view settled. Cycles took 13, 13, 34 and 16 s
(Metal, 4,096 samples). Timing rounds: (a) 4.508–4.555 against
2.724–2.784 ms; (b) 4.684–4.702 against 2.783–2.803; (c) 5.331–5.350
against 2.843–2.861; (d) 6.704–6.733 against 4.880–4.909.

Apple M4 (fanless MacBook Air), 640×360 (timed at 1280×720), release
build, run once at full size on 2026-11-16 (commit `7501a88`; binary
`experiments/room3d/src/bin/m20_gate2.rs`, stub committed in `f71ce50`;
output `docs/results/m20-gate2-run.txt`), after three minutes with no
game or indexing running and one more minute idle. Timings first.

## Method

DESIGN §3.4, "Depth of field" and its "For gate 2": `DepthOfField`
(aperture radius 0.1 m, the largest blur 4% of the height), gathered each
frame from the pinhole image and its depths, after gate 1's failure: a
pixel that sees a light takes the light's depth (a bright pixel beside
one, its radius); highlights are pixels 4× brighter than their tile's mean
and 8× brighter than its background, in focus or not, their excess over
the background added as light without coverage by every pixel their disc
covers; a disc's soft edge is no wider than the disc. Ours and the pinhole
at 4×4 sub-pixel offsets, each settled. References: the lens sampled
outright in our renderer (16 × 16 aperture points, the eye moved to each
with the image plane sheared so the plane in focus stays put, each
settled); Cycles' thin lens.

## Disclosures

- The bars were set with development numbers known (0.134×, 0.129× and
  0.025×; 1.025×, 1.053× and 0.472×; means within 0.5%; +1.78, +1.91 and
  +2.49 ms). View (c) is gate 1's, developed on after gate 1 failed; view
  (d) was not used in development.
- Views (c) and (d) are dominated by small, very bright sources (the
  night's point light; the sun in the metallic boulders), so criteria 2
  and 3 there measure highlights mostly. In (c) ours is nearer Cycles than
  the sampled lens is: its 256 copies of the light make a rougher disc
  than the exact one ours draws.
- The floor against Cycles is the shading's (our pinhole 7.08% from
  Cycles' pinhole on the day scene); criterion 3 measures against the
  sampled lens's own distance from Cycles for that reason.
- Not measured: aperture blades, blur seen through glass layers, movers.
- Cost grows with the scene's edges and highlights: the materials view
  costs as much as the day views, the night view (foliage, 3,347
  highlights) 0.6 ms more.
