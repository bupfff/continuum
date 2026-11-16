# M20 gate 1: depth of field

**FAIL** on criteria 2 and 3, in view (c) only. Criterion 1 passes, and
views (a) and (b) pass every criterion.

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Cost at 1280×720, view (a) (focused on the brick block) | **+1.657 ms** (4.424–4.439 against 2.756–2.783 ms, five rounds) | at most 2.0 ms | PASS |
| 1. Cost, view (b) (focused at 5 m) | **+1.752 ms** (4.529–4.546 against 2.779–2.796 ms) | at most 2.0 ms | PASS |
| 1. Cost, view (c) (night, focused at 3 m) | +2.124 ms | reported | |
| 2. (a) From the lens sampled in our renderer | **1.89%**, the pinhole 14.11% (**0.134×**) | at most 0.2× | PASS |
| 2. (b) | **1.95%**, the pinhole 15.10% (**0.129×**) | at most 0.2× | PASS |
| 2. (c) | **1368.72%**, the pinhole 4818.91% (**0.284×**) | at most 0.2× | FAIL |
| 3. (a) From Cycles (4,096 samples) | **7.15%**, the sampled lens 6.92% (**1.034×**); the pinhole 15.40% | at most 1.12× | PASS |
| 3. (b) | **6.92%**, the sampled lens 6.56% (**1.056×**); the pinhole 16.09% | at most 1.12× | PASS |
| 3. (c) | **1321.28%**, the sampled lens 107.61% (**12.279×**); the pinhole 4676.65% | at most 1.12× | FAIL |

Means against Cycles: (a) ours −2.41%, the sampled lens −2.40%; (b) ours
−2.44%, the sampled lens −2.40%; (c) ours **−6.83%**, the sampled lens
−2.84%. Highlights at 1280×720: 0, 10 and 531, no tile list overflowed.
Every view settled. Cycles took 13, 13 and 37 s (Metal).

Apple M4 (fanless MacBook Air), 640×360 (timed at 1280×720), release
build, run once at full size on 2026-11-16 (commit `e5ca6c5`; binary
`experiments/room3d/src/bin/m20_gate1.rs`, stub committed in `b188863`;
output `docs/results/m20-gate1-run.txt`), after three minutes with no
game or indexing running and one more minute idle. Timings first.

## Method

DESIGN §3.4, "Depth of field": `DepthOfField` (aperture radius 0.1 m, the
largest blur 4% of the height), gathered each frame from the pinhole image
and its depths. Ours and the pinhole at 4×4 sub-pixel offsets, each
settled. References: the lens sampled outright in our renderer (16 × 16
aperture points by the concentric map, the eye moved to each with the
image plane sheared so the plane in focus stays put, each settled); and
Cycles' thin lens.

## What failed

View (c), not used in development, sees the night scene's point light (10
cm across, 6.7 m away). The relative RMSE there is almost all the light's
disc (the image is otherwise dark), so the criteria measure the light's
blur. Ours draws it as a clipped, blocky shape, where the sampled lens and
Cycles draw a round disc (`experiments/room3d/out/m20_gate1_c_*.png`), and
loses 4% of the image's light. Two causes, found after the run:

- **Highlights against the tile's mean.** A pixel becomes a highlight
  (its excess drawn exactly as a disc) only when it is 8× brighter than
  its 16 × 16 tile's mean. The light covers so much of the tiles it falls
  in that only some of its pixels pass. The rest stay in the coarse levels,
  whose sparse taps scatter a very bright source into blotches, the defect
  highlights were made for.
- **A seen light's depth.** A pixel that sees a light takes its blur
  radius from the surface (here the sky) behind the light, not the light:
  14.4 px where the light's own depth gives about 8.

## Disclosures

- The bars were set with development numbers known (0.129× and 0.134×,
  1.055× and 1.033×, +1.77 and +1.71 ms). View (c) was not used in
  development; the development mode's run (160 × 90, four aperture points)
  showed its numbers dominated by the light, too coarse to read.
- The floor against Cycles is the shading's: our pinhole is 7.08% from
  Cycles' pinhole on this scene. Criterion 3 compares against the sampled
  lens's own distance from Cycles for that reason.
- Not measured: aperture blades, blur through glass layers, movers.
