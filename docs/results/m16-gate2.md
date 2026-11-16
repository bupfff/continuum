# M16 gate 2: thin transmissive glass at speed

**PASS** on both criteria.

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Speed: warm frame, transmissive / opaque panes | **1.217×** (4.35–4.45 ms against 3.56–3.65 ms, five rounds) | at most 1.3× | PASS |
| 2. Accuracy against Cycles | **5.08%** (panes opaque 85.41%, as alpha blending 13.90%) | at most 7% | PASS |
| 2. Over the 241,769 pixels transmission changes | **7.92%** against blending's 22.91% (**0.346×**; opaque 142.46%) | at most 0.5× | PASS |

Means over those pixels against Cycles: ours +2.90%, blending +14.92%,
opaque +91.77%. Every view settled (none needed the 20,000-frame report).
Cycles took 69 s (Metal, 4,096 samples).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `98d7def`; binary
`experiments/room3d/src/bin/m16_gate2.rs`, stub committed in `f77c831`;
output `docs/results/m16-gate2-run.txt`). Started after the machine had
been idle (no game, no indexing) for three minutes and one more minute;
the opaque frames (3.56–3.65 ms) match gate 1's (3.59–3.64 ms), the
control.

## What changed from gate 1

Gate 1 failed on speed only (1.389×): the extra-samples pass shaded each
pane pixel's own transmissive hit again to learn what it passes. Now the
shading pass keeps that in the colour's unused fourth word, 8 bits a
channel (`pack4x8unorm`), and the extra-samples pass starts from it and
shades only the layers behind. Accuracy is unchanged (5.08% against 5.09%).

## Method

As gate 1 (`docs/results/m16-gate1.md`): the textured scene with M10 gate
16's two panes as smooth thin glass, double-sided for Continuum and
one-sided for Cycles, bounce light on, ours at 4×4 sub-pixel offsets
against Cycles' reference material.

## Disclosures

- During development a game (Roblox) was running at about half a core;
  development timings in the gate's doc comment (1.097×) were taken
  then. It was closed before this run. Whether it ran during gate 1 is
  not known.
- Shadows under tinted glass are grey (the cache holds one channel per
  light); Cycles tints them. That is most of what is left over the pane
  pixels.
- In development, one 1280×720 run of `transmission_dev` did not settle
  within 20,000 frames and the next did; not reproduced here.
