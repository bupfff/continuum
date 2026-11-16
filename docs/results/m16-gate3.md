# M16 gate 3: solid glass (refraction through volumes)

**FAIL on criterion 1 (speed); criterion 2 (accuracy) passes.**

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Speed: warm frame, solid / opaque glass | **1.378×** (median of five rounds: 4.55/3.59, 5.12/3.91, 5.42/3.72, 5.13/3.72, 5.14/3.72 ms) | at most 1.3× | FAIL |
| 2. Accuracy against Cycles, clamped at 4× its mean | **8.34%** (opaque 87.21%, thin 22.24%) | at most 12% | PASS |
| 2. Over the 223,905 pixels transmission changes | **16.16%** against thin glass's 43.53% (**0.371×**; opaque 170.97%) | at most 0.5× | PASS |

Unclamped: 81.13% (opaque 131.10%, thin 83.55%); over the changed pixels
156.79% (thin 161.48%). Means over them: ours +4.77%, thin +18.97%, opaque
+109.74%. Pixels above the clamp: ours 103, Cycles 98. Every view settled.
Cycles took 90 s (Metal, 4,096 samples).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `afd4852`; binary
`experiments/room3d/src/bin/m16_gate3.rs`, stub committed in `28e8fa2`;
output `docs/results/m16-gate3-run.txt`), started after three minutes with
no game or indexing running and one more minute idle.

## Method

DESIGN §3.4, "Transmission" (volumes): the textured scene with a clear
glass sphere and an attenuating glass block (`Textured::with_glass_solids`),
bounce light on, ours at 4×4 sub-pixel offsets against Cycles' Principled
BSDF with its transmission weight and index, a Volume Absorption, shadow
rays meeting a Transparent BSDF at each surface, no caustics, transmission
bounces allowed beyond the one diffuse bounce. Both images clamped at 4×
the reference's mean, as fixed before the run (the sun's reflection in the
smooth sphere covers a few pixels at its full radiance).

## Disclosures

- **The machine's state moved during the timing.** The first round ran at
  1.267×; the next four at 1.31–1.46×, while the opaque frames rose from
  3.59 to 3.72–3.91 ms. The external displays' compositor (WindowServer)
  had been at about 40% of a core shortly before the run. The median over
  rounds is the criterion and it fails; development had measured 1.344×
  at this size, so the gate would most likely have failed either way.
- The glass covers about 11% of the image (105,412 records). Each such
  pixel traces about three rays (in, a check for a touching surface, on)
  with two interpolated normals in `hw_layers`, and the extra-samples
  pass shades what lies behind: about 1.2 ms at 720p.
- Shadows through the tinted block are grey (Cycles' are green, from the
  attenuation along its shadow rays); the sun's highlight on the sphere
  differs over a few pixels (unclamped error).

## Next

Speed levers, for a later gate: skip the touching-surface query for glass
whose box meets no other object's; trace the in and touching queries
together; shade the hit behind in `hw_layers`'s own records more cheaply.
Accuracy is settled.
