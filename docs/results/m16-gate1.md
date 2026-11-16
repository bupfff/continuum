# M16 gate 1: thin transmissive glass

**FAIL on criterion 1 (speed); criterion 2 (accuracy) passes.**

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Speed: warm frame, transmissive / opaque panes | **1.389×** (5.00–5.04 ms against 3.59–3.64 ms, five rounds) | at most 1.3× | FAIL |
| 2. Accuracy against Cycles | **5.09%** (panes opaque 85.40%, as alpha blending 13.92%) | at most 7% | PASS |
| 2. Over the 240,235 pixels transmission changes | **7.94%** against blending's 22.96% (**0.346×**; opaque 142.47%) | at most 0.5× | PASS |

Means over those pixels against Cycles: ours +2.93%, blending +15.02%,
opaque +92.05%. Cycles took 66 s (Metal, 4,096 samples).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (commit `3467332`; binary
`experiments/room3d/src/bin/m16_gate1.rs`, stub committed in `a0bc35c`;
output `docs/results/m16-gate1-run.txt`; started after Spotlight's
indexing had been idle for a minute and two more minutes idle).

## Method

DESIGN §3.4, "Transmission": `InstancedScene::geometry_transmission`, the
textured scene with M10 gate 16's two panes as smooth thin glass
(`Textured::with_transmission`), bounce light on; ours at 4×4 sub-pixel
offsets against Cycles' reference material (the opaque Principled BSDF
mixed by one less the factor with its specular layer plus a Transparent
BSDF passing the base colour times one less Schlick's Fresnel
reflectance, the same from either side).

## Why it is slow

The panes cover about a quarter of the image (240,235 pixels change).
Each such pixel is a layered record: the extra-samples pass shades the
pixel's own transmissive hit again (`vc_hit_color`, to learn what it
passes, which the shading pass had computed and dropped) and then each
layer behind it. That is about 1.4 ms at 720p, 5.8 ns a layered pixel.
In development at 640×360 the frame's fixed costs weigh more and the ratio
was 1.21× (disclosed in DESIGN §3.4; not run at full size before the
gate).

## Next

A new gate (M16 G2) with the shading pass keeping what a transmissive hit
passes (8 bits a channel in the colour's unused fourth word), so the
extra-samples pass shades only the layers behind.
