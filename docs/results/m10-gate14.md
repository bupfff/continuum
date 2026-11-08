# M10 gate 14: emissive textures

**FAIL on accuracy, mostly from the reference scene.** Speed and the
relative bound pass. Emissive textures are kept. M10 gate 15 re-runs it
with a one-sided sign.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Over surfaces: within 10% of Cycles, and at most 0.5× the error with the texture ignored | **62.86%**; ignored 432.10% (**0.15×**); 221 pixels apart | FAIL |
| 2. Warm frame at most 1.1× the texture ignored | 2.95 ms against 2.97 ms (**1.00×**) | PASS |

Means: ours −3.17%, ignored +22.89% from Cycles'. Settled in 604 and 603
frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (commit `c9af68d`; binary
`experiments/room3d/src/bin/m10_gate14.rs`, criteria in `3e93a9a`). Cycles:
one bounce, 4,096 samples per pixel, 84 s on Metal.

## Method

`InstancedScene::geometry_emissive_textures`:

- **Storage.** Per geometry, an sRGB texture in the base color array. Its
  layer goes in the materials buffer, and in the emission words bounce
  samples read.
- **Sampling.** The emission is sampled through the texture coordinates:
  at the pixel's footprint in the cached shading, and at their own
  footprints in reflections and bounce samples.
- **glTF and Cycles.** glTF import maps `emissiveTexture` (the base
  color's coordinates; not saved). The Cycles reference multiplies the
  emission by the same image.

Scene (`Textured::with_sign`): M6's textured scene at night with a
double-sided sign 5 m ahead, emission (4, 4, 4) times three colored
stripes.

## Why it failed

Diagnosis after the run (`M10_DEV=ss` in the binary: ours supersampled,
against the cached reference, in linear floats):

- **Ours matches the definition.** Across the sign, ours is 4 × the
  stripe's color plus a little reflected light. At the red stripe's centre:
  (4.02, 0.62, 0.41), the texture decoding to (1.0, 0.15, 0.1).
- **Cycles adds light that follows the sign's own color:** (4.89, 0.82,
  0.55) at the same point, +0.85 on each stripe's dominant channel, and
  +0.1 on the dark gaps. Ours over Cycles on the sign's bright pixels:
  0.828.
- **The likely cause is the reference scene.** The sign's front and back
  faces are exactly coincident (its back faces on the same plane). Shading
  the front face, Cycles light-samples the coincident back face as an
  emitter at zero distance, so the sign lights itself. M10 gate 12's glowing
  panel, a closed box, matched Cycles' mean to 0.2%.
- **A real limitation on top: the far ground is about 2× darker in ours**
  (0.009–0.014 against 0.017–0.034). Bounce samples (14° bins) undersample
  a distant emitter, as in M10 gate 12.

M10 gate 15 re-runs the criteria with a one-sided sign (no coincident
emitters).
