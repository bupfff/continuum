# M10 gate 8: exactly back-to-back faces

**FAIL on accuracy.** Speed passed: exactly back-to-back faces no longer
slow the cache.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Speed | GPU bake time to settle **3,865 ms** against 3,742 ms with 1 mm (**1.03×**, at most 1.5×); warm frame **3.39 ms** against 2.98 ms (**1.14×**, at most 1.25×); settled in 718 and 671 frames (5.8 s and 5.5 s) | PASS |
| 2. Accuracy | **5.01%** from Cycles (at most 5%; the control 3.28%); **3.78%** from the control (at most 0.5%) | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `e876b61`; binary
`experiments/room3d/src/bin/m10_gate8.rs`, criteria pre-registered in
`41bd563`). The Cycles reference is M10 gate 7's (the control's geometry,
cached in `out/m10_gate7_cycles`).

## The slow bakes (fixed)

Found in development (`backface_dev`, before the gate):

- Only bounce light was slow: 1,793 µs a baked texel against 2.67 µs with
  the back faces 1 mm behind (opaque cards; sun and sky bakes were normal).
- A mesh without normals gets smooth normals, summed over every triangle at
  a vertex's position. A front face and its exactly coincident back face
  cancel, so the shading normal is zero, and the shader's `normalize` makes
  it NaN.
- Bounce rays from those texels had NaN directions. A NaN ray passes every
  box's slab test, so each one visited the whole scene.

Fix (commit `e876b61`):

- smooth normals sum only the triangles at a position that face the same
  side as the corner's own (a positive dot of face normals);
- the shader takes the face normal where an interpolated normal is zero or
  NaN;
- bounce rays: of two hits on one mesh at the same distance within rounding,
  the one facing the ray wins (a back face counts as inside geometry);
- glTF double-sided materials put the back faces on the same plane (no 1 mm
  offset).

## Why accuracy failed

The leaves lost most card-to-card sun shadows, which showed as speckles and
stripes (`out/m10_gate8_ours.png` against `out/m10_gate8_control.png`).
Development runs after the gate isolated the cause:

- Per-pixel rendering (no cache) keeps the shadows. Software camera rays
  with the cache keep them too (mean difference 0.09 of 255 against the
  1 mm control, against 2.36 with hardware rays).
- **Metal's hardware ray queries ignore `CULL_BACK_FACING`.** Naga 30's
  Metal backend sets an opacity cull mode but no triangle cull mode (its
  source has a TODO on winding). Camera rays can therefore hit back faces,
  and of a card's two coincident faces either one.
- The cache then reads the back face's texels for a pixel on the front.
  The back face is turned away from the sun, so its stored light ratio is 1
  (unshadowed), and the pixel shows lit.
- With the back faces 1 mm behind, the front face is always nearer, so the
  wrong pick never happened.

Fix, for a new gate (M10 gate 9): a camera ray whose hardware hit is a
back face is traced again with its candidates tested for facing (the
nearest front face, as the software `camera_hit` finds). Other pixels keep
the single opaque query. This also makes the Metal path match the software
path and other backends for single-sided geometry seen from behind (now
culled, as documented). A development run matched software camera rays to
0.0006 of 255 (mean).
