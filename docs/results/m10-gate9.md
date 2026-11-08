# M10 gate 9: exactly back-to-back faces (gate 8 re-run)

**FAIL on accuracy, by its comparison with the 1 mm control.** Speed
passed. Against Cycles, the exactly back-to-back scene is as close as the
control (3.42% against 3.37%). Development runs after the gate show that the
failed comparison mostly measures the control's own 1 mm offset (below).
The fixes are kept.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Speed | GPU bake time to settle **2,192 ms** against 1,814 ms with 1 mm (**1.21×**, at most 1.5×); warm frame **3.06 ms** against 2.94 ms (**1.04×**, at most 1.25×); settled in 666 and 654 frames (5.5 s and 5.3 s) | PASS |
| 2. Accuracy | **3.42%** from Cycles (at most 5%; the control 3.37%); **1.08%** from the control (at most 0.5%) | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `a2a2f0d`; binary
`experiments/room3d/src/bin/m10_gate9.rs`; criteria and scene are gate 8's,
pre-registered in `41bd563`, and the binary in `c2c3628`). The Cycles
reference is M10 gate 7's (the control's geometry).

## What changed since gate 8

Metal's hardware ray queries ignore `CULL_BACK_FACING`: naga 30's Metal
backend sets no triangle cull mode. A camera ray could therefore hit either
of a card's coincident faces, and the cache read the back face's texels for
a pixel on the front (`m10-gate8.md`).

Fix: a camera ray whose hardware hit is a back face is traced again. The
second query reports its candidates, and only front faces are accepted:
the nearest front face, as the software `camera_hit` finds. Pixels whose
first hit is a front face keep the single opaque query. Single-sided
geometry seen from behind is now culled on Metal too, as on the software
path and the other backends.

## Where the images differ

The 1.08% sits at leaf edges and on cards seen nearly edge-on
(`out/m10_gate9_ours.png`, `out/m10_gate9_control.png`). Moving a card's
back faces changes its geometry exactly there.

Development runs after the gate (`backface_dev`; settled, one sample per
pixel, the PNGs' sRGB taken back to linear, so indicative only):

| Compared | Relative RMSE |
|---|---|
| 1 mm against a second 1 mm run | 0.18% |
| 0 mm against 1 mm | 1.77% |
| 2 mm against 1 mm | 1.82% |

Moving the back faces from 1 mm to 2 mm changes the image as much as moving
them from 1 mm to 0. A 0.5% bound against the 1 mm control could not tell a
fault from the control's own offset. A better criterion would compare both
scenes with Cycles, each against its own geometry. By the rules, that is for
a new gate, not a re-run of this one.

## Kept

- Smooth normals across same-side faces only; the face normal where an
  interpolated one is zero or NaN.
- Bounce rays prefer the coincident face that faces the ray.
- The hardware camera rays' back-face re-trace.
- glTF double-sided materials with the back faces on the same plane.
- The room3d foliage scene keeps 1 mm by default (`with_foliage()`, as
  gates 6–7 ran); `with_foliage_back(0.0)` builds it exactly back-to-back.
