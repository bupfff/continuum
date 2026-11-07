# M10 gate 6: alpha cutouts

**FAIL: the run never reached a criterion.** Its first step, settling the
control (the same scene with the cards opaque), did not settle within
50,000 frames, and the binary stopped. The fix is in a new gate (M10 gate
7, same criteria and scene).

Apple M4, 1280×720, run once at full size on 2026-11-07 (commit `b550a5d`;
binary `experiments/room3d/src/bin/m10_gate6.rs`, criteria pre-registered
in `43b4844`). The only line it reported: the cut turned 16 card triangles
into 33,878 in 19 ms of CPU time.

## Cause

A cache bug, not the method. With the cards opaque, about 10 pages were
requested in every frame and never baked, for example m 33, chart 8, mip 1,
page (5, 21), whose chart has a grid of 144 × 168 texels on the CPU.

- The CPU and the shader compute a chart's texel grid in different
  orders. For this shrub (1.4 m cards at scale 1.2, 100 texels per metre),
  the CPU gets 168 and the shader's f32 gets 169.
- The shader asked for page 21, which starts at texel 168. It was inside
  its own grid, but past the CPU's, so the CPU refused it every frame.

The plan now allows one texel past its own grid (commit after this one),
and the opaque scene settles in a development check. Scenes whose charts
times their instances' scale happen to land on an exact multiple of the
page size could hit this anywhere; earlier gates did not.

## Also found before the run (in development, disclosed in the binary)

Cards with exactly back-to-back faces (both windings coplanar) made the
cache's bakes about 100× slower, opaque or not. With the back faces 1 mm
behind, bakes are normal. The cause is not established. glTF import of
double-sided materials uses the same 1 mm offset.
