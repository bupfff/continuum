# M10 gate 23: emissive faces as area lights

**FAIL: criterion 2 not reached.** Speed passed; then the view with the
faces as lights never settled while supersampling (textures read for a
quarter of a pixel), and the binary stopped. Kept as an opt-in,
experimental function (`InstancedScene::emitters_to_lights`; the viewer's
`--emitters-as-lights`).

| Criterion | Result | Verdict |
|---|---|---|
| 1. Median over five interleaved rounds of the warm frame, faces as lights to emission through bounce, at most 1.5× | **1.286×** | PASS |
| 2. Over surfaces: within 5% of Cycles, and at most 0.5× the error of emission through bounce | not reached: "the view never settled" (50,000 frames) at a sub-pixel offset | FAIL |

| Round | Through bounce | Faces as lights | Ratio |
|---|---|---|---|
| 0 | 2.99 ms | 3.82 ms | 1.278× |
| 1 | 2.98 ms | 3.83 ms | 1.287× |
| 2 | 2.98 ms | 3.83 ms | 1.284× |
| 3 | 2.97 ms | 3.85 ms | 1.296× |
| 4 | 2.99 ms | 3.84 ms | 1.286× |

The panel's six faces made six lights (eight in all). Settled at one sample
per pixel in 3,841 frames with the faces as lights, 609 through bounce.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criteria committed as `e9835ab`; the
binary `experiments/room3d/src/bin/m10_gate23.rs` and the method are
committed with this result). The Cycles reference was not rendered.

## Method

Each geometry with uniform emission whose triangles pair into at most a
given number of planar rectangles becomes one one-sided rectangular light
per rectangle and instance, facing out, at the emission's radiance, 1 mm in
front; its emission is set to zero. A unit test checks a box: six outward
lights covering its area, 1 mm out, placed by the instance. A development
render before the gate looked like gate 12's Cycles image.

## Why it never settled (development, after the gate)

`emitters_dev` (the gate's scene, settled, then footprint 0.25 at the
first sub-pixel offset; statistics every 250 frames):

- **With bounce light:** no eviction (about 6,990 pages resident), but one
  full page (81 texels) baked in every sampled frame, and every 500 frames
  a burst of about 350 requests and 13 pages' bounce samples (117) re-baked.
  It never went quiet in 3,000 frames: a loop between bounce samples'
  "incomplete, re-bake later" path and some page near the panel, not yet
  found.
- **With bounce light off:** it settled within about 750 frames (after an
  initial settle of 7,208 frames).
- **Slow settling either way:** 3,800–7,200 frames against 609, with the
  bake budget at its floor (1,000) throughout: bakes near the panel's
  lights are expensive. Likely the receivers right by them (the lights sit
  1 mm off the faces, and the bottom edges touch the ground), whose cones
  toward the lights are nearly hemispherical.

Next: find the page in the loop (its key and mip, and why its bounce
samples stay incomplete), and measure the bake cost of texels near the
lights.
