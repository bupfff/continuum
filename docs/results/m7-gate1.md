# M7 gate 1: pixel cones for primary visibility

**FAIL on all three criteria.** Pixel cones as built are 48× too slow on the
textured scene and take 10–87 s per frame on the boulder field. They cut
the error of one ray per pixel only from 150 m up (0.76×). From standing
height they leave a banding artifact on the ground.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-06 (commit `9334185`; binary
`experiments/room3d/src/bin/m7_gate1.rs`, criteria pre-registered in
`a16961a`).

## Method

Each pixel traces a mask cone through its square: 8×8 bins of equal
image-plane area, front to back through the object and triangle
hierarchies, prefiltered at one bin as light cones are. Each object the cone
meets becomes a fragment, holding three things:

- its mean coverage;
- a mask of the bins it covers at least half of;
- its depth along a ray through its coverage.

Up to four fragments are kept. They are composited by mask, nearest first,
and each visible fragment is shaded once at a ray through its visible bins.
The sky fills what no fragment covers. Code: `mask_cones.wgsl`, section
"Pixel cones (M7)", entry points `render_cones` and `vc_shade_cones`.
`RendererOptions::pixel_cones` turns it on; it is off by default.

## Results

**Criterion 3, speed** (measured first): textured scene with the cache,
warm, median of 30 frames each waited for.

| | Median | Pipelined mean of 60 | Frames until warm |
|---|---|---|---|
| One ray (control) | 2.18 ms | 1.78 ms | 489 |
| Pixel cones | **105.17 ms** | 104.62 ms | not warm after 6,000 |

The criterion is at most 16.7 ms, so this is a **FAIL**.

The control's pipelined mean is 1.16× M6 gate 1's 1.53 ms, which was timed
the same way. The machine had run development renders in the half hour
before, so it was not fully cool. The verdict does not depend on it.

**Criterion 1, quality**: relative RMSE from the 16×16 area-integrated
reference, with per-pixel lighting.

| View | One ray | Pixel cones | Ratio | One-ray frame | Pixel-cone frame |
|---|---|---|---|---|---|
| Textured scene | 7.11% | 7.15% | 1.006× | 0.56 s | 0.77 s |
| Field, standing | 12.49% | 34.61% | 2.772× | 0.06 s | 86.8 s |
| Field, 150 m up | 23.49% | 17.91% | 0.763× | 0.26 s | 10.1 s |

The criterion is at most 0.35× in every view, so this is a **FAIL**. No pixel was
set apart: no reference sample saw a light.

**Criterion 2, stability**: turning the camera half a pixel from 150 m up
changes the images as follows.

| Image | Relative change | Against the reference |
|---|---|---|
| Reference | 21.91% | 1× |
| One ray | 45.21% | 2.06× |
| Pixel cones | 36.49% | **1.67×** |

The criterion is at most 1.5×, so this is a **FAIL**.

**Work per pixel** of the pixel cones' walk (profile stage 9):

| View | Object nodes | Meshes entered | Triangle-hierarchy nodes |
|---|---|---|---|
| Textured scene | 11 | 2.9 | 19 |
| Field, standing | 68 | 9.7 | 950 |
| Field, 150 m up | 73 | 6.2 | 364 |

## What went wrong

1. **Cost is inherent to exact 8×8 coverage per pixel.** A pixel cone must
   resolve every surface in its pixel down to an eighth of a pixel. That
   takes hundreds of hierarchy-node tests per pixel on the field, against a
   handful for a ray.
   - Horizon pixels reach far more. In a development run at 160×90, one
     pixel entered 12,656 boulders and tested 100,299 nodes.
   - Light cones avoid this because their apex sits on the surface. Near
     the apex the cone is narrow, and it is narrow where the geometry is.
     A pixel cone is widest where its geometry is.
2. **The textured scene's error is not at silhouettes.** One ray and pixel
   cones are equally far from the reference (7.11% and 7.15%). The textures
   are sampled at mip levels chosen for a whole pixel, which is blurrier
   than the area integral of finely sampled texture. Shading also varies
   within pixels, at shadow edges and in texture detail. Pixel cones change
   neither: they shade each fragment once.
3. **Banding on the field's ground from standing height** (a suspected
   cause, not yet confirmed). Whole pixel rows of the ground come out too
   dark or too bright, which is why the error rises to 34.61%.
   - The ground is two 2.1 km triangles. Clipped at the cone's near plane
     and projected into a pixel-sized window, their f32 coverage is
     inexact.
   - Light cones skip such triangles; pixel cones meet them head-on.
4. **The cache never warmed with pixel cones.** It was not warm after 6,000
   frames. Frames take 105 ms, so the bake budget sits at its floor. Each
   fragment may also need its own pages, and only the most visible one
   requests them.

## Development runs (disclosed)

These ran before the full-size run, with `m7_dev`, at small sizes, per-pixel
lighting, against 16×16 references. In the first two runs the reference was
rendered at 16 times the resolution.

| Run | One ray | Pixel cones | Pixel-cone frame |
|---|---|---|---|
| Textured scene, 320×180 | 11.66% | 11.22% | 114 ms |
| Textured scene, 320×180, cached | 11.69% | 11.26% | 36 ms (warm after 982 frames) |
| Field 150 m up, 320×180, before prefiltering | 36.89% | 21.58% | 6.2 s |
| Field 150 m up, 320×180, prefiltered | 36.89% | 21.09% | 4.6 s |

Work counts at 320×180 from 150 m up: 80 object nodes, 9.2 meshes and
1,108 triangle-hierarchy nodes per pixel.

The standing view at 160×90 took 7.7 s: 89 meshes per pixel on average,
and at most 12,656 meshes and 100,299 nodes.

Two variants changed none of the work counts, and were reverted:

- ordering children by where the cone enters their boxes;
- bounding the walk once bins were 90% covered.

## Changes before the full-size run

These are also listed in the binary's doc comment.

- **The reference.** A 20480×11520 render exceeds the 16384-pixel texture
  limit, so the reference averages 256 frames at 1280×720 instead. Their
  camera rays cross each pixel at the centers of a 16×16 grid
  (`InstancedRenderer::set_pixel_offset`). They sample textures for 1/16 of
  a pixel (`set_footprint`). This gives the same box filter and texture
  footprint as the 16× render.
- **"Half a pixel sideways"** is a turn of the camera by half a pixel's angle
  at the image's center.
- **Not reported:** fragments per pixel and the coverage left to the sky.
  Work per pixel is reported instead.
- **The field's warm frames with pixel cones are not measured**, because
  each frame takes seconds. Per-pixel frame times are reported instead.

## What next

Pixel cones stay in the code, off by default, as an experiment. The gate
points at three narrower problems, each for a new gate:

- **Texture filtering.** Mip levels are chosen per pixel, which is blurrier
  than area integration. On the textured scene this is the larger error,
  and it is cheap to address.
- **Visibility at silhouettes only.** Use one ray per pixel, and resolve
  coverage only where neighbouring pixels see different objects or depths.
  Most of the cost goes away, but sub-pixel objects that every neighbour
  misses would still flicker.
- **Precision of huge triangles in narrow cones.** Clip in 3D against the
  cone's side planes before projecting.

For v0.1, one ray per pixel stays, and aliasing remains a known limitation.
Offline renders can average frames over sub-pixel offsets, as the gate's
reference does.
