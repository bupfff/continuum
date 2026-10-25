# M6 gate 1: glTF import, automatic charts, base color textures

Date: 2026-10-25. Until now every scene was built in code. This gate imports
glTF 2.0 files: their geometry gets automatic charts for the view cache,
and their base color textures are sampled per pixel.

## What was built

- **`continuum-gltf`** loads `.gltf` and `.glb` files as an
  `InstancedScene`.
  - **Geometry.** Each primitive becomes a geometry, shared by primitives
    that read the same accessors and texture.
  - **Instances.** Each node with a mesh becomes an instance, with its world
    matrix split into rotation, uniform scale and translation.
    - A matrix with non-uniform scale, shear or a mirror is baked into a
      copy of the geometry instead (corners swapped for mirrors).
  - **Axes.** Y-up becomes Z-up.
  - **Materials.** Base color factor and texture are read; the camera is
    the file's first perspective camera, if any.
  - **Export.** `save` writes scenes back as `.glb`, for tests and for
    comparisons in other renderers.
- **Automatic charts** (`Surface::from_mesh`) for any triangle mesh.
  - Charts grow from a seed triangle over neighbours within 45° of its
    normal and are projected onto the seed's plane, so no triangle flips.
  - Each chart gets a grid of about two triangles per cell.
  - The view cache's chart cells now hold any number of triangles (cell
    lists), not one or two.
- **Base color textures.**
  - Textures are resampled into 1,024² layers of one texture array, with
    mip levels in linear light.
  - They are sampled per pixel at the level matching the pixel's footprint,
    in per-pixel shading and in the cache's shading.
  - The cache stores light, not color, so texture detail stays at pixel
    resolution however coarse the cache's texels are.
- **Cycles references take textures**: OBJ texture coordinates and PNG
  images, multiplied by each instance's albedo.
- **A textured test scene** (`room3d::textured`): a 60 m tiled ground,
  30 boulders with banded lichen, a brick block, and three generated 1,024²
  textures, lit by a sun 1 km away and a sky.
- **Also fixed:** the lightmap renderer had been broken since 0c6900e (it
  bound per-pixel-only buffers its layout lacked).

## Pass criterion

Fixed before the first run, in 76bc38f. Prefilter 1 bin, sky prefilter ×8,
cache defaults, hardware rays except where stated.

1. **Round trip.** The imported scene per pixel (640×360, one sample per
   pixel, compute rays) within 0.01% relative RMSE of the original scene.
2. **Accuracy with textures.** The imported scene per pixel, 16×16 samples
   per pixel at 320×180, within 3% relative RMSE of instanced Cycles at
   4,096 samples per pixel with the same textures.
3. **Automatic charts serve the cache.** At 1280×720, the warm cached image
   of the imported scene (automatic charts) against its per-pixel image is
   at most 1 point above the original scene's (parametric and box charts).
4. **Speed.** The imported scene's warm frame at 1280×720 at most 16.7 ms.

Real assets given on the command line, not in the repository, were reported
only.

**Verdict: FAIL** (criterion 1; criteria 2–4 pass).

## Results

Apple M4 (MacBook Air). Reproduce with
`cargo run --release -p room3d --bin m6_gate1 -- [asset.glb ...]`.

- **The round trip itself:** export 11 ms, import 30 ms, 2,893,496 bytes.
  The import has 3 geometries and 32 instances, with 188 charts after
  automatic charting.

| Criterion | Measured | Threshold | |
|---|---|---|---|
| 1. Round trip, per pixel | **0.02354%** | 0.01% | fail |
| 2. vs Cycles with textures | **0.43%** (ours +0.02% brighter; Cycles 4.8 s) | 3% | pass |
| 3. Cached vs per-pixel | imported **2.37%** (warm after 496 frames), original 1.52% (after 489) | original + 1 point | pass |
| 4. Imported warm frame | **1.53 ms** (original 1.57 ms) | 16.7 ms | pass |

**Disclosed before the gate.** One development run of the same comparison
as criterion 2, at smaller settings (`textured_dev`: 320×180, 4×4 samples
per pixel, Cycles at 256 samples), gave 3.01%.

**Machine state.** No control was run. The DisplayLink agent used about 50%
of a core throughout. Criterion 4's margin (11×) is far larger than the
machine's heat effects.

Real assets, reported only:

| Asset | Import | Triangles | Charts | Textures | Baked transforms | Cached vs per-pixel | Warm frame | Frames until warm |
|---|---|---|---|---|---|---|---|---|
| Suzanne (Blender monkey) | 1 ms | 968 | 147 | 0 | 0 | 3.28% | 1.28 ms | 133 |
| a 10-pack of suppressor models | 323 ms | 28,064 | 1,203 | 10 | 3 | 4.99% | 1.43 ms | 1,317 |

## Why criterion 1 failed

A diagnostic run after the gate (`m6_roundtrip_check`) swapped one imported
part at a time into the original scene, rendered exactly as criterion 1
renders.

| Imported part swapped in | vs original |
|---|---|
| instance order (the importer reverses it) | 0 (bit-identical) |
| textures (equal byte for byte) | 0 (bit-identical) |
| geometries (normals renormalized; positions and texture coordinates equal) | 0.00013% |
| instance transforms | **0.02354%** |
| everything (the gate's import) | 0.02354% |

- **Which values differ.** Imported rotation matrices differ from the
  originals by at most 6.0e-8 per entry (one f32 step) and scales by 1.2e-7
  relative.
  - The file stores rotation × scale as f32 values, and the importer
    divides by the columns' lengths to recover them, so the originals
    cannot come back bit for bit.
- **Where the error is.** Almost all of the squared error is in one pixel
  at (382, 198): 0.045 imported against 0.143 originally. There a camera
  ray at the edge of a boulder hits the boulder on one side and the ground
  on the other.
  - The next-largest pixel differences are about 0.004.
  - 10,474 pixels differ at all, by rounding.

So the importer reproduces the scene up to f32 rounding of its transforms.
At one sample per pixel, one edge pixel whose ray flips between two
surfaces exceeds the 0.01% bound. The criterion was too tight for
transforms stored as rounded matrices, but the gate stands as a failure.

A later round-trip gate should fix in advance a criterion that tolerates
f32 rounding of transforms. Two options:

- compare supersampled images, where a flipped sample weighs 1/n² of its
  pixel;
- or bound the number of pixels that change by more than a tolerance.

## What this means

- **Real files work.**
  - Imported glTF scenes render per pixel and through the cache.
  - The imported textured scene is within 0.43% of Cycles with textures.
  - Automatic charts serve the cache within a point of hand-made charts
    (2.37% against 1.52%).
- **Textures cost nothing in the cache.** The cache stores light and
  shading samples color per pixel, so texture resolution does not raise
  bake work.
  - The textured scene warms in 496 frames and runs warm at 1.5 ms.
- **The real assets warm slowly but run fast.** The suppressor pack (1,203
  charts, many small parts close together) takes 1,317 frames to warm and
  is 4.99% from per-pixel; warm frames are 1.3–1.4 ms.
- **Not yet supported:**
  - metallic and roughness (all surfaces are diffuse);
  - normal maps, alpha and emission;
  - skins and morph targets;
  - lights from the file (they are counted as skipped);
  - images other than 8-bit.
