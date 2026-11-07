# M10 gate 5: normal maps

**PASS.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Accuracy against Cycles with the normal maps | **3.63%** (at most 4%); ignoring the maps 26.42% (ours 0.14×, at most 0.6×) | PASS |
| 2. Speed | warm frame **2.90 ms** against 2.78 ms without maps (**1.04×**, at most 1.15×) | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `45b48b5`; binary
`experiments/room3d/src/bin/m10_gate5.rs`, criteria pre-registered in
`75baa38`). One development run, which saved the settled image without
Cycles, came before it.

## Method

- `InstancedScene::geometry_normal_maps` gives each geometry an optional
  index into the scene's textures: a tangent-space normal map, linear, in
  OpenGL's convention as glTF specifies (green toward the image's top).
  Normal maps go into their own linear layered texture, with mip levels.
- **The tangent frame per triangle** comes from its texture coordinates'
  derivatives: the tangent along increasing u, the bitangent toward
  decreasing v. Each is orthogonalized against the interpolated normal.
  This is exact for planar mappings; MikkTSpace differs on curved meshes.
- **The mapped normal** replaces the interpolated one in the cached
  shading for light from lights, specular and reflections. Light from
  lights is the cached shadowed share times the unshadowed light at the
  mapped normal. The cached sky and bounce light keep the interpolated
  normal.
- The map is read at the mip matching the pixel's footprint (isotropic;
  the base color's box filter is not used for normals).

## Scene

M6's textured scene (sun and sky), its camera, 1280×720, bounce light on,
Lambertian. The normal maps (`Textured::with_normal_maps`) match the color
textures: the ground's stone tiles and the block's bricks are beveled
toward their mortar (bevels 12% of a tile, sloped 30°).

## Results

| Image | Against Cycles with the maps |
|---|---|
| Ours with the maps | **3.63%** |
| Ours ignoring the maps | 26.42% |
| Cycles without the maps | 26.36% |

- The normal maps change the image by about 26%, and ours reproduces all
  but 3.6% of it.
- M10 gate 1's comparable error with flat surfaces was 1.79%. The added
  error is mostly the sky and bounce light, which ignore the maps.
- Settling took the same number of frames with and without the maps (602
  and 603). Cycles took 60 s on Metal.

## Caveats

- Planar mappings only (flat ground, a box). On curved meshes, our
  per-triangle tangents and Cycles' MikkTSpace tangents differ, which this
  gate does not measure.
- The sky and bounce light ignore the maps; a scene lit mainly by the sky
  would show more error.
- Only in the cached path; the per-pixel path ignores normal maps.
- Apple M4 only.

## Prior art

Tangent-space normal mapping is standard (Blinn 1978 for bump mapping;
glTF's convention). This gate makes no novelty claim.
