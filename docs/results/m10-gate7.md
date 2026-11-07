# M10 gate 7: alpha cutouts

**PASS.** This is gate 6 re-run after a cache fix; gate 6 reached no
criterion (`m10-gate6.md`).

| Criterion | Result | Verdict |
|---|---|---|
| 1. Accuracy against Cycles masking the cards | **3.29%** (at most 5%); with the cards opaque 34.07% (ours 0.10×, at most 0.5×) | PASS |
| 2. Speed | warm frame **3.34 ms** against 2.89 ms with the cards opaque (**1.15×**, at most 1.5×) | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `993089d`; binary
`experiments/room3d/src/bin/m10_gate7.rs`; the criteria are gate 6's,
pre-registered in `43b4844`). Development runs of the cut scene and the
opaque one came before it (settling only, no Cycles).

## Method

The cut becomes geometry. Camera rays, light and sky cones, the cache and
bounce light are unchanged, and each sees the cut exactly.

- `InstancedScene::geometry_alpha_cutoff` sets, per geometry, a cutoff for
  its base color texture's alpha.
- Before upload, `cut_out` replaces each masked triangle with the opaque
  part of its texture:
  - marching squares on texel centers, crossings interpolated linearly;
  - at the mip where the triangle spans at most 128 texels;
  - clipped to the triangle in texture space, then mapped back through its
    barycentric coordinates;
  - runs of fully opaque cells merged into rectangles.
- The reference is independent. Cycles keeps the original cards and masks
  them itself, with a Transparent BSDF where the texture's bilinear alpha
  is below the cutoff.

## Scene

M6's textured scene (sun and sky), its camera, 1280×720, bounce light on.
Added (`Textured::with_foliage`), alpha-masked at 0.5 with generated
textures:

- three shrubs of three crossed double-sided leaf cards each;
- a lattice fence panel.

The back faces sit 1 mm behind the front ones (see gate 6).

## Results

| | Cut | Opaque cards |
|---|---|---|
| Against Cycles | **3.29%** | 34.07% |
| Warm frame | 3.34 ms | 2.89 ms |
| Frames to settle | 688 | 513 |

- The cut turned 16 card triangles into 33,878 in 19 ms of CPU time.
- Cycles took 106 s on Metal.
- The error is close to the flat scene's against Cycles with one bounce
  (1.79%, M10 gate 1), plus leaf edges at 4×4 supersampling.

## Caveats

- The contour is made at a mip where a triangle spans at most 128 texels.
  A large card with a fine texture loses detail below that.
- Triangle counts grow with the contours' length: here about 2,100 per
  card.
- Blending (glTF BLEND) is not supported: glTF import treats it as MASK at
  0.5.
- Exactly back-to-back faces slow the cache's bakes (gate 6); double-sided
  geometry needs its back faces offset.
- One scene; Apple M4 only.

## Prior art

Converting alpha masks to geometry is common in offline and game
pipelines (for example, cutout meshes generated from sprites to cut
overdraw). Marching squares is standard. This gate makes no novelty claim.
