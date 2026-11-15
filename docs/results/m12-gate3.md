# M12 gate 3: glTF skins and morph targets, posed as Blender poses them

**PASS** on both criteria.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Posed as Blender poses it: eight times, every corner within 0.2 mm of Blender's own evaluation, mean below 0.05 mm | largest **0.0004 mm**, largest mean **0.0001 mm** over 4,788 corners (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.9 s) | PASS |
| 2. Played through `Renderer` (three loops at 24 fps, held at 1.33 s): within 1% of a renderer built with that pose, the pose showing (at least 5% from the rest pose) | **0.0000%** from the built pose; the rest pose 68.08% from it | PASS |

Reported: the file imported in 1.2 ms; a pose of 4,788 corners took
0.07 ms on the CPU.

Apple M4 (fanless MacBook Air), release build, run once on 2026-11-15.
The run's lines are in `docs/results/m12-gate3-run.txt`.

## What was built

- **Import** (`continuum-gltf`):
  - **Per primitive:** each vertex's joints and weights (`JOINTS_0`,
    `WEIGHTS_0`) and morph targets' position and normal deltas, kept per
    triangle corner in the geometry's order. Double-sided back faces
    reverse their corners and flip their normals.
  - **Skins:** joints and inverse bind matrices.
  - **Animation channels:** scale and morph weights as well as translation
    and rotation.
- **Placement:**
  - A node with a skin, or a mesh with morph targets, gets a geometry of
    its own built from its rest pose (default weights, joints at rest).
  - A skinned mesh sits at the world origin, since glTF ignores the
    skinned node's own transform; a morphing mesh keeps its node's
    transform.
  - Triangles degenerate at rest are left out of the geometry, and of the
    deformer's corners alike.
- **Posing:** `Animations::deform(animations, time)` applies the morph
  deltas by the weights, then linear blend skinning by the joints' world
  matrices times their inverse bind matrices, and converts to Z-up.
  Several animations play together: Blender exports a rig and a mesh's
  shape keys as two. `Imported::deforming` lists the geometries for
  `RendererOptions::deforming`.
- **Renderer:**
  - `Renderer::is_settled` also waits while deforming geometry's lighting
    lags its pose or movers' pages re-bake.
  - `Renderer::deforming_lag` reports the lag.
  - A scene whose every instance moves or deforms gets a static anchor,
    a triangle a tenth of a millimetre across under it: the movers'
    structures need a static object.
- **Viewer:** a glTF file's skinned and morphing meshes play from all its
  animations together, looping (P pauses).
- **Tests:**
  - `tests/skins.rs`: the import's shape on the committed test file.
  - GPU test `skinned_gltf_alone_plays`: a file whose only mesh is
    skinned renders and settles lit.

## The test file

`experiments/room3d/cycles/make_skinned_test.py` makes it in Blender 5.2
(no download), committed as `crates/continuum-gltf/tests/data/
skinned_tube.glb` (70 KB):
- a tube 2 m tall, 32 segments by 24 rings, skinned to three bones with
  automatic weights, with a shape key;
- a 48-frame animation at 24 fps that bends the middle bone 60° and back,
  twists the top bone, and drives the shape key 0 → 1 → 0.

`compare_skinned_pose.py` imports the same file in Blender, evaluates it
at a time and measures each of Continuum's corners against Blender's
evaluated vertices.

## Disclosed

- **Development fixed the reference script, not the importer.**
  - Blender's importer puts glTF time t at frame t × fps (its exporter
    wrote frame f at f / fps), not 1 + t × fps: the script was a frame off,
    75–95 mm where the bend changes fastest.
  - The importer leaves the shape keys' animation as a muted NLA track,
    which the script now makes active with its action slot (Blender 4.4
    on).

  The rest pose matched to 1.2e-7 m from the start.
- **`Renderer::is_settled`** read settled before the deforming lighting
  had caught up: 10% from the built pose in development, 0.0000% after.
- **One rig, one file, Blender's exporter.** Other exporters (cubic
  splines, several skins, sparse accessors, more than four influences)
  are untested. Tangents for normal maps stay the rest pose's.
