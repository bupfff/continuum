# M5 gate 3: object-level prefiltered coverage, against Cycles at scale

Date: 2026-10-24. M5 gate 2 matched Cycles to 0.7–1.2% with sky light on a
63-object scene. On the million-boulder field, a sky cone from the ground
looks across hundreds of thousands of boulders, and walking the object
hierarchy down to each one is unaffordable. Triangle hierarchies already
prefilter (M4 gate 1); this gate extends that to objects.

## What was built

- **Directional coverage for object nodes.** Each object-hierarchy node
  stores coverage for the 26 cube directions: the summed projected front area
  of its objects. Each object's own comes from its geometry's root coverage,
  seen along the direction rotated into the object and scaled by its scale
  squared (`hierarchy::object_coverage`).
  - It follows the triangle hierarchies' coverage in the same buffer, with its
    offset in the light uniform, so no new binding is needed.
  - Instanced scenes with prefiltering get it by default
    (`set_object_prefilter`).
- **Splatting object nodes.** A cone meeting a node that lies wholly between
  its near and far planes, and projects to at most the prefilter threshold in
  bins, splats the node's hull instead of descending.
  - Opacity is 1 − exp(−front area / silhouette area): the objects act as
    independent occluders, so overlap saturates instead of exceeding one.
  - It combines with other meshes like any mesh: max for sun cones,
    independent occluders for sky cones.
- **Instanced Cycles references** (`cycles::render_instanced`,
  `render_instanced_reference.py`). Each geometry is written once and
  instanced on a point cloud through geometry nodes, with per-instance
  rotation, scale and albedo. Cycles renders a million instances as
  instances. On the outdoor scene it is within 0.12% of the flattened
  reference, which is noise at 1,024 samples per pixel.
- **The field** gets `with_sun_at` and `sky()`, as the outdoor scene.

Prior art: prefiltered directional opacity for aggregate geometry is Lacewell
et al. 2008 (per-node, ray-traced). The independent-occluder opacity is the
standard model for randomly placed occluders. Splatting object nodes into
mask-cone bins is specific to Continuum and not claimed as novel.

## Pass criterion

Fixed before the first run, in b9566e2. The million-boulder field from its
standing viewpoint, 640×360, sun at 1 km, the outdoor scene's sky rule,
prefilter 1 bin, compute rays. Cycles: instanced, 1,024 samples per pixel,
direct light only, one-pixel box.

1. Sky only, 4×4 per pixel, within 3% of Cycles.
2. Sun and sky, the same.
3. The sky-only render at 1 sample per pixel at most a tenth of the time
   without object prefiltering. If the render without it exceeds 10 minutes,
   it is stopped and counted as more than 10×.
4. Gate 8's four sun-only viewpoints change by at most 0.5%.

**Verdict: PASS.**

| Criterion | Result |
|---|---|
| 1. Sky only vs Cycles | **1.15%** |
| 2. Sun and sky vs Cycles | **2.39%** |
| 3. Sky only, 1 per pixel, with object prefiltering | 23.7 s |
| 3. Without | stopped after 10 minutes: **more than 25×** |
| 4. Sun only, with vs without, worst of four viewpoints | **0.000%** |

- **Our times** at 4×4 per pixel (2560×1440 rendered): sky only 224 s, sun
  and sky 236 s.
- **Cycles:** 21.6 s per render on Metal, own noise 0.23% (sky only) and
  0.29% (sun and sky).
- **Not measured:** node splats per cone, which was pre-registered as
  reported. The kernel has no counter for it yet.
- **Reproduce** with `cargo run --release -p room3d --bin m5_gate3`.

## What this means

- **Sky light over a million objects matches Cycles to 1–2.4%** at 4×4 per
  pixel, with distant groups of boulders splatted as single prefiltered
  occluders.
- **Without object-level prefiltering, sky cones over this field are
  unaffordable:** more than 10 minutes for one 640×360 frame. With it, 23.7
  s. That is still 100 µs per pixel, so sky cones per pixel are not a
  real-time path. They belong in the lighting cache, where each texel is
  baked once. That is the next gate.
- **Sun cones never meet an object node small enough to splat on this
  field.** The sun's bins are 0.06°, so the sun-only images are bit-identical.
- **Sun and sky (2.39%) is less precise than sky alone (1.15%).** The sun
  part was not separated in this gate.
