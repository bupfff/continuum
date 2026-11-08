# M10 gate 13: MikkTSpace tangents for normal maps

**FAIL on accuracy: no measurable difference on this scene.** Speed
passes. The tangents are kept: they are what glTF specifies, and cost
nothing.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 5% of Cycles; over the pixels the tangents change, at most 0.8× the error of each triangle's own frame | **5.89%** (triangles' frames 5.88%); over 23,921 changed pixels **11.80%** against 11.75% (**1.00×**) | FAIL |
| 2. Warm frame at most 1.1× the triangles' frames | 3.17 ms against 3.16 ms (**1.00×**) | PASS |

Both are +1.80% on the mean from Cycles. Settled in 568 and 571 frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `569232f`; binary
`experiments/room3d/src/bin/m10_gate13.rs`, criteria in `dc79017`). Cycles:
one bounce, 4,096 samples per pixel, 60 s on Metal. The Cycles normal map
node uses MikkTSpace.

## Method

- **Computing them.** For each geometry with a normal map and texture
  coordinates, MikkTSpace tangents per triangle corner (`bevy_mikktspace`,
  no dependencies). The texture coordinates go in with v flipped to
  Blender's convention, as the Cycles export writes them, so the bitangent
  (handedness × normal × tangent) points toward the image's top, as glTF's
  green does.
- **Storing them.** Each corner's tangent is coded in the normals buffer's
  fourth component as an exact integer: octahedral coordinates at 11 bits
  each, plus one, its sign the handedness, 0 for none. No new binding,
  which the shading pass's 29-buffer limit would not allow. A round-trip
  test bounds the code's error at 0.15°.
- **Using them.** Shading interpolates the corners' tangents and keeps
  them orthogonal to the normal. Without codes, it keeps each triangle's
  own frame.

Scene (`Textured::with_curved_normal_maps`): M10 gate 5's normal maps, and
three low-poly spheres (16 × 8 segments) wearing the beveled tiles' map.

## Why no difference

The spheres look alike in all three images (`out/m10_gate13_ours.png`,
`_frames.png`, `_cycles.png`). Their texture coordinates follow the
triangle grid, so each triangle's own frame (its ∂p/∂u, made orthogonal to
the interpolated normal) already matches MikkTSpace's smoothed one. The
spheres' 12% from Cycles is common to both frames.

MikkTSpace matters for maps baked against it, from a high-poly model onto
a low-poly one, as glTF assets are. There, any other frame turns the baked
normals. Testing that needs a baked map and the high-poly model as the
reference: for a new gate.
