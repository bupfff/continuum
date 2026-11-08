# M10 gate 17: vertex colors

**FAIL on speed** (accuracy passed).

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 5% of Cycles, and at most 0.5× the error with the colors ignored | **1.29%**; ignored 57.63% (**0.02×**) | PASS |
| 2. Warm frame at most 1.1× the colors ignored | 3.49 ms against 3.05 ms (**1.15×**) | FAIL |

Means: ours +0.08%, ignored +16.76% from Cycles'. Settled in 604 and 604
frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criterion committed as `4097b70`; the
binary `experiments/room3d/src/bin/m10_gate17.rs` is committed with this
result). Cycles: one bounce, 4,096 samples per pixel, 51 s on Metal. No
stray processes before the run, but it started straight after a workspace
build and clippy run.

Method: `InstancedScene::geometry_colors`, a linear RGB color per
triangle corner, carried in the triangle buffer's unused w as an exact
integer (1 plus an sRGB byte per channel; 0: none). Shading decodes the
three corners, interpolates them, and multiplies the base color (and its
texture). No new binding. glTF import reads `COLOR_0`; save writes it.

Checks beside the gate (`crates/continuum-gltf/tests/gpu.rs`): a constant
vertex color renders as the instances' albedo times it, to 1.4e-7 relative
RMSE (textured and untextured, bounce light on); colors survive a glTF
round trip.

## The speed miss (development runs after the gate)

`vcolor_dev` (a development tool, not a gate) measured the same two
renderers again, same scene and size, shortly after the gate:

| Order | Plain | Colored | Ratio |
|---|---|---|---|
| Alternated, round 0 | 3.02 ms | 3.05 ms | 1.009× |
| Alternated, round 1 | 3.02 ms | 3.01 ms | 0.998× |
| The gate's order (plain settled and timed, then colored), run 1 | 3.03 ms | 3.04 ms | 1.004× |
| The gate's order, run 2 | 3.03 ms | 3.05 ms | 1.004× |

The gate's colored warm frame (3.49 ms) was not reproduced: a transient
during its 30 frames, most likely background work after the build. The
kernel's added work (three corner loads, decodes and an interpolation on
colored pixels) is small. Gate 18 re-measures speed with interleaved
rounds; these numbers were seen before it was written.
