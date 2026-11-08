# M10 gate 20: texture wrap modes (glTF samplers)

**PASS.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Median over five interleaved rounds of the warm-frame ratio, wrapped to repeating, at most 1.1× | **0.996×** | PASS |
| 2. Within 5% of Cycles, and at most 0.5× the error of the repeating scene | **2.18%**; repeating 67.44% (**0.03×**) | PASS |

| Round | Repeating | Wrapped | Ratio |
|---|---|---|---|
| 0 | 3.11 ms | 3.12 ms | 1.003× |
| 1 | 3.10 ms | 3.08 ms | 0.994× |
| 2 | 3.10 ms | 3.09 ms | 0.996× |
| 3 | 3.06 ms | 3.10 ms | 1.010× |
| 4 | 3.10 ms | 3.08 ms | 0.994× |

Means: ours +0.36%, repeating +40.99% from Cycles'. Both views settled in
606 frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criteria committed as `a719815`; the
binary `experiments/room3d/src/bin/m10_gate20.rs` is committed with this
result). Cycles: one bounce, 4,096 samples per pixel, 54 s on Metal; the
ground's image texture EXTEND, the lichen's and bricks' MIRROR. Spotlight
indexing was busy shortly before the run (44% of a core), idle at its
start.

## Method

`InstancedScene::texture_wrap`: per texture and axis, repeat, clamp or
mirror (`continuum_gpu::Wrap`). Each object's modes for its four texture
slots ride in a word per object appended to the object-texture buffer.
Every texture read applies them before sampling with the repeating
sampler:

- **Clamp** keeps the coordinates half a texel (at the coarser of the two
  mip levels read) inside the edges, so filtering never reaches the far
  edge.
- **Mirror** folds them into [0, 1] every two tiles, then clamps the same
  way: the edge texel repeats at each fold, as glTF's mirrored repeat does.
- **Level of detail.** Past a clamped edge the texture does not vary along
  that axis, so the box filter drops the footprint's extent along it, and
  the other reads use the base level where both axes lie past clamped
  edges.

Alpha cutouts index texels with the same modes. glTF import reads each
texture's sampler (the first sampler that reads an image wins); save
writes a sampler per distinct pair.

**Disclosed:** a development render (`M10_DEV=1`, no Cycles) before the
gate showed the first version choosing the level of detail from the
unclamped footprint, so distant stretched ground read coarse mips (lighter,
with bands where the level changed). The level-of-detail rule above was
added before the gate's only run.

## Limits

- With one axis past a clamped edge and the other varying, the level comes
  from the varying axis, so the clamped axis averages 2^level rows of the
  edge (isotropic mips cannot do better). Cycles holds the edge row.
- Cycles has one extension mode per image; the gate uses the same mode on
  both axes. Mixed modes are implemented but checked only through the
  shared code.
