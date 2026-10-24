# Flatland gate 1: mask cones vs rays

Date: 2026-10-23. Question: does one mask cone beat rays on error per unit of cost?

**Verdict: passed in 2D.** It still has to be repeated in 3D (milestone M1),
where projecting into 8×8 bins costs more than a 2D interval.

## Setup

- Fluence from a segment area light over a 2D scene of 41 segment occluders,
  384×384 receiver points, CPU, single thread, release build, brute force (no
  acceleration structure for any method).
- Reference: exact analytic visibility (union of blocked angular intervals),
  cross-checked against 8,192 rays per point in the test suite.
- Scene built to stress correlation: two fences of stacked slats, one with
  aligned gaps and one with offset gaps.
- Reproduce with `cargo run --release -p flatland`; images land in
  `experiments/flatland/out/`.

## Results

| Method | Rel. RMSE | Max abs error | Flicker | ns/pixel | Cost vs 64 jittered rays |
|---|---|---|---|---|---|
| Cone setup only (overhead in every row) | — | — | — | 19 | 0.01× |
| Exact (2D-only reference) | 0.00% | 0.0000 | 0 | 251 | 0.08× |
| **Mask cone, 64 bins** | **0.69%** | **0.0165** | **0** | **163** | **0.05×** |
| 64 rays, fixed (one per bin) | 0.69% | 0.0165 | 0 | 3036 | 1.02× |
| 4 rays, jittered | 14.00% | 0.4004 | 19.72% | 228 | 0.08× |
| 16 rays, jittered | 3.83% | 0.1019 | 5.42% | 798 | 0.27× |
| 64 rays, jittered | 0.97% | 0.0256 | 1.38% | 2989 | 1.00× |
| 256 rays, jittered | 0.24% | 0.0067 | 0.34% | 11959 | 4.00× |
| Transmittance cone, per occluder | 7.23% | 0.1351 | 0 | 127 | 0.04× |
| Transmittance cone, per layer (voxel-cone-tracing style) | 6.01% | 0.1257 | 0 | 131 | 0.04× |

Correlation error below each fence (relative to the reference mean there):

| Method | Aligned: bias | Aligned: RMSE | Offset: bias | Offset: RMSE |
|---|---|---|---|---|
| Mask cone, 64 bins | +0.0% | 2.6% | +0.0% | 2.8% |
| Transmittance, per occluder | +9.9% | 28.2% | +14.7% | 22.6% |
| Transmittance, per layer | −4.8% | 21.6% | −0.1% | 14.1% |

## What this shows

1. **Same answer as 64 rays at about 1/18 the cost.** The mask cone matched 64
   fixed rays on all but 2 of 147,456 pixels (float ties at bin edges), because
   it tests each occluder once instead of once per ray.
2. **Better than jittered rays at a fraction of the cost.** Lower error than 64
   jittered rays (0.69% vs 0.97%) with zero flicker, and cheaper than 4 jittered
   rays while being 20× more accurate.
3. **Correlation is the real problem with cheap cones.** Multiplied transmittance,
   the voxel-cone-tracing approach, is biased: per-layer coverage over-darkens
   where layers hide the same directions (−4.8%) and its per-pixel error reaches
   14–22% RMSE under the fences (the red and blue streaks in the error map).
   Masks have no bias.

## Caveats

- 2D only. In 3D the per-occluder projection is more expensive, so the speedup
  will shrink; how much is M1's question.
- No acceleration structure. A BVH speeds up both; masks still traverse once per
  cone rather than once per ray, but the ratio must be re-measured.
- Mask quantization: 1/64 steps across a penumbra appear as faint, structured
  banding in the error map. It is deterministic (no flicker), but smooth
  penumbrae at high contrast could show it. Candidate fix: analytic coverage for
  the two partially covered edge bins.
- The error maps use a square-root scale, which exaggerates small errors.
