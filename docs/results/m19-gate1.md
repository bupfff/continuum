# M19 gate 1: bloom and upscaling

**PASS** on all three criteria.

| Criterion | Result | Bound | |
|---|---|---|---|
| 1. Bloom's image against the same computed on the CPU from `read_colors` | **4.72e-8** (relative RMSE); total light **1.00000** of the unbloomed image's | at most 1e-3; within 1% | PASS |
| 2. Bloom's cost on the native frame | **+0.262 ms** (3.206–3.218 against 2.944–3.001 ms, five rounds) | at most 0.3 ms | PASS |
| 3. Upscaling from 640×360: Catmull-Rom's error against the native render | **10.65%**, nearest's 12.25% (**0.870×**); bilinear 11.19% | at most 0.9× nearest's | PASS |
| 3. Catmull-Rom's cost | **+0.291 ms** over nearest (1.493–1.502 against 1.200–1.208 ms); its frame **0.506×** native | at most 0.35 ms; at most 0.55× | PASS |

Apple M4 (fanless MacBook Air), 1280×720 `Rgba32Float` target, release
build, run once at full size on 2026-11-16 (commit `bd76a05`; binary
`experiments/room3d/src/bin/m19_gate1.rs`, stub committed in `6beaa79`;
output `docs/results/m19-gate1-run.txt`), after three minutes with no game
or indexing running and one more minute idle. The timings were taken
first, as medians of 40 warm frames in five interleaved rounds (native,
native with bloom, scale 2 with nearest, with Catmull-Rom, with bilinear).

## Method

DESIGN §3.4, "Bloom and upscaling". The textured scene with foliage, no
auto-exposure, exposure 1.
- **Bloom** (`Bloom::default()`: strength 0.04, radius 2% of the image's
  height): the image at a quarter of its size (4 × 4 means), a Gaussian
  across and down renormalized at the edges, mixed into the display in
  linear light before exposure as (1 − strength) of each pixel plus
  strength of the blur (bilinear). Criterion 1 compares the drawn float
  target with the same arithmetic done in f64 on the CPU from the
  renderer's linear colours, and the target's total light with the
  unbloomed target's.
- **Upscaling** (`Upscale`): rendered at scale 2 (640×360) and drawn to
  the 1280×720 target by the nearest pixel, bilinearly, or by Catmull and
  Rom's bicubic clamped to its four nearest pixels' range; each against
  the scene rendered at 1280×720 (scale 1, nearest).

## Disclosures

- The bars were set with development numbers known (`post_dev`: 0.87×,
  +0.29 ms, 0.50× native; bloom +0.25 ms), and the gate's development
  run (`M19_DEV=1`, 10 frames a round) gave the same images and timings
  within 0.02 ms.
- Upscaling recovers only what interpolation can: no temporal
  reconstruction (§4), so detail missing at 640×360 stays missing; the
  native render is the reference, not Cycles.
- Bloom is an artistic effect with no physical reference here; the gate
  checks that it is computed as designed and keeps the image's light,
  not that it matches a lens.
