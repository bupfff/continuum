# M4 gate 7: cones that pay only for what they see

Date: 2026-10-24. M4 gate 6's corrected diagnostic found the boulder field's
frame dominated by per-pixel sun cones, at a cost nearly independent of what a
cone contains. The hypothesis was that the bin weights were to blame: every
cone summed 64 bin weights (4 edge integrals each, with `atan2`) even when
nothing covered it. Two changes:

- **Weights.** A cone's visible light becomes the whole window's weight (one
  4-edge integral) minus the weights of the bins with coverage. That is the
  same quantity rearranged, used only when the whole window lies above the
  receiver's horizon, where the bins' weights sum to the window's.
- **Node test.** A node's box corners are projected once for both the cull
  test and the splat test, not twice.

Pass criterion, fixed before the first run (committed in c8e0b7e before
implementing):

1. **Same output.** The showcase's cached and per-pixel renders (1920×1080)
   and the 1,000,000-boulder field (1280×720, from 150 m up) are within 1e-4
   of peak of the pre-change kernel at every pixel.
2. **Playable.** The 1,000,000-boulder field at 1280×720 renders in at most
   33.3 ms from each viewpoint of gate 6, with prefiltered cones and the
   faster of compute or hardware rays.

**Verdict: FAIL on both.** The showcase matched (within 4e-6), but the field
differed by 2.0e-3 of peak, and the frame barely moved: 55.6–233.2 ms at 720p.
The hypothesis was wrong.

## Results

Apple M4 (MacBook Air, on battery). Reproduce with `cargo run --release -p
room3d --bin m4_gate7 -- capture` on the previous kernel, then without
`capture`.

| 1280×720, 1,000,000 boulders | Compute rays | Hardware rays |
|---|---|---|
| standing | 55.6 ms | 66.2 ms |
| 10 m up | 125.4 ms | 152.1 ms |
| 40 m up | 212.0 ms | 225.6 ms |
| 150 m up | 233.2 ms | 257.1 ms |

Max difference from the pre-change kernel:

| Render | Max difference |
|---|---|
| Showcase, cached | 2.8e-6 of peak |
| Showcase, per-pixel | 4.0e-6 of peak |
| Field | 2.0e-3 of peak |

## The measured breakdown

After three wrong guesses (gates 4, 6 and 7), the kernel now has a profiling
knob (`set_profile_stage`) that cuts cones short at successive stages. On the
field at 1280×720 with compute rays:

| Cumulative frame time | standing, exact | standing, prefiltered | 150 m up, exact | 150 m up, prefiltered |
|---|---|---|---|---|
| no cone (rays and shading) | 7.6 | 7.7 | 18.8 | 21.2 |
| + cone setup and weights | 11.9 | 16.2 | 25.6 | 20.7 |
| + object hierarchy | 15.9 | 16.2 | 45.4 | 38.0 |
| + triangle hierarchies (no rasterization) | 35.9 | 43.5 | **164.4** | 173.1 |
| full | 46.6 | 51.1 | 215.2 | 232.7 |

Times are in ms.

- **The cost is the cone's walk inside objects' triangle hierarchies.** From
  150 m up, nearly every pixel's cone starts on a boulder's surface. Every
  node containing the start point touches the cone and cannot be culled, so
  each cone descends its own 9,800-triangle boulder (and neighbors) to the
  leaves. Rasterizing what it finds adds the rest.
- **Rays and cone setup are small.** With lighting cached, this view would cost
  about 19 ms at 720p (the "no cone" row).

## What this means

- **Per-pixel cones every frame are the brute-force path.** M2 gate 3 found
  this for a small room, and it holds at scale. Node-level tricks (normal
  cones to skip a mesh's own back-facing clusters, cheaper first tests) might
  win 2×, not the 10× needed.
- **The structural answer is the paradigm's first rule:** cache what cannot
  change. The boulders and the sun are static, so their lighting needs
  computing once per visible surface patch, not once per pixel per frame. For
  a million instances, the cache must be view-dependent: texels allocated only
  for visible surfaces, coarser with distance, baked within a per-frame
  budget, kept across frames, and updated exactly when something moves (as in
  M2 gate 5). That is the next milestone.
- **The new form of the weights stays.** It was not the bottleneck, but it is
  cheaper for open cones and identical up to rounding where it applies. The
  field's 2.0e-3 difference is not explained yet.
