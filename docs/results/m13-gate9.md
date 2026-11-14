# M13 gate 9: colour in the coverage pass (appearance, first)

**FAIL** on both criteria, as development predicted.

| View | Colour error, ours | One ray | Ratio | Settled frame |
|---|---|---|---|---|
| Standing | 3.89% | 5.96% | **0.652× (at most 0.5×): FAIL** | **26.50 ms (at most 15): FAIL** |
| 150 m up | 9.87% | 18.05% | **0.547× (at most 0.5×): FAIL** | **22.05 ms (at most 15): FAIL** |

Reported: 20 m up looking 10° down, settled 28.24 ms (4,285 pages).
Standing settled in 10 frames (2,828 pages); from 150 m in 2 (10). Mean
colour: standing, reference (0.1897, 0.1828, 0.1678), ours (0.1897,
0.1828, 0.1677); 150 m up, reference (0.2212, 0.2090, 0.1782), ours
(0.2265, 0.2141, 0.1826), 2.4% bright.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `d32c042`; the method in `84dd03e`, before the
run). Machine free of other work; control just before, gate 4's path at
full size: 109.5 ms standing, 91.1 ms from 150 m (120.0 and 102.9 at gate
8: the machine ran cooler).

## Method (profile stage 20, on gate 8's pass)

- **One shading model everywhere, the reference included:** albedo ×
  (E_sun · max(0, n·l) · V + E_amb) / π, V a hard shadow toward the
  light's centre, E_amb a fifth of the sun's normal irradiance, a constant
  sky. The reference averages 16×16 samples a pixel, each one ray shaded
  by the model.
- **Background:** objects far larger than most (the ground) keep bit 0 of
  their hardware-structure mask. A pixel's background is what a ray seeing
  only those meets through its centre; its distance per bin comes from
  that hit's plane. Where the corners' background rays disagree (the
  ground's edge), each bin traces its own.
- **Large objects that are not background:** up to two a pixel, claimed by
  bin rays (pixels whose own corners hit one). Their colours, and the
  background's, come from the corners' shaded hits of the same object,
  bilinearly at each bin.
- **Small instances:** maps carry coverage, mean normal and mean depth;
  each bin is lit by its sample's normal. Records carry the instance's sun
  visibility (one ray), albedo and the light's irradiance at the instance.
  Bins showing the background skip the instance's parts behind it (the
  buried parts).
- **Pages:** texels carry colour (each bake ray's hit shaded, with a
  shadow ray). Hits behind the background don't count.
- Each entry adds the bin's transmittance × coverage × colour / 16; what a
  bin leaves shows its large object, the background or the sky.

## Development (`m13_gate9`'s `M13_DEV`, `M13_ACC`; accuracy on other views)

What the colour error was made of, as one ray's multiple, by turning the
model's parts off in both images. "Flat" is ambient only: colour then
measures coverage against the ground.

| Step (development views) | Flat | No shadows | Full model |
|---|---|---|---|
| Small field, one colour a pixel for large objects | — | — | 0.99× / 1.00× |
| Small field, a second large object a pixel | — | 0.65× / 0.90× | 0.72× / 0.94× |
| Small field, background through the centre past small hits | 0.38× / 1.05× | — | — |
| Small field, the light's irradiance at each instance | 0.41× / 0.43× | — | — |
| Small field, per-bin background where corners disagree | 0.28× / 0.27× | 0.62× / 0.62× | 0.69× / 0.72× |
| Full size, the same | 0.33× / 0.51× | 0.58× / 0.51× | 0.64× / 0.61× |
| Full size, background distance by its plane; bakes skip buried hits | 0.33× / 0.15× | 0.58× / 0.48× | 0.64× / 0.58× |
| Full size, background rays per bin where corners disagree | 0.29× / 0.15× | — | 0.64× / 0.58× |

(Standing / 120 m up.)

Lessons:
- **One colour per pixel for large objects is one ray.** Boulders and the
  ground are both large, so their shared silhouettes need per-bin
  ownership (two objects a pixel).
- **The light was 3 km away**, and its irradiance varies by 10–15% across
  the field. Lighting instances by the eye's irradiance made distant
  boulders bright. This was the largest single error (flat 1.0× → 0.43×
  from 120 m).
- **Buried parts.** The boulders sit partly below the ground. Maps and
  page bakes counted those parts until a depth test against the
  background removed them (flat 0.93× → 0.41× standing on the small
  field).
- **The horizon:** within one pixel the ground ends and the sky begins.
  Backgrounds must be per bin there.
- **Lighting inside a pixel is what is left.** Bumpy boulders' shading and
  sharp sun shadows vary below a pixel. Large objects get four corner
  samples a pixel, small ones a bin's mean normal and one sun visibility.
  Shading took flat 0.29× / 0.15× to 0.58× / 0.48×, and shadows to
  0.64× / 0.58×. A constant colour for small instances was far worse
  (0.96× from 120 m), so the maps' normals do work.
- **Speed:** colour adds about 13 ms to coverage's 9. It goes to every
  corner's shading and shadow ray, the background rays, and bin rays that
  shade.

## What the numbers say

- **The structure is right:** with the lighting held flat, colour through
  coverage, background and buried parts is within 0.15–0.3× one ray's
  error in development.
- **Sub-pixel lighting is the open problem,** and it is a different one:
  near large objects need shading anti-aliasing, not coverage. Far ones
  need light-dependent detail (shadowed parts of a boulder) that a mean
  normal and one visibility cannot carry.
- **Next:** a narrower design (DESIGN §3.4, "Appearance, second").
