# Real assets against Cycles (development, 2026-11-21)

Not a gate: development measurements while making real glTF assets look
right, recorded as CLAUDE.md asks.

## Setup

`gltf_dev` with `GD_CYCLES=1024`: Continuum's `Renderer` with its defaults
(bounce light, anti-aliasing, the time budget) at 2560×1440 (`GD_SS=4`),
averaged 4×4 down to 640×360 once settled (or after 400–500 frames),
against Cycles (Blender 5.2, Metal) rendering the same file through
**Blender's own glTF importer** (`cycles/render_gltf_reference.py`, an
independent reading of the file: its materials, textures and extensions),
1,024 samples, 4 bounces, no denoising, caustics off, at 640×360. Same
camera (the file's, or the viewer's default view), the same sun (irradiance
3, 0.5°, toward (−0.6, −0.3, 0.75)) and the uniform sky (0.18, 0.225, 0.3)
over the whole sphere.

Metrics: relative RMSE (RMS difference over the reference's mean) after
clamping both images at 4× the reference's mean (sun glints on smooth metal
are thousands of times the mean and dominate otherwise: unclamped 1.59 for
DamagedHelmet), and the share of pixels off by more than 10% of the mean in
any channel.

## DamagedHelmet

| Change | Sun and sky | Sky alone | Sun alone |
|---|---|---|---|
| Before (Karis's analytic split sum, metal seen in reflections black) | 8.78%, 3.71% | 3.84%, 3.76% | 58.6%, 4.71% |
| The split sum from a 32×32 table of GGX's directional albedo | 8.57%, 3.12% | 3.51%, 2.92% | 58.8%, 4.71% |
| And metal hits of reflection rays lit with their base color | **8.09%, 2.70%** | **3.13%, 2.64%** | 71.1%, 5.01% |

(Each cell: clamped relative RMSE, pixels off by > 10%.)

- Karis's fit (the mobile approximation) was 15–20% dark on metals of
  roughness 0.3–0.5 (e.g. f0 0.9, roughness 0.5: 0.65–0.67 against the
  integral's 0.76–0.82) and up to 3× dark for rough dielectrics at grazing
  angles; the table's worst interpolation error is 1.5% (metal), 0.9%
  (dielectric) at 150 random points.
- Cycles with single-scattering GGX instead of the importer's Multiscatter
  GGX: the same within 0.1% (8.73% against 8.78% before the changes), so
  multiple scattering is not what differed here.
- Bounces: Cycles at 1 bounce gives 8.70% against 8.78% at 4 (the helmet
  floats in the open).
- Sun alone is worse with lit metal hits: a metal lit by the sun alone
  reflects it in one direction, not as a Lambertian surface would. Sun
  alone (no sky) is not a lighting anyone uses; the full image improves.

## FlightHelmet and ToyCar (sun and sky; metal hits black → lit)

| Model | Black metal hits | Lit metal hits |
|---|---|---|
| FlightHelmet | 4.03%, 2.02% | **3.38%, 2.06%** |
| ToyCar | 39.2%, 28.4% | 39.9%, 28.3% |

ToyCar's difference lies mostly in the reference: its glass is glTF's
thin-walled transmission (no volume), which light crosses without
bending; Blender's importer makes it a solid refracting surface (IOR 1.5,
no exit), so near the window frame Cycles bends rays into the dark
interior where Continuum (as glTF defines it) sees the sky through a gap
(a bright green line), and with caustics off no light reaches the interior
through the glass in Cycles. A fair reference needs thin glass in Cycles
(as M16's reference script does), not done yet.

## Frame times (M4, 1280×720, `Renderer` defaults, settled or late frames)

Sponza 36 ms (after hardware reflection rays; 116 ms before), FlightHelmet
17–18 ms, ABeautifulGame 15 ms, ToyCar 32 ms, DamagedHelmet 4.6 ms.

## HDR environments (DamagedHelmet, Poly Haven HDRIs at 1k, CC0)

Before: the visible environment was 8-bit sRGB over its peak (an outdoor
HDRI's sky went black beside its sun; a studio's banded), and an outdoor
HDRI's sun lit only through the spherical harmonics (no shadow). Now: the
image is RGBM (8-bit colour times an 8-bit multiplier over the peak, mips
averaged from radiance), rough reflections read a mip by roughness, and
`Renderer` takes an outdoor HDRI's sun out of the image as a sun light
(`Environment::take_sun`: kloofendal_48d_partly_cloudy_puresky's at 48°,
irradiance about 4; none found in two studio HDRIs or venice_sunset).

| HDRI | Mean, ours / Cycles | Clamped relative RMSE | Pixels off > 10% |
|---|---|---|---|
| kloofendal_48d_partly_cloudy_puresky | 0.2121 / 0.2128 | 5.11% | 3.72% |
| studio_small_09 | 0.6812 / 0.6799 | 6.69% | 2.49% |

## Rough reflections and the hardware sky rays (Sponza)

- Rough surfaces' reflections blend from the mirror ray toward their
  cached sky and bounce light over π (roughness 0.45 → 0.8; no ray at
  all from 0.8). DamagedHelmet unchanged (8.09%, 2.72%); FlightHelmet
  3.38% → 3.17% (pixels off 2.06% → 1.80%).
- Sponza (viewer's courtyard view, 640×360, ours 2×2 supersampled, a
  fixed bake budget, Cycles 2,048 samples; Cycles is still noisy here, the
  image's mean being 0.005):

| Sponza | Sun and sky: clamped RMSE, pixels off | HDRI |
|---|---|---|
| Before | 71.6%, 58.6% | 69.5%, 51.2% |
| Rough reflections blended | 68.7%, 55.7% | |
| Sky rays turned per texel | 65.4%, 42.3% | 59.0%, 33.3% |
| Bounce rays turned per sample too (reverted) | 65.1%, 41.2% | |

- The bright strip down Sponza's floor (one tile wide, sharp-edged) was
  the hardware sky bake's 192 fixed directions (hemicube bin centres, the
  same at every texel): from a narrow courtyard the open sky is a few bins
  wide and stepped across them. Direct sky alone (no bounce, no
  reflections): floor centre 0.0239 → 0.0148, side 0.0113 → 0.0131,
  against Cycles' 0.0165 and 0.0145 (Cycles' include direct specular).
  Each texel now turns the hemicube about its normal by an angle of its
  own (every bin keeps its weight).
- Turning bounce rays per sample too was reverted: with nine samples a
  page its noise showed as blotches on Sponza's vaults in the viewer, for
  1.1 points.
- Left: Sponza's arches' undersides and the lion head stay brighter than
  Cycles', curtains darker; not yet explained.
- Found: with hardware rays off (the compute path's sky cones), Sponza's
  direct sky is nearly black (0.0003 at the floor centre, with or without
  object prefiltering and with meshes combined by the maximum). Not the
  default anywhere with ray queries; to look into.

## Ten Khronos showcase models under the studio HDRI (2026-11-21)

`gltf_dev`, studio_small_09 (1k), each file's own or the default view,
ours 4×4 supersampled to 640×360, Cycles 1,024 samples. The mean is mostly
the bright studio behind each model, so "pixels off" (by more than 10% of
the mean) undercounts errors on the models themselves.

| Model | Mean ours / Cycles | Clamped RMSE | Pixels off |
|---|---|---|---|
| AntiqueCamera | 0.7571 / 0.7561 | 2.11% | 0.89% |
| Lantern | 0.7580 / 0.7582 | 1.80% | 1.26% |
| Corset | 0.6881 / 0.6889 | 2.82% | 2.11% |
| BoomBox | 0.6671 / 0.6659 | 8.91% | 3.38% |
| WaterBottle | 0.6762 / 0.6792 | 9.56% | 5.02% |
| CarConcept | 0.7016 / 0.7008 | 12.40% | 7.10% |
| GlamVelvetSofa | 0.6433 / 0.6495 | 4.35% | 11.25% |
| DragonAttenuation | 0.6503 / 0.6386 | 10.00% | 14.09% |
| ChronographWatch | 0.6556 / 0.6762 | 24.23% | 15.73% |
| SheenChair | 0.6838 / 0.7005 | 8.08% | 18.30% |

- **Sheen** (`KHR_materials_sheen`) was not supported: both fabrics came
  out flat and dark (the navy velvet nearly black). With it (the Charlie
  distribution and visibility, an albedo table from
  `scripts/sheen_table.py`): GlamVelvetSofa 4.35% → **2.11%** (pixels off
  11.25% → **1.50%**, mean 0.6494 against 0.6495), SheenChair 8.08% →
  **5.92%** (18.30% → 12.78%; Cycles' Principled sheen is its own
  microfibre model, not glTF's, and shows the fabric's texture more).
- Left: ChronographWatch's metal reflections (darker and brighter in
  patches), DragonAttenuation's backdrop (its wall brighter, floor
  darker than Cycles').

## Studio lights taken out of the HDRI (2026-11-22)

`Environment::take_lights` (up to 4, `RendererOptions::
lights_from_environment`) generalizes the sun's extraction: from each
remaining peak, the pixels joined to it above 10% of its radiance, kept if
at most 0.2 sr, at least 2% of the light and ten times the median, made
distant square lights of the same solid angle. studio_small_09 gives its two
lamps (0.003 and 0.004 sr) and a softbox (0.103 sr); the outdoor HDRI its
sun alone; venice_sunset now its low sun (2% share, under take_sun's 20%).

| studio_small_09 | Before (clamped RMSE, pixels off) | Lights taken out |
|---|---|---|
| DragonAttenuation | 10.00%, 14.09% | **8.49%, 8.71%** |
| ChronographWatch | 24.23%, 15.73% | **15.83%, 14.64%** |
| SheenChair (with sheen) | 5.92%, 12.78% | 4.79%, 14.32% |
| DamagedHelmet | 6.69%, 2.49% | 9.80%, 2.46% |

Cost (DamagedHelmet, 1280×720, `Renderer` defaults): settled in 1.18 s with
no lights taken, 1.38 s with one, 1.75 s with two (the lamps), 3.66 s with
the softbox too (a wide light's cones); frames 4.7–5.1 ms once settled.

## Animated files under a sky (2026-11-22)

The viewer plays a file's animation only under the sun alone; under a sky
or an HDRI the movers' and deforming geometry's sky re-bakes cost too much
(`VIEWER_ANIMATE_UNDER_SKY=1` plays it anyway). Viewer, 2560×1440 window
rendered at half, 400 frames:

| File | Sun alone | Under the sky |
|---|---|---|
| Fox (1 skinned mesh) | 7.53 ms | 78.59 ms |
| BrainStem (59 skinned meshes) | 32.10 ms | over 1 s (stopped after 10 minutes) |

Characters are the game assets most often animated; making the sky cheap
around moving and deforming geometry is the next large item for them.
