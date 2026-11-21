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
