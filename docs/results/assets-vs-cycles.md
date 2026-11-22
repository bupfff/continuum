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
  Cycles', curtains darker; not yet explained. (The lion and the arches:
  sunlight through the walls, fixed 2026-11-22, below.)
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

### Sky rays with movers, and `sky_lag` (2026-11-22)

`anim_dev` (Fox, posed every frame, under the viewer's sun and sky, 1280×720):

| | Median frame | Sky bake pass |
|---|---|---|
| Sky through cones (movers had no ray path) | 42.18 ms | 40.29 ms |
| Sky rays with movers (`hw_vc_bake_sky_movers`) | 23.24 ms | 18.56 ms |
| And `CacheOptions::sky_lag` | **9.11 ms** | 2.25 ms |
| No sky | 7.52 ms | — |

BrainStem (59 skinned meshes) with both: 30.87 ms. Sky rays with movers
see the movers' structure as it is now; `sky_lag` lets a page in view
whose sky a mover may have changed past its ledger's threshold keep its
sky until a re-bake fits the budget (the bound charges a mover's whole
bounding sphere, so around anything that keeps moving every page in reach
re-baked every frame). The viewer now plays animations under the sky and
HDRIs with `sky_lag` on (Fox 9.18 ms, CesiumMan 12.43 ms, BrainStem 44.00
ms at 2560×1440 rendered at half).

Found and fixed (2026-11-22): in the viewer the animated Fox went unseen
by camera rays after a few frames (its shadow stayed right), also in
v0.3.1. The movers' top-level structure is rebuilt in place every frame
they move, and with frames pipelined (as a window renders them) the next
frame's rebuild overwrote it while the previous frame's rays still traced
it. Waiting for the GPU before each frame fixed it; so did waiting only
before the rebuild. `InstancedRenderer` now waits for the previous frame's
submission before rebuilding it (frames with moving movers only). Viewer,
2560×1440 rendered at half, under the defaults' HDRIs: Fox 10.90 ms,
CesiumMan 11.43 ms, BrainStem 32.56 ms a frame. (`anim_dev` never showed
it: its frames ran shallower.)

### Thin glass in the Cycles reference (2026-11-22)

`render_gltf_reference.py` now makes glTF's transmission without a volume
thin glass in Cycles too (as M16's reference does: the specular layer plus
a Transparent BSDF passing the base colour times one less Fresnel), where
Blender's importer made it a solid refracting surface with no exit.

| Model (lighting) | Clamped RMSE, pixels off: before | Thin glass |
|---|---|---|
| ToyCar (sun and sky) | 39.9%, 28.3% | 41.1%, 38.05% |
| ChronographWatch (studio) | 15.8%, 14.6% | 15.8%, 14.3% |

ToyCar's interior is now seen through the window in both, and differs:
its seats' sheen is pure red (`sheenColorFactor` [1, 0, 0] over a 0.15
grey), which glTF's Charlie sheen shows red and Blender's microfibre sheen
grey. Ours also shows a bright green line where the glass meets the
window frame, which Cycles does not; not the offset of the ray behind the
glass (0.1 mm, unchanged at 1e-7), and most likely the frame's inner lip
seen from behind: glTF culls single-sided materials' back faces, as
Continuum's camera rays do, and Cycles never culls them.

## Sunlight through walls (Sponza, 2026-11-22)

Sponza's lion head, and the edges of arches near the walls, were lit by the
sun from behind the wall they stand against (direct sun alone: lion 0.0070,
Cycles 0). A CPU ray from each of the lion's 317 triangles facing the sun
(`leak_probe`) meets the wall's front: no triangle sees the sun, and none
is blocked only by faces seen from behind. Two causes, both in the light
cones:

- **Clipping (all cones).** A wall triangle tens of metres across, a few
  centimetres behind a receiver, reaches behind the cone's apex. Clipped
  to the near plane (1e-4) and then projected, its corners landed millions
  of bins from the window, and the 2D clip in f32 lost the window: the
  isolated cone kernel let through up to 30% of the sun at the lion's
  receivers (largest 2.2e-5 per unit radiance, open 7.6e-5). Clipping to
  the window's pyramid in 3D first still lost about a thousandth of a
  window, interpolating between corners metres apart to points millimetres
  apart. Now a triangle reaching outside the window clips the window's
  square instead, against the planes through the apex and each of its
  edges (cross products of its corners, as in 2D homogeneous
  rasterization, Olano and Greer 1997), wound like the projected triangle
  (a mesh's pieces sum signed areas before the absolute value; an unwound
  square doubled the sky's error at contacts before it was fixed). Largest
  leak now 2e-7 there, a fully covered sun's window less its 64 bins in
  f32 (the cache takes such a cone as dark).
- **Texels facing away from the light (the cache).** A texel whose light
  is below its horizon kept a shadowed share of one, so pixels nearby
  facing the light (the lion's bumps, a terminator) were lit unshadowed:
  with the clipping fixed, the lion with only the wall behind it still
  showed 53% of its light. Such a texel now traces its light through a
  horizon facing it, past its own triangle (one where only its own
  surface turns away, as on a convex object's terminator; the shadow's
  where anything else covers it). Static lights only. Tracing only texels
  within 30° of edge-on saved half the cost and left faint outlines lit.

Viewer's courtyard view as above (640×360, ours 2×2 supersampled, fixed
bake budget 60,000 texels a frame; Cycles 512 samples for the direct sun,
2,048 otherwise, the same cached renders before and after):

| Sponza | Clamped RMSE, pixels off: before | After |
|---|---|---|
| Direct sun alone | 147.3%, 7.95% | 95.9%, 4.50% |
| Sun and sky | 65.4%, 42.3% | 46.4%, 27.4% |
| HDRI | 58.7%, 33.4% | 42.7%, 31.5% |

Direct sun alone, linear means: lion 0.0070 → 0.0000 (Cycles 0.0000),
far wall 0.0036 → 0.0029 (0.0028), left arch 0.00025 → 0 (0). The
curtains and the green fronds were darker than Cycles' before and still
are (curtain 0.0267, Cycles 0.0340; fronds 0.0269, 0.0399).

- Cost: warm-up at the fixed budget, sun, sky and bounce, 8.11 / 8.10 s →
  8.61 / 8.54 s (+6%, two runs each, interleaved); sun alone 2.34 / 2.34 →
  2.57 / 2.55 s (+9.5%). Frame times unchanged (33 ms, 16 ms).
- Unchanged elsewhere: the per-pixel golden image within 1e-4 (relative
  RMSE, no pixel off), light cones against the CPU reference 3.3e-6 of
  the light's weight; with boxes touching the receivers 3.3e-4 worst
  (before 5.1e-4). The sky at ground pixels near objects, against 2,304
  cosine-weighted CPU rays each: mean error 0.0079, worst 0.057, as before.
- Tests: `a_large_wall_shadows_receivers_close_to_it` (the cone kernel;
  7.6% before, 0.17% now, bar 0.5%) and
  `a_wall_shadows_a_sphere_close_to_it` (a settled cache; 2% of the lit
  sphere's brightest before, 0 now).
- Tried and dropped: back faces of open meshes occluding (meshes with
  unpaired edges flagged at build time): no change on Sponza.
