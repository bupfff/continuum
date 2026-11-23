# The basics scorecard (phase 1, from 2026-11-22)

Not a gate: a fixed set of scenes measured the same way after each change
that may move them (DESIGN §3.8, the user's phase 1). The binary is
`experiments/room3d/src/bin/scorecard.rs` (`cargo run --release -p room3d
--bin scorecard`, about 9 minutes; `SC_ONLY=sponza,field`, `SC_SPEED=0`,
`SC_QUALITY=0`); images and the cached Cycles references go to
`experiments/room3d/out/scorecard/`.

## What it measures

Scenes:

| Scene | What | Lighting | Motion (120 frames) |
|---|---|---|---|
| sponza | Khronos Sponza, the courtyard view (eye 5, 0.3, −3.4 from the center) | sun (irradiance 3) and the viewer's sky | walk 3 m, turn 30° |
| field | the million-boulder field, standing | its sun at 1 km, its sky | walk 10 m, turn 20° |
| room | the showcase's closed room (M10 G25) | one skylight panel, no sky | walk 1 m, turn 25° |
| movers | the same room with its ball and cube moving | the same | the movers play, camera still |
| helmet | DamagedHelmet, the viewer's default view | sun and sky | orbit 40° |
| fox | Fox (1 skinned mesh) over the viewer's ground | the viewer's sun and sky | the animation plays |
| brainstem | BrainStem (59 skinned meshes) over the ground | the same | the same |

- **Speed** (every scene first, so timings are as cool as the machine
  allows): `Renderer` with its defaults at 1280×720, every frame waited
  for. Warm-up: seconds (frames) until under 1% of the pixels lack their
  page, and until settled (32 quiet frames, at most 60 s). The settled
  frame (median, p90 of 60). Motion: median and p99 frame, mean and worst
  share of pixels without their page; then held still: the image at the
  stop against the settled one, and the time to settle again (at most
  30 s). GPU memory (`Renderer::gpu_bytes`).
- **Quality:** 2560×1440 at a fixed bake budget (120,000 texels a frame)
  until settled or 600 frames, averaged 4×4 to 640×360, against Cycles at
  the view (and pose) where the motion stopped: relative RMSE with both
  clamped at 4× the reference's mean, and pixels off by more than 10% of
  the mean in any channel. Cycles: Blender 5.2, 4 bounces (the rooms: 12
  bounces, 4,096 samples), 1,024 samples (Sponza 8,192); glTF files
  through Blender's own importer, posed at the stop time
  (`render_gltf_reference.py` gained `"time"` and `"ground"`).

Caveats: warm-up times follow the time budget, so they move with the
machine's heat (Sponza's under-1% mark was 23.2 s in a first run from cool
and 29.5 s in the baseline below, run after tests). "Pixels off" is harsh
on Sponza's dark interior (10% of a mean of 0.0015). Sponza against Cycles
at 2,048 samples was 30.98%, at 8,192 30.43%: Cycles' noise is not what
differs there.

## Found while building it: the sun was a square 50 radii away

Distant lights (`Light::sun`, `Light::distant`) are squares at a
distance, and every caller (the viewer, `Renderer`'s lights taken from an
HDRI, the example, the tools) put them 50 scene radii away. Across the
viewer's ground (20 radii wide) that sun's irradiance falls off by up to
±40%; Cycles' sun is at infinity. Now `continuum_gpu::SUN_RADII` (10⁴)
radii, used by all of them:

| Scene (against Cycles) | 50 radii | 1,000 radii | 10⁴ radii |
|---|---|---|---|
| fox | 8.73%, 23.37% | 2.17%, 0.67% | **2.13%, 0.68%** |
| brainstem | 9.47%, 25.21% | 3.26%, 1.72% | **3.22%, 1.73%** |
| helmet | 4.48%, 1.81% | 4.46%, 1.79% | 4.46%, 1.79% |
| sponza (2,048 samples) | 31.15%, 64.41% | 30.99%, 64.41% | 30.98%, 64.40% |

(Clamped RMSE, pixels off.) No precision loss at 10⁴ radii (Sponza's sun
200 km away). Tests: all pass (`continuum-gltf` GPU tests, `continuum-gpu`);
`motion_blur_smears_along_the_motion` failed once in a parallel run and
passed alone twice and in the next full run (a flake under load).

## Baseline (2026-11-22, after the sun fix)

Speed, 1280×720, `Renderer` defaults (minutes of load before each row):

| Scene | Load before | Warm-up: under 1% / settled | Settled frame (median, p90) | Motion frame (median, p99) | Without page in motion (mean, worst) | Stop vs settled | Settles again | GPU memory |
|---|---|---|---|---|---|---|---|---|
| sponza | 0.0 min | 29.48 s (713) / 43.70 s (1095) | 30.76, 31.11 ms | 38.24, 58.69 ms | 0.00%, 0.01% | 13.84% | 12.99 s (314) | 1314 MB |
| field | 1.2 min | 0.34 s (39) / not within 60 s | 10.35, 11.61 ms | 9.82, 11.04 ms | 0.07%, 0.08% | 5.72% | not within 30 s | 1017 MB |
| room | 2.8 min | 0.12 s (6) / 1.71 s (164) | 5.22, 7.14 ms | 9.11, 18.42 ms | 0.00%, 0.01% | 0.15% | 0.17 s (17) | 636 MB |
| movers | 2.9 min | 0.10 s (6) / 2.01 s (200) | 5.10, 5.18 ms | **159.06, 273.59 ms** | 0.00%, 0.13% | 2.51% | 0.81 s (80) | 656 MB |
| helmet | 3.3 min | 0.18 s (20) / 1.26 s (146) | 4.69, 6.23 ms | 6.92, 8.85 ms | 0.00%, 0.00% | 0.00% | 0.09 s (17) | 637 MB |
| fox | 3.3 min | 0.10 s (6) / 1.59 s (195) | 5.65, 7.51 ms | 14.67, 51.61 ms | 1.97%, 96.25% | 8.35% | 4.13 s (323) | 682 MB |
| brainstem | 3.5 min | 0.60 s (57) / 6.69 s (652) | 5.80, 5.99 ms | **49.21, 62.46 ms** | **38.18%, 85.92%** | **71.04%** | 9.21 s (646) | 709 MB |

Against Cycles, 640×360:

| Scene | Clamped relative RMSE | Pixels off > 10% | Settled at | Mean: ours, Cycles |
|---|---|---|---|---|
| sponza | **30.43%** | 62.35% | frame 104 | 0.0016, 0.0015 |
| field | 1.68% | 0.12% | not in 600 frames | 0.0300, 0.0301 |
| room | **46.55%** | 100.00% | frame 41 | 0.1550, 0.2624 |
| movers | **44.46%** | 100.00% | frame 51 | 0.1853, 0.3132 |
| helmet | 4.46% | 1.79% | frame 29 | 0.2073, 0.2070 |
| fox | 2.13% | 0.68% | frame 102 | 0.3322, 0.3355 |
| brainstem | 3.22% | 1.73% | frame 94 | 0.3272, 0.3297 |

Reading it (worst first):
- **The closed room, 41% dark:** one bounce by default (M10 G25: 8 bounces
  were 6.54% from the same reference, opt-in).
- **Movers in the room: 159 ms a moving frame** against 5 ms still (the
  known 80–175 ms of §3.4).
- **Animated characters with `Renderer`'s defaults:** BrainStem 49 ms a
  frame, 38% of pixels without their page while it plays, 71% from the
  settled image when it stops; Fox once 96% in a frame. (The viewer turns
  `CacheOptions::sky_lag` on; `Renderer`'s default leaves it off.)
- **Sponza:** 30% from Cycles (arch undersides and the lion's wall bright,
  the left vault dark), 44 s to settle, 31 ms a frame.
- **The field** never settles (bounce), though it is 1.7% from Cycles.
- Fine: the helmet, and the characters once settled.

## Fixes from the scorecard (2026-11-22)

**Movers: one atomic add instead of a compare-and-swap loop.** Every stale
texel of a moving frame appended its re-bake job through a compare-and-swap
loop on one counter (`append_jobs`); with about 76,000 stale texels a frame
in the room, the "stale texels" pass took about 285 ms (profiled, its own
submission) and now 0.43 ms. Past the capacity, the range an append took is
written as empty jobs, which the bake skips.

**`CacheOptions::sky_lag` on by default** (the viewer had it on): the sky
around moving and deforming things trails them by a few frames instead of
re-baking in the same frame. Without it, Fox took 92 ms and BrainStem 110
ms a moving frame (p99 745 ms) once the fix above let more work through.

| Row (motion: median, p99; without page: mean, worst; stop vs settled) | Baseline | After |
|---|---|---|
| movers | 159.06, 273.59 ms; 0.00%, 0.13%; 2.51% | **20.53, 27.58 ms**; 0.01%, 0.08%; 3.95% |
| fox | 14.67, 51.61 ms; 1.97%, 96.25%; 8.35% | **8.76, 10.54 ms; 0.00%, 0.02%; 3.77%** |
| brainstem | 49.21, 62.46 ms; 38.18%, 85.92%; 71.04% | **31.17**, 86.41 ms; 34.32%, 50.66%; **16.10%** |

Everything else within run-to-run spread; against Cycles unchanged. (Sponza's
warm-up moves most between runs: under 1% at 23–37 s, settled 36–50 s.)

**Tried and reverted: they hung the GPU and logged the user out** (three
times; macOS's WindowServer watchdog restarts it after 40 s without a frame;
reports in `/Library/Logs/DiagnosticReports`). Patches kept in
`experiments/room3d/out/scorecard/`.
- `APPEND_PAGES` 4,096 → 16,384: BrainStem's first moving frame asks for
  10,553 pages in view near its 59 meshes; past 4,096 they are invalidated
  (55% of the pixels unshadowed for about 90 frames). At 16,384 BrainStem's
  pixels without a page fell to 7.4% in motion, but one frame took 25.6 s.
- All deforming meshes' new structures built in one submission (one each
  before): BrainStem's render 32.4 → 22.9 ms a frame. Present in all four
  bad runs; a GPU report showed a page fault (an unmapped read) then a
  firmware-detected lockup. Shader buffer access is bounds-checked, so most
  likely ray traversal met an invalid structure (perhaps builds in one
  command buffer sharing scratch memory in wgpu's Metal path); not found.
- The scorecard now stops at any frame over 5 s.

**Investigated, not changed yet:**
- *More bounces* (`CacheOptions::bounces`; speed, then against Cycles):

| | 1 (default) | 4 | 8 |
|---|---|---|---|
| room | 46.55%; settled 1.7 s | **7.74%**; 3.4 s | 6.64%; 4.9 s |
| movers | 44.46%; moving 159 ms (before the fix) | 6.69%; 209 ms | 5.84%; 210 ms |
| field | 1.68% | 1.35% | 1.34% |
| helmet | 4.46%; settled 1.3 s | 4.46%; 2.4 s | 4.46%; 4.6 s |
| sponza | 30.43%; 30.8 ms, settled 44 s | 31.66%; 37.2 ms, not in 60 s | 31.66%; 46.6 ms, not in 60 s |

- *Sponza's 30%*, split at the original view (eye 5, 0.3, −3.4) against
  Cycles at 0 and 1 bounces, 4,096 samples: sky light alone with
  reflections 1.013× Cycles' mean (0.81× without reflections: Cycles'
  direct light includes the sky's specular), but by region 0.75–1.42×;
  direct light under the sun about right; bounce light 0.92× overall,
  0.48–1.63× by region (the right arcade's vaults bright, the far corners
  dark). Many regional errors of ±25–60%, not one bias.

## Deferred pages near deforming objects (2026-11-22)

A page in view that only deforming objects' catch-up made stale (lighting
near them may lag, M12 gate 2), left out of a frame whose 4,096 appended
pages are full, is no longer invalidated: it is marked deferred
(`slot_stale` 4), keeps showing its light, and the stale pass re-bakes it
whole (every light) in the first frame with room, running while any page
waits. A rigid mover reaching it, or the page leaving the view, still
invalidates it. `CacheStats::deferred` counts them.

BrainStem's first moving frame: 6,262 pages deferred (they were invalidated
before), all re-baked by frame 20. Full scorecard run, BrainStem moving:
pixels without their page 34.32% → **7.32%** mean, 50.66% → **15.66%**
worst; frames 31.17 / 90.51 ms (median, p99) against 31.17 / 86.41. Stop
against settled: 14.39% in a BrainStem-only run, 31.49% in the full run
(this measure moves with where the catch-ups stand at the stop; 16.10%
before). Other rows within run-to-run spread; against Cycles unchanged.
Tests pass (`movers_leave_no_stale_light` keeps rigid movers exact).

## Fewer mirror rays (2026-11-22)

Sponza's settled frame (1280×720, `gltf_dev` profile, each pass alone):
visibility 4.2 ms, shading 12.9, edge shading 3.0, anti-aliasing's
passes about 15 (its samples' shading 9.0). Without reflection rays at
all (`GD_GLOSSY=0`): shading 3.6 ms, the samples' shading 2.1 ms. So:
- Anti-aliasing's extra samples trace no mirror ray where the specular
  layer reflects less than 0.15 (dielectrics): their reflection is read
  from the cache as a rough surface's is (sky and bounce light over π).
  Samples' shading 9.0 → 3.8 ms.
- Any pixel skips its mirror ray where the ray's share of the reflected
  light (the specular albedo times the mirror's weight against the rough
  estimate) is under 3%: faint reflections on rough dielectrics. Metals,
  smooth surfaces and grazing views keep theirs. Shading 12.8 → 8.5 ms.

Full scorecard: Sponza's settled frame 30.74 → **22.41 ms**, moving 37.28
→ 30.47 ms, settled in 35.0 s (33–50 s before; under 1% at 17.5 s); against
Cycles 30.44% → **29.94%**. Other rows unchanged (helmet 4.46%, fox 2.15%).
FlightHelmet against Cycles (1,024 samples): 3.17% → 3.15%, pixels off 1.80%
→ 1.72%.

Also in (no effect at the default of one bounce): `CacheOptions::bounce_tail`
(0.15): with more bounces, a hit whose samples hold one asks for more only
where bounce light is over that share of its light. Four bounces with it:
the room 7.67% from Cycles (7.74% without), but Sponza still does not settle
within 60 s (39.7 ms frames before the change above): its arcades are mostly
lit by bounce, so the cascade is needed there. Next: four bounces by default
once settling is fast enough (M22 gate 6).

## Bounce rays per bin (2026-11-22, development)

Sponza's upper gallery, lit by bounce alone, shows coloured blotches where
Cycles is smooth: a bounce sample's 192 rays (one a hemicube bin, the same
directions in every sample) catch small sunlit sources across the courtyard
(the banners) in a few bins each, counted at a whole bin's weight, and
samples side by side catch different ones. `CacheOptions::bounce_rays`
(hardware ray queries) traces 1, 2 (the bin's diagonal quarters) or 4 rays a
bin. `pair_dev`, side by side against 1 ray (speed) and against Cycles:

| Rays a bin | Sponza | Room | Sponza settles (time budget) | Frames |
|---|---|---|---|---|
| 1 (default) | 29.94% | 7.68% | 28.6–31.2 s | — |
| 2 | 27.99% | 7.34% | 1.42× | 1.004× settled, 1.08× moving |
| 4 | 27.91% | 7.33% | 1.59× | 0.999×, 0.98× |

The field and the helmet are unchanged with 4. The gallery keeps some
blotches even with 4: a hit counts for its whole bin, and a few hits on a
small bright source cannot average it as Cycles does; averaging what a bin
covers across objects is beyond what rays give. Also tried: the four
quarters interleaved over a page's samples (one ray a bin): 28.35%, but a
visible grid of coloured dots in the gallery (rejected); every bin split
through the shared hemicube constant (`IND_R` 16): broke the sky bake,
whose loops assume 192 bins (50.67%).

Default left at 1 ray: two rays buy 2 points on Sponza for 1.42× its
settling.
