# M5 gate 4: sky light through the view-dependent cache, at scale

Date: 2026-10-24. M5 gate 3 matched Cycles on the million-boulder field with
sky light, but per-pixel sky cones cost 100–200 µs per pixel. This gate bakes
the sky into the view-dependent lighting cache and makes sky cones cheaper.

## What was built

- **Cache pages with two channels.** Each texel holds irradiance per unit sun
  radiance and per unit sky radiance; shading applies the radiances. Camera
  rays that miss see the sky. A pixel with no page uses unshadowed sun and an
  unoccluded upper-hemisphere sky, π(1 + n_z)/2, as a guess until its page
  arrives.
- **Coarser sky prefiltering** (`set_sky_prefilter`). Sky cones splat nodes
  up to 8 times the light's threshold; their bins are 11° wide.
  - Measured before the gate on the field at 320×180 (sky only, compared
    with factor 1):

    | Factor | Cost per pixel | Change vs ×1 |
    |---|---|---|
    | ×1 | 190 µs | — |
    | ×2 | 78 µs | 0.16% |
    | ×4 | 49 µs | 0.47% |
    | ×8 | 24.5 µs | 0.97% |

- **Cheaper weights.** Each sky face's light is its whole horizon-clipped
  window weight minus the weights of covered bins: the same sum, identical
  within 8e-8.
- **Profiling stages** cover sky cones. On the field they put the cost in
  finding coverage:

  | Stage | Cost per pixel | Share |
  |---|---|---|
  | Object hierarchy and object splats | 33 µs | 17% |
  | Triangle-hierarchy descents | 113 µs | 59% |
  | Rasterization | 45 µs | 24% |

## Pass criterion

Fixed before the first run, in 2ab670a.

1. Per-pixel sky only at the new sky prefilter, 4×4 per pixel, within 3% of
   Cycles.
2. The sun channel unchanged: warm cached sun-only images within 0.2 points
   of gate 8.
3. At each of gate 8's viewpoints, the warm cached sky-only image within 5% of
   the per-pixel sky-only render (1 sample per pixel).
4. With sun and sky:
   - warm, each viewpoint at most 16.7 ms;
   - three flights, each with mean at most 16.7 ms, 99th percentile at most
     33.3 ms, and fallback at most 10%.

**Verdict: FAIL on 3 and 4.** Criteria 1 and 2 pass. The timing results (4)
were confounded by the machine's state; see below.

## Results

Apple M4 (MacBook Air), 999,986 boulders. Reproduce with `cargo run
--release -p room3d --bin m5_gate4`.

**1.** Sky only, 4×4 per pixel, 640×360, sun at 1 km, against Cycles
(instanced, 1,024 spp): **1.87%**. Ours took 60 s, against 224 s at factor 1
in gate 3.

| Viewpoint | 2. Sun only vs per-pixel exact (gate 8) | 3. Sky only, cached vs per-pixel | 4. Warm sun and sky | Frames until warm: sun / sky / both |
|---|---|---|---|---|
| standing | 14.49% (14.49%) | 2.36% | 16.64 ms | 1,345 / 1,370 / 1,370 |
| 10 m up | 25.87% (25.87%) | 8.33% | 51.56 ms | 2,386 / 2,402 / 2,402 |
| 40 m up | 30.88% (30.88%) | 18.22% | 52.00 ms | 2,279 / 2,279 / 2,279 |
| 150 m up | 38.80% (38.80%) | 20.45% | 56.51 ms | 1,268 / 1,268 / 1,268 |

Flights with sun and sky:

| Run | Mean | 99th percentile | Fallback | Samples at wanted mip | Bake budget mean | Texels baked |
|---|---|---|---|---|---|---|
| 1 | 264.4 ms | 481.7 ms | 1.88% | 0.4% | 1,039 | 591,499 |
| 2 | 267.0 ms | 513.4 ms | 1.88% | 0.4% | 1,039 | 591,418 |
| 3 | 241.6 ms | 480.1 ms | 1.88% | 0.4% | 1,039 | 591,426 |

## What went wrong

**1. The machine was not in a state to measure speed.**

- **A control after the gate.** I repeated gate 11's sun-only warm-up at the
  standing viewpoint with hardware rays (`warm_check`):

  | Measure | This run | Gate 11 |
  |---|---|---|
  | Frames until warm | 1,379 | 102 |
  | Warm frame | 79.8 ms | 8.0 ms |
  | Visibility (staged profile) | 58 ms | about 2–3 ms |
  | Shading (staged profile) | 83 ms | about 2–3 ms |

  The same work took 10–40 times longer.
- **What was running alongside.** A Python process left over from earlier in
  this session had been at 100% of one CPU core for 9 hours, heating the
  fanless machine. A Roblox player had also been launched an hour before. It
  was idle in the tray when checked, but may not have been throughout.
- **What this means for criterion 4.** The timings here say little about the
  renderer. The sun-only warm-up counts match gate 11's compute-ray rows
  (1,295 / 2,402 / 2,279 / 1,266), because frames this slow pin the bake
  budget at its 1,000-cone floor.

**2. The budget floor is wrong for sky texels, whatever the machine.** The
floor counts texels, but a sky texel costs about 100 times a sun texel. In
the flights each frame baked about 1,000 texels, at roughly 250 µs each on
this machine. The floor must be a time, or scale with the measured cost per
texel.

**3. Criterion 3's 5% was too tight, and it fails for a known reason.** The
cached image is lit through texels about 2 pixels wide. Sky occlusion around
boulders that are a few pixels across varies within a texel, just as sun
shadows do. Gate 8 measured the sun channel at 14–39% from per-pixel cones
for the same reason. The sky, being softer, does better (2.4% standing,
8–20% from above), but not 5%. The threshold was set without weighing the
sun's precedent: a pre-registration mistake. The fail stands.

## What this means

- **Sky light works in the cache and stays accurate.** Prefiltering sky cones
  8× makes them 3.7× cheaper at 4×4 per pixel (60 s vs 224 s) and leaves them
  within 1.9% of Cycles. The sun channel is untouched (criterion 2), and the
  cached sky converges to the per-pixel sky where texels resolve the detail
  (2.4% standing).
- **Sky texels are about 100× costlier than sun texels.** Coverage within
  wide cones dominates. The cache's bake budget has to account for that: a
  floor in time, not texels. A cheaper far field would also help, for
  example coarser sky texels than sun texels, since sky light varies slowly.
- **Speed must be re-measured on a quiet machine** before anything is
  concluded about warm frame times with sky.
