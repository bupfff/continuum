# M5 gate 8: point and spot lights, and cheaper shading of lights

Date: 2026-10-25. Lights were rectangles only. This gate adds sphere lights
(point lights with a size) and spots, and retests gate 7's failed speed
criterion with cheaper per-pixel shading.

## What was built

- **Sphere lights** (`Emitter::Sphere`, `Light::point`, `Light::spot`),
  per pixel and in the view cache.
  - **The cone.** A sphere's cone is the square around its disk. Each bin
    is weighted by its part inside the disk (a fixed 8×8 table).
  - **Unshadowed light.** The projected solid angle of the sphere's cap,
    clipped to the horizon where it crosses it (Snyder 1996, in the form of
    Lagarde and de Rousiers 2014).
  - **What the cache stores.** The cap's visible share, as for rectangles
    since gate 7.
  - **Camera rays** see spheres.
- **Spots** use glTF's falloff: the square of a ramp in the cosine to the
  axis, between inner and outer half-angles. The falloff applies per
  receiver, and toward the camera for the emitter itself.
- **glTF punctual lights are imported** (`Imported::lights`). Intensities
  are converted at 683 lm/W, as Blender converts them. Point and spot
  lights get a radius chosen by the caller (the viewer uses 5 cm).
  Directional lights become suns 0.5° across.
- **Cycles references for spheres.**
  - **The emitter.** A 128 × 64 mesh sphere, scaled so its silhouette has
    the true sphere's area.
  - **The spot.** A node graph applies the same falloff toward the
    receiver.
- **Cheaper shading.** A rectangle's unshadowed light at a pixel comes from
  Lambert's formula over its corners, without building its cone, and is
  skipped where the cached share is zero. Cache images are unchanged.

## Pass criterion

Fixed before the first run, in 69b5bdd. Prefilter 1 bin with object-level
prefiltering, cache defaults, hardware rays at 1280×720 except criterion
1.

The two scenes:

- **The night scene.** The textured scene lit by `Textured::night_lights`:
  a point light 10 cm across, 2 m up; a spot 5 cm across, 4 m up.
- **The field.** The million-boulder field with the sun and
  `Field::sphere_lights`: a point light 3 m up and a spot 8 m up.

Control: the sun-only warm frame at the standing viewpoint at most 3.3 ms.

1. **Accuracy.** The night scene per pixel (16×16 at 320×180) within 3% of
   Cycles at 16,384 samples per pixel, over surfaces.
2. **The night scene's cache.** Within 2% of its per-pixel image at
   1280×720, over surfaces.
3. **The field's cache.** At each of gate 8's viewpoints, the cache error
   at most gate 7's sun-only figure plus 1 point.
4. **Speed with the sun and sphere lights.**
   - Warm: at most 16.7 ms at each viewpoint.
   - Three flights: each with mean at most 16.7 ms, 99th percentile at
     most 33.3 ms and fallback at most 10%.
5. **Speed with gate 6's three lights.** Three flights, likewise.

Development runs disclosed in the binary:

- the night scene, 3.53% from Cycles at 4,096 samples;
- one flight with the sphere lights, 14.34 ms;
- two flights with the three lights, 15.29 and 15.43 ms.

**Verdict: FAIL** (criteria 4 and 5; 1–3 pass).

## Results

Apple M4 (MacBook Air), plugged in. The control was 1.89 ms, and 1.88 ms
right after the run. Reproduce with
`cargo run --release -p room3d --bin m5_gate8`.

1. **Night scene against Cycles** (16,384 samples, 20.7 s): **1.35%** over
   surfaces, with 35 pixels set apart. Each alone at 4,096 samples: point
   1.66%, spot 2.50%.
2. **Night scene's cache** against its per-pixel image: **0.82%** over
   surfaces. It warms after 126 frames, and a warm frame takes 1.69 ms.

| Viewpoint | 3. Cache error, sun and sphere lights | Bound | 4. Warm frame | Frames until warm |
|---|---|---|---|---|
| standing | **5.44%** | 7.21% | 2.12 ms | 216 |
| 10 m up | **9.76%** | 11.63% | 3.42 ms | 401 |
| 40 m up | **14.68%** | 15.78% | 3.72 ms | 574 |
| 150 m up | **21.75%** | 22.75% | 4.04 ms | 419 |

Flights, 600 frames from an empty cache, in the order they ran:

| Lights | Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |
|---|---|---|---|---|---|---|---|---|
| sun and spheres | 1 | 14.48 ms | 25.16 ms | 51.10 ms | 1.74% | 3,432 | 8.65 ms | 4.96 ms |
| sun and spheres | 2 | **20.21 ms** | **82.47 ms** | 165.89 ms | 2.29% | 2,850 | 10.62 ms | 8.05 ms |
| sun and spheres | 3 | **19.69 ms** | **36.97 ms** | 62.52 ms | 3.27% | 1,931 | 8.86 ms | 9.07 ms |
| three lights | 1 | **24.60 ms** | **46.94 ms** | 66.85 ms | 4.32% | 1,158 | 13.32 ms | 9.53 ms |
| three lights | 2 | **22.29 ms** | **46.01 ms** | 48.86 ms | 3.85% | 1,308 | 13.02 ms | 8.06 ms |
| three lights | 3 | **22.27 ms** | **43.54 ms** | 86.97 ms | 3.54% | 1,308 | 12.79 ms | 7.87 ms |

## Why criteria 4 and 5 failed: sustained heat

- **The same flight slowed from run to run.** Within the gate, the sphere
  lights' identical flights went from 14.48 ms to about 20 ms mean, and
  their GPU camera passes from 4.96 to 9.07 ms. The three-light flights,
  which ran last, were slower still.
- **They ran at the end** of about ten minutes of full GPU load: Cycles at
  16,384 samples, 16×16 per-pixel renders, and four warm caches with
  per-pixel references.
- **A diagnostic after the gate** (`flight_repeat`, not a re-run for pass)
  isolates the cause.
  - **Not accumulation in the process.** Four flights with the sphere
    lights in one process averaged 13.04–13.48 ms, camera passes 3.4–3.7 ms.
  - **Heat.** After six warm caches, twelve flights back to back slowed
    steadily: means from 13.11 to 16.31 ms and camera passes from 3.75 to
    6.12 ms, in about two minutes.
  - **The control does not catch it.** A short warm-frame control right
    after the gate read 1.88 ms: the fanless M4 Air recovers in seconds
    when idle.

So on this machine the flights hold 60 Hz while it is cool, and do not
under sustained load. Both are real; a user flying for minutes gets the
second.

**Gate design.**

- Speed criteria should run first in a gate, or state the thermal state
  they measure.
- Sustained numbers should be gated on their own (CLAUDE.md asks for warm
  and sustained numbers).

## What this means

- **Point and spot lights work, per pixel and in the cache.**
  - The night scene is 1.35% from Cycles and its cache 0.82% from per-pixel
    cones.
  - glTF files bring their own lights.
- **Lights' shading is cheaper than in gate 7.** Camera passes were
  3.4–3.9 ms on a cool machine against gate 7's 4.3–6.2 ms, and cool flights
  stay under 16.7 ms. Gate 7's flights also ran at the end of a long run, so
  heat may account for part of its figures too.
- **Sustained performance on a fanless laptop is the open problem.**
  - The GPU slows by up to about 2× after minutes of load.
  - The bake budget follows GPU time, but its floor (1,000 units) and the
    camera passes do not shrink with the clock.
  - Next:
    - a sustained-speed gate with the machine's thermal state stated;
    - a lower budget floor only once timings are stable (a development run
      at 250 units was inconclusive while the machine was hot);
    - cheaper camera passes.
