# M4 gate 8: a view-dependent lighting cache

Date: 2026-10-24. M4 gate 7 showed that per-pixel sun cones cost 47–233 ms per
720p frame on the million-boulder field, because each cone descends the
triangle hierarchy of the boulder it starts on. The boulders and the sun are
static, so the paradigm's first rule applies: cache per visible patch of
surface what cannot change, instead of recomputing it per pixel per frame.

## What was built

`continuum_gpu::InstancedRenderer::with_cache` (`vcache.rs` and the
`vc_*` passes in `mask_cones.wgsl`):

- **Pages.** Pages are 8×8 texels, stored 9×9 so a bilinear lookup stays
  inside one page. They sit on each object's charts at power-of-two
  resolutions:
  - Texels at mip m are the base size (5 mm) × 2^m × the object's scale.
  - Each pixel wants the mip whose texels span about 2 pixels.
  - A chart's cell grid (`Surface::chart_grid`) maps a texel straight to the
    triangle under it. Every instance of a geometry shares these tables.
- **Page table.** An open-addressing hash table on the GPU, with keys
  (object, chart and mip, page). The CPU keeps a mirror and sends changed
  entries through a scatter pass each frame.
- **Shading** (`vc_shade`). Each pixel reads the visibility buffer and looks
  up its wanted page. Failing that, it uses the finest resident coarser page.
  Failing that, it uses unshadowed light (a "fallback" pixel). One pixel per
  4×4 tile, a different one each frame, reports its wanted page if it was
  missing.
- **Page manager** (CPU). Each frame it:
  - reads the requests back;
  - adds each request's missing coarser pages, so pixels with no page at all
    come first and refinement goes breadth-first;
  - allocates slots and lists the pages' texels within a budget of 20,000
    cones per frame;
  - evicts pages that no pixel has read for 64 frames.
- **Baking** (`vc_bake`, `vc_fill`). One thread per texel: the texel's center
  is placed on the surface through its chart's cells, then a prefiltered mask
  cone is traced from it. The page borders beyond the chart's grid are then
  filled.

Prior art: this is texture-space (object-space) shading on a sparse, virtual
page structure driven by GPU feedback:

- texture-space shading: Baker 2016, object space lighting; Hillesland & Yang
  2016, texel shading;
- sparse virtual textures: Barrett 2008; Mittring 2008;
- feedback-driven page allocation: UE5's virtual shadow maps.

None of the structure is new. Only its use as a cache for exact mask cones is
specific to Continuum.

## Pass criterion

Fixed before the first run. The criterion was committed in 23a443f; details
the stub left open were fixed in 5f5c424.

- **Scene:** the 1,000,000-boulder field at 1280×720, prefiltered cones, the
  faster of compute or hardware camera rays.
- **Quality:** warm, each of gate 6's four viewpoints is within 3% relative
  RMSE of per-pixel exact cones with the same rays.
- **Speed:** warm, each viewpoint renders in at most 16.7 ms (median of 30
  frames).
- **Flight:** along a 600-frame scripted flight from standing height up to
  150 m, from an empty cache, the mean frame takes at most 16.7 ms and on
  average at most 10% of pixels fall back to unshadowed light.

**Verdict: FAIL on all three.**

| Criterion | Result |
|---|---|
| Quality | 14.5–38.8% from the reference |
| Speed | 20.0–37.3 ms warm |
| Flight | 42.7 ms mean frame |

Fallback pixels averaged only 1.8%.

## Results

Apple M4 (MacBook Air, 24 GB), 999,986 boulders (the camera's own spot is
kept clear). Reproduce with `cargo run --release -p room3d --bin m4_gate8`.

| Rays | Viewpoint | Frames until warm | Warm median | Fastest | RMSE vs per-pixel exact | Pages | Cache memory |
|---|---|---|---|---|---|---|---|
| compute | standing | 78 | 20.0 ms | 16.0 ms | 14.49% | 33,011 | 212 MB |
| compute | 10 m up | 127 | 30.1 ms | 28.9 ms | 25.86% | 102,556 | 212 MB |
| compute | 40 m up | 123 | 37.3 ms | 35.5 ms | 30.88% | 101,146 | 212 MB |
| compute | 150 m up | 77 | 31.3 ms | 30.3 ms | 38.80% | 125,846 | 212 MB |
| hardware | standing | 78 | 25.8 ms | 24.4 ms | 14.49% | 33,013 | 212 MB |
| hardware | 10 m up | 127 | 27.6 ms | 21.9 ms | 25.87% | 102,820 | 212 MB |
| hardware | 40 m up | 123 | 29.8 ms | 26.2 ms | 30.88% | 100,081 | 212 MB |
| hardware | 150 m up | 77 | 26.5 ms | 25.2 ms | 38.80% | 125,856 | 212 MB |

- **Ray mode.** Hardware rays had the lower total of the four warm medians,
  so the flight used them.
- **Prefilter.** Per-pixel prefiltered cones differ from exact ones by under
  0.005% here. The sun is small, so its cones are narrow, and they rarely
  meet a node small enough to splat. All the cache's error is its own.
- **Flight** (hardware rays, 600 frames):

  | Measure | Result |
  |---|---|
  | Mean frame | 42.7 ms |
  | Median frame | 43.8 ms |
  | Worst frame | 96.1 ms (frame 462) |
  | Frames 0–299 | 49.2 ms mean |
  | Frames 300–599 | 36.2 ms mean |
  | Fallback pixels, mean | 1.77% |
  | Fallback pixels, worst | 50% (frame 0, empty cache: every surface pixel) |
  | Resident pages at the end | 75,119 |

  The per-frame times are in `out/m4_gate8_flight.csv`.

### Development runs, disclosed

Before the official run, the harness ran on a 2,000-boulder field to debug
binding lists and correctness, with the same settings:

- warm medians 2.5–4.2 ms with hardware rays;
- 7.7–13.6% from per-pixel exact.

A diagnostic run there with texels of half a pixel (`--pixels-per-texel 0.5`)
gave 2.7–4.9%. Error falling with texel size points to blur at texel
resolution, not misplaced texels. No setting was changed after these runs;
the column of per-pixel prefiltered error was added as a diagnostic.

## Where the time and the error go

`cargo run --release -p room3d --bin vcache_profile` times each stage of a
frame separately (each submitted and waited for), on the same field with
hardware rays:

| Viewpoint | Warm: bake / visibility (of which `queue.submit`) / shading / readback + planning | Busy frame, 20,000 texels baked: same stages |
|---|---|---|
| standing | 0.5 / 19.7 (15.2) / 3.4 / 0.9 | 15.5 / 17.5 (15.9) / 1.4 / 7.1 |
| 10 m up | 0.3 / 23.2 (17.4) / 3.2 / 0.8 | 14.1 / 27.9 (25.9) / 1.9 / 22.3 |
| 40 m up | 0.3 / 23.6 (17.0) / 2.9 / 0.8 | 14.2 / 28.1 (26.2) / 2.7 / 26.5 |
| 150 m up | 0.3 / 22.3 (16.2) / 2.6 / 0.9 | 18.4 / 31.3 (29.3) / 2.8 / 23.3 |

Times are in ms.

**Most of the visibility time is spent on the CPU, inside wgpu's
`queue.submit`.**

- **Measured.** At 64×36 pixels (2,304 rays), the hardware visibility pass
  still takes 16–18 ms: 15.4–16.3 ms in `submit`, 0.3 ms on the GPU. With
  1,000 instances, the same `submit` takes 0.03 ms.
- **Cause.** In wgpu-core 30.0.1, a TLAS records one `Arc<Blas>` per
  *instance* as its dependencies: a million entries here, all but two of them
  duplicates. Every submission that uses the TLAS walks that list twice:
  - once to validate (`validate_as_actions`, under a read lock per entry);
  - once to declare residency (`set_acceleration_structure_dependencies`).

  Keeping one entry per distinct BLAS would be semantically the same; a
  candidate upstream fix.
- **What the GPU actually spends.** Hardware camera rays at 720p take 4.5–6.6
  ms warm. Gate 6's finding that hardware rays are "no faster" than compute
  rays was skewed by this overhead (see the correction there).

Compute rays have no such overhead but are slower on the GPU. At 640×360,
compute visibility takes 4.6–10.1 ms warm.

It also compares both images with a 4×4-supersampled render with exact cones,
box filtered: each pixel's area, not its center.

| Viewpoint | Cached vs supersampled | Per-pixel exact vs supersampled |
|---|---|---|
| standing | 14.27% | 12.69% |
| 10 m up | 23.06% | 17.54% |
| 40 m up | 25.27% | 20.11% |
| 150 m up | 30.38% | 23.65% |

## What this means

- **The cache does its job on shading.** Warm shading costs 1.9–3.4 ms, where
  per-pixel cones cost 47–233 ms (gate 7). The cache holds 33k–126k pages in
  212 MB. Coarse-first allocation kept fallback pixels to 1.8% on average
  during the flight.
- **The floor is wgpu's per-submit overhead, not camera rays.** Hardware
  visibility costs 15–17 ms on the CPU in `queue.submit`, linear in instance
  count, and 4.5–6.6 ms on the GPU. Without that overhead, a warm frame would
  be about 0.3 + 6 + 3 + 1 ≈ 10 ms. That is an estimate from the stages, to
  be measured once the overhead is gone.
- **Busy frames are too slow, and the costs are fixable.**
  - Baking 20,000 cones takes 12–21 ms: about 0.8 µs per cone, against about
    0.23 µs amortized for a full screen of per-pixel cones. A budget-sized
    dispatch underfills the GPU and diverges.
  - The CPU page manager takes up to 31 ms. It reads back synchronously, uses
    std's SipHash maps, and walks every request's chain of coarser pages.
- **The quality criterion measured the wrong thing.** At 720p this field has
  detail below the pixel everywhere, such as boulders a few pixels across.
  One sample per pixel is itself 12.7–23.7% from the pixels' true (area)
  values. A 2-pixel texel cache is a blurred estimate of a point-sampled
  image, so 3% from point samples is not reachable at this texel size, and
  not the right target. Against the supersampled truth, the cache adds 1.6–6.7
  points to the per-pixel image's error.
  - Lesson for future gates: image quality is measured against the
    area-integrated image (supersampled exact cones, or Cycles with a proper
    pixel filter), not against one sample per pixel.

## Next

1. **Gate 9: the cache at speed.** Each fix is engineering, not a new idea:
   - remove the per-instance submit overhead: deduplicate the TLAS's
     dependencies in wgpu (a patched copy until upstream has it), or reduce
     the instance count;
   - pipeline the request readback, processing frame N−1 while frame N
     renders;
   - a fast hash map for the page manager;
   - a bake budget the frame can afford.
2. **Then primary visibility and anti-aliasing** (roadmap item 4).
   Pixel-footprint cones would integrate each pixel's area, which removes the
   12–24% sampling error. Quality gates from here on are measured against the
   area-integrated image.
3. **The viewer's field scene now uses the cache by default** (`--no-cache`
   for per-pixel cones). Until the submit overhead is gone, warm frames take
   about 20 ms at any resolution with hardware rays.
