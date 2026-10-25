# M4 gate 9: the lighting cache at speed

Date: 2026-10-24. M4 gate 8's view-dependent lighting cache cut warm shading
on the million-boulder field to 2–3 ms. Frames still took 20–43 ms, and its
diagnostics found three costs outside the cache's design:

- wgpu's per-submit overhead;
- a CPU and GPU that never overlapped;
- an expensive bake.

## What changed

The cache's design is unchanged. Four engineering changes:

- **wgpu-core patch** (`third_party/wgpu-core`, see `third_party/README.md`).
  A TLAS records each distinct BLAS once as a dependency, not once per
  instance. Before, every submit walked a million entries: 15–17 ms of CPU.
- **Pipelined requests.** Frame N plans its bake from frame N − 2's requests,
  read back asynchronously, so the CPU plans while the GPU renders the frame
  before. Slot-use readbacks for eviction are asynchronous too.
- **A faster page manager.** A cheap hash replaces SipHash for page keys.
  Walks to coarser pages stop at a page another request already listed.
  Partial selection replaces a full sort.
- **Bake budget** of 8,000 cones per frame, down from 20,000.

The wgpu patch was measured with `vcache_profile` before this gate was
written, which the pre-registration states.

## Pass criterion

Fixed before the first run, in ffeb9dc:

1. **Same images.** Warm, each viewpoint's relative RMSE against per-pixel
   exact cones is within 0.2 points of gate 8's.
2. **Speed.** Warm, each viewpoint takes at most 16.7 ms per frame: the wall
   time of 60 frames, ending with a GPU wait, divided by 60.
3. **Flight.** Gate 8's 600-frame flight from an empty cache:
   - mean frame at most 16.7 ms;
   - 99th-percentile frame at most 33.3 ms;
   - on average at most 10% of pixels fall back to unshadowed light.

**Verdict: FAIL, on the flight's 99th percentile.**

| Criterion | Result |
|---|---|
| Same images | pass |
| Warm speed | pass, 2.9–4.7 ms |
| Flight mean | pass, 12.8 ms |
| Flight fallback | pass, 1.6% |
| Flight 99th percentile | **fail**, 34.0 ms against 33.3 ms |

It is a near miss, and it is recorded as a fail with no re-run.

## Results

Apple M4 (MacBook Air, 24 GB), 999,986 boulders, 1280×720. Settings as gate
8 except the budget. Reproduce with `cargo run --release -p room3d --bin
m4_gate9`.

| Rays | Viewpoint | Frames until warm | Warm frame | RMSE vs per-pixel exact (gate 8) | Cached vs supersampled | Per-pixel exact vs supersampled | Pages |
|---|---|---|---|---|---|---|---|
| compute | standing | 177 | 13.95 ms | 14.49% (14.49%) | 14.27% | 12.69% | 18,093 |
| compute | 10 m up | 304 | 29.54 ms | 25.86% (25.86%) | 23.06% | 17.55% | 56,968 |
| compute | 40 m up | 291 | 37.26 ms | 30.88% (30.88%) | 25.28% | 20.11% | 51,554 |
| compute | 150 m up | 170 | 29.57 ms | 38.80% (38.80%) | 30.38% | 23.67% | 44,423 |
| hardware | standing | 177 | **2.86 ms** | 14.49% (14.49%) | 14.27% | 12.69% | 18,049 |
| hardware | 10 m up | 304 | **4.69 ms** | 25.87% (25.87%) | 23.06% | 17.54% | 57,350 |
| hardware | 40 m up | 291 | **4.73 ms** | 30.88% (30.88%) | 25.27% | 20.11% | 51,547 |
| hardware | 150 m up | 170 | **4.04 ms** | 38.80% (38.80%) | 30.38% | 23.65% | 44,438 |

- **Supersampled reference.** "Supersampled" is 4×4 exact cones per pixel,
  box filtered: the pixel's area.
- **Memory.** The cache holds 215 MB.
- **Warm-up.** Frames until warm went up from gate 8 (77–127) because the
  budget is smaller and requests are two frames late.

**Flight** (hardware rays, 600 frames from an empty cache):

| Measure | Result |
|---|---|
| Mean frame (wall time / 600) | 12.79 ms |
| 99th percentile | 34.00 ms |
| Worst frame | 91.87 ms (frame 81) |
| Fallback pixels, mean | 1.61% |
| Fallback pixels, worst | 50.09% (frame 0) |
| Planning on the CPU, mean | 11.67 ms |
| Planning on the CPU, max | 90.74 ms |
| Texels baked | 4.55 million |

Per-frame data is in `out/m4_gate9_flight.csv`. Frame time and the
planning time track each other almost exactly (correlation 1.00). In this
run, the planning time still included the wait for frame N − 2's readback.

## Diagnosis: the flight is bound by the CPU planner

After the gate, a flight-only diagnostic (`m4_gate9 -- --flight-only`) timed
the parts of each frame's CPU work separately. These are new counters:
`CacheStats::waiting`, `planning` and `evicting`.

| Measure | Run 1 | Run 2 |
|---|---|---|
| Mean frame | 24.8 ms | 21.9 ms |
| Waiting for the GPU (frame N − 2) | 1.67 ms | 1.67 ms |
| Planning | 20.6 ms | 18.2 ms |
| Eviction, within planning | 0.5 ms | 0.8 ms |

The same flight took 12.8 ms per frame in the gate run and 22–25 ms in these
runs, a run-to-run spread of about 2× on the fanless Air. The CPU work per
frame is nearly the same, so the spread is the machine's thermal state.
Report ranges, not single numbers, for CPU-bound work here.

What the planner does each frame, sampled every 25 frames of run 2:

- **Demand.** While flying, 25,000–32,000 request tiles per frame report a
  missing page. They add up to 28,000–53,000 distinct wanted pages, against
  1,000–2,000 pages the 8,000-cone budget can bake per frame.
- **Backlog.** Fallback stays low because coarse pages come first; most
  pixels are shaded from coarser pages than they want. But the backlog of
  wanted pages is re-derived from scratch every frame: 27,000–52,000 walk
  steps to coarser pages.
- **Cost of the walk.** It takes 5–28 ms of the planning time, about 0.2–0.6
  µs per step. The steps go through a page map that grows to 415,000
  resident pages.

**The GPU side fits.** In warm views, visibility, shading and readback take
2.9–4.7 ms. In flight, the GPU never made the CPU wait more than 1.7 ms on
average.

## What this means

- **Warm views are fast.** A million boulders (10^10 triangles) with exact
  soft sun shadows render at 720p in 2.9–4.7 ms per frame on a fanless M4
  Air. Gate 7 took 47–233 ms with per-pixel cones. This assumes the view has
  warmed, which takes 170–304 frames from an empty cache at this budget.
- **Flying is limited by the page manager's CPU work, not by rendering.** The
  fix is structural: stop re-deriving the backlog every frame.
  - The shader already walks to the finest resident mip. It could request the
    next page to refine (or the coarsest page when none is resident) instead
    of the wanted one, so the CPU never walks.
  - Requests could be deduplicated on the GPU: about 16 tiles of 4×4 pixels
    ask for the same page.
  - The backlog could persist between frames.
- **Quality is unchanged from gate 8.** At this resolution and density, the
  remaining error is sampling: one ray per pixel is 13–24% from the
  supersampled image. Anti-aliasing (roadmap item 4) addresses that.
- **The viewer.** The field scene uses the cache by default. In a smoke run
  at the full 2560×1440 Retina window, the median frame was 20.7 ms over 290
  frames, starting from an empty cache with the camera standing.
  - Caveat: `--frames N` runs stall when their window is hidden behind
    others. macOS stops presenting to occluded windows; this is not a renderer
    hang.
