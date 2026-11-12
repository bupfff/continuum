# M14 gate 6: gate 5's world within 60 fps

**FAIL** (on one frame). Gate 5's field of 36,864 unique rocks, with
lighting levels produced on demand, now flies at 6.2–6.5 ms a frame on
average (gate 5: 26–39 ms), with the 99th percentile at 10.2–11.9 ms. But
one frame of flight 2 took 42.3 ms against a 33.3 ms bar. Every other
criterion passed: the cache is still certified (1,416,782 pages compared,
none differing), and a rock's resident record shrank to 17.8 KB of GPU
memory.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Lighting levels against the slots / the CPU cache | 72.0× / 26.9× | | | at least 64× / 16× | PASS |
| 1. Frames over the slots / the cache | 0 / 0 of 21,012 | | | none | PASS |
| 2. Shown pages in the region compared in flight / differing | 515,324 / 0 | 470,184 / 0 | 385,408 / 0 | none differing (1e-5) | PASS |
| 2. Settled pages in common / differing | 13,831 / 0 | 18,516 / 0 | 13,519 / 0 | none differing | PASS |
| 2. Pages compared in all; longest shadow against the margin | 1,416,782; 7.89 m against 12 m | | | at least 100,000; under | PASS |
| 3. GPU memory per added rock | 17.8 KB, 0.086× a resident lighting level (207.1 KB) | | | at most 1/6 | PASS |
| 3. CPU heap per added rock | 14.9 KB, 0.051× a level as gate 4 kept it (291.7 KB) | | | at most 1/6 | PASS |
| 4. Mean frame | 6.16 ms | 6.53 ms | 6.18 ms | at most 16.7 ms | PASS |
| 4. 99th percentile | 10.17 ms | 11.92 ms | 11.15 ms | at most 16.7 ms | PASS |
| 4. Worst frame | 13.20 ms | 17.11 ms | **42.30 ms** | at most 33.3 ms | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-12 (stub `54aa3ab`; implementation and binary `5f37f4b`,
before the run). The flights ran after the scene's build (163 s on every
core) and two minutes idle.

## Details

- **Per frame in flight** (means): camera-level streaming 0.7–1.4 ms of
  CPU (2.4–4.4 loads), lighting updates 0.9–1.2 ms, the cache's planning
  0.5–1.0 ms, waiting for the GPU 0.8–1.8 ms. The pixel error never
  relaxed (the view wanted less than the camera budget).
- **Lighting levels:** 111,189, 45,975 and 48,516 loads; produced 1.9,
  16.7 and 13.1 frames after their request on average (the workers fall
  behind in flights 1 and 2); first views settled in 927, 1,313 and 2,036
  frames (gate 5: 413–1,313).
- **Memory:** 1,673 → 2,257 MB of GPU memory and 109 → 597 MB of CPU heap
  from 4,096 to 36,864 rocks; a resident lighting level 207.1 KB (1,024 →
  4,096 rocks: 1,716 → 2,353 MB). Gate 5's 27.7 KB a rock fell to 17.8 KB
  with per-rock proxy structures replaced by clusters.
- **Reference:** the 4,941 rocks within 212 m, every lighting level
  resident (2,527 MB), as in gate 5.

## Method

Profiled first (`sample` on the render thread, `stream_dev` `GATE5`): the
cost grew with the number of objects and uploads, not with the view.

1. **Lighting uploads through a staging belt** in the frame's encoder:
   `queue.write_buffer` made (and later freed) a Metal buffer per call,
   about 30% of the render thread.
2. **A persistent residency set** (patched `wgpu-hal`): one set per
   device, updated by difference, committed only when it changes; each
   submission had built and committed a new one holding every structure.
3. **Narrower scans in the camera streamer:** lists of the geometries
   seen in the latest feedback, of those holding levels and of produced
   levels in CPU memory, and a running total of resident triangles.
4. **Incremental top-level builds** (patched `wgpu-core`): only changed
   instances are resolved and encoded (no gain alone: other per-instance
   passes in wgpu remained).
5. **Clusters of proxies:** objects showing their proxy are seen through
   clusters of 32 (Morton order of position, one structure each, built
   once); an object showing a finer level takes one of a fixed number of
   top-level slots and its cluster is rebuilt with its triangles collapsed
   onto one vertex. A hit on a cluster names its object through a member
   list in the objects' buffer. The top level holds about 5,200 instances
   instead of 36,864. Retired cluster structures are reused.
6. `StreamOptions::fit_budget` relaxes the pixel error while a view wants
   more than the camera budget (the quality contract); it never engaged.

**Disclosed (before the run):** at 16,384 rocks in one thermal state,
clusters of 32 gave 6.6 and 9.2 ms against 12.6 and 17.5 ms with one
proxy per instance; two development bugs were fixed before the run
(degenerate triangles placed far below the field stretched clusters'
bounds; the Morton code scaled each axis by its own range, scattering
clusters). The settled image differs from gate 5's code in 386 of 921,600
pixels, along lines where rocks meet the ground (proxies placed in the
world on the CPU rather than by an instance transform).

## What the numbers say

- **Landmark 1's mechanism holds at 9× gate 4's field within a 60 fps
  mean and 99th percentile:** unique camera and lighting levels produced
  on demand, the cache exact, about 18 KB of GPU memory a rock.
- **Left: one 42 ms frame in 5,400,** and lighting levels arriving 13–17
  frames late when flights reach new ground (the workers' throughput).
  Likely spike sources, unmeasured: the residency set's periodic removal of
  retired structures (a large commit every 64 submissions), a burst of
  loads with cluster rebuilds, or the CPU cache's trim. A stability gate
  measures and bounds them next.
