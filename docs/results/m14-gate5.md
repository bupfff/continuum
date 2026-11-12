# M14 gate 5: lighting levels produced on demand, certified beyond one renderer

**FAIL** (on the frame budget only). Over 36,864 unique rocks (9× gate 4's
field), every lighting level is now produced on demand by worker threads
into a 256 MB CPU cache, and only a small record per rock stays resident
(27.7 KB of GPU memory, 15.2 KB of CPU heap). The cache stays certified
against a reference no single renderer could be: 676,080 pages compared,
none differing. But the flights run at 26–39 ms a frame against 16.7 ms:
the lighting slots churn at about 60 loads a frame (gate 4: 13–15).

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Lighting levels against the slots | 72.0× (35,389,440 / 491,520 triangles) | | | at least 64× | PASS |
| 1. Lighting packages against the CPU cache | 26.9× (7.21 GB / 256 MB) | | | at least 16× | PASS |
| 1. Frames over the slots / the cache | 0 / 0 of 17,145 | | | none | PASS |
| 2. Shown pages in the region compared in flight / differing | 162,372 / 0 | 212,987 / 0 | 252,222 / 0 | none differing (1e-5) | PASS |
| 2. Settled pages in the region in common / differing | 14,824 / 0 | 19,421 / 0 | 14,254 / 0 | none differing | PASS |
| 2. Pages compared in all | 676,080 | | | at least 100,000 | PASS |
| 2. Longest sun shadow against the margin | 7.89 m against 12 m | | | under the margin | PASS |
| 3. GPU memory per added rock | 27.7 KB, 0.128× a resident lighting level (216.8 KB) | | | at most 1/6 | PASS |
| 3. CPU heap per added rock | 15.2 KB, 0.052× a level as gate 4 kept it (291.7 KB) | | | at most 1/6 | PASS |
| 4. Mean frame | **26.23 ms** | **35.44 ms** | **38.88 ms** | at most 16.7 ms | FAIL |
| 4. 99th percentile | **50.51 ms** | **58.92 ms** | **66.37 ms** | at most 16.7 ms | FAIL |
| 4. Worst frame | **67.25 ms** | **90.32 ms** | **86.59 ms** | at most 33.3 ms | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-12 (stub `cec5b64`; implementation and binary `0ebe15f`,
before the run). The flights ran after the scene's build (153 s on every
core) and two minutes idle; a thermal slowdown of up to 2× would not
explain 4–6× gate 4's frame times (5.4–6.9 ms).

## Details

- **Scene:** 192 × 192 rocks 6 m apart (a 1,152 m square), each with a
  proxy (48 triangles); the scene holds each rock's chart, none of its
  triangles. Lighting levels (960 triangles, 196 KB packaged) and camera
  levels both come from the rocks' source. Gate 4's lighting slots (491,520
  triangles, 511 levels) and camera budget (8 × 2²⁰ triangles); the sun.
  Built in 153 s (mostly the levels' errors); the renderer in 10.3 s,
  including the stubs, written by producing every lighting level once.
- **Flights:** gate 4's three, at its speed, over the central 384 m; from
  there the camera sees across the whole field.
- **Lighting levels in flight:** 111,398, 95,424 and 61,385 loads (62, 53
  and 34 a frame, against a cap of 64), as many evictions; 47,348, 82,848
  and 53,407 produced, about one frame after their request; 5,860,026,
  1,591,831 and 213,622 wants refused; at most 478–498 levels pinned; the
  CPU cache at its cap (268 MB) throughout. 667,951 levels produced in the
  whole run. Settled first views in 831, 1,313 and 413 frames (gate 4:
  115–384).
- **Reference:** the 4,941 rocks within 212 m of the centre, every
  lighting level resident (2,558 MB of GPU memory), replaying each flight
  in step. Compared: pages on rocks within 200 m of the centre, and ground
  pages whose texels' box (widened by two texels) lies within that square.
  The longest shadow a rock's lighting level can cast toward a compared
  texel is 7.89 m (the sun at least 27.9° up over the region, 1 km away;
  rocks reach at most 1.59 m from their centres; 2.47 m from the lowest
  point to the highest), so no rock outside the reference can shadow a
  compared page. Every compared page was identical (worst difference 0),
  although the reference's object hierarchy differs.
- **Memory:** from 4,096 to 36,864 rocks the renderer at creation grew
  from 1,696 to 2,603 MB of GPU memory and from 110 to 609 MB of CPU heap
  (scene included). A resident lighting level costs 216.8 KB of GPU memory
  (1,024 → 4,096 rocks, every level resident: 1,709 → 2,375 MB). Every
  lighting level resident at 36,864 rocks would take about 8.0 GB more.
  Most of a rock's record is its proxy (triangles and hardware structure)
  and its chart's cells.

## Method

1. **Packages built alone** (`stream::lit_package`): a geometry's
   triangles, hierarchy and coverage, chart coordinates and cell lists
   with triangles, nodes and entries counted from zero, exactly what the
   whole scene's buffers would hold for it rebased (unit test
   `a_lighting_package_built_alone_matches_the_scene_buffers`).
2. **Records at creation:** a streamed geometry whose surface holds no
   triangles takes its lighting level from `LevelSource::lighting`; the
   renderer produces each once, on every core, and keeps its record (the
   root node and its coverage as the stub, its bounds, its sizes).
3. **On demand:** a wanted level not in CPU memory is asked of worker
   threads (`StreamOptions::workers`) and loads when it arrives; produced
   packages stay in a CPU cache capped at
   `StreamOptions::lighting_cache_bytes`, the least recently wanted
   dropped first. Gate 4's policy is unchanged (pages wait for every level
   their bakes found absent; wanted levels protected oldest first; at most
   64 loads a frame).

**Disclosed (before the run):** the shadow's reach is computed from each
rock's lighting level, not a sphere around its box's corner (13.1 m in
development against the 12 m margin); the CPU cache is checked after each
frame's trim. The development run (2,304 rocks, a smaller region) gave 0
differing of 358,969 pages, 31.3 KB (0.144×) and 17.5 KB (0.060×) a rock,
and flights of 5.9–6.6 ms mean. Gate 4's packaged way still matches
(`stream_dev` `PAGES`, 0 differing).

## What the numbers say

- **On-demand lighting works and is exact at scale:** a field whose
  lighting levels would take 8 GB of GPU memory and 7.2 GB packaged runs
  on 2.6 GB, with CPU memory for lighting capped, and every compared page
  matches an independent reference.
- **A unique object's resident cost is now about 28 KB of GPU memory**,
  mostly its proxy, so a million unique objects would need about 28 GB:
  the next limit on object count, after this gate's speed.
- **Speed fails:** the lighting slots churn (34–62 loads a frame, at the
  per-frame cap in flight 0), and settling slows 2–7×. Where the time goes
  (uploads, the cache's planning over tens of thousands of waiting pages,
  or work that grows with the larger field and object count regardless of
  lighting streaming) is not yet measured; that comes first in the next
  gate.

## After the run (diagnosis, not part of the gate)

`stream_dev` `GATE5=192` (flights 0 and 1, 900 frames each, after 600
settling): with 4× and 16× the lighting slots, lighting loads fall to
13–31 a frame but frames stay 29–47 ms, so lighting churn is not the main
cost. Per frame: camera-level streaming 7–19 ms of CPU (with only 4–14
loads), lighting updates 2.3–4.5 ms (about 150 µs a load, mostly
uploads), waiting for the GPU 3–6 ms, GPU camera passes 4–8 ms, and
roughly 15 ms more unaccounted. A `sample` profile at 16,384 rocks shows
the main thread mostly waiting on mutexes (the workers producing levels:
`Surface::parametric`'s `smooth_normals` hashing, `lit_package`'s
hierarchy), plus `Streamer::begin_frame` and wgpu's acceleration-structure
validation. Suspects for the next gate: per-frame scans over every
geometry in the streamer (eviction choice, CPU-cache trim, statistics),
the top-level structure rebuilt over all 36,864 instances on each load,
and lock contention between the render thread and level production.
