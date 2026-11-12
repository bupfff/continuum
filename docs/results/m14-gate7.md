# M14 gate 7: frame-time stability at scale

**PASS.** Gate 6's world (36,864 unique rocks, camera and lighting levels
produced on demand) now flies with no frame over 16.7 ms: means 4.8–5.9
ms, 99th percentiles 8.3–10.1 ms, worst 15.8 ms. The cache stays
certified against a reference no single renderer could be (908,545 pages
compared, none differing), and a rock costs 17.8 KB of GPU memory.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Lighting levels against the slots / the CPU cache | 72.0× / 26.9× | | | at least 64× / 16× | PASS |
| 1. Frames over the slots / the cache | 0 / 0 of 29,177 | | | none | PASS |
| 2. Shown pages in the region compared in flight / differing | 322,661 / 0 | 302,580 / 0 | 238,061 / 0 | none differing (1e-5) | PASS |
| 2. Settled pages in common / differing | 13,804 / 0 | 18,058 / 0 | 13,381 / 0 | none differing | PASS |
| 2. Pages compared in all; longest shadow against the margin | 908,545; 7.89 m against 12 m | | | at least 100,000; under | PASS |
| 3. GPU memory per added rock | 17.8 KB, 0.086× a resident lighting level (207.1 KB) | | | at most 1/6 | PASS |
| 3. CPU heap per added rock | 14.9 KB, 0.051× a level as gate 4 kept it (291.7 KB) | | | at most 1/6 | PASS |
| 4. Mean frame | 5.86 ms | 5.26 ms | 4.76 ms | at most 16.7 ms | PASS |
| 4. 99th percentile | 10.09 ms | 9.08 ms | 8.30 ms | at most 16.7 ms | PASS |
| 4. Worst frame | 15.84 ms | 10.92 ms | 9.38 ms | at most 33.3 ms | PASS |
| Frames over 16.7 ms / 33.3 ms (reported) | 0 / 0 | 0 / 0 | 0 / 0 | | |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-12 (stub `c0c062d`; implementation and binary `dc377bf`,
before the run). The flights ran after the scene's build (157 s on every
core) and two minutes idle.

## Details

- **Per frame in flight** (means): camera-level streaming 0.4–0.5 ms of
  CPU (2.0–3.7 loads), lighting updates 0.5–1.0 ms, the cache's planning
  0.4–0.5 ms, waiting for the GPU 2.6–2.9 ms (the frames are now mostly
  GPU time). The slowest frames (flight 0: 15.8, 14.1, 13.2 ms) came with
  64 lighting loads each.
- **Lighting levels:** 97,339, 35,241 and 37,417 loads; produced 4.4,
  21.0 and 20.9 frames after their request on average (gate 6: 1.9, 16.7,
  13.1). With the producers at a lower priority, lighting arrives later on
  new ground, and views settle more slowly: first views in 927, 1,584 and
  2,702 frames; the last views, during certification, in 958, 4,791 and
  3,458.
- **Memory** as gate 6 (the changes hold no new data): 1,690 → 2,274 MB of
  GPU memory, 109 → 597 MB of CPU heap from 4,096 to 36,864 rocks.

## Method

Gate 6's flights replayed from views not yet settled (`stream_dev`
`SPIKES=192`) gave 58 frames over 33.3 ms in 5,400, their time spread over
unrelated stages (uploading levels, choosing evictions, trimming the CPU
cache, lighting updates, the cache's planning). The render thread was
being starved or blocked, so:

1. **Producer threads at utility quality of service** (macOS
   `pthread_set_qos_class_self_np`): the eight threads producing camera and
   lighting levels no longer compete with the render thread for the four
   performance cores.
2. **Frees off the render thread:** levels dropped from the CPU caches
   (megabytes each) are freed on a background thread.
3. **32 MB staging chunks** for camera and lighting uploads, so a frame's
   uploads rarely need a new mapped buffer.

**Disclosed (before the run):** the same replay with these changes gave 3
frames over 33.3 ms instead of 58 (one burst early in flight 0) and means
of 8.7, 8.8 and 7.1 ms instead of 12.6, 17.5 and 14.7. The workspace's
lints were brought in line with the newer clippy CI uses (two loops).

## What the numbers say

- **Landmark 1's world runs at 60 fps with headroom:** 36,864 unique
  rocks whose detail is produced on demand (lighting alone would need 7.6
  GB more GPU memory), every frame under 16.7 ms on a fanless laptop, the
  lighting exact against an independent reference.
- **The cost moved to settling:** lighting levels now arrive about 20
  frames after they are wanted on new ground, and a view can take several
  thousand frames to settle completely. Production throughput (more or
  faster workers at low priority, or lighting levels kept longer) is the
  next lever, with the sky and bounce light under streamed lighting.
