# M14 gate 10: bounce rays through hardware ray queries, and the frame budget with a sky and bounce at scale

**PASS.** With a sky and bounce light, 36,864 unique rocks (camera and
lighting levels produced on demand) now fly within 60 fps: means 9.2–10.2
ms, 99th percentile 14.5–15.0 ms. Bounce rays trace through the camera's
hardware structures instead of the lighting hierarchy, so they never need a
lighting level; direct light stays certified, bounce converges, and bounce
accuracy against Cycles is unchanged (3.09%, against gate 22's 3.07%).

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. (b) Mean frame | 9.18 ms | 10.18 ms | 9.61 ms | at most 16.7 ms | PASS |
| 1. (b) 99th percentile | 14.60 ms | 15.00 ms | 14.50 ms | at most 16.7 ms | PASS |
| 1. (b) Worst frame | 17.73 ms | 17.69 ms | 28.31 ms | at most 33.3 ms | PASS |
| 1. (b) Frames over 16.7 ms (reported) | 4 | 2 | 3 | | |
| 2. (a) Sun / sky pages compared in flight, differing | 23,992, 0 | 52,213, 0 | 64,257, 0 | none differing (1e-5) | PASS |
| 2. (a) Settled shown pages compared, differing | 16,891, 0 | 680, 0 | 19,342, 0 | none differing | PASS |
| 2. Pages compared in all | 177,375 sun and 177,375 sky | | | at least 100,000 and 30,000 | PASS |
| 3. (a) Settled image's mean difference | 0.067% | 0.577% | 0.025% | at most 1% | PASS |
| 4. M10 gate 22's scene against Cycles over surfaces | 3.09% | | | at most 5% | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `da9e63e`; implementation and binary `64cc04d`,
before the run), after the field's build (144 s) and two minutes idle.

## Details

- **Flights (b):** bake budget 1,068–3,486 units on average (at its cap),
  11–35 pages baked a frame; lighting levels 44,528–81,404 loads and
  35,695–122,528 produced a flight (gate 9: up to 192,584 produced, many
  for bounce samples). The frames are GPU-bound at about 10 ms.
- **Settling (a):** flight 1's last view settled in neither renderer within
  8,000 frames (680 shown pages in common); flights 0 and 2 settled here in
  8,000 and 6,026 frames, there in 5,592 and 4,504.
- **Accuracy:** gate 22's night scene (four lights, one bounce, no
  streaming), supersampled 4×4, against its cached Cycles reference (4,096
  samples per pixel), the lamps' disks set apart as there.

## Method

1. **Bounce rays through hardware ray queries** (`hw_vc_bake_indirect`):
   where camera rays use them, bounce samples trace their 192 rays through
   the camera's top-level structure (camera levels, clusters of proxies)
   and shade the hit as before (its page, albedo, emission). They never
   need a lighting level, so bounce samples want none. Where the sample's
   own object shows a camera level or a proxy, hits on it are skipped
   (that surface differs from the lighting level the sample sits on by up
   to the level's error). The compute bake and the hardware one share the
   per-bin gathering (`bounce_gather`) and the reduction.
2. **The frame budget at 11 ms** (`CacheOptions::frame_budget_ms`' new
   default): gate 9's cap on the bake budget, aimed lower so the frames'
   variance under heavy streaming (about 5 ms) stays within 16.7 ms.

**Disclosed (before the run):** gate 9's flights in `stream_dev`
`FLIGHTS9` with hardware bounce gave 12.7–12.9 ms mean (99th percentile
19.2–19.7 ms) at a 15 ms budget and 9.2–11.0 ms at 11 ms (99th percentiles
14.6/15.6/18.2 ms in one run, 15.0/15.1/14.9 in another); the development
run passed with views settling and bounce within 0.02–0.03%.

## What the numbers say

- **Landmark 1's world now has a sky and bounce light within 60 fps:**
  36,864 unique rocks, detail produced on demand, direct light certified
  against an independent reference, bounce within half a percent settled
  and within 3.1% of Cycles on a reference scene.
- **Bounce through the camera's structures** is both faster and frees
  lighting streaming from far-reaching rays; it applies to every scene with
  hardware camera rays, streamed or not.
- **Left:** frames are GPU-bound at about 10 ms, so the bake runs near its
  floor in heavy flights (11–35 pages a frame) and views settle slowly.
