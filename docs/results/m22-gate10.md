# M22 gate 10: coarse texels average their area — PASS

**PASS.** Texels in the top three mips of their chart that curve take
their light from 16 points over their area instead of their centre alone.
The field seen from above comes 0.753× as far from Cycles (13.83% →
10.41%, without bounce light), its shortfall in brightness 0.736× the
current's; the field's start view 1.79% → 1.29% and the room 7.54% →
6.93% from Cycles; every other scene within its bar; speed side by side
within its bars, Sponza's settling exactly at its bar (1.25×).

Apple M4 (fanless MacBook Air), release build, run once at full size on
2026-11-22 (stub `d6c031f`; candidate and binary `f3a4d0b`, before the
run; gate 9, the same candidate, parked unrun: `docs/results/m22-gate9.md`).
12.9 minutes. Output: `experiments/room3d/out/m22_gate10_run.txt`.

## Criteria (fixed before the run)

| Criterion | Result | Bar | |
|---|---|---|---|
| 1. From above (no bounce here or in Cycles): error | 10.41% against 13.83% (0.753×) | at most 0.85× | PASS |
| 1. Mean short of Cycles' 0.0313 | 2.85% (0.0304) against 3.87% (0.0301): 0.736× | at most 0.8× | PASS |
| 2. Elsewhere (scorecard, bounce on) | see below | each within 0.2 points (Sponza 0.5) | PASS |
| 3. Speed (side by side) | see below | settled ≤ 1.05×, moving ≤ 1.10×, settling ≤ 1.25× | PASS |

| Scene | Against Cycles: current | Candidate | Bar |
|---|---|---|---|
| sponza | 25.53% | 25.35% | at most 26.03% |
| field | 1.79% | **1.29%** | at most 1.99% |
| room | 7.54% | **6.93%** | at most 7.74% |
| movers | 5.40% | 5.40% | at most 5.60% |
| helmet | 4.46% | 4.46% | at most 4.66% |
| fox | 2.17% | 2.17% | at most 2.37% |
| brainstem | 3.28% | 3.28% | at most 3.48% |

| Scene | Settled frame (ratio) | Moving frame (ratio) | Settled at: current, candidate (ratio) |
|---|---|---|---|
| sponza | 22.50, 22.50 ms (1.000×) | 27.22, 27.13 ms (0.997×) | 41.72 s, 52.15 s (**1.25×**, at the bar) |
| field | 10.48, 10.38 ms (0.991×) | 9.77, 10.02 ms (1.026×) | never, never |
| room | 5.20, 5.20 ms (1.000×) | 8.05, 8.06 ms (1.000×) | 1.90 s, 1.95 s (1.03×) |
| movers | 5.15, 5.16 ms (1.002×) | 20.48, 20.06 ms (0.979×) | 2.11 s, 1.58 s (0.75×) |
| helmet | 4.67, 4.66 ms (0.999×) | 6.63, 6.71 ms (1.012×) | 0.90 s, 0.78 s (0.86×) |
| fox | 5.02, 5.03 ms (1.001×) | 9.03, 8.99 ms (0.996×) | 1.38 s, 1.26 s (0.92×) |
| brainstem | 5.88, 5.89 ms (1.001×) | 31.93, 29.61 ms (0.927×) | 5.83 s, 5.19 s (0.89×) |

## Method

- **The candidate** (`CacheOptions::area_texels`, where sky samples use
  hardware rays: not with movers, streamed levels or alpha-blended
  objects). `vc_bake` lists a texel of a static light in the top three mips
  of its chart (tagged in the wide lights' list, `wide_list`, to stay
  within Metal's 29 storage buffers a stage); `hw_vc_bake_area`, a
  workgroup a texel, places 16 points over the texel's cell of the chart
  (4 × 4, each placed as a texel's centre is; points off the surface
  skipped) and keeps the texel's point sample unless one lies more than 30°
  from the centre's normal. A texel that curves: 4 rays a point toward the
  light (2 × 2 over its rectangle), the visible share over the points that
  face it, each weighted by its unshadowed light, replaces the cone's
  ratio. Its sky sample's 192 rays go from its points in turn (bin b from
  point b mod 16, around that point's normal).
- **The view from above:** `scorecard::field_from_above(40, 15, true)`:
  the scorecard's field, 40 m higher, 15° down, bounce light off here and
  Cycles at 0 bounces (direct light and the sky: what texels hold),
  measured as `scorecard::quality` measures (640×360, 4×4 supersampled,
  fixed bake budget 120,000, at most 600 frames).

## Development (disclosed before the run)

- The flat dark boulders (the viewer, 2026-11-22): each boulder is one
  chart (10,000 triangles, 3.57 × 1.46); its top texel's centre lies at
  0.52 of its height with a normal 0.36 down (`chart_centres`), so a
  boulder shown at that mip was unlit. With the candidate, all lit
  (`viewer_repro`, the start view).
- Gate 9's bar compared with Cycles at four bounces, where our one bounce
  (which never settles on the field) leaves the image 3–12% short; parked.
- A rehearsal averaging every coarse texel (flat ones too) failed on
  Sponza: 25.53% → 27.55% (its walls' coarse texels, read by bounce rays,
  averaged darker; the image 6% darker, without bounce 0.9996×; on the
  field the near ground's bounce darker too, 0.123 → 0.068 there). Only
  texels that curve are averaged since; why flat texels averaged read
  darker through bounce is not known.
- The second rehearsal (Sponza and the field) passed every criterion.
