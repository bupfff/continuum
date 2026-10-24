# Flatland gate 4: the worst case, everything moving

Date: 2026-10-24 (overnight run). Design rule 1 says the incremental path must
never cost meaningfully more than a conventional full recompute, even when
nothing can be reused.

Pass criterion, fixed before the first run: in every configuration, at ε = 0
and ε = 1%, zero soundness violations and incremental time at most 1.1× the full
recompute time (same thread count).

**Verdict: FAIL**, because of one configuration. Geometric-bound charging meets
the rule with margin (0.90–0.95× a full recompute). Projected charging costs
1.84–1.96× in the wide-cone worst case.

## Setup

All 41 occluders drift every frame (no statics), under two lights (direct, 256²
receivers) or under the sky (8 wide cones, 128² receivers), with the per-tile
budget policy. 10 threads for both paths. Reproduce with
`cargo run --release -p flatland --bin gate4`.

## Results

| Scenario | Visibility | Charging | ε | Violations | Cones recomputed | Incremental / full time |
|---|---|---|---|---|---|---|
| Direct light | binary | geometric | 0% | 0 | 75.8% | 0.95 |
| Direct light | binary | geometric | 1% | 0 | 75.8% | 0.94 |
| Direct light | continuous | geometric | 0% | 0 | 75.8% | 0.95 |
| Direct light | continuous | geometric | 1% | 0 | 74.3% | 0.94 |
| Sky | continuous | geometric | 0% | 0 | 66.7% | 0.94 |
| Sky | continuous | geometric | 1% | 0 | 61.1% | 0.90 |
| Sky | continuous | projected | 0% | 0 | 66.7% | **1.96** |
| Sky | continuous | projected | 1% | 0 | 50.5% | **1.84** |

## What this shows

1. **Geometric charging meets rule 1 even when everything moves.** Charging is
   cheap, and tiles with a clear view of the light (or of open sky in a sector)
   still need no work, so the incremental path stays slightly faster than a
   full recompute.
2. **Projected charging breaks rule 1 under heavy motion.** Its cost grows as
   movers × receivers in each charged tile. With 41 movers that exceeds the cost
   of simply recomputing, even though it still recomputes fewer cones (50.5% vs
   61.1% at ε = 1%).

## Fix (not tried; recorded for the next session)

Charging must never cost more than the recompute it is trying to avoid. A
per-tile rule: estimate the projected-charge cost (movers touching the tile ×
receivers) against the recompute cost (receivers × occluders), and when
charging would cost more, use the geometric bound or mark the tile for
recompute directly. Projected charges then keep their gate 5 advantage in
typical scenes and fall back gracefully under heavy motion.
