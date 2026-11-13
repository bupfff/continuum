# M14 gate 12: coarser levels shade with the finest's normals

**FAIL on criterion 3** (a fault of the method, not the change); criteria 1
and 2 pass. Coarser rock levels now carry the finest surface's normals at
their vertices: settled images streamed are 0.56–0.65× as far from those
with every finest level resident, and producing the levels got cheaper.
Criterion 3, the resident (finest-only) images identical with and without
the change, failed at one view by 0.0045%, from the time-based bake budget's
run-to-run variation; with a fixed budget they are identical (diagnostic
below). Gate 13 repeats the gate with a fixed budget.

| Criterion | View 0 | View 1 | View 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Streamed against resident, finest's normals / levels' own | 1.484% / 2.287% (0.649×) | 1.246% / 1.945% (0.641×) | 1.424% / 2.538% (0.561×) | at most 0.75× | PASS |
| 2. Levels 1 and 2 produced, per rock | 2.30 ms against 2.66 ms (0.87×) | | | at most 1.5× | PASS |
| 3. Resident images apart | 0.000000% | **0.004465%** | 0.000000% | identical | FAIL |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `cacc89a`; implementation and binary `f11b00a`,
before the run). M14 gate 1's field (256 unique rocks 6 m apart, sun only),
a renderer streaming camera levels within 2.6 million triangles against one
with every finest level resident, settled at the three flights' last views
and averaged over 4×4 sub-pixel offsets, default cache options; a first
pass rendered and discarded.

## Diagnostic after the run (not the gate)

- The same binary with both passes' rocks unchanged (`M14_SAME=1`): the
  resident images identical at all three views.
- With a fixed bake budget (`M14_FIXED=1`, `frame_target_ms` 0) and the
  change: the resident images identical at all three views, criteria 1 and 2
  as above (0.649×, 0.641×, 0.561×; 0.87×).

The resident renderer loads only finest levels, which the change does not
touch (the finest level's code path is unchanged). With the time-based
budget, the first pass's different streamed work shifts the GPU's timing,
and the resident cache bakes in a slightly different order.

## Method

`Detail::finest_normals` (on by default): a rock's camera levels coarser
than the finest take the finest surface's unit normal at each grid vertex,
from the displacement's derivatives (`Shape::finest_normal`, checked within
0.23° of central differences), computed once per grid vertex; these levels
skip their own smooth normals (`Surface::parametric_unshaded`), which made
production cheaper. Positions are unchanged.

**Disclosed (before the run):** the development path (averaged normals with
a length factor in the shader, then one sample per vertex, then derivatives
and caching) and the method fixes after the development mode (median of five
production rounds; a discarded first pass) are in the binary.

## What the numbers say

- The colour gap between streamed and fully resident rocks falls by a third
  to nearly half, from the dropped octaves' tilt restored at the coarse
  levels' vertices; the length of averaged normals (a Toksvig-like factor)
  added nothing measurable here.
- Equality checks need a fixed bake budget: with the time-based one, a
  cache's settled images vary a little with the machine's timing.
