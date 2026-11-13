# M14 gate 15: coarser levels for meshes that come without them

**PASS.** Any surface can now be given coarser levels
(`continuum::simplify::levels`; `InstancedScene::simplify_geometries` for a
scene's imported meshes). On 256 unique rocks whose camera levels 1 and 2
were replaced by levels simplified from their finest, streamed images stay
as close to the finest-resident ones as with the rocks' analytic levels,
and every level's stated error holds against a dense measurement.

| Criterion | View 0 | View 1 | View 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Streamed against resident: simplified / analytic levels | 1.435% / 1.484% (0.967×) | 1.375% / 1.246% (1.103×) | 1.103% / 1.425% (0.774×) | at most 1.25× | PASS |
| 2. Dense distance over stated error, 512 levels | at most **0.490×** | | | at most 1.1× | PASS |
| Reported: resident images apart | 0.000000% | 0.000000% | 0.000000% | | |
| Reported: lighting level simplified too, from the analytic resident image | 6.399% | 3.758% | 6.106% | | |

| Per rock (average) | Level 1 | Level 2 | Lighting level |
|---|---|---|---|
| Triangles, simplified (analytic) | 4,096 (3,968) | 16,384 (16,132) | 1,023 (960) |
| Stated error, simplified | 0.0383 m | 0.0199 m | 0.0743 m |
| Stated error, analytic | 0.0322 m | 0.0103 m | 0.0952 m |

Simplification: 1.45 s a rock on one thread (65,536 triangles to 16,384,
4,096 and 1,024, with the distance bounds); lighting levels' dense distance
at most 0.94× their stated error.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub `931cf2b`; implementation and binary `e47b020`,
before the run). Gate 13's scene and renderers (256 unique rocks, sun only,
a 2.6-million-triangle streaming budget against every finest level
resident, fixed bake budget, the three flights' last views over 4×4
sub-pixel offsets, a first pass discarded).

## Method

`continuum::simplify::levels(surface, targets)`:
- **Half-edge collapses ordered by quadric error** (Garland & Heckbert
  1997): a vertex moves onto a neighbour, so a coarser level keeps a subset
  of the finest's vertices with their shading normals (the finest's normals
  at its vertices, as gate 13's rock levels carry), chart coordinates and
  texture coordinates.
- **Seams and borders kept:** an edge where chart, chart or texture
  coordinates or the shading normal change is a seam; a vertex on one seam
  or border moves only along it, a vertex where they meet never moves, and
  their quadrics hold heavily weighted planes along them.
- **Rejected collapses:** any that would break manifoldness (the link
  condition), fold a triangle in space (its normal turning by more than
  about 75°) or in chart coordinates.
- **A distance bound:** each level's triangles are measured on grids a
  quarter of their mean edge apart to the finest's nearest triangle, and
  the finest's on grids half of theirs to the level's, each triangle adding
  the farthest any of its points lies from its grid (distance to a surface
  changes no faster than the point moves).
- **The error level choice uses:** the larger of that bound and half the
  level's mean edge.

**Disclosed (before the run):** with the stated error the largest distance
found from 7 sample points a triangle, the development mode (36 rocks)
failed both criteria: simplified levels stated about a third of the analytic
levels' error at the same counts, so the streamer showed them where their
triangles were several pixels across (2.4–3.4× the analytic colour
difference), and the dense check found 1.12× the stated distance. The
simplifier was written before the stub but not run.

**Also disclosed (after the run):** the run's executable was built just
before a reporting line (each view's streamed working set) was added to the
committed binary, so the gate did not print working sets; criteria and
method are the same. The development mode, run afterwards with that line:
streamed working sets of 1.19, 1.57 and 1.96 million triangles with the
simplified levels against 0.63, 0.83 and 1.35 million with the analytic ones
(36 rocks, the budget not reached).

## What the numbers say

- **Imported meshes can stream now.** A mesh with one level gets a chain of
  coarser ones whose stated errors hold (dense distances under half of
  them), with the finest's normals at their vertices.
- **The error that works for shading is about triangle size, not only
  distance.** Quadric collapses keep the shape with a third of the analytic
  levels' distance, but a level then shown by distance alone has triangles
  several pixels across; with half the mean edge in the error, levels show
  where triangles are about a pixel across, and shade as well as the
  analytic ones. Where the budget allows, that uses more triangles than the
  analytic levels (1.45–1.9× in development); in the gate both share the
  2.6-million-triangle budget.
- **Lighting levels are the open part.** With the lighting level simplified
  too, images are 3.8–6.4% from the analytic resident ones: shadows cast by
  a different coarse shape (the cache bakes on the lighting level). That
  compares two approximations, not one against the truth; a lighting-level
  gate needs a reference with the finest as lighting.

## Caveats

- One family of shapes (smooth rocks with one parametric chart); textured
  and many-chart imports are not yet measured.
- 1.45 s a rock is slow for large imports (about 23 s a million
  triangles, estimated); it runs once at load.
