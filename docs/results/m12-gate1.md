# M12 gate 1: deforming geometry

**FAIL on cost.** The cache stays exact (criterion 1) and the refit
geometry is as accurate as a fresh build (criterion 2), but frames with
the blob deforming cost about 10× still ones (criterion 3).

| Criterion | Result | Verdict |
|---|---|---|
| 1. Exact: 30 `verify_cache` checks over 300 frames | 0 texel values differing (pages checked: 1,640 → 6,781) | PASS |
| 2. Accuracy at frame 150's pose: per-pixel cones (16×16, 320×180) vs Cycles (4,096 spp), surfaces | **1.08%** (bar 3%; 313 pixels apart) | PASS |
| 3. Cost: deforming runs, 600 frames from an empty cache | means **113.55, 110.68, 107.59 ms**: 9.6–10.1× the still runs' 11.23 ms (bar 1.25× and 16.7 ms); 99th percentiles 167.7–180.6 ms (bar 33.3); fallback 0.23–0.37% | FAIL |

- **Still runs:** means 11.02, 11.98 and 10.68 ms; 99th percentiles
  31.33, 31.35 and 32.89 ms; fallback 0.26–0.27%.
- **Reported:** per-pixel cones through the refit against a renderer
  built with frame 150's pose as static geometry: 0.000% over surfaces.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (a million boulders). The run's lines are in
`docs/results/m12-gate1-run.txt`.

## Method (DESIGN §3.4, "Deforming geometry")

- **The API:**
  - `RendererOptions::deforming` lists geometries whose vertices change,
    and their instances become movable.
  - `Renderer::set_vertices(geometry, triangles, normals)` poses one from
    the next frame. Topology, charts and texture coordinates stay the rest
    pose's, so a texel follows its triangle: its point is computed from
    the triangles at bake time.
- **Per pose:**
  - **Upload:** triangles and normals (keeping vertex colours' and
    tangents' words).
  - **Refit:** the rest pose's triangle hierarchy on the CPU: boxes and
    the 27 directional coverage samples, the same tree and leaf order
    (`hierarchy::refit_triangle_hierarchy`).
  - **Hardware structure:** a fresh one per pose, built in a submission of
    its own before the frame's. Built in the frame's encoder before the
    top-level build that places it, it was never hit: the blob vanished
    (development). The last few are kept for frames in flight.
  - **Bounds:** the geometry's new box gives its instances' world boxes,
    which flow through M9 G1's movers path (subtree, movers' structure,
    stale pass, same-frame re-bake).
- **Coarser pages:** a deforming object's cache pages take one mip coarser
  (`DEFORMING_MIPS`, the object's albedo w read by the shading), a quarter
  of the texels to re-bake.

## Why the cost fails

Exactness means re-baking, every frame, every texel whose cone toward a
light can meet the blob's old or new box:
- **The blob's own pages.**
- **Receivers in its shadows** toward the three lights: the ground and
  boulders around it, out to its shadows' length.

In development on 10,000 boulders the re-bake grew from about 10 to 4,200
texel–light cones a frame by frame 95, mostly receivers. Each costs about
5 µs: a cone passing near the blob walks its 16,384 triangles at prefilter
1 bin. Over 600 frames at full size, more pages near the blob become
resident and the re-bake keeps growing: 108–114 ms means. Making the
blob's own pages a mip coarser barely helped (1.95× → 1.79× in
development); the receivers are most of it.

M9 G1's rigid movers cost 1.06–1.09× because they were small and far
(64 boulders along a flight). A deforming object near the camera with
large shadows is the opposite case.

**What would fix it:** not re-baking what a pose barely changes. A pose
moves the surface a few millimetres a frame, so a texel's light changes
by about the area its shadow edge sweeps, not by the whole box. This is
the ledger's bounded deferral (M9 G6 for a moving light): a per-texel
bound on the swept change, re-baking past ε. Two cheaper alternatives,
neither exact:
- re-bake stale texels for deforming objects under the frame's bake
  budget, read pages first, and report the error;
- prefilter coarser for movers' shadows.

## Disclosed

- **Development before the run** (in the binary's doc comment):
  - the own-submission build;
  - the coarser pages;
  - the dev numbers on 10,000 boulders (criterion 3 failing at
    1.79–1.95×).
- **Criterion 2** is the same 1.08% as in development on 10,000
  boulders: the 320×180 view's near field, the blob and its surroundings,
  is the same.
- **The machine:** the run went alone.
