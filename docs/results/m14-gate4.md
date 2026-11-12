# M14 gate 4: lighting levels streamed, the cache still certified

**PASS.** On gate 3's 4,096 unique rocks, the lighting levels (what mask
cones, bakes and texel lookups see) now stream into slots holding an
eighth of them. Every cache page shown in three flights and at their
settled views is bit-identical to a renderer with every lighting level
resident (2.68 million page comparisons, none differing). GPU memory at
creation is 0.486× gate 3's way, and flights stay within a 60 fps budget.

| Criterion | Flight 0 | Flight 1 | Flight 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Lighting levels against the lighting budget | 8.00× (3,932,164 / 491,520 triangles) | | | at least 8× | PASS |
| 1. Frames over the lighting budget | 0 of 12,439 | | | none | PASS |
| 2. Shown pages compared in flight / differing | 824,868 / 0 | 965,621 / 0 | 887,970 / 0 | none differing (1e-5) | PASS |
| 2. Settled pages in common / differing | 14,511 / 0 | 19,063 / 0 | 16,138 / 0 | none differing | PASS |
| 3. GPU memory at creation | 1,705 MB against 3,509 MB: 0.486× | | | at most 0.5× | PASS |
| 4. Mean frame | 5.36 ms | 6.76 ms | 6.92 ms | at most 16.7 ms | PASS |
| 4. 99th percentile | 10.29 ms | 12.75 ms | 13.31 ms | at most 16.7 ms | PASS |
| 4. Worst frame | 12.35 ms | 18.23 ms | 18.05 ms | at most 33.3 ms | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-12 (stub `055f087`; method and binary `1740e01`, before
the run).

## Details

- **Scene:** gate 3's 64 × 64 rocks (camera levels produced on demand into
  an 8 × 2²⁰-triangle budget), each with a proxy level (8 × 4 segments, 48
  triangles), lighting levels (about 984 triangles each) streamed into 499
  slots (an eighth). Sun only.
- **Lighting levels in flight:** 24,329, 26,928 and 22,922 loads (as many
  evictions; 13–15 a frame), at most 475–488 levels pinned; 71,196,
  206,112 and 24,740 wants refused (beyond what the slots hold). Settled
  first views in 384, 257 and 115 frames (gate 3: 107–126). Pages hidden
  while waiting, at the sampled frames: 463, 1,357, 338.
- **Settled views:** here 16,626, 20,189, 16,304 pages; the reference
  14,541, 19,206, 16,146 (the two renderers' views took different paths
  to their settled state; every page they share is identical).
- **Memory:** the remaining 1,705 MB is mostly what both renderers share
  (the cache's pool and the camera levels' pool of 8 × 2²⁰ triangles); the
  margin under 0.5× is thin (0.014×), and owes about 145 MB to skipping the
  hardware vertices of proxied geometries (a fix made before the run).

## Method

1. **Proxy levels:** each streamed geometry gets an always-resident proxy
   (`Levels::proxy`) packed at the camera pool's start, its structure built
   once; camera rays see it until a finer level arrives, so they never see
   lighting levels, which need no hardware structure.
2. **Lighting slots** (`StreamOptions::lighting_triangles`,
   `stream::LightingPool`): the streamed geometries (the scene's last)
   leave every per-triangle and hierarchy buffer for packages; each keeps
   a *stub* node (its root's bounds and prefiltered coverage). Loading
   copies a package into a slot, rebasing node, item and cell indices, and
   points the objects at it; evicting points them back at the stub.
3. **Certification** (`mask_cones.wgsl`): where a cone would descend into
   a stub (`add_mesh_coverage_if`), a ray enter one (`nearest_in_mesh`),
   or a texel sit on an absent level (`vc_surface`), the bake marks the
   page's slot with this frame's mark (`vc_fill` keeps it, so shading
   treats the page as missing) and lists (page, object). The cache keeps
   such pages waiting for every level their bakes found absent, and
   re-plans them once all are resident.
4. **Policy:** pages asked for on an absent level stay pending and are
   planned once it arrives; only as many levels are wanted as the bake
   budget can use; wanted levels are protected, the longest-waiting first,
   while the slots hold them; loaded levels stay at least 4 frames.

**Disclosed (before the run):** lighting levels stay packaged in CPU
memory (streamed on the GPU only), not produced on demand as the stub's
scene line says. The policy above was settled in development on 64 and 256
rocks (`stream_dev` `PAGES`, `STILL`; `M14_DEV=1`): earlier versions
thrashed or deadlocked at an eighth. The development run measured memory
at 0.821× (256 rocks, where the shared pools dominate). `read_cache_pages`
read pages at the wrong channel stride without a sky; it is fixed, and was
used only by diagnostics before.

## What the numbers say

- **The paradigm's guarantee survives streaming:** a cache page is never
  shown unless baked against every lighting level its cones needed, even
  with seven eighths of them absent at any moment.
- **Memory per unique object is now small:** a stub node, the proxy (48
  triangles), the chart cells and the slots' share; the shared pools
  dominate what remains.
- **The cost is churn and slower settling:** 13–15 lighting loads a frame
  and up to 3× gate 3's settle at the first view. Not yet: lighting levels
  produced on demand (CPU memory), a sky, bounce light or movers with
  streamed lighting, and a reference beyond memory for scenes larger than
  this one.
