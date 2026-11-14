# Continuum design

Living document. It is the source of truth for what we are building and why;
update it whenever a decision changes. Last updated 2026-10-25.

## 1. Goal

A real-time renderer built on an original paradigm, shipped first as a library
(Bevy integration first, a full engine later). It must match modern engines
(UE5: Nanite + Lumen + TSR) in its worst case and beat them decisively in its
typical case, and every claim must be measured.

## 2. The paradigm in one sentence

Ask cones what they see, answer in bits, and never ask again until the answer
could have changed by more than you can see.

## 3. Pillars

### 3.1 Error ledger and certified frames (the invention)

Modern engines amortize lighting by blending previous frames (TAA, temporal
accumulation in Lumen). That error is unbounded: it shows as ghosting, smearing
and lighting lag. Continuum amortizes by proof instead.

Every cached lighting result (a probe cone, a shadow query) stores:

- **Dependency region**: the part of space it actually depends on, which is the
  unblocked part of its cone, read from its mask.
- **Ledger**: accumulated worst-case error since it was last computed.

Every scene change (object moves, light dims, sun shifts) computes a conservative
worst-case effect on each cached result whose dependency region it touches and
charges it to that result's ledger. Some charges are exact (lighting is linear in
emitted power); others come from bounds. The tightest charges found so far are
projected: project the mover's old and new positions into the cached cone and
charge the larger of the coverage gained and lost (gate 5), which is far
cheaper than recomputing the cone. When a ledger exceeds ε, scaled by how
much the result contributes to visible pixels, the result is recomputed, largest
first. Indirect changes propagate through the same rule and die out because
energy shrinks every bounce.

Consequences:

- Frame cost is roughly screen work plus change work.
- Each pixel has a certified bound on its distance from a full recompute.
- With ε below a just-noticeable difference, staleness is invisible by construction.
- Occlusion-aware invalidation: a change behind a wall does not touch lighting
  in front of it, because the wall's bins end the cone's dependency region.

**Exact updates where they are cheap (gate 6b).** Bounds are not the only way
to avoid recomputing. When a change has a cheap exact update, apply it:
caching each gather cone's G-buffer (nearest static surface and depth per bin)
turns black movers into a depth composite and radiance changes into a re-sum
of only the cones that show the changed surface. In gate 6b this was bit-exact
and 7.4× faster than recomputing, while the bounded ledger managed 1.7×. The
combined mechanism: exact incremental updates where they are cheap (static
structure plus dynamic overlays), ledger-bounded deferral where they are not
(moving lights, slow global changes, budget pressure).

Prior art closest to this (none found doing it end to end in real time):
incremental radiosity with line-space hierarchies (Drettakis & Sillion 1997,
Granier & Drettakis 2001), the render cache (Walter et al. 1999), incrementally
baked GI (2019), path verification (2021), temporally adaptive shading reuse
(Mueller et al. 2021, heuristic gradients), JND-driven temporal reuse (2025).

### 3.2 Mask cones (the primitive)

Every visibility query is a cone carrying a direction mask (64 bins on the CPU;
see open questions for the GPU layout). Occluders set the bins they cover;
combining is bitwise OR, which keeps correlated occlusion exact up to bin
resolution, unlike multiplied transmittance. The continuous variant
(`CoverageMask`) keeps a fraction for partially covered bins: 7.7× more
accurate in gate 1 and continuous in occluder position, which the ledger needs
to defer small changes (gate 5). One primitive serves anti-aliasing
(pixel footprint), soft shadows (light size), glossy reflections (roughness)
and GI (wide cones, radiance-cascade style).

On the GPU, one cone maps to one subgroup: lanes own bins, the subgroup walks
the hierarchy in lockstep (no divergence), and a ballot builds the mask.

### 3.3 Quality contract (the quality model)

The caller sets ε and a frame budget; the library guarantees one and reports the
other. When the budget is short, cones widen: the image softens but stays stable,
with no aliasing, noise or ghosting. Foveation falls out for free.

**The error while flying, reported (design, 2026-11-13; Landmark 2).** At
a frame budget the cache shows coarser pages than pixels want while it
catches up (M4 G12: 20.8% from settled on a fast flight), and nothing
reported it. The renderer now reports, each frame, an estimate of the
image's relative RMSE from its settled image and an upper bound on it.
- *Why an estimate, not a certificate.* A coarser page holds point samples
  at its texel centres, not means over its texels' areas, so a pixel near a
  shadow edge can be off by the light's whole value; a certified bound per
  pixel would be the light itself. (Pages whose texels held area means
  would make one: values in [0, 1] with mean `c` deviate by at most
  `c(1 − c)` in mean square. A possible later step.)
- *How.* `N` probe pixels a frame, on a low-discrepancy pattern that moves
  each frame, are shaded twice: as shown, and with each light's share and
  the sky's computed at the pixel's own surface point by the bake's own
  tests (the light's mask cone; the sky's 192 hardware rays) instead of read
  from pages. Over the probes, the mean squared difference and the mean
  value give the estimate (`sqrt(3·mean d)/mean s`, as `relative_rmse`), and
  a one-sided 95% bound from the squared differences' spread. Pixels on the
  sky count with no error.
- *What it measures.* The shown image against lighting exact at each
  pixel: the settled image plus the settled cache's own interpolation
  error, whose size the estimate shows at rest. Bounce light is not
  probed (both shadings take the page's), so the estimate leaves out the
  bounce's error; the default has no bounce.
- *Cost.* `N` cones per light and `192 N` rays; 512 probes, estimated
  at under half a millisecond.
- *Next, the dial's other side:* an error-first mode in which the caller
  sets ε, the renderer raises the bake budget while the bound exceeds it,
  and the frame time is reported.

Gate: **M4 G13, passed** (`docs/results/m4-gate13.md`): the bound held in
10 of 10 checks on M9's flight, the estimate 1.215× the error against
settled renderers (it includes the settled cache's own 14–23% against exact
lighting), 1.023× the flight. The lights' shares are taken by 64 hardware
rays each over the light (the stub's cone walks were latency-bound, 6.6 ms).
Off by default (about 1 ms of GPU time a frame).

### 3.4 Prefiltered hybrid hierarchy (infrastructure)

One hierarchy whose leaves are triangle clusters (imported assets), field bricks
(procedural, CSG, destruction) or volumes. Interior nodes store prefiltered
directional coverage and appearance; a cone stops descending once a node is
smaller than its footprint, so cost is bounded by screen size, not scene size.
Builds on Lacewell et al. 2008, Bako et al. 2022, Weier et al. 2023; ours is
non-neural, fitted for mask cones, and portable. Not a novelty claim.

#### The far field for sub-pixel coverage (M13's design, 2026-11-13)

The goal is Landmark 3's coverage pass, about 10 ms at 1280×720 on the
million-boulder field, from M13 gate 4's 122–135 ms at the same accuracy
(0.218× one ray's coverage error standing, 0.132× from 150 m).

**Where the time goes at full size** (`docs/results/m13-far-field-profile.md`,
development).
- *From 150 m:* gate 4's path (with the packet walk) takes 104 ms: the
  bins' hardware rays about 51, the maps' reads about 24, and a walk from
  the root to the instances in every pixel about 27 (67 object nodes a
  pixel). Every boulder there is 2–5 pixels across, so no node above the
  instances is small enough to stop at. **Nodes holding coverage cannot
  help this view**; it needs fewer rays, cheaper reads and a shared walk.
- *Standing:* 122 ms. Pixels above the horizon walked about 40 nodes each
  for nothing, because the side planes' test cannot reject boxes straddling
  the eye's plane; testing the boxes' faces removes that exactly (36.0 →
  18.6 nodes a pixel, the same coverage, 113 ms). The rest is the horizon:
  a pixel there sees boulders from about 540 m to the field's edge, each
  about a pixel across, and walks up to thousands of nodes and reads their
  maps. **This is the far field, and what nodes holding coverage are for.**
  Its depth is the problem: even nodes exactly one pixel wide would line a
  grazing pixel's frustum about f·ln(far/near) deep (f is 772 pixels a
  radian: about a thousand), so a far pixel must stop at nodes that are
  deep along the view and thin on screen.

**The pass, in five parts** (A, B, C and E for both views; D for the far
field):

- **A. Exact culling by the boxes' faces.** Also reject a box when the
  pixel's (or tile's) frustum lies wholly on the eye's side of one of its
  faces: every frustum edge direction points away from that face and the
  eye is beyond it. Exact; done in development (above).
- **B. The bins' rays only near large objects.** The corner rays (one per
  pixel, 2 ms) already say which corners hit a large object (more than 8
  pixels across). A pixel traces its 16 bin rays only if one of the 16
  corners of its 3×3 block hits one (and its own four do not all hit the
  same one). It misses only slivers of large objects whose silhouettes cross
  between corners, measured by the gate's accuracy. Today every unresolved
  pixel traces them, including the sky and from 150 m 96% of pixels whose
  instances are all small.
- **C. One walk per tile** (a beam, Heckbert & Hanrahan 1984): an 8×8 tile
  walks the object hierarchy once against its frustum, nearest first, and
  lists its entries (small instances, nodes splatted today, D's pages) in
  workgroup memory in chunks; after each chunk each pixel evaluates the
  entries over its open bins, and the walk stops when every pixel's bins
  are full. The top of the hierarchy, today walked by each of the tile's 64
  pixels, is walked once. The entries and their per-bin combination are the
  per-pixel walk's, so the coverage is the same up to rounding.
- **D. Hierarchy nodes holding coverage: view-dependent coverage pages.**
  Below.
- **E. Lighter reads of the instances' maps:** 8-bit texels in a texture
  array (a layer per direction), one hardware-filtered sample per bin and
  direction instead of four loads with address arithmetic each.

**D in detail.**
- *Representation:* a node's **page** is its coverage seen from the eye,
  at one texel per bin: 512 8-bit texels in a shape from 32×16 to 128×4 (or
  transposed), so a slab seen at grazing angles, wide and a pixel tall,
  fits. Each texel holds the share of 4×4 rays from the bake eye `E`
  through its square that meet the node's contents. The page's header holds
  `E`, its plane (through the node's centre, facing `E`), the plane's axes
  and texel sizes. A pixel reads it per bin where the bin's ray crosses the
  plane, bilinearly: about 64 byte loads for a whole node, against 256
  loads for one instance's map today.
- *Correlation:* within a node it is exact, since the rays see its
  contents together (two boulders in line count once), and it is kept to a
  bin in the pixel, since texels are no larger than bins in either axis
  (M13 gate 2's saturation came from one number per node). Across entries,
  pages combine per bin as independent occluders, as instances do today.
- *Which nodes:* during C's walk, a node whose projected box fits a page
  shape at one texel per bin is an entry if its page is resident and valid;
  otherwise it is descended as today, and requested if it fits (a request
  list deduplicated on the GPU, as the view cache's). Nodes holding movers
  (their subtree is never prefiltered) never get pages. Instanced geometry
  keeps its shared per-geometry maps for instances: pages for 30,000
  instances from 150 m would cost millions of rays a frame to keep valid
  in motion, and the shared maps need no bakes.
- *Baking:* by hardware ray queries, each ray clipped to the node's box,
  a hit accepted only if its instance lies in the node's item range (a
  subtree's items are contiguous in the hierarchy's item list; one u32 an
  instance maps it back). 8,192 rays a page; baked within a time budget
  each frame. No preprocessing beyond the item ranges.
- *Reuse and its bound:* read from an eye `E′`, a point of the node's
  contents (at most `L/2` from the plane, `L` the box's depth seen from
  `E`) lands within `|ΔE⊥| L / 2D` of where `E` saw it (`D` its distance, `D ≫ L`),
  that is `2 f |ΔE⊥| L / D²` bins (a bin is `D / 4f` wide at `D`). A page
  is valid while that is at most a quarter of a bin and its texels stay
  within 0.7–1.4 bins (zoom, approach). A page past its bound is not read
  (the node is descended) until re-baked, so accuracy never depends on
  staleness, only frame time. Estimates: at the standing view's horizon,
  nodes of 32–64 boulders (`L` about 16 m at 1 km) stay valid for about 5
  m of the eye's motion; seen from above, a slab's `L` is its thickness and
  pages last tens of metres.
- *Memory:* bounded by the screen, not the scene: a pool of pages (32,768
  pages, 16 MB plus headers, in the first gate), nodes beyond it descended.
  Estimates: the standing view needs 2,000–5,000 pages (30–60 entries a
  horizon pixel against about 1,200 instances), the view from 150 m none,
  a view from 1 km up over the whole field 60,000–115,000. Precomputed
  maps for every node would need a million unique maps (64 GB at the
  instances' resolution).
- *Streaming (M14):* pages bake against whatever levels are resident;
  unique objects small on screen have no shared maps, so pages (a leaf of
  one object) serve them too. Pages want their contents' levels at a bin's
  resolution, finer than camera feedback's half pixel: open, for a later
  gate.
- *Later, appearance:* texels add premultiplied lit colour from the
  lighting cache at each ray's hit, so far pixels shade from the same
  reads; it is also the root fix for the error while flying (far pixels
  stop asking every object's lighting pages, M4 G12).

**Considered and set aside:**
- *Precomputed directional maps per node*, as the instances': a million
  unique nodes (memory above), and a square over the bounding sphere leaves
  a grazing slab one texel row tall, M13 gate 2's saturation again.
- *An opacity volume per node*: voxels combined along the ray as
  independent (exponential) count a boulder spanning two voxels twice;
  voxels larger than the boulders lose where in the pixel the coverage
  sits. That is the correlation problem Vicini et al. 2021 address with
  non-exponential transmittance.
- *Scalars per node* (Lacewell et al. 2008; M13 gate 2): saturation.

**Prior art (checked 2026-11-13 from abstracts, not full papers):**
- Shade et al. 1996, *Hierarchical image caching*: images of a BSP tree's
  nodes cached, drawn on quads and re-rendered when an error metric is
  exceeded. **The closest found.** D differs in holding coverage at
  sub-pixel bins rather than colour, baking by rays restricted to the
  node's contents, combining per bin with the other entries, and
  bounding reuse by position.
- Schaufler & Stürzlinger 1996 (a 3D image cache) and Yoon et al. 2006
  (R-LODs: planes in kd-tree nodes as levels of detail for rays): to check.
- Gobbetti & Marton 2005, *Far Voxels*: inner nodes hold a view-dependent
  appearance fitted to rays cast against the full-resolution model;
  precomputed, voxels, rasterized.
- Lacewell et al. 2008: directional opacity per BVH node, composited.
- Heitz & Neyret 2012: sparse voxel octrees whose voxels hold sub-voxel
  Gaussian descriptors; cone rays compute occlusion within the pixel's
  footprint and combine fragments as an A-buffer.
- Loubet & Neyret 2017: macroscopic surfaces as meshes, sub-resolution
  ones as a microflake volume with an occlusion-preserving voxelization.
- Vicini et al. 2021: non-exponential transmittance for correlation across
  volume elements. Weier et al. 2023: neural, correlation along rays in a
  sparse multi-level voxel grid.

No novelty claimed. View-dependent coverage pages at sub-pixel bins over a
hierarchy's cut, baked by restricted hardware rays and reused within a
positional bound, alongside shared per-geometry maps, is not known to us in
this form; that stays hedged until the papers are read in full.

**Gates** (stubs written before any implementation, criteria in their doc
comments):
- **M13 G5** (A, B, C, E): accuracy at most 0.25× one ray's error
  standing and 0.15× from 150 m (gate 4: 0.218× and 0.132×, a stated loss
  of about 15%); at most 25 ms from 150 m and 50 ms standing (the horizon
  is left for G6).
- **M13 G6** (D on G5): the same accuracy with pages baked; at most 20 ms
  in both views with pages resident; pages within 32 MB; and, along a
  walk standing with pages re-baked within a budget, every frame checked
  within 0.3× one ray's error.
- Later: the 10 ms bar, flights, then appearance in the same pages.
  (G5–G7 passed in the eleventh session; G7 kept pages valid in flight.)

**The 10 ms bar (design, 2026-11-13, after G7).** Profiled at full size
(`docs/results/m13-10ms-profile.md`): settled, standing 16.1 ms, 150 m up
17.2, 20 m up 25.4; the pixels' pass is 72–76% of each. From 150 m it
splits into the walk 2.4 ms, the instances' tests 2.7, their maps' set-up
5.0, texture samples 2.2, the rest 0.9. Standing and 20 m up, the set-up is
about 4 ms and the walk with page reads 7–9. The waste is shared: an
instance's set-up (where it lies on screen, which directions' maps, the
maps' coordinates across the pixel) is the same for every pixel, yet each
subgroup touching it computes it again, as each pixel reading a page
re-tests the page's validity and re-derives its plane.

- **Instance records, once a frame.** A pass with one invocation per object
  writes, for each small instance in front of the eye and inside the view,
  a record: its projected centre and radius, its maps' level, its two
  directions' layers and weights, and the affine map from screen position
  to each map's coordinates (2×2 and offset per direction; exact to under
  0.3% of the map, as G5's linearization). Others are marked out (outside
  the view, behind, large or mapless). The walk's items then cost a record
  load and the bins' samples. 64 bytes an object (64 MB for the million
  boulders; a pool bounded by the screen is left for later).
- **Page records, once a frame.** A pass over the pool's pages in use
  decides each page's validity from the eye (one bit, and the refresh
  request) and the homography from screen position to its texels (3×3);
  a pixel reading a page then costs a record load and per bin three dot
  products, a division and the bilinear read.
- **The other passes** (corners, roots, bins, bin rays, page passes: 4–5 ms
  as profiled, each with about 0.15 ms of submission in the profile)
  trimmed where it is free: skipping the bakes' dispatch when nothing is
  asked, the roots pass sharing its projections.

Expected from the profile: about 5 ms off the pixels' pass from 150 m and
4–5 standing, so 11–12 ms; the rest from the walk (each node's eight
corner projections) if needed, measured in development. Gate **M13 G8**:
gate 6's accuracy held, at most 10 ms settled standing and from 150 m.

**Appearance: colour in the same structures (design, 2026-11-13, after
G8).** Coverage is about 9 ms; Landmark 3 asks for coverage and colour.
Two things change at once:
- **A shading model a bin can afford.** The regular renderer lights a hit
  with a mask cone per light and the sky's cones, far too heavy per bin,
  per instance and per page texel. The first appearance gate uses one
  model everywhere, the reference included: colour = albedo × (E_sun ·
  max(0, n·l) · V + E_amb) / π, with l and E_sun from the light's
  rectangle (its centre, its solid angle), V a hard shadow (one hardware
  ray toward the light's centre), E_amb a constant (a fifth of the sun's
  normal irradiance), shading normals; rays that miss see a constant sky.
  The gate then measures sub-pixel aggregation, not the lighting model;
  soft shadows and the lighting cache come with the move into the
  renderer.
- **The ground (large objects as a background with depth).** Without the
  ground, a pixel whose corners hit one large object skipped its walk. With
  it, most pixels' corners hit the ground behind the boulders. So a large
  object no longer fills bins: a pixel with large bins (corner and bin
  rays) shades one large hit by the model (its centre's, else its first
  large bin's) and keeps its depth; walks cull nodes behind that depth;
  small entries in front composite over the large bins; what is left of
  each bin shows the large colour or the sky.

Per bin, entries composite in walk order (nearest first) as before, with
colour: the pixel's colour accumulates `T_b · α · c / 16` per entry and bin
(T_b the bin's transmittance before it), so bins keep no colour of their
own.
- **Small instances:** maps become RGBA8: coverage and the texels'
  coverage-weighted mean normal (object space, octahedral); records (48
  bytes) add the light's direction in the object's frame, the instance's
  sun visibility (one hardware ray from its sphere's sunward point, a frame)
  and its albedo; each bin lights its sample's normal.
- **Pages:** texels add the mean colour of their rays' hits, each shaded by
  the model at bake (a shadow ray a hit): 4 bytes a texel (coverage and
  colour), a pool of 8,192 pages of 2,048 texels (64 MB).
- **Splats** (rare since pages): their node's first instance's albedo under
  the ambient and half the sun.

Gate **M13 G9**: the field with its ground, the model above; colour within
0.5× one ray's error of the 16×16 reference in both views; settled at most
15 ms; the 20 m view reported.

## 3.5 What the 2D phase (M0) taught us

Status after the flatland gates (2026-10-23). Full results are in `docs/results/`.

| Gate | Question | Result |
|---|---|---|
| G1 | Mask cones vs rays | Pass: same answer as 64 rays at about 1/18 the cost; continuous masks 7.7× more accurate |
| G2 | Ledger soundness | Pass: zero violations in every run, including forced-deferral stress tests |
| G3 | Work skipped, typical scene | Pass: 87.7%, output identical to a full recompute |
| G4 | Worst case ≤ conventional | Fail: geometric charging fine (0.90–0.95×), projected charging 1.96× |
| G5 | Deferral for wide cones | Pass on attempt 2: 25.4% of exact-invalidation work |
| G6 | One-bounce GI, bounded charges | Fail: 66.9% (oracle with exact charges: 27.4%) |
| G6b | One-bounce GI, cone G-buffers | Pass: bit-exact, 7.4× faster |
| G7 | Moving light | Fail, narrowly: 25.1% vs ≤ 25% (exact invalidation needs 100%) |

Synthesis:

1. **Most of the speed comes from exact incremental work**: caching what static
   geometry contributes per cone (static masks, static coverage, cone
   G-buffers) and applying dynamic changes exactly to only the cones they can
   reach. That is where 17×, 9× and 7.4× results came from, with zero error.
2. **The ledger's own value is in what exact updates cannot reach cheaply**:
   slow changes (gate 5: a further 2.9× beyond exact invalidation), moving
   lights (gate 7: 4× less work than exact invalidation), and frame-budget
   pressure. It has never produced a soundness violation.
3. **Bounds are loose** (actual error is typically 5–10% of the certified
   bound), and **charging must never cost more than recomputing** (gate 4).
4. Refined one-sentence paradigm: *cache per cone what cannot change, apply
   what changed exactly where that is cheap, and certify with a bound what you
   choose not to recompute.*

## 3.6 Where we stand (updated 2026-11-13, eleventh session)

Against the goal in §1, honestly:

- **Proven in test scenes.**
  - Mask cones as the primitive, on the CPU and the GPU.
  - The paradigm's first two rules: caches of what cannot change, and exact
    incremental updates for moving objects and bounce light, bit-exact on
    every frame (M2 G5–G6, M3 G3).
- **Proven at scale (M4).**
  - A million instanced boulders (10^10 triangles) in a large world.
  - A view-dependent lighting cache that flies over them at 720p in 14.6–15.0
    ms per frame on the fanless M4 Air (M4 G11). Lighting is softer in motion
    and sharpens when the view settles.
  - Hardware camera rays take about 2–3 ms. This needed a patched wgpu-core
    (`third_party/`).
- **Unique geometry streamed (M14 G1, failed narrowly on speed).** Camera
  levels of each geometry load by need into a fixed pool: a field of 256
  unique rocks whose finest levels are 6.35× the budget settles to within
  a tenth of one ray's coverage error of fully resident, with 3.4× less
  GPU memory, at 1.07–1.12× the frame time (loads are too many and cost
  about 1 ms each). G2 (failed narrowly again, 1.110–1.113× in three of
  four flights) made loads cheaper (paged pool, staging belt, level chains;
  29 ns a triangle, most of it building the structure) and showed detail
  following the zoom: 50° to 1.5° onto a rock 103 m away settles exact.
  Streaming costs 0.33–0.42 ms a frame, 2–2.5% of a 60 fps frame. **G3
  passed:** 4,096 unique rocks (finest 32× the budget, 33 GB packed,
  beyond the machine's 25.8 GB) generated on demand by worker threads,
  flown within a 60 fps budget (worst frame 14.5 ms) and settling within
  0.03× one ray's error of the ideal levels. **G4 passed:** lighting
  levels stream too (an eighth resident), and the cache stays certified:
  every page shown is bit-identical to one baked with every lighting level
  resident (2.68 million compared), at 0.486× gate 3's GPU memory. **G5
  failed on speed only:** lighting levels are produced on demand too
  (36,864 unique rocks, lighting CPU memory capped at 256 MB, about 28 KB
  of GPU memory a rock), still certified against a reference holding the
  flights' region (676,080 pages identical), but flights took 26–39 ms a
  frame. **G6 failed on one frame only:** after profiling (uploads through
  a staging belt, a persistent residency set and incremental top-level
  builds in patched wgpu, clusters of 32 proxies, narrower scans) the same
  flights take 6.2–6.5 ms on average, 99th percentile 10–12 ms, with
  about 18 KB of GPU memory a rock; one frame took 42 ms. **G7 passed:**
  with producer threads at a lower priority and large frees off the
  render thread, no frame of the three flights exceeds 16.7 ms (means
  4.8–5.9 ms, 99th percentile 8.3–10.1 ms), still certified. Settling is
  slower (lighting levels about 20 frames late on new ground). **G8 failed
  on the frame budget only:** a sky and bounce light now work with
  streamed lighting (sun and sky certified on 9,216 rocks, bounce within
  0.5% settled), but with them the 36,864-rock flights had a 20–25 ms
  99th percentile. **G10 passed:** bounce rays now trace through the
  camera's hardware structures (never needing a lighting level) and the
  bake's frame budget is 11 ms: with a sky and bounce the flights run at
  9.2–10.2 ms mean, 99th percentile 14.5–15.0 ms, still certified, bounce
  3.09% from Cycles. Not yet with movers. **G11 passed:** the bake's
  frame-period cap times the renderer's own frame call, so a caller's
  stalls (reading images back) no longer shrink the budget. **G12 failed
  on a method fault, G13 passed:** the rocks' coarser camera levels carry
  the finest surface's normals at their vertices, taking the settled
  colour gap to the finest-resident images from 1.9–2.5% to 1.2–1.5%, at
  no production cost (equality checks need a fixed bake budget: the
  time-based one varies with the machine's timing). Imported meshes have
  no coarser levels yet (there is no simplifier; streaming takes levels a
  caller provides).
- **Measured against an independent path tracer** (Cycles, compared
  correctly: supersampled 4×4 against Cycles' area-integrated pixels):
  - sun and sky on an outdoor scene: 0.7–1.2% (M5 G2);
  - the million-boulder field with sky: 1.2–1.9% sky only (M5 G3–G5), with
    object-level prefiltered coverage. (G3's "sun and sky 2.4%" lit only the
    horizon with the sun: its sun stood 1 km from the field's center, which
    left the camera's surroundings behind the one-sided panel; M5 G6.)
  - three colored lights around the camera (a sun, a 4° fill, a nearby lamp)
    on the field: 1.22%, and with sky 1.03% (M5 G6).

  The earlier "3.8% vs Cycles" (M3 G3) compared point samples with
  area-integrated pixels and is an overestimate. The sky's figures are a
  favourable case: turning the sky's cube of faces moves its error against
  Cycles between 1.9% and 4.2% on the boulder field (M5 G5), because
  coverage is weighted per 11° bin.
- **Sky light in the view cache runs at frame rate but converges slowly**
  (M5 G5, failed on warm-up). **The sky later** (M5 G9, passed): pages bake
  their lights first and their sky alone from the next frame, shown as the
  open sky until then. Views come within 5% of their warm image 10–20×
  sooner (frame 120–300), the error while warming is 0.11–0.17× the
  baseline's, they warm in 0.61–0.65× the frames, and the warm image is
  identical. Not with bounce light, movers or moving lights.
  - Flights over a million boulders with sun and sky average 14.8 ms on the
    fanless Air (99th percentile under 25 ms), and warm frames take 2.2–2.6 ms.
    The bake budget follows GPU time measured with timestamp queries.
  - Each geometry's occlusion of its own sky is baked once for all its
    instances (own-object masks, faithful to 0.06–0.15%).
  - But a view warms with sky in 1,365 frames standing and 6,900–12,200 from
    above, against 82–132 with the sun alone. Lighting stays at coarse pages
    while flying. The viewer's sky stays opt-in (`--sky`).
  - **Solved for static scenes by the sky through hardware rays** (M5 G13,
    passed): a sky sample traces 192 rays (a hemicube's bins, first hit)
    at 0.72 µs against a cone sample's 10.3 µs; the field's views warm in
    166–400 frames against 1,827–15,905, and standing is closer to Cycles
    (2.53% against 4.67%). Movers, streamed levels and blended objects keep
    cones (rays see camera levels, which would trade M14's exact sky for a
    converging one).
- **Moving objects in the view cache** (M9 G1, passed). The second rule now
  runs in `Renderer`, not only in the lightmap path.
  - 65 objects moved every frame among a million static boulders, with
    three lights. In 30 checks of up to 137,260 pages, no value differed
    from a from-scratch re-bake.
  - Re-baking followed the change: 0.84% of the pages read per frame.
  - Flights cost 1.06–1.09× the same flights with the movers still
    (15.8–16.1 ms).
  - Method: a movers subtree under the hierarchy's root and a small second
    acceleration structure; per-page then per-texel tests of each cone's
    hull against moved boxes; same-frame re-bake of stale texels read in
    the frame before; other stale pages are marked invalid. Not yet with a
    sky, whose wide cones are the ledger's job.
- **A moving light through the ledger** (M9 G2, failed on fallback only).
  The third rule ran on the GPU in 3D for the first time: a sun drifting at
  a 24-minute day's speed over the million-boulder field.
  - Sound: in 30 checks of up to 57,409 pages, no page shown differed from
    a re-bake at all.
  - 0.69% of the pages read re-baked per frame.
  - Means 14.5 ms, against 15.6 ms with the sun still.
  - But fallback was 33–34%, against at most 10%: off-screen pages past
    their guard are set aside and shown missing until wanted. About
    30,000 a frame, more than the guards' lifetime explains, cause not
    yet established.
  - Method: per-texel guard cones (three times the light's radius) mark
    clear texels, exact while the light stays inside; mixed texels re-bake
    whenever it moves; pages keep a slack and re-bake whole past it.
  - What it certified was exactness: the ε went into snapping estimator
    noise, and no texel was kept with a non-zero bound. Bounded deferral
    within ε (the invention proper) is still unshown in 3D. Movers with
    moving lights are not supported yet.
  - **A moving sun at real speed, bounded** (M9 G6, passed): a rect
    light's cone bins sit on a lattice fixed in the page's version frame,
    so a mixed texel's ratio moves only with the light's footprint and is
    held within 1.5κ(Pδ + πδ²)/Ω until its share of ε. Flights cost
    0.98–1.02× the still sun's (gate 4: 1.6×), and 886,506 pages checked
    at three speeds were all within their bounds. This is bounded deferral
    within ε, the invention proper, shown in 3D for one moving light.
    Not yet under a sky, with movers, or with streamed lighting.
- **Movers under a sky** (M9 G3, failed on work and speed). The cache now
  accepts movers with a sky. Sky charges are a telescoping bound (each
  mover's sphere's solid angle at the page's bake plus now); a page read
  past ε re-bakes its sky in the same frame. It was sound in 30 checks, but
  99.7% of pages read re-baked their sky every moving frame: 53–58 ms
  against 14.7 ms still, with 30% fallback. The per-page sphere bound is
  about 5,000× loose; texel-level charges with the cosine are next.
- **Several colored lights** (M5 G6, passed): a list of RGB rect lights per
  pixel and in the cache, flights at 14–15 ms with three lights. A cache
  texel stores light per unit radiance, so re-coloring a light bakes nothing
  and is bit-identical to a fresh cache.
- **Lights in the cache keep each pixel's normal** (M5 G7, failed on flight
  speed only).
  - Texels store each light's shadowed share of its unshadowed light, and
    shading multiplies by the unshadowed light at the pixel (a ratio
    estimator).
  - The cache's error against per-pixel cones fell to 0.41–0.68× at gate 8's
    viewpoints.
  - But it doubles the GPU camera passes. Flights with three lights averaged
    16.2–17.6 ms against 16.7, because the bake budget sits at its floor
    while flying.
  - The sky did not gain from the same idea (`m5-sky-bake-diagnostics.md`).
- **Point and spot lights** (M5 G8, failed on sustained speed only).
  - Sphere lights with a size and spots with glTF's falloff work per pixel
    and in the cache, and come in with glTF files.
  - A night scene with both is 1.35% from Cycles, and its cache 0.82% from
    per-pixel cones.
- **Sustained speed on the fanless Air is the open performance problem**
  (M5 G8).
  - Cool, flights with three lights hold 13–15 ms. After minutes of load
    the GPU slows by up to about 2×, and flights averaged 20–25 ms.
  - The bake budget follows GPU time but cannot go below its floor, and the
    camera passes do not shrink.
- **A library front door** (not gated: no claim beyond its use).
  - `continuum_gpu::Renderer` wraps the kernel and the instanced renderer
    with the measured defaults.
  - `continuum-gltf`'s `render` example turns a glTF file into a PNG.
  - `docs/GUIDE.md` covers setup, concepts and limits; rustdoc covers the
    public API, with experiment-only items hidden.
  - The viewer's field and glTF scenes run on it.
  - A Bevy 0.20 plugin (`bevy_continuum`) renders a camera's view with it,
    on Bevy's own device.
  - CI runs every test on Linux (Vulkan, lavapipe) and compiles every
    shader for Windows (DX12, DXC); WARP cannot run the kernel (§3.8).
- **Imported content** (M6 G1, failed on round-trip precision only).
  - glTF 2.0 files load as instanced scenes, with automatic charts for any
    mesh and base color textures sampled per pixel at the pixel's footprint.
  - A textured scene imported from `.glb` is 0.43% from Cycles with the same
    textures. Automatic charts serve the cache within a point of hand-made
    ones, and warm frames take 1.5 ms.
  - The cache stores light, not color, so texture detail costs no bake work.
  - The round trip reproduces the scene up to f32 rounding of its transforms,
    which flipped one edge pixel at one sample per pixel: 0.024% against a
    0.01% bound.
  - Imports base color, metallic and roughness factors, emission (factor
    times strength; not shown by other surfaces' lighting) and normal maps
    (M10 G5: 3.63% from Cycles with them), metallic-roughness textures,
    and alpha masks cut into geometry (M10 G7: 3.29% from Cycles).
    Since then: emissive textures (M10 G15), BLEND materials as blending
    (M10 G16), double-sided materials as back-to-back faces, node
    animations (played through movers),
    `KHR_texture_transform` (Blender's sign), and vertex colors (M10
    G17–G18: 1.29% from Cycles, 1.00× the frame), and sampler wrap modes
    (M10 G20: 2.18% from Cycles). Saving writes all of these back. Not
    yet: a second texture coordinate set.
- **One bounce of light in the view cache** (M10 G1, failed on its
  re-coloring check only).
  - On M6's textured scene, the cached image with bounce light is 1.79%
    from Cycles with one bounce, against 7.09% without it.
  - Warm frames cost 1.07×, and views settle in 1.21× the frames.
  - Method: 3×3 hemicube samples per page (192 rays each), reading the
    direct light the cache holds at the hits; hits with no page request
    one and re-bake later.
  - Re-coloring alone bakes nothing (M10 G4, passed): gate 1's failure was
    its own settle helper, which returned after one frame. M10 G3's
    secondary requests stopped request cascades (materials settle in 1.12×
    the frames instead of 6.9×). Not yet with movers or moving lights.
  - **More than one bounce** (M10 G25, passed; opt-in through
    `CacheOptions::bounces`): samples gather the bounce light of the pages
    they hit and record how many bounces they hold, pages re-baking until
    their hits hold enough. The showcase's closed room goes from 45.4% to
    6.5% of Cycles' 12-bounce image with 8, settling in 80 frames. Past the
    first bounce carries under 1% outdoors and two fifths indoors, at
    about 7× the bounce samples, so the default stays one. Untried with
    streamed lighting and movers.
- **The error while flying, measured** (M4 G12, passed): against renderers
  settled at the same cameras, a fast flight over the million boulders is
  21–26% off (14.6% at a tenth of the speed). Pixels now request pages up
  to three levels finer than the coarse ones shown (`CacheOptions::
  refine_levels`), taking it from 25.4% to 20.8% and warming views in
  0.70–0.76× the frames. Its root is per-object page demand: about 100,000
  visible boulders each need their own pages (16,000–20,000 wanted a frame
  against 800–1,500 baked), which Landmark 3's prefiltered appearance
  addresses. Measured, not yet bounded.
- **Metallic-roughness materials and glossy reflections** (M10 G2 failed
  on accuracy; **M10 G10 passed** after a fix). GGX specular from lights (a
  representative point times the cached shadowed share), a split-sum
  directional albedo, and one reflection ray per pixel reading the cache at
  the lobe's mip. 5.85% from Cycles against at most 6% (gate 2: 8.94%; the
  light-size normalization was applied twice), warm frame 1.68×. Metallic-roughness textures scale the factors (tested, not gated).
- **Normal maps** (M10 G5, passed): tangent-space maps with per-triangle
  tangents bend the normal for light from lights, specular and
  reflections. 3.63% from Cycles with the maps against 26.42% ignoring
  them, at 1.04× the warm frame. The sky and bounce light ignore the maps.
  Tangents are MikkTSpace's per corner since M10 G13 (failed: no
  measurable difference on its scene, whose maps were not baked from a
  high-poly model; kept as glTF specifies, no cost).
- **Alpha cutouts** (M10 G7, passed; G6 reached no criterion). Masked
  triangles become the opaque part of their texture, as geometry, before
  upload (`cut_out`: marching squares on the bilinear alpha), so every path
  sees the cut exactly. 3.29% from Cycles masking the cards itself, against
  34.07% with opaque cards, at 1.15× the warm frame.
- **Exactly back-to-back faces** (M10 G8–G9, failed on a comparison with a
  1 mm control that mostly measured the control's own offset; kept). A
  double-sided card's front and back faces on one plane now bake at normal
  speed (1.21× the 1 mm scene's bake time, 1.04× its warm frame; they were
  100× or more slower) and are as close to Cycles (3.42% against 3.37%).
  Two bugs: smooth normals cancelled to zero (NaN bounce rays visited the
  whole scene), and Metal's ray queries ignore `CULL_BACK_FACING`, so
  camera rays now re-trace back-face hits through tested candidates.
- **Emission** (exact by construction where seen directly and in
  reflections). It lights other surfaces through bounce light (M10
  G11–G12, failed on the absolute bound, kept): 7.46% from Cycles over
  surfaces near a glowing panel at night against 18.00% without, at no
  cost; the rest is bounce light's 3×3 samples a page near an emitter.
  **Emissive textures** (M10 G15, passed; G14 failed from its reference
  scene's coincident faces): 6.35% from Cycles on a striped sign, no cost.
- **Alpha blending** (M10 G16 failed on speed only, 1.33× the frame
  against 1.3×; **M10 G19 passed at 1.23×**, where a fifth of the pixels
  see a pane). With hardware camera rays, up to four layers behind blended
  surfaces are traced in the visibility pass and composited front to back
  (the extra-samples list, shared with silhouette rays, shaded by an
  indirect dispatch over the list alone); shadows keep the opacity
  factor's share of light.
  4.71% from Cycles on glass and a blind, against 57.31% drawn opaque.
  Without hardware rays, or with movers, a textured blend is cut at 0.5.
  Bounce and reflection rays see blended surfaces opaque.
- **Environment light** (M5 G11 failed, **M5 G12 passed**): a sky whose
  radiance varies with direction, from an equirectangular image.
  Surfaces scale the cached uniform-sky visibility by the environment's
  nine-harmonic irradiance over the open sky's, so the cache bakes nothing
  new; rays that miss read the image. 1.37% from Cycles on a gradient sky
  (the best uniform sky: 19.92%), 1.04× the frame. Seen directly it has
  8-bit range (a sun belongs in a light); bounce light and the per-pixel
  path use its mean. The viewer takes `--env` (a gradient, or an `.hdr` or
  `.png` image through `continuum_gltf::load_environment`).
- **Texture filtering for the area integral** (M7 G2, passed): a box
  filter over the pixel's footprint (ray differentials, four taps) halves
  the textured scene's error against the area-integrated image (7.11% to
  3.99%) at 1.03× the warm frame. On by default.
- **Sub-pixel detail, coverage** (M13 G1–G8; G5–G8 passed): per-geometry
  directional coverage maps (opacity from 256 directions, 16×16 texels
  with mips) read for instances up to 8 px, 4×4 sub-pixel bins, the object
  hierarchy walked per pixel with prefiltered splats for far nodes, and
  hardware rays (pixel corners, then bins near large objects) for larger
  surfaces. M13 G5: a pass with one invocation per 8×8 tile lists subtree
  roots, pixels walk below them as packets, two directions' maps read with
  coordinates affine across the pixel, open bins as bit masks. On the
  million-boulder field: 0.219× one ray's coverage error standing in 38.25
  ms, 0.135× from 150 m in 20.35 ms (one ray: 8.69% and 57.99%; gate 4:
  122–135 ms). M13 G6: far hierarchy nodes read from view-dependent
  coverage pages (a node's coverage from the bake eye, a texel a bin,
  baked by hardware rays restricted to its instances, reused within a
  quarter-bin shift): settled, 0.212× standing in 16.12 ms with 10,224
  pages, 0.135× from 150 m in 17.11 ms. M13 G7: pages keep up with a
  10 m/s flight 20 m up (frames 1.191× the settled frame at their place,
  95th percentile; 56 bakes a frame in 1.39 ms) by baking one invocation
  per ray, leading bake eyes along the motion and refreshing pages near
  their bound. That view settles at 21.6–29.3 ms. M13 G8: the coverage
  pass in about 9 ms settled (9.01 standing, 9.10 from 150 m; 0.159× and
  0.128× one ray's error), by once-a-frame records for pages, nodes and
  instances, maps of 32×32 directions read nearest, 2,048-texel pages and
  tile lists of leaf instances; the flight view 11.8 ms. Not in the renderer yet:
  a diagnostic path (profile stage 18), coverage only. Left: bakes that
  keep up in motion, the 10 ms bar, appearance.
- **Silhouette rays** (M7 G3–G4, failed on speed): four rays at edge
  pixels halve the field's error from 150 m. As compute rays they cost
  13–24× the frame there. Through ray queries, in their own passes (M7 G4),
  they cost 1.29× on the textured scene (passed) and 3.03× on the field
  (11.08 ms, against 1.5×), with the same quality (`CacheOptions::edge_rays`,
  off by default).
- **Not started:**
  - sharing the sky's far field across texels (cascades of wide cones), the
    remaining lever for the total sky work (the sky later, M5 G9, orders
    the work rather than cutting it);
  - anti-aliasing of silhouettes that runs in real time. One ray per pixel
    is 12–23% from the area-integrated image on the boulder field (M4 G8,
    M7 G1), the largest quality gap. Pixel cones (M7 G1) and compute
    silhouette rays (M7 G3) both failed on cost; silhouette rays through
    ray queries (M7 G4) cost 3× where half the pixels are edges;
  - the quality contract (§3.3) beyond the bake's frame-time target;
  - coarser levels for imported meshes (a simplifier).
- **Against UE5.** No direct comparison has been made, so none is claimed.
  - Our measured edge: exact soft shadows from area lights and noise-free
    sky occlusion, bounce light that updates exactly in the frame the change
    happens, and light colors and intensities that change for free and
    exactly, with no noise, denoiser or temporal lag.
  - Moving objects update the cache exactly in the frame they move (M9
    G1), with no stale light.
  - UE5 leads on everything else: material accuracy and features beyond
    glTF's metallic-roughness basics, lights, GI bounces, anti-aliasing and
    upscaling, platforms and tooling.

## 3.7 Road to v1 (2026-10-24)

A **batch** below is the size of one long working session in this project's
history: a few gates with their implementation and write-ups. These are
estimates, and gates fail often, so treat them as ranges.

**v0.1, experimental** (a library people can try on their own scenes): about
11–15 batches.

| # | Work | Why | Batches |
|---|---|---|---|
| 1 | View-dependent lighting cache: texels only on visible surfaces, coarser with distance, a per-frame bake budget, kept across frames, exact incremental updates for movers, instanced objects | Real-time at scale. M4 gate 7 shows per-pixel cones are the bottleneck. **Done for static scenes (M4 G8–G11; G11 passed):** flying over a million boulders at 720p averages 14.6–15.0 ms with the bake budget following the frame time; lighting is softer in motion and sharpens when the view settles. Sky in this cache since M5 G5 (at frame rate, slow to converge; the sky later, M5 G9, brings views within 5% 10–20× sooner). Not yet: movers in this cache | 2–3 (3 done) |
| 2 | Sky light and several lights (sun, sky environment, point and spot lights) | Outdoor scenes had black shadows. **Sky done per pixel and matched to Cycles (M5 G2–G3); in the view cache at frame rate (M5 G5), but slow to converge. Several colored rect lights done (M5 G6), cached as ratios (M5 G7); point and spot lights (M5 G8).** The sky later (M5 G9) shows shadows first and warms views 10–20× sooner to within 5%. Not yet: sustained speed with several lights | 3 (3 done) |
| 3 | glTF import: meshes, instances, base-color textures, automatic charts | Real content instead of procedural scenes. **Done (M6 G1, failed on round-trip precision only):** a textured import is 0.43% from Cycles, automatic charts are within a point of hand-made ones in the cache, warm frames take 1.5 ms. Since then (M10): metallic-roughness, normal, emissive and metallic-roughness textures, alpha masks and blending, double-sided materials, the file's lights. Not yet: `KHR_texture_transform`, vertex colors | 2–3 (2 done) |
| 4 | Pixel-footprint cones for primary visibility: anti-aliasing and far-field level of detail | The far field shimmers; one ray per pixel. **Tried (M7 G1, failed on all three):** exact 8×8 coverage per pixel costs hundreds of node tests per pixel at scale (105 ms on the textured scene, 10–87 s on the field) and gains 0.76× only from 150 m up. Kept off by default; v0.1 ships with one ray per pixel | 2 (1 spent) |
| 5 | Library API, Bevy plugin, examples, documentation | Something people can use. **Done:** `Renderer`, a glTF-to-PNG example, `docs/GUIDE.md`, the viewer's instanced scenes on `Renderer`, rustdoc for the public API (`missing_docs` warns), experiment-only items hidden from the docs; a Bevy 0.20 plugin (`bevy_continuum`, on Bevy's own device; not gated) | 2 (done) |
| 6 | Windows/NVIDIA: CI (WARP, lavapipe), a cloud GPU benchmark run | The day-one platform requirement. **CI done:** Linux (lavapipe) passes every test; Windows compiles every shader with DXC, but WARP cannot run the kernel (`docs/results/ci-portability.md`). The cloud GPU benchmark is prepared (`docs/BENCHMARK.md`: the gates' timing parts in one headless binary, a setup script, an M4 baseline); not yet run on NVIDIA | 1–2 (1 spent) |

### 3.8 Next steps, in order (updated 2026-11-13, eleventh session)

**Start here (updated 2026-11-13, eleventh session).** The state in one
place, for a new conversation. Read this block, then the milestone table
(§6) and the results files it names. Item 1 of "Next, in order" below (the
design for M13's far field, §3.4) and items 2.1–2.3 are done (M4 G13 and
M14 G15 passed; M14 G14, several bounces with streaming, failed one view's
convergence check with a disclosed confound). **M13 gates 5 and 6 passed**
in the eleventh session (gate 6: coverage pages for far hierarchy nodes;
settled 0.212× / 0.135× one ray's coverage error in 16.12 / 17.11 ms,
`docs/results/m13-gate6.md`) and **gate 7** (pages keep up with a 10 m/s
flight: 1.191× the settled frame against 1.2×, `docs/results/m13-gate7.md`)
and **gate 8** (the coverage pass in about 9 ms settled in both views,
`docs/results/m13-gate8.md`). The next task is item 3's remainder:
**colour (appearance) in the same structures**: designed (§3.4,
"Appearance"), stub `m13_gate9`; then the showcase (item 4).

- **The user's direction.** Core first (2026-11-08), sharpened 2026-11-12:
  be excellent at three things (**scale, detail, lighting**) and adequate
  at the rest. "Scale and detail" means huge, extremely detailed worlds
  where **detail follows the screen**: zoom in or come closer and
  geometry, lighting and textures resolve further instead of pixelating;
  zoom out and what cannot be seen costs less. One primitive (cones over a
  prefiltered hierarchy, with a certified cache) serves shadows, level of
  detail, anti-aliasing, reflections and GI, so strengthening it
  strengthens all of them. A **quality dial** grows alongside (§3.3: an
  error bound and a frame budget, one guaranteed, the other reported).
  Keep usage low. The NVIDIA benchmark waits for funds (not soon).
- **Decided 2026-11-12 (the user):** streaming's speed is measured
  against a 60 fps frame budget (16.7 ms) with a streaming time budget,
  not as a ratio against unthrottled frames (which are about 4 ms on this
  scene).
- **Releases:** `v0.2.0` tagged (2026-11-08). The repository is public
  (2026-11-09) under PolyForm Noncommercial 1.0.0 ("Copyright 2026 Gabriel
  Kirrane, doing business as Penumbral Labs"); commercial requests through
  issues.

**The four landmarks (agreed 2026-11-12) and where each stands:**

1. **A unique world beyond memory:** unique geometry several times GPU
   memory, flown at 60 fps on the M4, never over budget, lighting streamed
   too and still certified, within a stated bound of a reference, at any
   zoom. *Most of the mechanism is done (M14 G1–G6):*
   - camera levels stream by need (feedback per pixel, half a pixel of
     error, a paged pool, level chains, a 2 ms-a-frame time budget);
   - levels are produced on demand by worker threads into a capped CPU
     cache (`LevelSource`): 4,096 unique rocks, 33 GB of detail, beyond
     the laptop's 25.8 GB, within 60 fps (G3);
   - lighting levels stream too (an eighth resident), with the cache still
     bit-exact (G4); proxy levels stand in for camera rays;
   - detail follows the zoom (G2: 50° to 1.5° onto a rock 103 m away,
     settled exact);
   - lighting levels produced on demand too (G5), and 36,864 unique rocks
     (9× G4's field) at 6.2–6.5 ms mean, 99th percentile 10–12 ms, about
     18 KB of GPU memory a rock, the cache certified against a reference
     holding the flights' region (G6: 1.4 million pages identical);
     objects showing proxies are seen through clusters of 32.
   - no frame over 16.7 ms in those flights (G7: means 4.8–5.9 ms, 99th
     percentile 8.3–10.1 ms, worst 15.8 ms), still certified.
   - a sky and bounce light with streamed lighting (G8), within 60 fps
     (G10: 99th percentile 14.5–15.0 ms), direct light certified, bounce
     converging; bounce rays through hardware ray queries.
   - coarser rock levels shade with the finest's normals at their
     vertices (G13: settled colour gap 1.9–2.5% → 1.2–1.5%).
   - imported meshes get coarser levels by simplification
     (`InstancedScene::simplify_geometries`, M14 G15: shading as close to
     the finest as the rocks' analytic levels).
   *Left:* settling (views take thousands of frames to settle fully at
   this scale); movers with streamed lighting; simplified lighting levels
   measured against a finest-lit reference.
2. **Lighting you can trust in motion:** the cache always converges, the
   sky warms in a few hundred frames, the error while flying is measured
   and bounded, a moving sun at real speed. *Done:* the loop that never
   settled with bounce light is fixed (M10 G24); a moving sun at real speed
   costs what a still one does, every shown page within a certified bound
   (M9 G6: rect lights' cones on a lattice fixed in the page's frame); the
   sky warms in a few hundred frames (M5 G13: sky samples through hardware
   rays, 166–400 frames against 1,827–15,905); the error while flying is
   measured against settled references (M4 G12: 20.8% on a fast flight,
   from 25.4%, after requesting pages three levels finer); more than one
   bounce (M10 G25, opt-in); the error while flying reported each frame
   with a 95% bound (M4 G13: held in 10 of 10 checks, opt-in, 1 ms of GPU
   time). *Left:* lowering that error (its root is per-object page demand,
   Landmark 3), perhaps an error-first mode on the report, several bounces with
   streaming (item 2.2) and perhaps by default, the moving sun under a sky
   and with streamed lighting, the ray sky with movers and streamed levels.
3. **Detail without shimmer at frame rate:** sub-pixel coverage and colour
   in about 10 ms at 720p. *Done:* coverage accuracy (M13 G3–G4: 0.13–0.22×
   one ray's error). *Designed (§3.4):* the far field and the pass's
   speed, in two gates (stubs `m13_gate5`, `m13_gate6`). *Done since:*
   gate 5 (A, B, C, E): 38.25 ms standing, 20.35 ms from 150 m, accuracy
   held; gate 6 (D, coverage pages): settled 16.12 ms standing, 17.11 ms
   from 150 m; gate 7: pages keep up with a 10 m/s flight; gate 8: the
   coverage pass in about 9 ms settled (0.159× and 0.128× one ray's
   error). *Left:* appearance (colour), then into the renderer.
4. **One showcase scene** with all three, at 60 fps against Cycles, the
   quality dial working. *Not started.*

**Next, in order (the user's order, 2026-11-13):** the design step for
M13's far field first (done), then the three smaller items, then the
design's gates.

1. **Done (2026-11-13, tenth session): the design step for M13's far
   field.** Profiled first at full size (development,
   `docs/results/m13-far-field-profile.md`): from 150 m (104 ms) half is
   the bins' rays, the rest the maps' reads and a walk from the root to the
   instances in every pixel, and no node above the instances is small
   enough to stop at, so nodes holding coverage cannot help that view;
   standing (122 ms) the sky's pixels walked boxes straddling the eye's
   plane for nothing (removed exactly in development: 36 → 18.6 nodes a
   pixel) and the horizon is the far field. The design (§3.4, "The far
   field for sub-pixel coverage") has five parts: A exact culling by the
   boxes' faces, B the bins' rays only near large objects, C one walk per
   8×8 tile, E lighter map reads (gate 5, both views), and D hierarchy
   nodes holding coverage as **view-dependent coverage pages** (a node's
   coverage seen from the eye at one texel per bin, baked by hardware rays
   restricted to its contents, reused within a positional bound, a pool
   bounded by the screen; gate 6, the horizon). Prior art checked from
   abstracts; the closest is Shade et al. 1996's hierarchical image
   caching. Stubs: `m13_gate5`, `m13_gate6`.
2. **The smaller items, in this order:**
   1. **Done (tenth session): the error while flying reported** (M4 G13,
      passed; §3.3). Each frame 512 probe pixels shaded as shown and with
      lighting exact at their points give an estimate of the image's
      relative RMSE and a 95% bound; on M9's flight the bound held the
      error against settled renderers in 10 of 10 checks, the estimate
      1.215× it, for 1.023× the flight. It measures against exact lighting,
      so it includes the settled cache's own error (14–23% on the field).
      Not done: an error-first mode (bake until the bound is within ε,
      report the frame time), left for when the dial needs it.
   2. **Several bounces with streamed lighting: M14 G14 failed on one
      view's convergence check** (tenth session). Eight bounces on 36,864
      streamed rocks with a sky flew within 60 fps (p99 14.8–15.3 ms) with
      direct light certified (222,416 pages identical); flight 1's image
      after 8,000 frames was 1.038% from all resident (bar 1%), neither
      renderer settled, and I ran other GPU work during that phase, which
      shrinks the time-based bake budget (disclosed). Outdoors more bounces
      change these views by 0.2–0.3%. Left: a new gate for convergence at a
      fixed bake budget with the machine free; whether more bounces become a
      default (outdoors they add little, indoors two fifths).
   3. **Done (tenth session): coarser levels for imported meshes** (M14
      G15, passed). `continuum::simplify` makes coarser levels of any
      surface by half-edge collapses ordered by quadric error, so they keep
      the finest's vertices with their normals (gate 13's finest normals,
      for free) and chart coordinates; `InstancedScene::simplify_geometries`
      gives a scene's large geometries levels and a simplified lighting
      level. Found: an error of distance alone made levels show with
      triangles several pixels across (2.4–3.4× the analytic levels' colour
      difference in development); half the mean edge in the error fixed it,
      at up to 1.9× the triangles where the budget allows. Left: a gate for
      simplified lighting levels against a finest-lit reference; speed
      (1.45 s for 65,536 triangles).
3. **M13's far-field gates** from the design (§3.4): gate 5 (A, B, C, E),
   gate 6 (coverage pages), then the 10 ms bar, flights and appearance
   (colour in the same pages). **Gate 5 passed** (eleventh session,
   `docs/results/m13-gate5.md`): 0.219× one ray's error standing in 38.25
   ms, 0.135× from 150 m in 20.35 ms. A pass with one invocation per tile
   lists subtree roots; pixels walk below them as packets; bin rays run in
   their own pass; maps read two directions with coordinates affine across
   the pixel; open bins are bit masks. From 150 m about 3 ms is corners and
   roots, 6 bin rays, 3.5 the walk, 8 map reads; standing is mostly the
   horizon (walks up to 3,275 nodes). **Gate 6 passed** (eleventh session,
   `docs/results/m13-gate6.md`): the highest node on a path that fits a
   page (512 texels, a texel a bin) and holds at least 8 instances is read
   from its page, found in the roots pass and in the pixels' walks; pages
   bake after the pixels' pass (eight workgroups a page, rays from the
   node box's entry: from the eye they cost about 50× more, traversal
   meeting everything a grazing ray passes first). Settled: standing 16.12
   ms with 10,224 pages, 150 m up 17.11 ms (no pages; 150 m was sped up
   first: bin rays compacted, bounding spheres, stored root boxes). Walk:
   0.213–0.224× at 20 bakes a frame, frames 34 ms while a cold pool
   filled; from a settled start (development) a walk needs 1.5 bakes a
   frame (16.2 ms), but a 10 m/s flight 20 m up outruns 20 bakes a frame
   (pages baked together expire together; frames up to 41 ms).
   **Gate 7 passed** (`docs/results/m13-gate7.md`): bakes one invocation
   per ray (0.02–0.03 ms a page), bake eyes led along the motion by half a
   page's lifetime, refreshes past 80% of the bound on a second list, a
   parallel arguments pass; a 10 m/s flight 20 m up at 1.191× the settled
   frame at its place (bar 1.2×; the settled reference itself varies by
   15%), with a wave of 600 requests a frame at 4–5 s. **Gate 8 passed**
   (`docs/results/m13-gate8.md`): about 9 ms settled in both views.
   **Next:** colour (appearance) in the same structures; then the 10 ms
   bar's follow-ups
   (150 m: corner and bin rays about 5 ms), flights, then colour in the
   same pages (appearance).
4. **The showcase;** the NVIDIA benchmark when funded.

**Open, not scheduled** (after the above, or when a landmark needs them):

- **Settling at scale (studied, parked):** G7's views take up to 4,791
  frames to settle. Measured on 16,384 rocks (`stream_dev` `SETTLE`;
  first views settle in 400–1,600 frames): phases bound by production
  (four lighting workers at utility QoS, sharing the efficiency cores
  with four camera workers, deliver 13–30 levels a frame against ~270
  waiting) alternate with phases bound by the slots (64 lighting loads a
  frame but 8–45 pages baked, waiting pages pinning ~340 of 511 slots for
  neighbours' levels that bakes find only one round at a time). Done:
  normal smoothing by sorting (bit-identical, generation 1.6× faster:
  0.20 ms a lighting level, 9.9 ms the finest camera level), producers
  skipping requests not wanted for 8 frames (no measured gain). Levers
  left: predicting the neighbours a page's sun cones can reach (from the
  records' bounds) so a page's levels are wanted together, more slots, or
  a production budget on performance cores. Frame rate and exactness do
  not depend on it.
- **Lighting in motion, the rest:** the moving sun under a sky (skies
  re-bake with a moving light today only as pages re-bake) and at M14's
  scale; movers with streamed lighting; the ray sky with movers (their
  structure) and streamed levels (own-object hits on a camera level). On
  the moving sun: a tighter bound (Steiner's P δ is π/2–3× a translation's
  true change; the largest change seen was an eighth of the largest
  bound), and the ledger's appended work, which halves the bake budget for
  new pages (fallback 2.4% → 5.5%). Not the error while flying's cause:
  baking skies with their pages raised it slightly.
- **Parked features** (untouched unless a landmark needs them): emissive
  faces as lights (M10 G23–G24: 19.71% from Cycles, the faces showing
  emission without the light they reflect, worse supersampled for an
  unexplained reason), a second texture coordinate set, movers under a
  sky, moving lights' dark state, new material or glTF features, Bevy
  work; the ray sky with movers and streamed levels (it would trade M14's
  exact sky for a converging one: rays see camera levels, not lighting
  levels).

**The eleventh session (2026-11-13):** M13 gates 5, 6, 7 and 8 passed.
Dev tools: `m13_gate5` takes `M13_STAGE=16` (gate 4's path as a timing
control) and `M13_ACC=1` (accuracy on other views than the gate's) with
`M13_DEV=1M`.

- *M13 G5:* **passed** (0.219× / 0.135× in 38.25 / 20.35 ms). From the
  tenth session's 116 / 42 ms: tile roots in their own pass (86 / 29),
  full bins skipped and two directions (51–53 / 25.5), set-up linearized
  (49.8 / 22.6), bit masks (42.6 / 22.8), bin rays in their own pass
  (38.8 / 20.4). Packets still beat independent walks (42.5 standing).
- *M13 G8:* **passed** (9.01 / 9.10 ms settled, 0.159× / 0.128×). Development:
  `docs/results/m13-gate8-development.md`; `m13_gate8` takes `M13_DEV=1M`
  and `M13_ACC=1`; `m13_gate7`'s `M13_PROFILE` prints each pass's time.
- *M13 G7:* **passed** (flight 1.191× against 1.2×, bakes 1.39 ms, accuracy
  0.153–0.158×, cold walk 16.50 ms). `m13_gate7` dev modes: `M13_SECONDS`,
  `M13_END`, `M13_COST`. Lesson: on this device a pass's timestamps did
  not bracket its work; time passes by `frame_profiled` (now twelve).
- *M13 G6:* **passed** (settled 0.212× / 0.135× in 16.12 / 17.11 ms;
  10,224 pages; walk 0.213–0.224×). Profile stage 18; `m13_gate6` takes
  `M13_DEV=1M` with `M13_ACC=1`, `M13_WALK=1` or `M13_BAKECOST=1`.
  Development: `docs/results/m13-gate6-development.md`.

**The tenth session (2026-11-13), step by step:** the far-field design
(§3.4), the error while flying reported, several bounces with streaming,
levels for imported meshes, and gate 5 begun. Dev tools added:
`simplify_dev`; `m13_gate4` and `m13_gate5` take `M13_DEV=1M` (gate 5) for
full-size development.

- *M13's far field designed:* profiled at full size first (from 150 m half
  the time is bin rays, standing the sky's wasted walk and the horizon);
  five parts, coverage pages for hierarchy nodes the far field's.
- *M4 G13, the error while flying reported:* **passed** (512 probe pixels
  shaded with exact shares by hardware rays; bound held 10 of 10, estimate
  1.215×, 1.023× the flight). The settled cache is itself 14–23% from exact
  lighting on the field.
- *M14 G14, eight bounces with streamed lighting:* **failed** criterion 3 at
  one view (1.038% against 1%, neither renderer settled; I ran other GPU
  work during that phase, disclosed). Frame budget and certification pass.
- *M14 G15, levels for imported meshes:* **passed** (`continuum::simplify`;
  shading within 0.77–1.10× the analytic levels'; stated errors hold). The
  error needed half the mean edge as well as the distance.
- *M13 G5:* built in part, not run (above).

**The ninth session (2026-11-13), gate by gate:** a CI test failing since
M14 G9 exposed a real bug, fixed; then seven gates, six passed, and M13's
far field scoped. Dev tools added: `sky_dev`, `sky_rays_dev`,
`flight_error_dev`, `bounces_dev`, `normals_dev`, `normals_cmp_dev`,
`m9_dev` modes (`real`, `nocap`, `budget=`, `max=`, `profile_from=`,
`step=`, `work`).

- *Fixed after G10 (a CI test failing since G9):* with movers under a sky,
  a page marked invalid while its sky was still the guess was not listed
  for eviction, and a later frame's partial re-bake cleared its mark,
  showing texels stale from the earlier move. Any small bake budget showed
  it (G9's frame-period cap had shrunk the test's budget). Marked pages are
  now always listed and never re-baked in part; the cap applies only with
  a frame target (`frame_target_ms` 0 keeps the budget fixed, as
  documented). A test at a small budget guards it.
- *M9 G6, a moving sun at real speed:* **passed** (0.98–1.02× the still
  flights against gate 4's 1.6×; 886,506 pages within their bounds; at rest
  0.054% from the fixed sun). The light cone's bins wavered as they
  re-centred on a moving light, which no motion bound covers; bins on a
  lattice fixed in the page's version frame make the ratio move only with
  the light's footprint. Development found set-aside pages re-baking whole
  and a growth check costing 4 ms a frame.
- *M5 G13, the sky through hardware rays:* **passed** (views warm in
  166–400 frames with default options against 1,827–15,905; closer to
  Cycles standing, 2.53% against 4.67%; flights 0.947×). A cone sky sample
  cost 10.3 µs (17 sun texels, priced at 10); 192 rays cost 0.72 µs.
- *M14 G11, the cap times the renderer's own frames:* **passed** (gate 10's
  flights still within budget; readbacks every 20 frames no longer slow a
  warm-up: 0.96–1.05×).
- *M4 G12, pages requested three levels finer:* **passed** (error while
  flying 0.819× at a fixed budget, 20.77% against 25.36%; warm-up 0.70–0.76×,
  identical images; flights 1.014×). First measure of the error while
  flying, against renderers settled at the same cameras.
- *M10 G25, more than one bounce:* **passed** (the showcase room 6.54%
  from Cycles at 12 bounces with 8, against 45.36% with one; settled in 80
  frames; the textured scene unharmed, 2.11% against 2.36%). Measured first
  that past the first bounce outdoors carries under 1%, indoors two fifths.
- *M14 G12–G13, coarser levels with the finest's normals:* G12 failed on a
  method fault (an equality check under the time-based budget), G13 with a
  fixed budget **passed** (0.56–0.65× the colour gap to the finest-resident
  images, production 0.87×). The gain is the normals' direction; a length
  factor for averaged normals added nothing measurable.
- *M13's far field scoped (development, not a gate):* at 10,000
  boulders M13 G4's path takes 25.5 ms standing and 57.7 ms from 150 m;
  the walk to instances and the bins' rays carry it, not the maps'
  reads. The walk as a packet (committed) is 14% faster from 150 m;
  larger splat thresholds changed nothing. The route left needs a
  design step (item 1 above).

**The eighth session (2026-11-12), gate by gate:**

- *M14 G5, lighting levels produced on demand, certified beyond one
  renderer:* failed on frame rate only (26–39 ms); passed scale (36,864
  rocks, lighting levels 72× the slots and 26.9× a 256 MB CPU cache),
  certification (676,080 pages identical to a reference holding the
  flights' region with every lighting level resident) and memory (27.7 KB
  of GPU memory a rock).
- *M14 G6, gate 5's world within 60 fps:* failed on one frame (42.3 ms
  against 33.3 ms in flight 2); passed everything else: means 6.2–6.5 ms,
  99th percentile 10.2–11.9 ms, 1,416,782 pages identical, 17.8 KB of GPU
  memory a rock. Fixes, found by profiling: lighting uploads through a
  staging belt, a persistent Metal residency set and incremental top-level
  builds (patched wgpu, `third_party/`), clusters of proxies, the
  streamer's scans narrowed to tracked lists.
- *M14 G7, frame-time stability at scale:* **passed** (no frame over 16.7
  ms in the three flights: means 4.8–5.9 ms, p99 8.3–10.1 ms, worst 15.8
  ms; 908,545 pages identical; 17.8 KB a rock). A replay from unsettled
  views showed spikes spread over unrelated stages; producer threads at
  utility QoS, frees off the render thread and larger staging chunks took
  them from 58 to 3 in 5,400 frames there. Settling became slower.
- *M14 G8, a sky and bounce light with streamed lighting:* failed on the
  frame budget only (p99 20.5–25.1 ms with a sky and bounce on 36,864
  rocks); passed certification (436,004 sun and 436,002 sky pages
  identical on 9,216 rocks), bounce (settled images within 0.07–0.52%),
  memory (20.0 KB a rock). Found in development: pins held by retries that
  could not progress deadlocked the slots; a bounce retry cleared marks.
- *M14 G9, the frame budget with a sky and bounce:* failed on the frame
  budget only (run twice, the first disturbed by a game; p99 19.8–21.6
  ms): a frame-period cap on the bake budget held the mean, not the tail.
- *M14 G10, bounce rays through hardware ray queries:* **passed** (p99
  14.5–15.0 ms with a sky and bounce on 36,864 rocks; 177,375 sun and sky
  pages identical; bounce 3.09% from Cycles on M10 G22's scene). Profiling
  showed bounce samples' lighting-level wants churning the slots and their
  compute rays costing about 2 ms; rays through the camera's structures
  need no lighting level. The frame budget is 11 ms by default.

**The seventh session (2026-11-12), gate by gate** (numbers in
`docs/results/`, design notes in the milestone table):

- *M14 G1, camera levels streamed by need:* failed on frame rate in one
  flight of three (1.124× the resident renderer against 1.10×); passed
  scale (6.35× a 2.5M-triangle budget), settled coverage (0.062–0.096×
  one ray's error), 3.4× less GPU memory.
- *M14 G2, fewer and cheaper loads, plus a zoom:* failed on frame rate in
  three of four flights (1.110–1.113×); passed coverage and the zoom. A
  load costs about 29 ns a triangle (22 of it building the structure,
  about 45 million triangles a second on the M4); the volume is mostly
  inherent to flying past unique detail. Led to the user's 60 fps decision.
- *M10 G24, the cache settles with bounce light:* liveness passed at all 16
  sub-pixel offsets (pages baked for bounce hits stay resident while
  bounce samples wait); failed on the parked emissive faces' accuracy.
- *M14 G3, beyond CPU memory within 60 fps:* **passed** (4,096 rocks, 33 GB
  of detail on demand, worst frame 14.5 ms, settled within 0.018–0.030×
  one ray's error of the ideal levels).
- *M14 G4, lighting levels streamed, certified:* **passed** (8× the
  lighting budget; 2.68 million shown pages bit-identical to an
  all-resident renderer; memory 0.486× gate 3's way, a thin margin; worst
  frame 18.2 ms; 13–15 lighting loads a frame, settling up to 3× slower).

**Where the M14 code is** (see also CLAUDE.md): `stream.rs` holds camera
levels (`Streamer`: feedback readback, paged pool, level chains, proxies,
the time budget, `LevelSource` workers) and lighting levels
(`LightingPool`, `LightingLayout`, `LitPackage`); `stream.wgsl` is the
feedback pass; `hw_visibility.wgsl`'s `cam_tri` maps a hit through the page
table (after the objects' ranges in `meshes`); `mask_cones.wgsl` has
`lit_missing` (set where a walk meets a stub) and `LIT_MARK` (a page baked
against an absent level, missing until re-baked); `vcache.rs` has the
`lit_*` fields (pending pages, waiting pages, per-page needs, pins) and
`pinned` (bounce pages, M10 G24); `instanced_buffers` (lib.rs) and
`VirtualCache::new` cut streamed geometries out of the buffers when
lighting streams. Lighting levels on demand (G5): `LevelSource::lighting`,
`stream::lit_package` (a package built alone), `LitRecord` (the stub),
`LitProducer` and the CPU cache in `LightingPool`. Clusters of proxies
(G6): `hw::proxy_clusters`, `Clusters` in `hw.rs` (cluster structures,
slots for finer levels, `rebuild_clusters`), `hw_object` in
`hw_visibility.wgsl`, the member lists after the objects' ranges in
`meshes` (before the page table). The rock field is `room3d::rocks`
(`Rocks::new`, `with_detail`, `on_demand`, `with_proxies`,
`lighting_on_demand`, `region`; `flight(k, t, half, fov)`). `stream_dev`
`GATE5=<side>` times gate 5's flights stage by stage; `RECORDS` measures
GPU memory per rock.

**M14 caveats to keep in mind:** lighting streaming needs the streamed
geometries last in the scene, a proxy on each, no sky, bounce, movers or
moving lights; clusters hold at most about 1.6 million objects (member
lists addressed in 23 bits); a page's box (`page_bounds`) is empty for
lighting levels produced on demand (used only with movers);
camera levels carry no vertex colors, tangents, cutouts or blending; the
G2 run had about 0.1 ms a frame unexplained and 10–14 ms CPU spikes later
traced to waits counted as CPU time; `read_cache_pages` was wrong without
a sky until G4 (fixed; only diagnostics used it).

- **The fourth session's features** (all in `docs/results/` and the
  milestone table): viewer `--env` (HDR/PNG environment images);
  `KHR_texture_transform` (Blender's sign); vertex colors (M10 G17–G18:
  1.29% from Cycles, 1.00× the frame); glTF save of emissive textures,
  BLEND and colors; blending speed (M10 G19: 1.23× the frame); glTF node
  animations through movers; Bevy `ContinuumInstance` movers;
  `Renderer::set_exposure`; texture wrap modes (M10 G20: 2.18%); bounce
  light with any number of lights (M10 G22: 3.07%, speed 1.167× against
  1.15×, kept); emissive faces as lights (M10 G23 failed, experimental).
- **Third session's results** (for reference): passed M5 G9 (the sky
  later), M10 G10 (materials, 5.85%), M10 G15 (emissive textures, 6.35%),
  M5 G12 (environment light, 1.37%); failed but kept M10 G8–G9
  (back-to-back faces), M7 G4 (silhouette rays), M10 G11–G12 (emission via
  bounce, 7.46%), M10 G13 (MikkTSpace), M10 G16 (blending, since fixed by
  G19); `RendererOptions::environment`; the NVIDIA benchmark prepared
  (`docs/BENCHMARK.md`).
- **Earlier candidates, as left:**
  - *Movers under a sky:* deferred. A per-sample bound with the cosine
    and the horizon would clear ground beyond about 10 m from the movers,
    but not nearer. The sky later with movers (M9 G5) is sound but gains
    nothing in flight: both caches lag 25–30% behind the settled image in
    motion.
  - *Moving lights' mixed texels:* 97.5% of texels store exactly 0 or 1,
    so a tighter band bound helps almost none. A dark state cut the cost
    to 1.25× but failed soundness, because coverage summed within a mesh
    can over-count. Ways out were assessed:
    - a bin counted only when one triangle covers it is sound but almost
      never full;
    - an exact union per bin needs polygon clipping in the hot loop;
    - a guard traced at the light cone's resolution costs about 9×.
  - *Silhouette rays:* nothing reaches 1.5× where half the pixels are
    edges. Four quarters give 0.41× the error at 2.62× the frame. A cap
    per frame would flicker without blending frames.
- **Known issues:**
  - the total sky work is large (the sky later orders it, but does not cut
    it);
  - sustained speed on the fanless Air (M5 G8);
  - only Metal has been measured;
  - blending and silhouette rays need hardware ray queries and no movers;
  - Naga 30's Metal backend ignores ray flags' face culling. It is worked
    around in the camera's passes and worth reporting upstream (the
    user's call);
  - the 29-buffer limit per stage on Metal. Every new kernel input so far
    has been packed into an existing buffer (emission words after the
    object textures, the blend clear in the transforms' row 3, the
    environment as an extra base color layer; M14: a camera level's page
    table after the objects' ranges in `meshes`, its entry in the
    transforms' row 3 `w`, missing lighting levels' pairs after the
    `incomplete` list). The bounce bake binds all 29; `vc_bake` binds 27
    since M14 G4.
- **Needs the user:**
  - running the NVIDIA benchmark (needs funds; not soon);
  - any publishing or upstream report.
- **Working notes:**
  - Gate binaries are `experiments/room3d/src/bin/m<milestone>_gate<n>.rs`.
  - Test scenes are in `room3d::textured`: `Textured::new()` with
    `.with_materials()`, `.with_normal_maps()`, `.with_curved_normal_maps()`,
    `.with_foliage()`, `.with_foliage_back(m)`, `.with_glow()`,
    `.with_sign()` / `.with_sign_sided()` and `.with_glass()`. The field
    scene is `room3d::field`.
  - Cycles jobs (`room3d::cycles`) take emission, emissive textures,
    blending and an environment image (`sky_map`, written as HDR).
    References are cached per work directory (`out/<gate>_cycles`) and
    re-rendered when the job changes.
  - Settle a view by waiting for 32 quiet frames rendered after the call.
  - When lights are visible, score over surfaces: their visible disks
    decide all-pixel scores.
  - Keep coincident emissive or transparent faces out of Cycles scenes.
  - Gate commits run
    `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
    first and commit only if it passes (two lint slips reached `main`).
    CI uses the latest stable clippy, newer than the local default (1.98):
    also run `CARGO_TARGET_DIR=target/stable cargo +stable clippy
    --workspace --all-targets -- -D warnings` (stable 1.100 is installed).
  - Streaming tools: `stream_dev` (working-set sizing; `LOAD_COST`,
    `BASELINE`, `MEMORY`, `PAGES` page-by-page cache comparison, `STILL` a
    still view's progress, `SNAPSHOT`); `CONTINUUM_TRACE_LIT` traces the
    plan's lighting-level decisions; `emitters_dev dump` writes a settled
    linear image. Gates `m14_gate1`…`m14_gate4` share helpers (settle,
    4×4 samples, coverage error) by copy.
  - Development tools: `backface_dev`, `ledger_dev`, `edges_dev`,
    `spikes_dev` (work and passes modes), `vcolor_dev`, `blend_dev` (each
    camera pass alone) and `bench`.
  - Speed criteria comparing two renderers: settle both, then five
    interleaved rounds of 30 warm frames each, median ratio (M10 G18). A
    single back-to-back pair met a transient in G17. CPU timings the same
    way (five interleaved rounds, medians; M14 G12).
  - Equality criteria between renderers need a fixed bake budget
    (`CacheOptions::frame_target_ms` 0): with the time-based one, settled
    images vary a little with the machine's timing (M14 G12). A process's
    first renderer settles to images 0.004–0.037% from later ones; render
    and discard a first pass.
  - Save a gate's output as it runs (`… | tee file`), and never show it
    only through `tail`: M10 G21 lost two criteria's lines to a crash.
  - Give a gate binary a development mode (`M13_DEV=1`: a smaller scene,
    no reference) and run it once before the gate: a shader that fails to
    compile otherwise costs the gate its only run.
  - M13's code: `mask_cones.wgsl`, sections "M13 gate 2" (the scalar walk,
    `View3`, screen boxes, the pixel frustum), "M13 gate 3" (bins,
    `cov_maps`, `map_coverage`) and "M13 gate 4" (`m13_large`);
    `hw_visibility.wgsl` (`hw_m13_corners`, `hw_m13_near`); the far
    walk as a packet is `map_coverage_from` (`packet_any`, `packet_first`);
    `coverage_maps.rs` (building the maps). Profile stages select them:
    10–11 (G1), 12–14 (G2), 15 (G3), 16 (G4); `kernel.set_pixel_cones(true)`
    with the stage set before the renderer is made.

**Decided 2026-11-07: a more complete paradigm before the tag.** The
user's order (all attempted; results above and below):

1. **Moving objects in the view cache** (M9 gate 1): the second rule
   (exact incremental updates) in `Renderer`, not only the lightmap path.
   **Done (2026-11-07), M9 G1 passed** (`docs/results/m9-gate1.md`).
2. **Moving lights, and movers' effect on sky light, through the error
   ledger** (M9 gates 2–3): the third rule, the invention (§3.1), on the
   GPU in 3D. **Moving lights tried (2026-11-07), M9 G2 failed on fallback
   only** (`docs/results/m9-gate2.md`): sound and exact in every check,
   0.69% of pages re-baked, but 33–34% fallback from pages set aside off
   screen. **M9 G4** (no work
   limit; `docs/results/m9-gate4.md`) kept soundness and cut fallback to
   about 5%, but costs 1.6× at a real sun's speed: mixed texels re-bake
   every frame the light moves, as their bound (with ε for snapping, 7.5×
   loose) always exceeds ε. A tight bound for them is next. **Movers under
   a sky tried (2026-11-07), M9 G3 failed on work and speed**
   (`docs/results/m9-gate3.md`): sound, but the per-page sphere bound
   re-bakes 99.7% of pages' sky each moving frame (3.6–4.0× the still
   flights).
3. **Tag v0.1.** **Done** (2026-11-07): `v0.1.0` on commit `41c0df2`, CI
   green on Linux and Windows.
4. **Bounce light in the view cache** (M10). **Tried (2026-11-07), M10 G1
   failed on its re-coloring check only** (`docs/results/m10-gate1.md`):
   1.79% from Cycles with one bounce (7.09% without), warm frames 1.07×.
   Then M10 G3 (secondary requests: no cascades; materials settle in 1.12×
   the frames instead of 6.9×) and **M10 G4 passed** the re-coloring check:
   gate 1's failure was its own settle procedure, which returned after one
   frame (`docs/results/m10-gate4.md`).
5. **Glossy reflections and PBR materials** (M10): roughness cones,
   metallic and roughness, normal maps, emission, alpha. **Tried
   (2026-11-07), M10 G2 failed on accuracy** (`docs/results/m10-gate2.md`):
   8.94% from Cycles against 6% (12.71% without reflection rays), warm
   frame 1.82×. (With M10 G3's fix it settles in 1.12× the frames; still
   9.01% from Cycles.) **Emission** added (2026-11-07, no gate: exact by
   construction, tested in `emission_adds_to_the_cached_image`), and glTF
   import of metallic, roughness and emission factors. **Normal maps done
   (2026-11-07), M10 G5 passed** (`docs/results/m10-gate5.md`): 3.63% from
   Cycles with the maps (26.42% ignoring them), warm frame 1.04×. **Alpha
   cutouts done (2026-11-07), M10 G7 passed** (`docs/results/m10-gate7.md`;
   gate 6 reached no criterion from a cache bug, now fixed): cut into
   geometry, 3.29% from Cycles (34.07% with opaque cards), 1.15× the frame.
6. **Anti-aliasing, the narrower way** (M7 gate 2 onward): texture
   filtering for the area integral, then coverage only at silhouettes.
   **Texture filtering done (2026-11-07), M7 G2 passed**
   (`docs/results/m7-gate2.md`): a box filter over the pixel's footprint
   cut the textured scene's error from the area integral from 7.11% to
   3.99%, at 1.03× the warm frame. **Silhouette rays tried, M7 G3
   failed** (`docs/results/m7-gate3.md`): four rays at edge pixels halve
   the field's error from 150 m (0.49×) but cost 13× there in compute rays;
   on the textured scene edges are 1% of pixels and the gain is 0.91×.

The steps below are the earlier v0.1 plan (2026-10-25, after M5 gate 7).

The user's budget allows about a third again of the work so far (about 5–7
sessions). That fits a v0.1 release (§3.7) with the scope below; v1.0 does
not fit.

1. **Finish the library's surface** (item 5). **Done** (2026-11-06):
   `Renderer`, the `render` example, `docs/GUIDE.md`; the viewer's field and
   glTF scenes run on `Renderer`; every public item has rustdoc
   (`#![warn(missing_docs)]` in the three library crates); experiment-only
   items (`InstancedRenderer`, the kernel's tuning switches, diagnostics)
   are `#[doc(hidden)]`. `LiveRenderer` stays public as the only path with
   moving objects.
2. **Pixel-footprint cones for primary visibility** (item 4). **Tried
   (2026-11-06), M7 G1 failed**; v0.1 keeps one ray per pixel.
   - Exact 8×8 coverage per pixel costs hundreds to thousands of node tests
     per pixel at scale.
   - Narrower follow-ups, each for a new gate: texture mips chosen for the
     area integral (cheap, the larger error on textured content); coverage
     only at silhouette pixels; 3D clipping of huge triangles in narrow
     cones.
3. **Bevy plugin.** **Done** (2026-11-06, not gated): `bevy_continuum`
   renders a Bevy 0.20 camera's view with `Renderer` on Bevy's own device.
   Bevy 0.20 uses wgpu 30, so the workspace's patched wgpu-core serves
   both. `Renderer::compute` and `encode_display` let the engine order the
   draw; float targets get linear light for Bevy's tone mapping.
4. **CI** (2026-11-06/07, `.github/workflows/ci.yml`, private repository
   `github.com/bupfff/continuum`).
   - Linux, Vulkan on lavapipe: passes. Format, clippy, rustdoc and every
     test, the GPU tests included. The golden image is 0.09% from Apple
     M4's, and cones match the CPU to 1.6e-5.
   - It found two portability bugs, both fixed. Naga's SPIR-V output
     panicked on a pointer to a struct's member (`bound_cone`). A Windows
     checkout's CRLF line endings broke the shader's text patch for
     subgroups.
   - `continuum-gpu/tests/shaders.rs` translates every entry point to
     SPIR-V, HLSL and MSL on the CPU, on every machine.
   - Windows, DX12: every entry point compiles with DXC as wgpu calls it
     (0.05–4.8 s each, 23 s in all). The kernel cannot run on WARP, the
     software adapter: 31–41 s per pipeline, then out of memory. That is
     WARP's own shader compilation, so the GPU tests run on Linux only.
     Real Windows GPUs are untested (`docs/results/ci-portability.md`).
5. **Needs the user:**
   - a cloud GPU benchmark (NVIDIA) needs an account;
   - publishing is theirs to decide.
6. **Known limitations for v0.1:**
   - **Sustained speed** (M5 G8): flights with several lights exceed 16.7 ms
     after minutes of load on the fanless Air. A sustained-speed gate should
     state the thermal state; levers are a lower bake floor and cheaper
     camera passes.
   - **Sky warm-up** (M5 G5): 13–95× the sun's frames. Sharing the far field
     and sparser samples did not pay. The sky later (M5 G9) shows shadows
     first and brings views within 5% in 120–300 frames. Sky samples
     through hardware rays (M5 G13) warm the field's views in 166–400
     frames with default options.
   - **Sky accuracy:** binned coverage is orientation-dependent and always
     too dark. Candidate: first moments per bin.
7. **Open questions:**
   - offering the wgpu-core TLAS-dependency fix upstream (needs the user's go
     ahead before posting anything);
   - the world-space lightmap path (`LiveRenderer`) still refuses a sky and
     takes one light;
   - wgpu compiles Metal shaders with fast math, so moving code between
     kernels changes rounding (0.02% on the sun channel in M5 G5). Results
     stay deterministic within a build;
   - Metal pass timestamps: an end-of-pass timestamp followed only by copies
     can land too late, and a render pass's start is not ordered after
     compute passes (M5 G5); the cache times its camera passes up to an
     empty compute pass.

**v1.0, competitive in its niche** (a render library, not a full engine):
another 20 or more batches.

- Specular and glossy reflections (roughness cones) and PBR materials with
  normal maps.
- Multi-bounce GI.
- Moving lights through the error ledger, the paradigm's invention (§3.1),
  which is still unproven in 3D.
- World streaming.
- Deforming geometry (M8).
- Performance tuning per GPU vendor.

**Not planned to match UE5:** UE5 is a whole engine (editor, animation,
physics, audio, networking, tooling). Continuum aims to be a render library
whose lighting is exact, stable and incremental, used inside other engines
(Bevy first).

## 4. Design rules

1. **Worst case parity, typical case win.** The everything-moving fallback must
   match modern engines on quality and cost. Gate every milestone on it.
2. **Never worse, using our own paths.** No rasterization. Hard cases get a
   slower path inside the paradigm (deforming meshes: per-frame subtree refit,
   traced by the same cones, feeding the ledger through swept bounds).
3. **Frame-complete.** No blending of previous frames for image quality. Reuse is
   allowed only when the ledger proves it is within ε.
4. **Measured, not claimed.** Independent reference (Blender Cycles or Mitsuba 3),
   FLIP for image error, fixed benchmark protocol for timings.
5. **Portable.** wgpu only, compute only: Metal, DX12, Vulkan, later WebGPU.

## 5. Library shape

- `continuum`: core types (masks, bounds, ledger, hierarchy), CPU reference paths.
- `continuum-gpu`: wgpu backend. `Renderer` is the front door for v0.1:
  instanced scenes with moving objects, lights and sky, through the view
  cache (`docs/GUIDE.md`).
- `continuum-gltf`: glTF import and export.
- `bevy_continuum`: Bevy 0.20 integration (a camera's view rendered by
  `Renderer` on Bevy's device; `docs/GUIDE.md`).

The direction for v1, once the ledger drives moving objects and lights (not
final):

```rust
let mut renderer = Renderer::new(&device, QualityContract {
    max_error: Epsilon::Jnd(1.0),
    frame_budget_ms: 16.6,
});
renderer.apply(&scene_changes); // moves, light edits: charged to ledgers
let frame = renderer.render(&camera); // frame.image, frame.certified_error
```

## 6. Milestones and gates

| Milestone | Content | Gate |
|---|---|---|
| M0 flatland (CPU, 2D) | Mask cones, ledger, worst case | G1 masks beat rays on error per cost (**passed**, `docs/results/flatland-gate1.md`); G2 ledger soundness: certified bound ≥ actual error, every pixel, every frame (**passed**, including forced-deferral stress runs); G3 ledger skips ≥ 80–90% of lighting work in typical animated scenes (**passed: 87.7%**, all from dependency regions, see `docs/results/flatland-gates2-3.md`); G4 everything-moving case costs no more than a conventional equivalent (**failed** on one configuration: geometric charging 0.90–0.95× a full recompute, projected charging 1.84–1.96×, `docs/results/flatland-gate4.md`); G5 magnitude deferral pays off for wide cones with continuous visibility (**passed on attempt 2: 34% of ε = 0 work**, `docs/results/flatland-gate5.md`); G6 one-bounce GI with charges that propagate through surfaces (**failed**: best implementable charges 66.9% vs ≤ 50%, zero violations; an oracle with exact charges reaches 27.4%, so the gap is charge tightness, `docs/results/flatland-gate6.md`); G6b exact incremental indirect via cone G-buffers (**passed: bit-exact, 7.4× faster than a full recompute**, `docs/results/flatland-gate6b.md`); G7 moving light, the ledger's own job (**failed narrowly**: 25.1% of cones recomputed vs ≤ 25% at a realistic sun speed, zero violations; exact invalidation needs 100%, `docs/results/flatland-gate7.md`) |
| M1 3D CPU reference | 3D mask cones over a BVH, reference path tracer, scene import | G1 repeated in 3D with 8×8 bins (**passed on attempt 2**: lower error than 64 jittered rays at 0.22× their cost, no flicker, `docs/results/m1-gate1.md`); G2 hierarchy culling (**failed on the letter**: identical candidate sets, but float summation order differed by 2.5e-6 vs a 1e-6 tolerance; culling scales 6.3× for ~100× triangles, `docs/results/m1-gate2.md`); G3 tile-shared traversal (**failed**: identical candidates but only 20–24% cheaper vs a 67% target, `docs/results/m1-gate3.md`); G4 exact incremental direct light in 3D (**passed: bit-exact, 14.9× faster**, `docs/results/m1-gate4.md`); agreement with Cycles/Mitsuba |
| M2 GPU primary visibility | wgpu, tile frustum traversal, mask coverage AA | G1 mask cones in a WGSL compute shader (**passed on attempt 2**: matches the CPU to 6.4e-6, 7.5× faster than 10 CPU threads on the M4, `docs/results/m2-gate1.md`); G2 mesh hierarchy on the GPU (**failed**: identical output, rays 7.5× faster, but the frame only 1.6× faster because cone setup and rasterization dominate, `docs/results/m2-gate2.md`); G3 960×540 at 60 fps with per-pixel cones every frame (**failed**: 33.6 → 26.0 ms, identical output; per-pixel cones every frame are the brute-force path, `docs/results/m2-gate3.md`); G4 world-space lighting cache (**passed on attempt 2**: 1080p in 5.66 ms at 0.58% RMSE vs per-pixel cones, `docs/results/m2-gate4.md`); G5 moving objects with exact soft shadows (**passed**: incremental block re-bake on the GPU equals a full re-bake exactly on every frame; 9.85 ms per 1080p frame with 6.1% of blocks re-baked, vs 21.9 ms for a full re-bake alone, `docs/results/m2-gate5.md`); G6 one-bounce indirect light from cone G-buffers (**passed on attempt 2**: exact against a from-scratch trace on every frame after padding hierarchy boxes, 3.6% RMSE vs 16× the directions, 11.66 ms per 1080p frame with moving boxes, `docs/results/m2-gate6.md`); real-time viewer with moving boxes and indirect light (`apps/viewer`, about 10.8 ms per frame at 2560×1440 on the M4 Air); frame time at 1080p on M4 Air; image-diffed in CI |
| M3 general meshes and a first showcase | Per-mesh triangle hierarchies, chart-based lightmaps with smooth normals, static coverage cache, probes on moving objects, procedural showcase, Cycles reference pipeline | G1 triangle hierarchies on the GPU (**failed**: identical output but 3.9× vs a 10× target on a 40k-triangle sphere, because cone cost grows with the triangles really inside cones; that is the prefiltered hierarchy's job, `docs/results/m3-gate1.md`); G2 general meshes in the lighting cache (**passed**: 1.61% RMSE vs per-pixel cones with the same normals, 4.99 ms at 1080p, `docs/results/m3-gate2.md`); G3 the showcase in real time vs Cycles (**failed narrowly on speed**: bit-exact on all 300 frames and 3.80% RMSE vs Cycles, but 17.76 ms vs 16.7 ms per 1080p frame, `docs/results/m3-gate3.md`) |
| M4 scale | Prefiltered cones, instancing, large-world coordinates, a virtual lighting cache | G1 prefiltered cones (**passed**: flat cost from 40k to 2.56M triangles, 86 ms vs 13.7 s exact, 0.55% from exact, `docs/results/m4-gate1.md`); G2 ten billion instanced triangles in a large world (**failed** on two of three: the 64 km-offset field renders identically with rebasing, and 10^10 triangles fit in 152 MB, but instanced vs flattened is 0.22% vs a 0.1% target because the flattened reference is the less precise one, and 10^6 vs 10^3 instances cost 3.04× vs 2× because camera rays near the horizon dominate; cones to a small sun are nearly free, `docs/results/m4-gate2.md`); G3 nearest-first ray traversal (**passed**: identical output, 10^10 vs 10^7 triangles 1.40×, 2.83× faster than the fixed order, 100.8 ms at 1080p for a million boulders, `docs/results/m4-gate3.md`); G4 visibility buffer (**failed**: identical output but 0.88× on the field and 1.11× slower on the showcase; register pressure was not the limit, ray traversal itself is, `docs/results/m4-gate4.md`); G5 traversal stacks in workgroup memory (**failed**: identical output but 2.6× slower, as 12 KB per workgroup limits occupancy, `docs/results/m4-gate5.md`); G6 hardware ray queries for camera rays (**failed**: no faster, 0.9×, and 67–270 ms per 720p frame. A corrected diagnostic shows per-pixel sun cones, not rays, dominate; gates 2 and 4 had blamed rays from a broken "light turned away" test, `docs/results/m4-gate6.md`); G7 cones that pay only for what they see (**failed**: no speedup; a stage-by-stage profile shows cones descending the triangle hierarchies of the objects they start on dominate, 119 of 215 ms from 150 m up at 720p, so per-pixel cones every frame are the brute-force path at scale too, `docs/results/m4-gate7.md`); G8 view-dependent lighting cache (**failed** all three: 14–39% from per-pixel exact cones, 20–37 ms warm, 42.7 ms mean in flight; but warm shading fell from 47–233 ms to 2–3 ms and only 1.8% of pixels lacked a page in flight. Diagnostics: one sample per pixel is itself 13–24% from the supersampled image, so later quality gates compare with the area-integrated image; and wgpu spends 15–17 ms of CPU time per `submit` walking the TLAS's million per-instance dependencies, while hardware rays take 4.5–6.6 ms on the GPU, `docs/results/m4-gate8.md`); G9 the cache at speed (**failed** on the flight's 99th percentile, 34.0 vs 33.3 ms. Same images as G8; warm frames 2.9–4.7 ms at 720p with hardware rays after a wgpu-core patch that keeps one TLAS dependency per distinct BLAS, pipelined requests and a faster page manager; flight mean 12.8 ms, 1.6% fallback. The flight is bound by CPU planning, which re-derives a backlog of ~50k wanted pages every frame, `docs/results/m4-gate9.md`); G10 smooth flight (**failed**: progressive requests from the shader and GPU deduplication cut CPU planning to 1.9 ms per frame, but the flight is then GPU-bound at 17.6–17.9 ms mean, 38 ms 99th percentile on a hot machine; next, baking by frame time instead of a fixed cone count, `docs/results/m4-gate10.md`); G11 baking to a frame-time target (**passed**: same images; warm 6.5–8.0 ms at 720p; three flights over a million boulders at 14.6–15.0 ms mean, 26–32 ms 99th percentile, 1.7% fallback. Caveat: the budget sits near its floor while flying, so only 0.6% of samples reach their wanted mip in motion and lighting is softer until the view settles, `docs/results/m4-gate11.md`); G12 pages requested three levels finer (**passed**: the error while flying, against renderers settled at the same cameras, 20.77% against 25.36% at a fixed budget (0.819×); views warm in 0.70–0.76× the frames with identical images; flights 1.014×, `docs/results/m4-gate12.md`); G13 the error while flying reported each frame (**passed**: 512 probe pixels shaded as shown and with lighting exact at their points by hardware rays give an estimate and a 95% bound (`CacheOptions::error_probes`, `CacheStats::error_estimate`/`error_bound`); on gate 12's flight the bound held the error against settled renderers in 10 of 10 checks, the estimate 1.215× it (it includes the settled cache's own 14–23% against exact lighting), 1.023× the flight, about 1 ms of GPU time, `docs/results/m4-gate13.md`); zoom and flythrough with no shimmer |
| M5 lights (v0.1 item 2) | Several lights, a sky environment, colored light | G1 sky light through six cube-face mask cones (**failed**: sky alone 2.74% from Cycles, sun and sky 5.12% against 3%; the reference was at fault: Blender 5.2's Cycles always box-filters at least one pixel, so it area-integrates while we point-sampled, and it loses 1.9% of a small sun 3 km away; sky interiors match Cycles to 0.1–0.5%, `docs/results/m5-gate1.md`); G2 the same with the reference fixed (**passed**: 4×4 per pixel against Cycles' one-pixel box, sun at 1 km: sky only 0.74%, sun and sky 1.19% exact and prefiltered; interiors within 0.0–0.5%, `docs/results/m5-gate2.md`); G3 object-level prefiltered coverage (**passed**: on the million-boulder field against instanced Cycles, sky only 1.15% and sun and sky 2.39% at 4×4 per pixel; more than 25× cheaper than walking every object; sun-only images bit-identical, `docs/results/m5-gate3.md`); G4 sky through the view cache (**failed** on cached-vs-per-pixel sky, 2.4–20.5% against 5%, the texel blur the sun channel also shows, and on speed; the speed numbers are confounded, since a control showed the machine running gate 11's own warm frame 10× slower than in gate 11; real finding: sky texels cost about 100× sun texels, so the bake budget's 1,000-texel floor must become a time, `docs/results/m5-gate4.md`); G5 sky through the cache at speed (**failed** on warm-up: with own-object sky masks baked once per geometry (faithful to 0.06–0.15%), sky faces as separate jobs, a horizon cull and a bake budget that follows GPU time, three sun-and-sky flights average 14.8 ms with 99th percentile under 25 ms and 4.5% fallback, warm frames 2.2–2.6 ms, sun channel unchanged; but views warm with sky in 13–95× the sun's frames, against 10×. Reported: the sky's error against Cycles depends on the cube's orientation, 1.9–4.2%, `docs/results/m5-gate5.md`; diagnostics in `docs/results/m5-sky-bake-diagnostics.md`); G6 several colored lights (**passed**: three RGB lights around the camera within 1.22% of Cycles at 16×16 samples per pixel over surfaces; re-coloring a warm cache bakes nothing and is bit-identical to a fresh one; flights with three lights 14.1–14.9 ms mean, 99th percentile 29–31 ms; light cones gained bounding-cone culling and nearest-first walks, which took a grazing lamp's flight from 579 to 14 ms; reported: with sky the 99th percentile is 35 ms, and gate 3's sun lit only the horizon, `docs/results/m5-gate6.md`); G7 lights cached as ratios, shaded with each pixel's normal (**failed** on flight speed: the cache's error against per-pixel cones fell to 0.41–0.56× with the sun and 0.57–0.68× with three lights; per-pixel still 1.22% from Cycles; but two of three flights with three lights averaged 17.09 and 17.58 ms, one with 99th percentile 36.4 ms, as camera passes doubled to 4.3–6.2 ms with the bake budget at its floor, `docs/results/m5-gate7.md`); G8 point and spot lights, and cheaper shading of lights (**failed** on sustained speed: a night scene with a point light and a spot is 1.35% from Cycles and its cache 0.82% from per-pixel; field cache errors within gate 7's; but flights ran last after about ten minutes of load and slowed from 14.5 to 20–25 ms mean, camera passes from 5 to 9.5 ms; a diagnostic showed the same flight slowing steadily under sustained load (13.1 → 16.3 ms over twelve flights) and none in one process when cool, `docs/results/m5-gate8.md`); G9 the sky later (**passed**: pages bake lights first, sky alone from the next frame; at gate 5's four viewpoints with a fixed budget, error while warming 0.11–0.17× the baseline's, within 5% at frame 120–300 against 1,200–4,220, warm in 0.61–0.65× the frames, warm image identical, `docs/results/m5-gate9.md`); G10 a guessed sky from coarser pages (development only, **failed**: 1.00× the baseline's flight error against 0.8×; removed, `docs/results/m5-gate10.md`); G11 environment light (**failed** both: 22.64% from Cycles, 1.16× the frame; nine harmonics cannot show a sky that ends at the horizon, `docs/results/m5-gate11.md`); G12 environment light, rays that miss reading the image (**passed**: 1.37% from Cycles against the uniform sky's 19.92%, 1.04× the frame, `docs/results/m5-gate12.md`); G13 the sky through hardware rays (**passed**: a sky sample traces the 192 bins of a hemicube around the normal, 0.72 µs against cones' 10.3 µs; with default options the field's four views warm in 166–400 frames against 1,827–15,905; sun and sky 2.53% from Cycles standing against 4.67%, within 1.05× the cones' from above; flights 0.947× the cones'; movers, streamed levels and blended objects keep cones, `docs/results/m5-gate13.md`) |
| M6 content (v0.1 item 3) | glTF import, base-color textures, automatic charts | G1 glTF import with automatic charts and textures (**failed** on criterion 1 only: the `.glb` round trip is 0.024% from the original per pixel against 0.01%, all in one edge pixel flipped by f32 rounding of instance transforms; the imported textured scene is 0.43% from Cycles with textures at 16×16 per pixel, its automatic charts give 2.37% cached vs per-pixel against the hand-made charts' 1.52%, warm frame 1.53 ms at 720p; real assets reported: a 28k-triangle textured pack warms in 1,317 frames at 1.43 ms, `docs/results/m6-gate1.md`) |
| M7 primary cones (v0.1 item 4) | Pixel-footprint cones: anti-aliasing and far-field level of detail | G1 pixel cones for primary visibility (**failed** all three: against the 16×16 reference the textured scene is 7.15% vs one ray's 7.11%, the field from 150 m 17.91% vs 23.49% (0.76× against ≤ 0.35×), from standing height 34.61% vs 12.49% with a banding artifact on the 2.1 km ground triangles; half a pixel's turn changes the image 1.67× as much as the reference's (≤ 1.5×); warm cached frames 105 ms on the textured scene (≤ 16.7 ms), per-pixel frames 10–87 s on the field; the textured scene's error is texture filtering, not silhouettes, `docs/results/m7-gate1.md`); G2 texture filtering for the area integral (**passed**: a box filter over the pixel's footprint from ray differentials, 3.99% from the 16×16 reference against 7.11%, warm frame 1.03×, `docs/results/m7-gate2.md`); G3 silhouette rays (**failed**: field from 150 m 0.49× the error but 13× the frame; textured scene 0.91×, 1.66×, `docs/results/m7-gate3.md`); G4 silhouette rays through ray queries (**failed** on the field's speed: quality as the compute path's, 1.29× the frame on the textured scene, 3.03× on the field from 150 m against 1.5× (compute path 24.15×), `docs/results/m7-gate4.md`); no shimmer in flythroughs |
| M8 library (v0.1 items 5–6) | Library API, Bevy plugin, examples; Windows/NVIDIA CI and a cloud benchmark | A tagged v0.1 (**done**: `v0.1.0`, 2026-11-07; the cloud benchmark is still open) |
| M9 dynamic scenes | Moving objects in the view cache (exact invalidation, same-frame re-bake); the ledger on the GPU: movers' effect on sky light, moving lights | G1 moving objects in the view cache (**passed**: 65 movers among a million boulders with three lights; no stale value in 30 checks of up to 137,260 pages; 0.84% of pages read re-baked per frame; flights 15.76–16.14 ms, 1.06–1.09× the movers held still, `docs/results/m9-gate1.md`); G2 a moving light through the ledger (**failed** on fallback only: a sun drifting 0.25° per second over the field; no page beyond its bound or ε in 30 checks, none changed at all; 0.69% of pages read re-baked per frame; means 14.45–14.58 ms against 15.64 ms still; fallback 33.3–34.2% against 10%, from off-screen pages set aside; the ε went into snapping, so no deferral within ε was exercised, `docs/results/m9-gate2.md`); G4 the ledger without its work limit (**failed** on speed: sound in 30 checks; at a real sun's speed fallback 4.9–5.0% but means 24.7–25.3 ms against 15.8 ms still, mixed texels re-baked every frame, `docs/results/m9-gate4.md`); G3 movers under a sky (**failed** on work and speed: sound in 30 checks, but 99.73% of pages read re-baked their sky every moving frame; means 53.0–58.3 ms against 14.66 ms still; fallback 30%, 21–22% even still from slow sky streaming; the per-page sphere bound about 5,000× loose, `docs/results/m9-gate3.md`); G5 the sky later with movers (development only, **failed**: 1.02× the error while moving against 0.5×; sound, cost 1.01×; kept, `docs/results/m9-gate5.md`); G6 a moving sun at real speed on a lattice (**passed**: cones' bins fixed in the page's version frame so a mixed texel's ratio moves only as the light's footprint does, held within 1.5 κ (P δ + π δ²) / Ω until a share of ε; three flights 15.16–15.70 ms against 15.43 ms still (gate 4: 1.6×), p99 27.5–28.3 ms, fallback 5.2–5.9%; 1,837–2,125 texels re-baked a frame against gate 4's 51,463; 886,506 pages checked at three speeds, none beyond its bound or ε; at rest 0.054% from the sun fixed there; set-aside pages checked texel by texel, `docs/results/m9-gate6.md`); a light switch with no lag |
| M10 GI and reflections | More bounces, cascades of mask cones, roughness cones | G1 one bounce in the view cache (**failed** on criterion 3 only: on the textured scene, 1.79% from Cycles with one bounce against 7.09% without; warm frame 1.07×, settling 1.21×; but 794 pages baked in the 30 frames after re-coloring, from cache activity resuming after supersampling, not from re-coloring, `docs/results/m10-gate1.md`); G3 secondary requests for hits (**failed** on its re-coloring check, a procedure bug; settling with materials 1.12× instead of 6.9×; accuracy kept, 1.45%/9.01%, `docs/results/m10-gate3.md`); G4 re-coloring with a corrected settle (**passed**: 0 pages baked, `docs/results/m10-gate4.md`); G2 metallic-roughness materials and glossy reflections (**failed** on accuracy: 8.94% from Cycles against 6%, 12.71% without reflection rays; warm frame 1.82× against 2×, `docs/results/m10-gate2.md`); G5 normal maps (**passed**: 3.63% from Cycles with the maps, 26.42% ignoring them; warm frame 1.04×, `docs/results/m10-gate5.md`); G6 alpha cutouts as geometry (**failed**, no criterion reached: the opaque control never settled, from a CPU/shader texel-grid mismatch now fixed, `docs/results/m10-gate6.md`); G7 gate 6 re-run (**passed**: 3.29% from Cycles masking the cards, 34.07% with them opaque; warm frame 1.15×, `docs/results/m10-gate7.md`); G8 exactly back-to-back faces (**failed** on accuracy: speed fixed, 1.03× the 1 mm scene's bake time; but Metal ray queries ignore `CULL_BACK_FACING`, so camera rays hit either coincident face, 5.01% from Cycles, `docs/results/m10-gate8.md`); G10 gate 2 re-run (**passed**: 5.85% from Cycles, warm frame 1.68×; specular from lights had applied Karis's normalization twice, so sun highlights on metal were low, `docs/results/m10-gate10.md`); G9 gate 8 re-run with back-face hits re-traced (**failed** on its 0.5% bound against the 1 mm control, 1.08%, which mostly measures that control's own offset: 1 mm to 2 mm moves the image as much; 3.42% from Cycles against the control's 3.37%; bake 1.21×, warm frame 1.04×; fixes kept, `docs/results/m10-gate9.md`); G11 emission lights its surroundings (**failed**: scored over all pixels, which the lights' visible disks decide, 109.67%; 1.00×, `docs/results/m10-gate11.md`); G12 the same over surfaces (**failed** on the absolute bound: 7.46% from Cycles against 5%, 18.00% without (0.41×); 1.00×; kept, `docs/results/m10-gate12.md`); G13 MikkTSpace tangents (**failed**: 11.80% against the triangles' frames' 11.75% where they differ; 1.00×; kept, `docs/results/m10-gate13.md`); G14 emissive textures (**failed** from the reference: 62.86%, Cycles lit the double-sided sign from its coincident faces, `docs/results/m10-gate14.md`); G15 re-run with a one-sided sign (**passed**: 6.35% over surfaces, texture ignored 491.34%; 1.00×, `docs/results/m10-gate15.md`); G16 alpha blending (**failed** on speed: 4.71% from Cycles, panes opaque 57.31%; 1.33× the frame against 1.3×; kept, `docs/results/m10-gate16.md`); G17 vertex colors (**failed** on speed: 1.29% from Cycles, colors ignored 57.63%; but 1.15× the frame against 1.1×, a transient that development runs did not reproduce, `docs/results/m10-gate17.md`); G18 its speed re-measured in five interleaved rounds (**passed**: 1.004×, `docs/results/m10-gate18.md`); G19 blending's speed (**passed**: 1.226× the frame against 1.3×, layers traced in the visibility pass and the list shaded by an indirect dispatch; 4.70% from Cycles, `docs/results/m10-gate19.md`); G20 texture wrap modes (**passed**: 2.18% from Cycles with clamped and mirrored textures, 67.44% repeating; 0.996× the frame, `docs/results/m10-gate20.md`); G21 bounce light with more than three lights (**failed**, no result: crashed at re-coloring on a missing buffer usage, output lost, `docs/results/m10-gate21.md`); G22 its re-run (**failed** on speed: 1.167× the frame with bounce against 1.15×, bounce light's own cost on that night scene, 1.165× with three lights too; accuracy 3.07% from Cycles against 17.78% without bounce; re-coloring a later light 0.407% from a fresh renderer; kept, `docs/results/m10-gate22.md`); G23 emissive faces as area lights (**failed**, accuracy not reached: 1.286× the frame, then the view never settled at a sub-pixel offset, a loop with bounce samples; settling 6× slower; kept as experimental, `docs/results/m10-gate23.md`); G24 its re-run with the loop fixed (**failed** on accuracy, 19.71% from Cycles against 5%, the faces showing their emission without the light they reflect; **liveness passed**: both renderers settle at all 16 sub-pixel offsets in 132–869 frames, since pages baked for bounce hits stay resident while bounce samples wait, `docs/results/m10-gate24.md`); G25 more than one bounce (**passed**: bounce samples gather the bounce light of the pages they hit and record the bounces they hold, pages re-baking until their hits hold enough; the showcase's closed room 6.54% from Cycles at 12 bounces with 8 bounces against 45.36% with one, settled in 80 frames; the textured scene 2.11% against 2.36%; opt-in, `docs/results/m10-gate25.md`) |
| M11 showcase | Bistro/Sponza, UE5 side-by-side | Published comparisons in motion |
| M12 deforming geometry | Subtree refit | Characters in the showcase |
| M13 prefiltered appearance (Landmark 3) | Hierarchy nodes keep lit appearance; pixel cones stop at pixel-size nodes | Sub-pixel detail without aliasing, cost bounded by the screen. G1 coverage only (**failed**: from 150 m 0.38× one ray's coverage error, passing, but 1.2–2 s a frame, about 100× too slow: each node visit rasterizes into 8×8 bins; standing failed mostly on a flawed measure, buried boulder parts the ground hid in the reference only, `docs/results/m13-gate1.md`); G2 a scalar per-node test (**failed**: 257–281 ms a frame, about 26× over; accuracy 0.38× one ray's from 150 m, 0.59× standing, where one horizon row holds 84% of the error: independent splats saturate a pixel whose boulders all sit low in it, correlation that sub-pixel bins keep, `docs/results/m13-gate2.md`); G3 directional coverage maps and 4×4 bins (**failed** on speed only: accuracy **passes** both views, 0.264× one ray's error standing and 0.231× from 150 m; but 178–707 ms a frame, long tails where partly covered pixels never stop, `docs/results/m13-gate3.md`); G4 near surfaces by hardware rays at pixel corners and bins (**failed** on speed only: accuracy 0.218× and 0.132×; 122–135 ms against a 50 ms intermediate bar, the far field's walk left, `docs/results/m13-gate4.md`); scoped after (development, not a gate: at 10,000 boulders the walk to instances and 16 bin rays per unresolved pixel carry the cost, the maps' reads little; the walk as a packet is 14% faster from 150 m); profiled at full size (development, `docs/results/m13-far-field-profile.md`: from 150 m 104 ms, about half the bins' rays, the rest the maps' reads and a root-to-instance walk, with no node above the instances small enough to stop at; standing 122 ms, the sky's pixels walking boxes straddling the eye's plane for nothing, removed exactly, and the horizon's far field); **far-field design** (2026-11-13, §3.4): exact culling by the boxes' faces, the bins' rays only near large objects, one walk per 8×8 tile, lighter map reads (gate 5), and hierarchy nodes holding coverage as view-dependent coverage pages at one texel per bin, baked by hardware rays restricted to the node's contents and reused within a positional bound (gate 6); G5 and G6 stubs written; G5 exact culling, bin rays near large objects, tile roots and lighter map reads (**passed**: 0.219× one ray's coverage error standing in 38.25 ms, 0.135× from 150 m in 20.35 ms, against 0.25×/0.15× and 50/25 ms; a pass with one invocation a tile listing subtree roots, pixels walking below them as packets, bin rays in their own pass, two directions' maps with affine coordinates, open bins as bit masks; `docs/results/m13-gate5.md`); G6 view-dependent coverage pages for hierarchy nodes (**passed**: settled 0.212× in 16.12 ms standing with 10,224 pages, 0.135× in 17.11 ms from 150 m with none, against 0.25×/0.15× and 20 ms; a 10 s walk from a cold pool at 20 bakes a frame held 0.213–0.224× against 0.3×, frames 34 ms while the pool filled; from a settled start a walk needs 1.5 bakes a frame (16.2 ms), but a 10 m/s flight outruns 20 bakes a frame (41 ms); `docs/results/m13-gate6.md`); G7 pages that keep up in flight (**passed**, criterion 1 narrowly: a 10 m/s flight 20 m up at 1.191× the settled frame at its place against 1.2× (95th percentile), 56 bakes a frame in 1.39 ms, accuracy 0.153–0.158×, a cold walk's median 16.50 ms; bakes one invocation per ray, bake eyes led along the motion, refreshes past 80% of the bound; criteria 1–2 amended before any run; `docs/results/m13-gate7.md`); G8 the coverage pass in 10 ms (**passed**: settled 9.01 ms standing at 0.159×, 9.10 ms from 150 m at 0.128×, against 10 ms and 0.25×/0.15×; once-a-frame page, node and instance records, maps of 32×32 directions read nearest, 2,048-texel pages, tile lists of leaf instances, bin rays by the pixel's own corners; `docs/results/m13-gate8.md`) |
| M14 unique geometry at scale (Landmark 1) | Geometry and hierarchy streamed by need, prefiltered nodes standing in | A scene several times GPU memory at frame rate. G1 camera levels streamed by need (**failed** on frame rate in one flight of three: 256 unique rocks whose finest levels are 6.35× a 2.5M-triangle budget, never over it; settled coverage 0.062–0.096× one ray's error from fully resident, 0.013–0.029% of pixels differ in flight, 3.4× less GPU memory; but 1.074×, 1.085× and **1.124×** the resident frame against 1.10×, from about half a load a frame at about 1 ms each; settled colour 1.3–1.6% off, smoother normals at coarser levels, `docs/results/m14-gate1.md`); G2 fewer, cheaper loads and a zoom (**failed** on frame rate in three flights of four: 1.091×, **1.110×, 1.111×, 1.113×** against 1.10×, streaming 0.33–0.42 ms a frame, 2.0–2.5% of a 60 fps frame; passed: never over budget, settled coverage 0.072–0.087× one ray's error, 0.011–0.028% of pixels off in flight, the zoom from 50° to 1.5° on a rock 103 m away exact settled with the working set 1.55 → 0.52 million triangles; paged pool, staging belt, level chains; a load about 29 ns a triangle, 22 of it the structure's build, `docs/results/m14-gate2.md`); G3 beyond CPU memory within a 60 fps budget (**passed**: 4,096 unique rocks, finest levels 32× an 8.4-million-triangle budget and 33 GB packed against 25.8 GB of memory, generated on demand by worker threads into a 2 GiB cache; three 1,800-frame flights at 4.8–5.3 ms mean, 99th percentile 7.9–8.8 ms, worst 14.5 ms, never over budget or cap; settled within 0.018–0.030× one ray's error of every rock at its ideal level; a 2 ms-a-frame time budget deferred 384–926 loads a flight, `docs/results/m14-gate3.md`); G4 lighting levels streamed, certified (**passed**: lighting levels 8× their budget; 2.68 million shown cache pages in flight and every settled page in common bit-identical to a renderer with all lighting levels resident; GPU memory at creation 0.486× gate 3's way; flights 5.4–6.9 ms mean, worst 18.2 ms; proxy levels for camera rays, stubs and slots for lighting levels, pages baked against an absent level hidden and re-baked, `docs/results/m14-gate4.md`); G5 lighting levels produced on demand, certified beyond one renderer (**failed** on the frame budget only: 36,864 unique rocks, lighting levels produced by worker threads into a 256 MB CPU cache, 72× the slots and 26.9× the cache, never over either; 676,080 pages in the flights' region identical to a reference holding the region's 4,941 rocks with every lighting level resident, the longest shadow 7.89 m inside a 12 m margin; 27.7 KB of GPU memory and 15.2 KB of CPU heap a rock, 0.128× and 0.052× a lighting level resident; but **26–39 ms** mean frames, p99 51–66 ms, worst 67–90 ms, with 34–62 lighting loads a frame, `docs/results/m14-gate5.md`); G6 gate 5's world within 60 fps (**failed** on one frame: means 6.16, 6.53, 6.18 ms, p99 10.2–11.9 ms, worst 13.2, 17.1 and **42.3 ms** against 33.3; 1,416,782 pages identical to the subset reference; 17.8 KB of GPU memory and 14.9 KB of CPU heap a rock; lighting uploads through a staging belt, a persistent residency set and incremental top-level builds in patched wgpu, clusters of 32 proxies, the streamer's scans narrowed, `docs/results/m14-gate6.md`); G7 frame-time stability at scale (**passed**: gate 6's world with no frame over 16.7 ms, means 5.86, 5.26, 4.76 ms, p99 8.3–10.1 ms, worst 15.8 ms; 908,545 pages identical; producer threads at utility QoS, CPU-cache frees on a background thread, 32 MB staging chunks; lighting levels now 21 frames late on new ground and views settle in up to 4,791 frames, `docs/results/m14-gate7.md`); G8 a sky and bounce light with streamed lighting (**failed** on the frame budget only: on 9,216 rocks 436,004 sun and 436,002 sky pages identical to every lighting level resident, settled images within 0.07–0.52% (bounce); on 36,864 rocks means 14.1–15.9 ms but p99 **20.5–25.1 ms**; 20.0 KB of GPU memory a rock; the sky's bake marks pages, every absent object listed once a frame, stubs of four hierarchy levels, bounce counting absent levels as missing and retrying, retries and skies baked alone waiting unpinned, `docs/results/m14-gate8.md`); G9 the frame budget with a sky and bounce (**failed** on the frame budget only; run twice, the first disturbed by a game on the machine and stopped by the usage limit, the second clean: a frame-period cap on the bake budget took means to 12.8–13.1 ms but p99 stayed **19.8–21.6 ms**, one frame 163 ms; 286,042 sun and sky pages identical, bounce within 0.065–0.079%, `docs/results/m14-gate9.md`); G10 bounce rays through hardware ray queries, the frame budget with a sky and bounce (**passed**: 36,864 rocks with a sky and bounce at 9.2–10.2 ms mean, p99 14.5–15.0 ms, worst 28.3 ms; 177,375 sun and sky pages identical; bounce within 0.025–0.58% settled; M10 G22's scene 3.09% from Cycles (3.07% with compute rays); the frame budget 11 ms by default, `docs/results/m14-gate10.md`); G11 the frame-period cap times the renderer's own frame calls (**passed**: gate 10's flights 9.2–9.9 ms mean, p99 14.4–15.3 ms; reading the image back every 20th frame warms M5 G13's views in 0.96–1.05× the undisturbed frames, against up to 4.5× before, `docs/results/m14-gate11.md`); G12 coarser levels shade with the finest's normals (**failed** on criterion 3, a method fault: streamed rocks 0.56–0.65× as far from the finest-resident images (1.25–1.48% against 1.95–2.54%), levels produced in 0.87× the time; but the resident images, which the change does not touch, differed by 0.0045% at one view from the time-based budget's timing, identical with a fixed budget, `docs/results/m14-gate12.md`); G13 gate 12 with a fixed bake budget (**passed**: 0.649×, 0.641×, 0.561× the colour difference from the finest-resident images; production 0.87×; resident images identical, `docs/results/m14-gate13.md`); G14 several bounces with streamed lighting (**failed** on criterion 3 at one view: 8 bounces on 36,864 rocks with a sky flew at 9.3–10.0 ms mean, p99 14.8–15.3 ms, 222,416 sun and sky pages identical to all resident; but flight 1's image after 8,000 frames 1.038% from the resident renderer's against 1%, neither settled, with other GPU work run during criteria 2–3 (disclosed); outdoors past the first bounce changes views by 0.2–0.3%, `docs/results/m14-gate14.md`); G15 coarser levels for meshes without them (**passed**: `continuum::simplify`, half-edge collapses ordered by quadric error keeping the finest's vertices, normals and chart coordinates, seams in place, a distance bound per level and an error of the larger of it and half the mean edge; rocks' levels simplified from their finest shade within 0.77–1.10× the analytic levels' colour difference from the finest-resident images, dense distances at most 0.49× the stated errors over 512 levels; 1.45 s a rock; with the lighting level simplified too 3.8–6.4% from the analytic resident image, `docs/results/m14-gate15.md`) |

## 7. Benchmark protocol

- M4 MacBook Air is fanless and throttles: report a short warm run and a separate
  sustained number, plugged in, same scene, same camera path.
- Correctness CI on Windows (WARP) and Linux (lavapipe); NVIDIA performance on a
  rented cloud machine per benchmark round.
- Image quality against an independent path tracer, scored with FLIP; flicker
  measured as frame-to-frame difference on a static camera.

## 8. Risks and open questions

- **Ledger bound tightness** (biggest risk): too-pessimistic charges mean
  everything recomputes and the edge disappears. Flatland: geometric bounds used
  0.2–8% of the certified bound; projected charges about 6% and halved the work.
  Remaining sources: per-tile charges, accumulation over oscillating motion.
- **Static coverage memory**: continuous recomputes reuse cached static
  coverage (264 bytes per cone on the CPU), which made continuous masks the
  fastest configuration in gate 5. A GPU layout needs something far smaller
  (quantized partials, or partials only for edge bins).
- **Charging cost vs recompute cost**: projected charges do 3× less recompute
  work than geometric bounds but cost more time in 2D, where a recompute is
  cheap, and break rule 1 under heavy motion (gate 4: 1.96× a full recompute
  with 41 movers). Required: a per-tile rule that never spends more on charging
  than the recompute it avoids (fall back to geometric bounds or recompute).
- **Charges for indirect light are too loose** (gate 6): implementable charges
  skip 73% of gathers at ε = 1%, while exact charges would skip 90%. Cone
  G-buffers sidestep this for black movers and radiance changes (gate 6b:
  exact, 7.4×), at 384 bytes per cone and with a binned estimator that is 2.5%
  off the exact gather. Open: coloured or emissive movers, moving reflective
  surfaces, and the GPU memory layout.
- **Moving lights** (gate 7): the ledger recomputes a quarter of the cones
  that exact invalidation must (realistic sun speed, ε = 1%), narrowly missing
  its target. Next: occlusion-aware light-motion charges; and with mask
  estimators, a bound term for bins realigning as the cone moves.
- **Depth-correct mask gathers**: binned gathers with partial coverage depend on
  processing order when several surfaces share a bin, which breaks charge
  bounds. Flatland used an exact angular sweep; 3D needs a depth-correct
  binned gather (per-bin depth layers, or a cone G-buffer).
- **What ε is relative to**: per-pixel total brightness (perceptual) vs per-term
  means. Gate 6 used 1% of indirect light, about 5× stricter than 1% of total.
  Decide in the quality contract.
- **Mask quantization**: 64 bins quantize penumbrae into 1/64 steps (faint
  banding in gate 1) and make visibility discontinuous, so any mover touching a
  cone costs at least one bin. That floor exceeded realistic ε, which is why the
  ledger never deferred direct shadows. Fix candidate for both: analytic
  coverage for partially covered edge bins (continuous visibility).
- **Where magnitude deferral pays**: direct shadows of moving objects change
  visibly every frame, so recomputing them is correct. Wide cones are where it
  pays: gate 5 cut work 2.9× beyond dependency regions at ε = 1%. Slow global
  changes (moving sun) are still untested.
- **3D projection cost**: resolved for triangles by an accumulation rasterizer
  (exact per-bin area, M1 gate 1: 1.1 µs per receiver for about 28 candidate
  triangles). Open: projecting hierarchy nodes (prefiltered coverage), and
  culling cost without a BVH (15 µs per receiver in M1 gate 1).
- **Bins shared by separate meshes**: max-combining under-counts their union,
  which leaks light (the dominant error in M1 gate 1).
- **GPU mask layout**: partly resolved. M2 gate 1 keeps one float of coverage
  per bin in private memory (no 64-bit integers needed) and runs on Metal at
  7.5× the CPU. Open: register pressure at scale, subgroup-per-cone
  traversal, and testing on NVIDIA/Windows.
- **Prefiltered wide cones**: resampling a node's stored masks into a cone's bins
  loses precision; correlation is only exact down to node resolution.
- **Binned coverage for wide cones depends on orientation** (M5 G5): turning
  the sky's cube moves its error against Cycles between 1.9% and 4.2%, and it
  is always too dark. Coverage is an area fraction times the bin's average
  cosine weight; occluders near the horizon cover the low-weight part of a
  bin. Candidate: first moments of covered area per bin.
- **Wide cones are latency-bound on the GPU**: one sky face walk takes
  milliseconds, and neighbouring texels' walks diverge (10× in M5 G5's
  diagnostics). Packet traversal helps coherent work (pixels) and hurts
  incoherent work (cache texels far apart).
- **Fully dynamic scenes, foliage in wind, a continuously moving sun.**
- **Memory** for caches and dependency data.
- **NVIDIA RT cores** make individual rays cheap; our bet is fewer, smarter queries.

## 9. References

- Amanatides 1984, Ray tracing with cones. Heckbert & Hanrahan 1984, Beam tracing.
- Carpenter 1984, The A-buffer.
- Lacewell et al. 2008, Raytracing prefiltered occlusion for aggregate geometry.
- Crassin et al. 2011, Interactive indirect illumination using voxel cone tracing.
- Laine et al. 2011, Clipless dual-space bounds.
- Bako, Sen, Kaplanyan 2022, Deep appearance prefiltering.
- Weier et al. 2023, Neural prefiltering for correlation-aware levels of detail.
- Therrien et al. 2023, Screen space indirect lighting with visibility bitmask.
- Sannikov 2023, Radiance cascades.
- Drettakis & Sillion 1997; Granier & Drettakis 2001 (incremental GI).
- Walter, Drettakis, Parker 1999, The render cache.
- Mueller et al. 2021, Temporally adaptive shading reuse.
- Keeter 2020, Massively parallel rendering of complex closed-form implicit surfaces.
- Sloan, Kautz, Snyder 2002, Precomputed radiance transfer.
- Zhou et al. 2005, Precomputed shadow fields for dynamic scenes.
- Kontkanen & Laine 2005, Ambient occlusion fields.
- Valient 2008, Stable rendering of cascaded shadow maps (ShaderX6): the
  shadow map's projection snapped to whole texels so edges do not shimmer
  as the view moves; the closest known to M9 G6's lattice, which fixes a
  cone's bins as the light moves so a motion bound bounds the cached ratio.
