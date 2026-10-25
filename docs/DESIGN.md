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

### 3.4 Prefiltered hybrid hierarchy (infrastructure)

One hierarchy whose leaves are triangle clusters (imported assets), field bricks
(procedural, CSG, destruction) or volumes. Interior nodes store prefiltered
directional coverage and appearance; a cone stops descending once a node is
smaller than its footprint, so cost is bounded by screen size, not scene size.
Builds on Lacewell et al. 2008, Bako et al. 2022, Weier et al. 2023; ours is
non-neural, fitted for mask cones, and portable. Not a novelty claim.

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

## 3.6 Where we stand (updated 2026-10-25, after M6 gate 1)

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
  (M5 G5, failed on warm-up).
  - Flights over a million boulders with sun and sky average 14.8 ms on the
    fanless Air (99th percentile under 25 ms), and warm frames take 2.2–2.6 ms.
    The bake budget follows GPU time measured with timestamp queries.
  - Each geometry's occlusion of its own sky is baked once for all its
    instances (own-object masks, faithful to 0.06–0.15%).
  - But a view warms with sky in 1,365 frames standing and 6,900–12,200 from
    above, against 82–132 with the sun alone. Lighting stays at coarse pages
    while flying. The viewer's sky stays opt-in (`--sky`).
- **Not yet shown in 3D or on the GPU.** The ledger, the third rule and "the
  invention" (§3.1). It has only run in 2D, with mixed results, and moving
  lights depend on it.
- **Several colored lights** (M5 G6, passed): a list of RGB rect lights per
  pixel and in the cache, flights at 14–15 ms with three lights. A cache
  texel stores light per unit radiance, so re-coloring a light bakes nothing
  and is bit-identical to a fresh cache.
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
  - Diffuse only: no metallic or roughness, normal maps, alpha, emission or
    the file's lights yet.
- **Not started:**
  - point and spot lights (only rectangles so far);
  - sharing the sky's far field across texels (cascades of wide cones), which
    is what sky warm-up needs;
  - mask cones for primary visibility and anti-aliasing. One ray per pixel is
    13–24% from the area-integrated image on the boulder field (M4 G8), now
    the largest quality gap;
  - glossy reflections, materials beyond Lambert (imported PBR parameters are
    ignored), more than one bounce;
  - the quality contract (§3.3) beyond the bake's frame-time target.
- **Against UE5.** No direct comparison has been made, so none is claimed.
  - Our measured edge: exact soft shadows from area lights and noise-free
    sky occlusion, bounce light that updates exactly in the frame the change
    happens, and light colors and intensities that change for free and
    exactly, with no noise, denoiser or temporal lag.
  - UE5 leads on everything else: materials, lights, GI bounces,
    anti-aliasing and upscaling, platforms and tooling.

## 3.7 Road to v1 (2026-10-24)

A **batch** below is the size of one long working session in this project's
history: a few gates with their implementation and write-ups. These are
estimates, and gates fail often, so treat them as ranges.

**v0.1, experimental** (a library people can try on their own scenes): about
11–15 batches.

| # | Work | Why | Batches |
|---|---|---|---|
| 1 | View-dependent lighting cache: texels only on visible surfaces, coarser with distance, a per-frame bake budget, kept across frames, exact incremental updates for movers, instanced objects | Real-time at scale. M4 gate 7 shows per-pixel cones are the bottleneck. **Done for static scenes (M4 G8–G11; G11 passed):** flying over a million boulders at 720p averages 14.6–15.0 ms with the bake budget following the frame time; lighting is softer in motion and sharpens when the view settles. Sky in this cache since M5 G5 (at frame rate, slow to converge). Not yet: movers in this cache | 2–3 (3 done) |
| 2 | Sky light and several lights (sun, sky environment, point and spot lights) | Outdoor scenes had black shadows. **Sky done per pixel and matched to Cycles (M5 G2–G3); in the view cache at frame rate (M5 G5), but slow to converge. Several colored rect lights done (M5 G6).** Not yet: point and spot lights; sky warm-up | 3 (3 done) |
| 3 | glTF import: meshes, instances, base-color textures, automatic charts | Real content instead of procedural scenes. **Done (M6 G1, failed on round-trip precision only):** a textured import is 0.43% from Cycles, automatic charts are within a point of hand-made ones in the cache, warm frames take 1.5 ms. Not yet: PBR parameters, normal maps, alpha, the file's lights | 2–3 (1 done) |
| 4 | Pixel-footprint cones for primary visibility: anti-aliasing and far-field level of detail | The far field shimmers; one ray per pixel | 2 |
| 5 | Library API, Bevy plugin, examples, documentation | Something people can use | 2 |
| 6 | Windows/NVIDIA: CI (WARP, lavapipe), a cloud GPU benchmark run | The day-one platform requirement, untested so far | 1–2 |

### 3.8 Next steps, in order (2026-10-25, after M6 gate 1)

1. **Sky warm-up.** M5 G5 failed on warm-up (13–95× the sun's frames), and
   with three lights and sky the flights' 99th percentile is 35 ms (M5 G6,
   reported). Sharing the far field per page was tried and is 1.4–3.8×
   slower: with object-level prefiltering the far field is already cheap, and
   the cost is each face's walk near its apex (about 107 object nodes,
   `m5-sky-bake-diagnostics.md`). Candidates: a wider hierarchy, faces
   turned with the normal (one and four halves), packets for large bakes.
2. **Sky accuracy: weight coverage within bins.** The sky's error against
   Cycles depends on the cube's orientation (1.9–4.2%) and is always too
   dark. Accumulating first moments of covered area per bin would weight
   coverage by the cosine to first order.
3. **Pixel-footprint cones for primary visibility** (item 4). Measure against
   area-integrated references. Packet traversal (M5 G5) suits them, and M5
   G6 found 4×4 samples per pixel too few for small lit boulders on a dark
   ground (the lamp alone: 30% at 4×4, 4.7% at 16×16 against Cycles).
4. **Point and spot lights**, after the above (the rect light covers area
   lights and suns), then the file's lights in glTF import.
5. **Library API and Bevy plugin** (item 5), then Windows/NVIDIA CI (item 6).
6. **Open questions:**
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

## 5. Library shape (planned)

- `continuum`: core types (masks, bounds, ledger, hierarchy), CPU reference paths.
- `continuum-gpu`: wgpu backend.
- `bevy_continuum`: Bevy integration.

API direction, not final:

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
| M4 scale | Prefiltered cones, instancing, large-world coordinates, a virtual lighting cache | G1 prefiltered cones (**passed**: flat cost from 40k to 2.56M triangles, 86 ms vs 13.7 s exact, 0.55% from exact, `docs/results/m4-gate1.md`); G2 ten billion instanced triangles in a large world (**failed** on two of three: the 64 km-offset field renders identically with rebasing, and 10^10 triangles fit in 152 MB, but instanced vs flattened is 0.22% vs a 0.1% target because the flattened reference is the less precise one, and 10^6 vs 10^3 instances cost 3.04× vs 2× because camera rays near the horizon dominate; cones to a small sun are nearly free, `docs/results/m4-gate2.md`); G3 nearest-first ray traversal (**passed**: identical output, 10^10 vs 10^7 triangles 1.40×, 2.83× faster than the fixed order, 100.8 ms at 1080p for a million boulders, `docs/results/m4-gate3.md`); G4 visibility buffer (**failed**: identical output but 0.88× on the field and 1.11× slower on the showcase; register pressure was not the limit, ray traversal itself is, `docs/results/m4-gate4.md`); G5 traversal stacks in workgroup memory (**failed**: identical output but 2.6× slower, as 12 KB per workgroup limits occupancy, `docs/results/m4-gate5.md`); G6 hardware ray queries for camera rays (**failed**: no faster, 0.9×, and 67–270 ms per 720p frame. A corrected diagnostic shows per-pixel sun cones, not rays, dominate; gates 2 and 4 had blamed rays from a broken "light turned away" test, `docs/results/m4-gate6.md`); G7 cones that pay only for what they see (**failed**: no speedup; a stage-by-stage profile shows cones descending the triangle hierarchies of the objects they start on dominate, 119 of 215 ms from 150 m up at 720p, so per-pixel cones every frame are the brute-force path at scale too, `docs/results/m4-gate7.md`); G8 view-dependent lighting cache (**failed** all three: 14–39% from per-pixel exact cones, 20–37 ms warm, 42.7 ms mean in flight; but warm shading fell from 47–233 ms to 2–3 ms and only 1.8% of pixels lacked a page in flight. Diagnostics: one sample per pixel is itself 13–24% from the supersampled image, so later quality gates compare with the area-integrated image; and wgpu spends 15–17 ms of CPU time per `submit` walking the TLAS's million per-instance dependencies, while hardware rays take 4.5–6.6 ms on the GPU, `docs/results/m4-gate8.md`); G9 the cache at speed (**failed** on the flight's 99th percentile, 34.0 vs 33.3 ms. Same images as G8; warm frames 2.9–4.7 ms at 720p with hardware rays after a wgpu-core patch that keeps one TLAS dependency per distinct BLAS, pipelined requests and a faster page manager; flight mean 12.8 ms, 1.6% fallback. The flight is bound by CPU planning, which re-derives a backlog of ~50k wanted pages every frame, `docs/results/m4-gate9.md`); G10 smooth flight (**failed**: progressive requests from the shader and GPU deduplication cut CPU planning to 1.9 ms per frame, but the flight is then GPU-bound at 17.6–17.9 ms mean, 38 ms 99th percentile on a hot machine; next, baking by frame time instead of a fixed cone count, `docs/results/m4-gate10.md`); G11 baking to a frame-time target (**passed**: same images; warm 6.5–8.0 ms at 720p; three flights over a million boulders at 14.6–15.0 ms mean, 26–32 ms 99th percentile, 1.7% fallback. Caveat: the budget sits near its floor while flying, so only 0.6% of samples reach their wanted mip in motion and lighting is softer until the view settles, `docs/results/m4-gate11.md`); zoom and flythrough with no shimmer |
| M5 lights (v0.1 item 2) | Several lights, a sky environment, colored light | G1 sky light through six cube-face mask cones (**failed**: sky alone 2.74% from Cycles, sun and sky 5.12% against 3%; the reference was at fault: Blender 5.2's Cycles always box-filters at least one pixel, so it area-integrates while we point-sampled, and it loses 1.9% of a small sun 3 km away; sky interiors match Cycles to 0.1–0.5%, `docs/results/m5-gate1.md`); G2 the same with the reference fixed (**passed**: 4×4 per pixel against Cycles' one-pixel box, sun at 1 km: sky only 0.74%, sun and sky 1.19% exact and prefiltered; interiors within 0.0–0.5%, `docs/results/m5-gate2.md`); G3 object-level prefiltered coverage (**passed**: on the million-boulder field against instanced Cycles, sky only 1.15% and sun and sky 2.39% at 4×4 per pixel; more than 25× cheaper than walking every object; sun-only images bit-identical, `docs/results/m5-gate3.md`); G4 sky through the view cache (**failed** on cached-vs-per-pixel sky, 2.4–20.5% against 5%, the texel blur the sun channel also shows, and on speed; the speed numbers are confounded, since a control showed the machine running gate 11's own warm frame 10× slower than in gate 11; real finding: sky texels cost about 100× sun texels, so the bake budget's 1,000-texel floor must become a time, `docs/results/m5-gate4.md`); G5 sky through the cache at speed (**failed** on warm-up: with own-object sky masks baked once per geometry (faithful to 0.06–0.15%), sky faces as separate jobs, a horizon cull and a bake budget that follows GPU time, three sun-and-sky flights average 14.8 ms with 99th percentile under 25 ms and 4.5% fallback, warm frames 2.2–2.6 ms, sun channel unchanged; but views warm with sky in 13–95× the sun's frames, against 10×. Reported: the sky's error against Cycles depends on the cube's orientation, 1.9–4.2%, `docs/results/m5-gate5.md`; diagnostics in `docs/results/m5-sky-bake-diagnostics.md`); G6 several colored lights (**passed**: three RGB lights around the camera within 1.22% of Cycles at 16×16 samples per pixel over surfaces; re-coloring a warm cache bakes nothing and is bit-identical to a fresh one; flights with three lights 14.1–14.9 ms mean, 99th percentile 29–31 ms; light cones gained bounding-cone culling and nearest-first walks, which took a grazing lamp's flight from 579 to 14 ms; reported: with sky the 99th percentile is 35 ms, and gate 3's sun lit only the horizon, `docs/results/m5-gate6.md`); next: the sky's far field shared per page |
| M6 content (v0.1 item 3) | glTF import, base-color textures, automatic charts | G1 glTF import with automatic charts and textures (**failed** on criterion 1 only: the `.glb` round trip is 0.024% from the original per pixel against 0.01%, all in one edge pixel flipped by f32 rounding of instance transforms; the imported textured scene is 0.43% from Cycles with textures at 16×16 per pixel, its automatic charts give 2.37% cached vs per-pixel against the hand-made charts' 1.52%, warm frame 1.53 ms at 720p; real assets reported: a 28k-triangle textured pack warms in 1,317 frames at 1.43 ms, `docs/results/m6-gate1.md`) |
| M7 primary cones (v0.1 item 4) | Pixel-footprint cones: anti-aliasing and far-field level of detail | Against supersampled references; no shimmer in flythroughs |
| M8 library (v0.1 items 5–6) | Library API, Bevy plugin, examples; Windows/NVIDIA CI and a cloud benchmark | A tagged v0.1 |
| M9 ledger on GPU | Moving lights, ledger-bounded deferral | G2/G3 on GPU; light switch with no lag |
| M10 GI and reflections | More bounces, cascades of mask cones, roughness cones | FLIP vs reference |
| M11 showcase | Bistro/Sponza, UE5 side-by-side | Published comparisons in motion |
| M12 deforming geometry | Subtree refit | Characters in the showcase |

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
