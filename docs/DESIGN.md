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

## 3.6 Where we stand (updated 2026-11-08, after M10 gate 19)

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
  - more than one bounce;
  - the quality contract (§3.3) beyond the bake's frame-time target.
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

### 3.8 Next steps, in order (updated 2026-11-08, fourth session)

**Start here (end of the fourth session, 2026-11-08).** The state in one
place, for a new conversation:

- **The user's direction:** make the engine complete. Pick the next
  feature by judgment, finish usable features before new research, and keep
  usage low. The NVIDIA benchmark is ready but waits until the user can pay
  for a rented GPU (not soon, as of 2026-11-08).
- **`v0.2.0` is tagged** (2026-11-08, approved by the user) on the commit
  that bumps the workspace version, after CI passed.
- **Going public (the user's decision, 2026-11-08):** the repository is to
  become public under the PolyForm Noncommercial License 1.0.0
  (`LICENSE.md`), branded Penumbral Labs, with the copyright line
  "[legal name], doing business as Penumbral Labs". The README was rebuilt
  for a wide audience (diagrams, engine GIFs, Cycles comparisons, an honest
  comparison of methods). Before flipping it public: the legal name in
  `LICENSE.md`, and the commercial-license contact (the README says "open
  an issue").
- **The fourth session** took the third session's list in order (results
  in `docs/results/`):
  - *viewer `--env`*: `--env gradient|sky.hdr|sky.png`, `--env-strength`;
    `continuum_gltf::load_environment` reads Radiance HDR (flat or RLE)
    and PNG. Not gated (M5 G12's path); a test reads back the HDR Cycles
    was checked against.
  - *glTF `KHR_texture_transform`*: baked into texture coordinates, with
    Blender's rotation sign (the extension's GLSL sample, read
    column-major, is its transpose); checked against Blender's importer's
    route. One set and transform per geometry; other slots are skipped and
    counted.
  - *vertex colors*: `InstancedScene::geometry_colors`, carried in the
    triangle buffer's unused w (no new binding). M10 G17 accuracy passed
    (1.29% from Cycles) but speed failed on a transient (1.15×); M10 G18
    re-measured it in interleaved rounds: 1.004×. glTF `COLOR_0` import
    and save.
  - *glTF save* of emissive textures, BLEND and vertex colors, with
    round-trip tests.
  - *blending speed*: M10 G19 passed, 1.23× the frame (G16: 1.33×), 4.70%
    from Cycles. The layers are traced in the visibility pass, and the
    extra-samples list is shaded by an indirect dispatch over the list.
  - *glTF animations*: `Imported::animations` (`Animations::instances`
    and `pose`: translation and rotation channels, step, linear and
    cubic-spline, through the node hierarchy), tested against poses worked
    out from the spec; the viewer plays a file's first animation through
    movable instances (not with a sky). Not gated: movers are M9 G1's.
  - *Bevy movers*: `ContinuumInstance(index)` makes a movable instance
    follow its entity's `GlobalTransform` (a unit test checks the change
    of axes for rotations); the `boxes` example's `--spin`.
  - *exposure*: `Renderer::set_exposure` scales the drawn colors before
    the tone curve (a test reads a float target back: exactly the colors
    times the factor); the viewer's `[`, `]` and `--exposure`.
  - *texture wrap modes*: `InstancedScene::texture_wrap` (repeat, clamp,
    mirror per axis), from glTF samplers and saved back. M10 G20 passed:
    2.18% from Cycles (67.44% repeating), 0.996× the frame.
  - *`v0.2.0` tagged* after CI passed.
  - *bounce light with any number of lights*: lights past the third
    bounce through the absolute emission channel; re-coloring one clears
    bounce light, which re-bakes. M10 G21 crashed (no result); M10 G22:
    3.07% from Cycles, re-coloring 0.407% from a fresh renderer, but 1.167×
    the frame against 1.15× (bounce light's own cost at night, the same
    with three lights). Kept.
  - *emissive faces as area lights*, experimental:
    `InstancedScene::emitters_to_lights` and the viewer's
    `--emitters-as-lights`. M10 G23 failed before its accuracy check (the
    view never settled with bounce light at a sub-pixel offset).
- **Core first (the user's direction, 2026-11-08):** before more
  features, make the fundamentals hold up at extreme scale and detail
  (unique geometry close up and kilometers of open world, both), then
  branch out: in this paradigm one primitive (cones over a prefiltered
  hierarchy, with a certified cache) serves shadows, anti-aliasing, level
  of detail, reflections and GI, so strengthening it strengthens all of
  them. A **quality dial** grows alongside: the quality contract (§3.3), an
  error bound and a frame budget, one guaranteed and the other reported,
  each core item exposing its knob to it rather than fixed presets.
- **Next, in the order I would take them (core first):**
  1. *Sub-pixel detail without aliasing (M13, prefiltered appearance).*
     One ray per pixel aliases wherever a pixel holds more than one
     surface (the field from 150 m is noise: 23.49% from the
     area-integrated image, M7 G1). M7 G1's pixel cones resolved every
     pixel to an eighth of a pixel and cost hundreds of node tests per
     pixel (105 ms on the textured scene, 10–87 s on the field). The plan
     §3.4 always had: hierarchy nodes keep prefiltered **appearance**
     (color and lit radiance) beside the coverage they keep for shadows,
     and a pixel's cone stops at nodes about its own size, so cost is
     bounded by the screen, not the scene. Steps, each gated: (a) the walk
     stopping at pixel-size nodes, its cost on the field from 150 m and
     standing; (b) per-instance and per-node lit appearance from the
     cache's coarse pages; (c) accuracy against M7 G1's area-integrated
     references, at a frame cost the quality dial can trade. **M13 G1
     (step a) failed:** prefiltering at the pixel works from 150 m (0.38×
     one ray's coverage error) but the light cones' per-node test (an 8×8
     bin rasterization) costs about 9 ns a visit per pixel: 1.2–2 s a
     frame. **M13 G2 failed:** a scalar per-node test is accurate (0.26–0.29×
     one ray's error off the horizon) but loses correlation within a pixel
     at the horizon (one row, 0.632 covered, ours 1.0), and still costs
     about 2 ns a node visit (257–281 ms): latency- and occupancy-bound. Next:
     its own entry point (it ran inside M7's heavy one), a few sub-pixel
     bins (4×4) for correlation, then a walk shared by a tile of pixels.
  2. *Unique geometry at scale (M14).* The 10¹⁰ triangles are a million
     copies of one boulder; everything must be resident. Stream geometry
     (and its hierarchy nodes) by need, with the prefiltered nodes of item
     1 standing in for what is not loaded, so a scene several times GPU
     memory flies at frame rate.
  3. *Frame-time stability:* flight spikes (the slowest 1% of frames about
     2× the mean; cost per page varies 2–3× with content,
     `flight-spikes-diagnostics.md`).
  4. *Lighting fundamentals:* the cache that never settles with bounce
     light (M10 G23's loop, a core bug), the sky's slow warm-up, more than
     one bounce, bounce light with movers, cheaper bounce shading (1.16×
     the frame at night, M10 G22).
  5. *Speed beyond one laptop:* the NVIDIA benchmark, when the user can
     fund it.
  - Parked features (after the core): emissive faces as lights (M10 G23,
    blocked by item 4's loop), a second texture coordinate set, movers
    under a sky, moving lights' dark state.
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
    environment as an extra base color layer).
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
  - Development tools: `backface_dev`, `ledger_dev`, `edges_dev`,
    `spikes_dev` (work and passes modes), `vcolor_dev`, `blend_dev` (each
    camera pass alone) and `bench`.
  - Speed criteria comparing two renderers: settle both, then five
    interleaved rounds of 30 warm frames each, median ratio (M10 G18). A
    single back-to-back pair met a transient in G17.

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
     first and brings views within 5% in 120–300 frames. Remaining levers
     for the total work are a wider hierarchy and packets for large bakes.
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
| M4 scale | Prefiltered cones, instancing, large-world coordinates, a virtual lighting cache | G1 prefiltered cones (**passed**: flat cost from 40k to 2.56M triangles, 86 ms vs 13.7 s exact, 0.55% from exact, `docs/results/m4-gate1.md`); G2 ten billion instanced triangles in a large world (**failed** on two of three: the 64 km-offset field renders identically with rebasing, and 10^10 triangles fit in 152 MB, but instanced vs flattened is 0.22% vs a 0.1% target because the flattened reference is the less precise one, and 10^6 vs 10^3 instances cost 3.04× vs 2× because camera rays near the horizon dominate; cones to a small sun are nearly free, `docs/results/m4-gate2.md`); G3 nearest-first ray traversal (**passed**: identical output, 10^10 vs 10^7 triangles 1.40×, 2.83× faster than the fixed order, 100.8 ms at 1080p for a million boulders, `docs/results/m4-gate3.md`); G4 visibility buffer (**failed**: identical output but 0.88× on the field and 1.11× slower on the showcase; register pressure was not the limit, ray traversal itself is, `docs/results/m4-gate4.md`); G5 traversal stacks in workgroup memory (**failed**: identical output but 2.6× slower, as 12 KB per workgroup limits occupancy, `docs/results/m4-gate5.md`); G6 hardware ray queries for camera rays (**failed**: no faster, 0.9×, and 67–270 ms per 720p frame. A corrected diagnostic shows per-pixel sun cones, not rays, dominate; gates 2 and 4 had blamed rays from a broken "light turned away" test, `docs/results/m4-gate6.md`); G7 cones that pay only for what they see (**failed**: no speedup; a stage-by-stage profile shows cones descending the triangle hierarchies of the objects they start on dominate, 119 of 215 ms from 150 m up at 720p, so per-pixel cones every frame are the brute-force path at scale too, `docs/results/m4-gate7.md`); G8 view-dependent lighting cache (**failed** all three: 14–39% from per-pixel exact cones, 20–37 ms warm, 42.7 ms mean in flight; but warm shading fell from 47–233 ms to 2–3 ms and only 1.8% of pixels lacked a page in flight. Diagnostics: one sample per pixel is itself 13–24% from the supersampled image, so later quality gates compare with the area-integrated image; and wgpu spends 15–17 ms of CPU time per `submit` walking the TLAS's million per-instance dependencies, while hardware rays take 4.5–6.6 ms on the GPU, `docs/results/m4-gate8.md`); G9 the cache at speed (**failed** on the flight's 99th percentile, 34.0 vs 33.3 ms. Same images as G8; warm frames 2.9–4.7 ms at 720p with hardware rays after a wgpu-core patch that keeps one TLAS dependency per distinct BLAS, pipelined requests and a faster page manager; flight mean 12.8 ms, 1.6% fallback. The flight is bound by CPU planning, which re-derives a backlog of ~50k wanted pages every frame, `docs/results/m4-gate9.md`); G10 smooth flight (**failed**: progressive requests from the shader and GPU deduplication cut CPU planning to 1.9 ms per frame, but the flight is then GPU-bound at 17.6–17.9 ms mean, 38 ms 99th percentile on a hot machine; next, baking by frame time instead of a fixed cone count, `docs/results/m4-gate10.md`); G11 baking to a frame-time target (**passed**: same images; warm 6.5–8.0 ms at 720p; three flights over a million boulders at 14.6–15.0 ms mean, 26–32 ms 99th percentile, 1.7% fallback. Caveat: the budget sits near its floor while flying, so only 0.6% of samples reach their wanted mip in motion and lighting is softer until the view settles, `docs/results/m4-gate11.md`); zoom and flythrough with no shimmer |
| M5 lights (v0.1 item 2) | Several lights, a sky environment, colored light | G1 sky light through six cube-face mask cones (**failed**: sky alone 2.74% from Cycles, sun and sky 5.12% against 3%; the reference was at fault: Blender 5.2's Cycles always box-filters at least one pixel, so it area-integrates while we point-sampled, and it loses 1.9% of a small sun 3 km away; sky interiors match Cycles to 0.1–0.5%, `docs/results/m5-gate1.md`); G2 the same with the reference fixed (**passed**: 4×4 per pixel against Cycles' one-pixel box, sun at 1 km: sky only 0.74%, sun and sky 1.19% exact and prefiltered; interiors within 0.0–0.5%, `docs/results/m5-gate2.md`); G3 object-level prefiltered coverage (**passed**: on the million-boulder field against instanced Cycles, sky only 1.15% and sun and sky 2.39% at 4×4 per pixel; more than 25× cheaper than walking every object; sun-only images bit-identical, `docs/results/m5-gate3.md`); G4 sky through the view cache (**failed** on cached-vs-per-pixel sky, 2.4–20.5% against 5%, the texel blur the sun channel also shows, and on speed; the speed numbers are confounded, since a control showed the machine running gate 11's own warm frame 10× slower than in gate 11; real finding: sky texels cost about 100× sun texels, so the bake budget's 1,000-texel floor must become a time, `docs/results/m5-gate4.md`); G5 sky through the cache at speed (**failed** on warm-up: with own-object sky masks baked once per geometry (faithful to 0.06–0.15%), sky faces as separate jobs, a horizon cull and a bake budget that follows GPU time, three sun-and-sky flights average 14.8 ms with 99th percentile under 25 ms and 4.5% fallback, warm frames 2.2–2.6 ms, sun channel unchanged; but views warm with sky in 13–95× the sun's frames, against 10×. Reported: the sky's error against Cycles depends on the cube's orientation, 1.9–4.2%, `docs/results/m5-gate5.md`; diagnostics in `docs/results/m5-sky-bake-diagnostics.md`); G6 several colored lights (**passed**: three RGB lights around the camera within 1.22% of Cycles at 16×16 samples per pixel over surfaces; re-coloring a warm cache bakes nothing and is bit-identical to a fresh one; flights with three lights 14.1–14.9 ms mean, 99th percentile 29–31 ms; light cones gained bounding-cone culling and nearest-first walks, which took a grazing lamp's flight from 579 to 14 ms; reported: with sky the 99th percentile is 35 ms, and gate 3's sun lit only the horizon, `docs/results/m5-gate6.md`); G7 lights cached as ratios, shaded with each pixel's normal (**failed** on flight speed: the cache's error against per-pixel cones fell to 0.41–0.56× with the sun and 0.57–0.68× with three lights; per-pixel still 1.22% from Cycles; but two of three flights with three lights averaged 17.09 and 17.58 ms, one with 99th percentile 36.4 ms, as camera passes doubled to 4.3–6.2 ms with the bake budget at its floor, `docs/results/m5-gate7.md`); G8 point and spot lights, and cheaper shading of lights (**failed** on sustained speed: a night scene with a point light and a spot is 1.35% from Cycles and its cache 0.82% from per-pixel; field cache errors within gate 7's; but flights ran last after about ten minutes of load and slowed from 14.5 to 20–25 ms mean, camera passes from 5 to 9.5 ms; a diagnostic showed the same flight slowing steadily under sustained load (13.1 → 16.3 ms over twelve flights) and none in one process when cool, `docs/results/m5-gate8.md`); G9 the sky later (**passed**: pages bake lights first, sky alone from the next frame; at gate 5's four viewpoints with a fixed budget, error while warming 0.11–0.17× the baseline's, within 5% at frame 120–300 against 1,200–4,220, warm in 0.61–0.65× the frames, warm image identical, `docs/results/m5-gate9.md`); G10 a guessed sky from coarser pages (development only, **failed**: 1.00× the baseline's flight error against 0.8×; removed, `docs/results/m5-gate10.md`); G11 environment light (**failed** both: 22.64% from Cycles, 1.16× the frame; nine harmonics cannot show a sky that ends at the horizon, `docs/results/m5-gate11.md`); G12 environment light, rays that miss reading the image (**passed**: 1.37% from Cycles against the uniform sky's 19.92%, 1.04× the frame, `docs/results/m5-gate12.md`) |
| M6 content (v0.1 item 3) | glTF import, base-color textures, automatic charts | G1 glTF import with automatic charts and textures (**failed** on criterion 1 only: the `.glb` round trip is 0.024% from the original per pixel against 0.01%, all in one edge pixel flipped by f32 rounding of instance transforms; the imported textured scene is 0.43% from Cycles with textures at 16×16 per pixel, its automatic charts give 2.37% cached vs per-pixel against the hand-made charts' 1.52%, warm frame 1.53 ms at 720p; real assets reported: a 28k-triangle textured pack warms in 1,317 frames at 1.43 ms, `docs/results/m6-gate1.md`) |
| M7 primary cones (v0.1 item 4) | Pixel-footprint cones: anti-aliasing and far-field level of detail | G1 pixel cones for primary visibility (**failed** all three: against the 16×16 reference the textured scene is 7.15% vs one ray's 7.11%, the field from 150 m 17.91% vs 23.49% (0.76× against ≤ 0.35×), from standing height 34.61% vs 12.49% with a banding artifact on the 2.1 km ground triangles; half a pixel's turn changes the image 1.67× as much as the reference's (≤ 1.5×); warm cached frames 105 ms on the textured scene (≤ 16.7 ms), per-pixel frames 10–87 s on the field; the textured scene's error is texture filtering, not silhouettes, `docs/results/m7-gate1.md`); G2 texture filtering for the area integral (**passed**: a box filter over the pixel's footprint from ray differentials, 3.99% from the 16×16 reference against 7.11%, warm frame 1.03×, `docs/results/m7-gate2.md`); G3 silhouette rays (**failed**: field from 150 m 0.49× the error but 13× the frame; textured scene 0.91×, 1.66×, `docs/results/m7-gate3.md`); G4 silhouette rays through ray queries (**failed** on the field's speed: quality as the compute path's, 1.29× the frame on the textured scene, 3.03× on the field from 150 m against 1.5× (compute path 24.15×), `docs/results/m7-gate4.md`); no shimmer in flythroughs |
| M8 library (v0.1 items 5–6) | Library API, Bevy plugin, examples; Windows/NVIDIA CI and a cloud benchmark | A tagged v0.1 (**done**: `v0.1.0`, 2026-11-07; the cloud benchmark is still open) |
| M9 dynamic scenes | Moving objects in the view cache (exact invalidation, same-frame re-bake); the ledger on the GPU: movers' effect on sky light, moving lights | G1 moving objects in the view cache (**passed**: 65 movers among a million boulders with three lights; no stale value in 30 checks of up to 137,260 pages; 0.84% of pages read re-baked per frame; flights 15.76–16.14 ms, 1.06–1.09× the movers held still, `docs/results/m9-gate1.md`); G2 a moving light through the ledger (**failed** on fallback only: a sun drifting 0.25° per second over the field; no page beyond its bound or ε in 30 checks, none changed at all; 0.69% of pages read re-baked per frame; means 14.45–14.58 ms against 15.64 ms still; fallback 33.3–34.2% against 10%, from off-screen pages set aside; the ε went into snapping, so no deferral within ε was exercised, `docs/results/m9-gate2.md`); G4 the ledger without its work limit (**failed** on speed: sound in 30 checks; at a real sun's speed fallback 4.9–5.0% but means 24.7–25.3 ms against 15.8 ms still, mixed texels re-baked every frame, `docs/results/m9-gate4.md`); G3 movers under a sky (**failed** on work and speed: sound in 30 checks, but 99.73% of pages read re-baked their sky every moving frame; means 53.0–58.3 ms against 14.66 ms still; fallback 30%, 21–22% even still from slow sky streaming; the per-page sphere bound about 5,000× loose, `docs/results/m9-gate3.md`); G5 the sky later with movers (development only, **failed**: 1.02× the error while moving against 0.5×; sound, cost 1.01×; kept, `docs/results/m9-gate5.md`); a light switch with no lag |
| M10 GI and reflections | More bounces, cascades of mask cones, roughness cones | G1 one bounce in the view cache (**failed** on criterion 3 only: on the textured scene, 1.79% from Cycles with one bounce against 7.09% without; warm frame 1.07×, settling 1.21×; but 794 pages baked in the 30 frames after re-coloring, from cache activity resuming after supersampling, not from re-coloring, `docs/results/m10-gate1.md`); G3 secondary requests for hits (**failed** on its re-coloring check, a procedure bug; settling with materials 1.12× instead of 6.9×; accuracy kept, 1.45%/9.01%, `docs/results/m10-gate3.md`); G4 re-coloring with a corrected settle (**passed**: 0 pages baked, `docs/results/m10-gate4.md`); G2 metallic-roughness materials and glossy reflections (**failed** on accuracy: 8.94% from Cycles against 6%, 12.71% without reflection rays; warm frame 1.82× against 2×, `docs/results/m10-gate2.md`); G5 normal maps (**passed**: 3.63% from Cycles with the maps, 26.42% ignoring them; warm frame 1.04×, `docs/results/m10-gate5.md`); G6 alpha cutouts as geometry (**failed**, no criterion reached: the opaque control never settled, from a CPU/shader texel-grid mismatch now fixed, `docs/results/m10-gate6.md`); G7 gate 6 re-run (**passed**: 3.29% from Cycles masking the cards, 34.07% with them opaque; warm frame 1.15×, `docs/results/m10-gate7.md`); G8 exactly back-to-back faces (**failed** on accuracy: speed fixed, 1.03× the 1 mm scene's bake time; but Metal ray queries ignore `CULL_BACK_FACING`, so camera rays hit either coincident face, 5.01% from Cycles, `docs/results/m10-gate8.md`); G10 gate 2 re-run (**passed**: 5.85% from Cycles, warm frame 1.68×; specular from lights had applied Karis's normalization twice, so sun highlights on metal were low, `docs/results/m10-gate10.md`); G9 gate 8 re-run with back-face hits re-traced (**failed** on its 0.5% bound against the 1 mm control, 1.08%, which mostly measures that control's own offset: 1 mm to 2 mm moves the image as much; 3.42% from Cycles against the control's 3.37%; bake 1.21×, warm frame 1.04×; fixes kept, `docs/results/m10-gate9.md`); G11 emission lights its surroundings (**failed**: scored over all pixels, which the lights' visible disks decide, 109.67%; 1.00×, `docs/results/m10-gate11.md`); G12 the same over surfaces (**failed** on the absolute bound: 7.46% from Cycles against 5%, 18.00% without (0.41×); 1.00×; kept, `docs/results/m10-gate12.md`); G13 MikkTSpace tangents (**failed**: 11.80% against the triangles' frames' 11.75% where they differ; 1.00×; kept, `docs/results/m10-gate13.md`); G14 emissive textures (**failed** from the reference: 62.86%, Cycles lit the double-sided sign from its coincident faces, `docs/results/m10-gate14.md`); G15 re-run with a one-sided sign (**passed**: 6.35% over surfaces, texture ignored 491.34%; 1.00×, `docs/results/m10-gate15.md`); G16 alpha blending (**failed** on speed: 4.71% from Cycles, panes opaque 57.31%; 1.33× the frame against 1.3×; kept, `docs/results/m10-gate16.md`); G17 vertex colors (**failed** on speed: 1.29% from Cycles, colors ignored 57.63%; but 1.15× the frame against 1.1×, a transient that development runs did not reproduce, `docs/results/m10-gate17.md`); G18 its speed re-measured in five interleaved rounds (**passed**: 1.004×, `docs/results/m10-gate18.md`); G19 blending's speed (**passed**: 1.226× the frame against 1.3×, layers traced in the visibility pass and the list shaded by an indirect dispatch; 4.70% from Cycles, `docs/results/m10-gate19.md`); G20 texture wrap modes (**passed**: 2.18% from Cycles with clamped and mirrored textures, 67.44% repeating; 0.996× the frame, `docs/results/m10-gate20.md`); G21 bounce light with more than three lights (**failed**, no result: crashed at re-coloring on a missing buffer usage, output lost, `docs/results/m10-gate21.md`); G22 its re-run (**failed** on speed: 1.167× the frame with bounce against 1.15×, bounce light's own cost on that night scene, 1.165× with three lights too; accuracy 3.07% from Cycles against 17.78% without bounce; re-coloring a later light 0.407% from a fresh renderer; kept, `docs/results/m10-gate22.md`); G23 emissive faces as area lights (**failed**, accuracy not reached: 1.286× the frame, then the view never settled at a sub-pixel offset, a loop with bounce samples; settling 6× slower; kept as experimental, `docs/results/m10-gate23.md`) |
| M11 showcase | Bistro/Sponza, UE5 side-by-side | Published comparisons in motion |
| M12 deforming geometry | Subtree refit | Characters in the showcase |
| M13 prefiltered appearance (core first, item 1) | Hierarchy nodes keep lit appearance; pixel cones stop at pixel-size nodes | Sub-pixel detail without aliasing, cost bounded by the screen. G1 coverage only (**failed**: from 150 m 0.38× one ray's coverage error, passing, but 1.2–2 s a frame, about 100× too slow: each node visit rasterizes into 8×8 bins; standing failed mostly on a flawed measure, buried boulder parts the ground hid in the reference only, `docs/results/m13-gate1.md`); G2 a scalar per-node test (**failed**: 257–281 ms a frame, about 26× over; accuracy 0.38× one ray's from 150 m, 0.59× standing, where one horizon row holds 84% of the error: independent splats saturate a pixel whose boulders all sit low in it, correlation that sub-pixel bins keep, `docs/results/m13-gate2.md`) |
| M14 unique geometry at scale (core first, item 2) | Geometry and hierarchy streamed by need, prefiltered nodes standing in | A scene several times GPU memory at frame rate; not started |

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
