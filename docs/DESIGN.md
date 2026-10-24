# Continuum design

Living document. It is the source of truth for what we are building and why;
update it whenever a decision changes. Last updated 2026-10-23.

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
emitted power); others come from bounds (an occluder of a given size moving a
given distance can flip at most k bins). When a ledger exceeds ε, scaled by how
much the result contributes to visible pixels, the result is recomputed, largest
first. Indirect changes propagate through the same rule and die out because
energy shrinks every bounce.

Consequences:

- Frame cost is roughly screen work plus change work.
- Each pixel has a certified bound on its distance from a full recompute.
- With ε below a just-noticeable difference, staleness is invisible by construction.
- Occlusion-aware invalidation: a change behind a wall does not touch lighting
  in front of it, because the wall's bins end the cone's dependency region.

Prior art closest to this (none found doing it end to end in real time):
incremental radiosity with line-space hierarchies (Drettakis & Sillion 1997,
Granier & Drettakis 2001), the render cache (Walter et al. 1999), incrementally
baked GI (2019), path verification (2021), temporally adaptive shading reuse
(Mueller et al. 2021, heuristic gradients), JND-driven temporal reuse (2025).

### 3.2 Mask cones (the primitive)

Every visibility query is a cone carrying a direction mask (64 bins on the CPU;
see open questions for the GPU layout). Occluders set the bins they cover;
combining is bitwise OR, which keeps correlated occlusion exact up to bin
resolution, unlike multiplied transmittance. One primitive serves anti-aliasing
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
| M0 flatland (CPU, 2D) | Mask cones, ledger, worst case | G1 masks beat rays on error per cost (**passed**, `docs/results/flatland-gate1.md`); G2 ledger soundness: certified bound ≥ actual error, every pixel, every frame (**passed**, including forced-deferral stress runs); G3 ledger skips ≥ 80–90% of lighting work in typical animated scenes (**passed: 87.7%**, all from dependency regions, see `docs/results/flatland-gates2-3.md`); G4 everything-moving case costs no more than a conventional equivalent (preview: 1.1× faster); G5 (new) magnitude deferral pays off for wide GI cones with continuous visibility |
| M1 3D CPU reference | 3D mask cones over a BVH, reference path tracer, scene import | G1 repeated in 3D with 8×8 bins; agreement with Cycles/Mitsuba |
| M2 GPU primary visibility | wgpu, tile frustum traversal, mask coverage AA | Frame time at 1080p on M4 Air; image-diffed in CI |
| M3 prefiltered hierarchy | Directional coverage per node, detail to roughness | Zoom and flythrough with no shimmer; flat cost |
| M4 direct light + ledger on GPU | Area lights, cone shadows, ledger | G2/G3 on GPU; light switch with no lag |
| M5 GI and reflections | Cascades of mask cones, roughness cones | FLIP vs reference |
| M6 showcase | glTF import, Bistro/Sponza, UE5 side-by-side | Published comparisons in motion |
| M7 deforming geometry, Bevy | Subtree refit, `bevy_continuum` | Characters in the showcase |

## 7. Benchmark protocol

- M4 MacBook Air is fanless and throttles: report a short warm run and a separate
  sustained number, plugged in, same scene, same camera path.
- Correctness CI on Windows (WARP) and Linux (lavapipe); NVIDIA performance on a
  rented cloud machine per benchmark round.
- Image quality against an independent path tracer, scored with FLIP; flicker
  measured as frame-to-frame difference on a static camera.

## 8. Risks and open questions

- **Ledger bound tightness** (biggest risk): too-pessimistic charges mean
  everything recomputes and the edge disappears. Flatland: actual error used only
  0.2–8% of the certified bound. Sources: one charge per tile applied to all its
  receivers, bin terms, accumulation over oscillating motion.
- **Mask quantization**: 64 bins quantize penumbrae into 1/64 steps (faint
  banding in gate 1) and make visibility discontinuous, so any mover touching a
  cone costs at least one bin. That floor exceeded realistic ε, which is why the
  ledger never deferred direct shadows. Fix candidate for both: analytic
  coverage for partially covered edge bins (continuous visibility).
- **Where magnitude deferral pays**: direct shadows of moving objects change
  visibly every frame, so recomputing them is correct; the gain must come from
  wide, low-frequency queries (GI) and slow global changes. Unproven until G5.
- **3D projection cost**: projecting a triangle or node into 8×8 bins costs more
  than a 2D interval, so gate 1's speedup will shrink in 3D. Re-measure in M1.
- **GPU mask layout**: WGSL has no 64-bit integers without native-only features;
  options are 32-bin masks (one bin per lane at subgroup size 32) or `vec2<u32>`.
  Subgroup ops need wgpu's subgroup feature on every target.
- **Prefiltered wide cones**: resampling a node's stored masks into a cone's bins
  loses precision; correlation is only exact down to node resolution.
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
