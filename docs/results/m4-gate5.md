# M4 gate 5: traversal stacks in workgroup memory

Date: 2026-10-24. M4 gate 4 showed that ray traversal itself, not register
pressure from the cone code, limits camera rays. Every traversal (rays and
cones, through object and triangle hierarchies) used a 48-entry stack in a
per-thread array indexed at runtime, which Apple GPUs tend to keep in slow
memory.

**The change.** Each invocation's stack becomes a slice of a workgroup array
(64 invocations × 48 entries, 12 KB per workgroup), interleaved so that
neighboring invocations hit different memory banks. A pipeline constant
selects workgroup or per-thread stacks from the same code.

Pass criterion, fixed before the first run (committed in 827388b before
implementing): 1920×1080, best of 5, both variants in the same session.

1. **Same output.** The 1,000,000-boulder field and the showcase's cached
   render match the per-thread versions within 1e-5 of peak at every pixel.
2. The 1,000,000-boulder frame is at least 1.3× faster.

**Verdict: FAIL on (2).** The output is identical, but workgroup stacks are
2.6× *slower* on the field, and slower everywhere else.

## Results

Apple M4 (MacBook Air, on battery). Reproduce with
`cargo run --release -p room3d --bin m4_gate5`.

| | Per-thread stacks | Workgroup stacks |
|---|---|---|
| 1,000,000 boulders, 1920×1080 | 100.6 ms | 255.5 ms (0.39×) |
| Showcase cached render, 1920×1080 | 5.20 ms | 130.30 ms |
| M2 render scene, per-pixel cones, 960×540 | 31.5 ms | 62.1 ms |
| Showcase bake | 72.8 ms | 122.2 ms |

Max difference: **0.0** on both scenes.

## Why it failed

- **Workgroup memory limits occupancy.** 12 KB of workgroup memory per 64
  invocations caps how many workgroups a GPU core can keep in flight, so
  memory latency is no longer hidden.
- **Development confirmed it before the gate ran.** With the 12 KB array
  declared but unused (the per-thread branch selected), the showcase's cached
  render took 127 ms. Shrinking the unused array brought it back to 4.6 ms.
  Interleaving the stacks to avoid bank conflicts changed nothing.
- **The gate's baseline is fair:** the per-thread variant compiles without the
  workgroup array.
- **Workgroup stacks only pay with far fewer entries.** 48 entries are needed
  for a million objects plus a 10k-triangle mesh beneath them.

## Decision

Per-thread stacks stay the default (`KernelOptions::workgroup_stacks` is
false). The option remains for other GPUs, where the balance may differ; that
is untested.

## Where the ray floor stands

After gates 3–5, camera rays cost about 70–100 ms at 1080p on the field: not
register pressure (gate 4), not stack placement (gate 5). The remaining levers:

- **Hierarchy quality.** Surface-area-heuristic builds, or 4-wide nodes.
- **Fewer node visits for the far field.** This is the paradigm's own answer:
  primary visibility through pixel-footprint cones that stop at nodes smaller
  than a pixel and use their prefiltered coverage (DESIGN §3.2 and §3.4). Cost
  would follow pixels, as Nanite's does, and the far field would stop
  aliasing.
- **Hardware ray queries** where wgpu exposes them. That is a portability and
  originality question for the design (DESIGN rule 5).
