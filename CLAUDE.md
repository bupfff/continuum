# Working on Continuum

- `docs/DESIGN.md` is the source of truth for the paradigm, design rules and
  milestones. Read it before proposing changes; update it when a decision changes.
- Results of every experiment go in `docs/results/` with the exact numbers and caveats.
- Where things live:
  - **Current state and next steps:** DESIGN.md §3.6–3.8 and the milestone
    table (§6). Start a new conversation at §3.8's "Start here" block, and
    keep it current at the end of each session.
  - **Library front door:** `Renderer` in
    `crates/continuum-gpu/src/renderer.rs`; usage in `docs/GUIDE.md`; the
    example `crates/continuum-gltf/examples/render.rs`.
  - **Renderer:** `crates/continuum-gpu`. `mask_cones.wgsl` holds the kernel;
    `vcache.rs` is the view-dependent cache; `instanced.rs` the instanced
    renderer; `live.rs` the lightmap renderer with movers; `hierarchy.rs`
    the prefiltered hierarchies; `cutout.rs` alpha cutouts made geometry.
  - **Materials and textures:** `InstancedScene` in `continuum-gpu/src/lib.rs`
    (`geometry_materials`, `_emission`, `_normal_maps`,
    `_metallic_roughness`, `_alpha_cutoff`, `_emissive_textures`,
    `_blend`); `Environment` (the environment sky) in the same file;
    `tangents.rs` MikkTSpace; glTF import and save in
    `crates/continuum-gltf/src/lib.rs`.
  - **Core types:** `crates/continuum`.
  - **2D experiments:** binaries in `experiments/flatland/src/bin/`; every
    ledger experiment runs on `flatland::incremental`.
  - **3D gates and diagnostics:** binaries in `experiments/room3d/src/bin/`
    (`m<milestone>_gate<n>.rs`), with scenes in `room3d::{showcase, field,
    outdoor, textured}` (`Textured::new()` and its `.with_materials()`,
    `.with_normal_maps()`, `.with_foliage()`, `.with_glow()`,
    `.with_sign()`, `.with_glass()`).
  - **Viewer:** `apps/viewer` (`cargo run --release -p viewer -- --scene
    field --scale 2`).
  - **Patched wgpu-core:** `third_party/` (see its README).
  - **Sub-pixel detail (M13):** `coverage_maps.rs` builds directional
    coverage maps; `mask_cones.wgsl` sections "M13 gate 2/3/4" and
    `hw_visibility.wgsl`'s `hw_m13_*` read them; profile stages 10–16
    select the diagnostic paths (DESIGN §3.8, working notes).
  - **README media:** diagrams from `scripts/readme_diagrams.py`
    (`docs/images/how/`), GIFs from the `readme_media` binary's frames
    through `scripts/make_gif.py`, Cycles comparisons in
    `docs/images/compare/`, the logo in `docs/images/brand/`. License:
    PolyForm Noncommercial 1.0.0 (`LICENSE.md`), by Penumbral Labs.
- Gate workflow:
  1. Write the gate's binary as a stub whose doc comment states the pass
     criterion, and commit it before implementing or running.
  2. Run it once at full size. Development runs on smaller scenes are fine
     but must be disclosed.
  3. Record pass or fail with exact numbers in `docs/results/<gate>.md`.
     Never tune and re-run after a near miss; fixes go into a new gate.
     Save the run's output as it runs (`| tee`), and give the binary a
     development mode run once before the gate (a shader that fails to
     compile would otherwise cost the gate its only run).
  4. Update the milestone table, §3.6–3.8 and the README, and commit.

  Commits end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>".
- Cycles references (Blender 5.2 at `/Applications`; `room3d::cycles`):
  - Cycles always averages each pixel's area (at least a one-pixel box), so
    compare our renders supersampled (4×4), never one sample per pixel.
  - Keep small bright emitters within about 1 km: Cycles loses 1.9% of a
    0.5° sun at 3 km.
  - Use `render_instanced` for large instanced scenes.
- Design rules that are easy to forget: no rasterization anywhere (hard cases get
  slower paths inside the paradigm); no blending of previous frames for image
  quality; wgpu and compute only; every performance or quality claim is measured
  against an independent reference before it is stated.
- Novelty claims must cite the closest prior art (see DESIGN.md section 9) and
  stay hedged until a literature check is done.
- Timings come from release builds, single-threaded unless stated. The dev
  machine is a fanless M4 MacBook Air: report warm and sustained numbers.
  - Its heat moves timings by 2× or more. Before a timed gate, check `ps
    -axo pid,etime,%cpu,command -r | head` for stray processes (games,
    leftover scripts), and repeat an earlier gate's number as a control. If
    the control is far off, the machine is not fit to measure (M5 gate 4).
  - Sustained load heats it within minutes, and a short warm-frame control
    does not catch that. The same flight slows about 1.25× after two minutes
    of flights and about 2× after ten minutes of heavy work (M5 gate 8;
    `flight_repeat` measures it). Run timed criteria first in a gate, or
    state the thermal state they measure.
