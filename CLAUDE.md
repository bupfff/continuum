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
  - **Streamed geometry (M14):** `stream.rs` (camera levels, the pool,
    the policy; `LightingPool` for lighting levels) and `stream.wgsl` (the
    feedback pass); `cam_tri` in `hw_visibility.wgsl`; `lit_missing` and
    `LIT_MARK` in `mask_cones.wgsl` and the `lit_*` fields in `vcache.rs`
    (pages waiting for lighting levels); the rock field `room3d::rocks`
    (`with_proxies`, `on_demand`); `stream_dev` sizes working sets and has
    `PAGES`/`STILL`/`MEMORY`/`LOAD_COST`/`BASELINE` modes.
  - **README media:** diagrams from `scripts/readme_diagrams.py`
    (`docs/images/how/`), GIFs from the `readme_media` binary's frames
    through `scripts/make_gif.py`, Cycles comparisons in
    `docs/images/compare/`, the logo in `docs/images/brand/`. License:
    PolyForm Noncommercial 1.0.0 (`LICENSE.md`), by Penumbral Labs; the repository is public.
- Gate workflow:
  1. Write the gate's binary as a stub whose doc comment states the pass
     criterion, and commit it before implementing or running.
  2. Give the binary a rehearsal mode that runs every criterion exactly as
     the gate measures it (full size, the same baselines and bake-budget
     mode, timing interleaved in one process) on fewer frames or views.
     Rehearse before the gate: a missed bar means more development, not a
     gate run, and a shader that fails to compile shows up here.
     Development on smaller scenes is fine but never stands in for the
     rehearsal, and is disclosed.
  3. Run the gate once at full size, saving its output as it runs (`|
     tee`, under a time limit). Record pass or fail with exact numbers in
     `docs/results/<gate>.md`. Never tune and re-run after a near miss;
     fixes go into a new gate.
  4. Per gate, update only the results file and the milestone table, in
     one commit with a one-line subject (verdict and key numbers; details
     live in the results file). README, GUIDE and DESIGN §3.6–3.8 are
     brought current once a session or at a milestone's end.
  5. Park after two failed gates on the same criterion: write the levers
     left into DESIGN, then take the next item, unless the criterion is
     one of the four landmarks (DESIGN §3.8) or the user says otherwise.

  Commits end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>".
- Session habits (agreed with the user 2026-11-20):
  - DESIGN §3.8's "Start here" stays about 200 words: the state, the next
    three items, current pitfalls. History lives in git and
    `docs/results/`.
  - One milestone per conversation where possible; in a long run, bring
    "Start here" current at each milestone so a compaction or a fresh
    start loses nothing.
  - While a gate holds the laptop, only read and write: the next design
    and stub, the results draft, literature checks (network only).
    Schedule a check on the run instead of ending the turn idle.
- Metal pitfalls: at most 29 storage buffers a stage (`vc_shade` is at
  it); ray queries run about 50× slower when tmin skips a long grazing
  path (start rays at the box entry); real-frame pass times come from
  Instruments (`xcrun xctrace record --template 'Metal System Trace'`,
  table `metal-gpu-intervals`), and traces fill the disk (delete them
  after export).
- Cycles references (Blender 5.2 at `/Applications`; `room3d::cycles`):
  - Cycles always averages each pixel's area (at least a one-pixel box), so
    compare our renders supersampled (4×4), never one sample per pixel.
  - Keep small bright emitters within about 1 km: Cycles loses 1.9% of a
    0.5° sun at 3 km.
  - Use `render_instanced` for large instanced scenes.
  - Glass needs transmission bounces (one bounce renders it black); clamp
    highlight pixels at 4× the mean for smooth glass. Blender 5.2 removed
    `material.cycles.homogeneous_volume`. `cycles.rs` reuses a cached EXR
    while the job's JSON is unchanged: delete its directory after changing
    a script. Blender's glTF importer maps t seconds to frame t × fps.
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
