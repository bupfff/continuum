# Working on Continuum

- `docs/DESIGN.md` is the source of truth for the paradigm, design rules and
  milestones. Read it before proposing changes; update it when a decision changes.
- Results of every experiment go in `docs/results/` with the exact numbers and caveats.
- Design rules that are easy to forget: no rasterization anywhere (hard cases get
  slower paths inside the paradigm); no blending of previous frames for image
  quality; wgpu and compute only; every performance or quality claim is measured
  against an independent reference before it is stated.
- Novelty claims must cite the closest prior art (see DESIGN.md section 9) and
  stay hedged until a literature check is done.
- Timings come from release builds, single-threaded unless stated. The dev
  machine is a fanless M4 MacBook Air: report warm and sustained numbers.
