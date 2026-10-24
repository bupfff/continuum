# Patched dependencies

## wgpu-core 30.0.1

A copy of `wgpu-core` 30.0.1 from crates.io (MIT OR Apache-2.0, licenses
included), used through `[patch.crates-io]` in the workspace manifest. One
change, in `src/command/ray_tracing.rs` (marked "Continuum patch"):

- **What.** A top-level acceleration structure records each distinct
  bottom-level structure once as a dependency, not once per instance.
- **Why.** Every submission that uses the TLAS walks that list, to validate it
  and to declare residency. With a million instances of two BLAS, the walk
  cost about 16 ms of CPU per `queue.submit` (M4 gate 8,
  `docs/results/m4-gate8.md`).
- **Safety.** Validation (each BLAS built, none newer than the TLAS),
  lifetime and residency all need each BLAS only once.

Remove this copy once upstream wgpu has an equivalent fix.
