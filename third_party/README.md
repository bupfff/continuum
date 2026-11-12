# Patched dependencies

## wgpu-core 30.0.1

A copy of `wgpu-core` 30.0.1 from crates.io (MIT OR Apache-2.0, licenses
included), used through `[patch.crates-io]` in the workspace manifest.
Changes, in `src/command/ray_tracing.rs`, `src/resource.rs` and
`src/device/ray_tracing.rs` (marked "Continuum patch"):

- **What.** A top-level acceleration structure records each distinct
  bottom-level structure once as a dependency, not once per instance.
- **Why.** Every submission that uses the TLAS walks that list, to validate it
  and to declare residency. With a million instances of two BLAS, the walk
  cost about 16 ms of CPU per `queue.submit` (M4 gate 8,
  `docs/results/m4-gate8.md`).
- **Safety.** Validation (each BLAS built, none newer than the TLAS),
  lifetime and residency all need each BLAS only once.

- **Incremental TLAS builds (M14 gate 6).** A TLAS keeps each instance's
  resolved BLAS and its encoded bytes; a build resolves, validates and
  encodes only the instances whose BLAS, transform, custom data or mask
  changed, and updates the distinct-BLAS list by reference counts. With
  tens of thousands of instances, re-doing all of them cost milliseconds
  per build.

Remove this copy once upstream wgpu has an equivalent fix.

## wgpu-hal 30.0.1

A copy of `wgpu-hal` 30.0.1 from crates.io (MIT OR Apache-2.0), used the
same way. One change, in `src/metal/command.rs` (marked "Continuum patch"):

- **A persistent residency set (M14 gate 6).** Every submission that
  traces a TLAS made a new Metal residency set holding every BLAS it
  depends on and committed it. Now one set per device is kept, updated by
  difference (a structure stays resident for 64 calls after it was last a
  dependency, so command buffers in flight keep it) and committed only
  when its members change. With tens of thousands of BLAS the old way cost
  milliseconds per frame.

Remove this copy once upstream wgpu has an equivalent fix.
