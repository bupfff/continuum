//! M14 gate 2: fewer, cheaper loads, and detail that follows the zoom.
//!
//! M14 gate 1 streamed camera levels correctly (6.35× the budget, settled
//! coverage within 0.062–0.096× one ray's error, 3.4× less GPU memory) but
//! failed on frame rate in one flight of three (1.124× the fully resident
//! renderer against 1.10×). Diagnosis after it (`stream_dev`, disclosed in
//! the results): a load cost about 37 ns a triangle (structure build, then
//! uploads through fresh staging buffers); the buddy allocator's
//! fragmentation forced evictions; rocks briefly out of view were evicted
//! and reloaded; over-detailed rocks dropped to their lighting level and
//! reloaded. Most of the volume is inherent: a flight passes about 120
//! rocks close enough to need their finest level once.
//!
//! Changes (made while diagnosing, before this stub):
//! - The pool is fixed pages of 4,096 triangles; a level's bottom-level
//!   structure has one geometry per page, and a hit's geometry index names
//!   the page through a page table after the objects' ranges. Nothing
//!   fragments.
//! - Uploads go through a reused staging belt.
//! - Eviction order: unseen for 30 frames first; then the most over-detailed
//!   seen geometry, downgraded in place to the level it wants (never shown
//!   coarser than it needs); then recently seen ones.
//!
//! Scene, budget, flights and reference are gate 1's (256 unique rocks 6 m
//! apart, finest levels 6.35× a budget of 2.5 × 2²⁰ triangles, half a
//! pixel of error), plus a fourth flight that zooms: standing 10 m outside
//! the field, the field of view narrows from 50° to 1.5° (exponentially)
//! onto a rock about 100 m away. As the view narrows, geometry and lighting
//! resolve further; what leaves the view costs nothing.
//!
//! Pass criterion, fixed before the run (gate 1's, over four flights):
//! 1. **Scale:** the rocks' finest levels hold at least 5× the budget, and
//!    the resident camera-level triangles never exceed it (every frame of
//!    every run).
//! 2. **Frame rate:** on each flight, the streamed renderer's mean frame at
//!    most 16.7 ms, its 99th percentile at most 33.3 ms, and its mean at
//!    most 1.10× the fully resident renderer's (three interleaved rounds per
//!    flight, each settled at the flight's first view; timed before
//!    everything else).
//! 3. **Coverage, settled:** at each flight's last view (the zoom's at
//!    1.5°), the streamed renderer's coverage error against the resident
//!    renderer at most 0.25× one ray's (gate 1's measure, 4×4 samples).
//! 4. **Coverage in flight:** replaying each flight, every 10th frame, the
//!    share of pixels whose visible object differs from the resident
//!    renderer's at most 1% on average.
//!
//! Reported: loads by reason (upgrades, reloads, downgrades) and evictions;
//! the working set at the zoom's start and end; the settled images' colour
//! difference; memory (the GPU device's allocations).
//!
//! `cargo run --release -p room3d --bin m14_gate2`
//!
//! `M14_DEV=1` runs a development check (36 rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    unimplemented!("M14 gate 2: stub committed with its criteria before the run");
}
