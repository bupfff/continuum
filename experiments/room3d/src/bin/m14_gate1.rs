//! M14 gate 1: unique geometry streamed by need (camera levels).
//!
//! Until now every triangle of a scene was resident: the 10¹⁰ triangles of
//! M4 were a million copies of one boulder. Unique geometry at that density
//! does not fit, so it must be streamed: kept on the GPU only where the
//! view needs its detail, within a fixed budget, with something standing in
//! for the rest.
//!
//! Method (the first step of M14):
//! - Each geometry may carry finer *camera levels* (`Levels`), coarse to
//!   fine, sharing its charts, each with its largest distance from the
//!   finest. The geometry itself stays the *lighting level*: always
//!   resident, what mask cones, the cache's bakes and bounce light see, and
//!   the camera's stand-in.
//! - Camera rays (hardware ray queries) see each geometry at its finest
//!   resident level. After the visibility pass, a feedback pass turns each
//!   pixel's hit distance into the level its object needs (the coarsest
//!   whose distance from the finest stays under `pixel_error` pixels) and
//!   keeps the highest per geometry. The CPU reads that back two frames
//!   later, loads the most needed levels into a fixed pool (triangles,
//!   normals, chart coordinates; a bottom-level structure each), evicts the
//!   least recently needed, and points the geometry's top-level instances at
//!   the new level.
//! - Lighting is unchanged and certified as before, for the lighting
//!   level. Streaming the lighting level by cone footprint is a later gate.
//!
//! Scene: unique procedural rocks (each its own geometry: band-limited
//! octaves, one more per level; four levels, the lighting level always
//! resident) on a ground, lit by a sun; 1280×720, the view cache on,
//! hardware camera rays. The fully resident renderer (every rock's finest
//! level uploaded at start) is the reference.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Scale:** the rocks' finest levels hold at least 5× the camera
//!    budget's triangles, and the resident camera-level triangles never
//!    exceed the budget (checked every frame of every run).
//! 2. **Frame rate:** on three flights, the streamed renderer's mean frame
//!    is at most 16.7 ms and its 99th percentile at most 33.3 ms, and its
//!    mean at most 1.10× the fully resident renderer's on the same flight
//!    (resident, then streamed, per flight; timed before everything else).
//! 3. **Coverage, settled:** at each flight's last view, once streaming and
//!    the cache are quiet, the streamed renderer's coverage error against
//!    the resident renderer is at most 0.25× one ray's coverage error.
//!    Coverage error: per pixel, half the sum over objects (a miss counted
//!    as one) of the absolute difference in the share of 4×4 sub-pixel
//!    samples each covers, averaged over pixels. One ray's: the resident
//!    renderer's pixel centre against its own 4×4 samples.
//! 4. **Coverage in flight:** replaying each flight frame by frame, every
//!    10th frame, the share of pixels whose visible object differs from the
//!    resident renderer's at the same frame is at most 1% on average.
//!
//! Reported: memory (the pool's buffers, and bottom-level structures from
//! the process's footprint), loads and evictions per flight, the cache's
//! fallback in flight against the resident renderer's, and the settled
//! images' colour difference (4×4 supersampled).
//!
//! `cargo run --release -p room3d --bin m14_gate1`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    unimplemented!("M14 gate 1: stub committed with its criteria before the implementation");
}
