//! M14 gate 6: gate 5's world within 60 fps.
//!
//! Gate 5 produced lighting levels on demand over 36,864 unique rocks and
//! stayed certified and small (27.7 KB of GPU memory a rock), but flew at
//! 26–39 ms a frame. It was not the lighting churn: with 16× the lighting
//! slots frames stayed at 29–47 ms. A profile of the render thread at
//! 16,384 rocks (`sample`, `stream_dev` `GATE5=128`) found work that grows
//! with the number of objects or of uploads, not with the view:
//! - about 25%: lighting uploads through `queue.write_buffer`, which
//!   allocates a new Metal buffer for every call (a dozen a level), and
//!   about 7% more freeing them;
//! - about 13%: every submission that traces the top-level structure builds
//!   a new Metal residency set holding every bottom-level structure (one
//!   per rock's proxy, plus loaded levels) and commits it;
//! - about 12%: the camera streamer's own work, scanning every geometry's
//!   levels to trim the CPU cache (once per level dropped) and to choose an
//!   eviction (once per eviction).
//!
//! Method:
//! - lighting levels upload through a staging belt into the frame's
//!   encoder (as camera levels already do);
//! - the patched wgpu (`third_party/`) keeps one residency set across
//!   submissions and updates it by difference, committing only when the
//!   set of structures changed;
//! - the streamer keeps lists of the levels held in CPU memory and of the
//!   geometries with levels resident, and scans only those.
//!
//! Scene, options, flights, reference and criteria: gate 5's, unchanged
//! (36,864 unique rocks, camera and lighting levels produced on demand,
//! 491,520 triangles of lighting slots, a 256 MB lighting cache, an 8 ×
//! 2²⁰-triangle camera budget, the sun, 1280×720; gate 4's three flights
//! over the central 384 m).
//!
//! Pass criterion, fixed before the implementation (gate 5's four):
//! 1. **Scale:** lighting levels at least 64× the slots' triangles and 16×
//!    the CPU cache's bytes; neither ever exceeded; the scene holds no
//!    rock's lighting triangles.
//! 2. **Certified beyond one renderer:** every page shown in flight (every
//!    60 frames) and settled at each flight's last view, on a rock within
//!    200 m of the centre or on the ground inside that square, identical
//!    (1e-5) to a reference holding the rocks within 212 m with every
//!    lighting level resident; at least 100,000 compared; the longest sun
//!    shadow within the 12 m margin.
//! 3. **Memory per unique rock** (4,096 → 36,864 rocks, same pools, at
//!    creation): GPU at most 1/6 of a resident lighting level's, CPU heap
//!    at most 1/6 of a level as gate 4 kept it.
//! 4. **Frame budget:** each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//!
//! Reported besides: per frame, the CPU time of camera streaming and of
//! lighting updates, the wait for the GPU, and the GPU's camera and bake
//! passes.
//!
//! `cargo run --release -p room3d --bin m14_gate6`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    println!("M14 gate 6: not implemented yet");
}
