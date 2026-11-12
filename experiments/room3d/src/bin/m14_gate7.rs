//! M14 gate 7: frame-time stability at scale.
//!
//! Gate 6 flew 36,864 unique rocks at 6.2–6.5 ms a frame on average (99th
//! percentile 10–12 ms) but one frame took 42.3 ms. Replaying its flights
//! from views not yet settled (`stream_dev` `SPIKES=192`, a harsher case)
//! gave 58 frames over 33.3 ms in 5,400, their time in unrelated stages:
//! uploading and building camera levels (up to 112 ms), choosing
//! evictions (24 ms), trimming the CPU cache (42 ms), lighting updates
//! (22 ms), the cache's planning (25 ms), waiting for the GPU (up to 22
//! ms). Work that varies that much in unrelated places points at the
//! render thread being starved or blocked, not at one algorithm:
//! - eight producer threads (camera and lighting levels) run at the render
//!   thread's priority on a chip with four performance cores;
//! - the render thread frees large data itself (levels dropped from the
//!   CPU caches, megabytes each);
//! - a frame's uploads can outgrow the staging belts' chunks, which then
//!   allocate new mapped buffers mid-frame.
//!
//! Method:
//! - producer threads run at a lower quality of service (utility, on
//!   macOS), so the system keeps the performance cores for the render
//!   thread;
//! - data dropped from the CPU caches is freed on a background thread;
//! - the staging belts' chunks hold a frame's typical uploads (32 MB).
//!
//! Scene, options, flights, reference and criteria: gate 6's, unchanged.
//!
//! Pass criterion, fixed before the implementation (gate 6's four):
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
//! Reported besides: frames over 16.7 ms and over 33.3 ms per flight, and
//! the slowest frames' parts.
//!
//! `cargo run --release -p room3d --bin m14_gate7`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    println!("M14 gate 7: not implemented yet");
}
