//! M14 gate 5: lighting levels produced on demand, certified against a
//! reference no single renderer could be.
//!
//! Gate 4 streamed lighting levels (what mask cones, the cache's bakes and
//! texel lookups see) into GPU slots, but kept every one packaged in CPU
//! memory, beside the scene's own copy of its triangles: about 330 KB a
//! rock, growing with every unique object. And its reference held every
//! lighting level at once, which only works while the world fits one
//! renderer.
//!
//! Method:
//! - A streamed geometry's lighting level comes from the scene's
//!   `LevelSource` (the scene holds only its charts): worker threads
//!   produce its package (triangles, hierarchy and coverage, chart cell
//!   lists) into a CPU cache with a fixed cap, the least recently wanted
//!   dropped first; the slot pool loads from that cache, and a level not
//!   yet produced waits as an absent one does (pages wait for it).
//! - What stays resident per unique object is its record: the stub (its
//!   hierarchy root's box and prefiltered coverage, written once at
//!   creation as an offline build would), its proxy, its chart's cells,
//!   and its camera levels' page table.
//!
//! Scene: gate 4's rocks over a field 9× larger (192 × 192 = 36,864
//! unique rocks), each with a proxy, camera and lighting levels both
//! produced on demand; gate 4's lighting slots (491,520 triangles) and
//! camera budget (8 × 2²⁰ triangles); lighting packages cached in 256 MB
//! of CPU memory; the sun; 1280×720, the view cache, hardware camera rays.
//! Gate 4's three flights, at its speed, over the central 384 m (they see
//! the whole field).
//!
//! Pass criterion, fixed before the implementation:
//! 1. **Scale:** the lighting levels hold at least 64× the lighting slots'
//!    triangles and at least 16× the CPU cache's bytes (packaged as gate 4
//!    kept them); neither the slots nor the cache is ever exceeded; the
//!    scene holds no rock's lighting triangles.
//! 2. **Certified beyond one renderer:** each flight replayed beside a
//!    reference renderer holding only the rocks within 212 m of the field's
//!    centre (axis-aligned), every lighting level resident (gate 4's
//!    reference way); every 60 frames, every page shown here on a rock
//!    within 200 m of the centre, or on the ground with its texels' box
//!    inside that square, and shown there too, holds the same texels
//!    (within 1e-5 relative); likewise every such page both hold settled at
//!    each flight's last view. No compared page differs, at least 100,000
//!    are compared in all, and the margin (12 m) exceeds the longest sun
//!    shadow any rock casts (from the rocks' bounds and the sun's
//!    elevation).
//! 3. **Memory per unique rock:** from 4,096 to 36,864 rocks with the same
//!    pools, measured right after creation (renderer and scene), GPU
//!    memory grows by at most a sixth of what a resident lighting level
//!    takes (measured the same way from 1,024 to 4,096 rocks with every
//!    lighting level resident; 217 KB in development), and CPU heap memory
//!    by at most a sixth of a lighting level as gate 4 kept it (its package
//!    and its surface).
//! 4. **Frame budget:** each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//!
//! Reported: lighting levels produced and their latency, the CPU cache's
//! peak, loads, evictions and refusals, pages hidden while waiting, settled
//! frames, the time to write the stubs, and what a renderer holding every
//! lighting level would take.
//!
//! Disclosed before the implementation: with gate 4's code, a rock's
//! resident record took 27 KB of GPU memory (16,384 rocks: 26.8 KB a rock,
//! `stream_dev` `RECORDS`), 0.124× a resident lighting level; most of it
//! is the proxy (its triangles and its hardware structure) and the chart's
//! cells.
//!
//! `cargo run --release -p room3d --bin m14_gate5`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    println!("M14 gate 5: not implemented yet");
}
