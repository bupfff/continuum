//! M14 gate 3: unique geometry beyond CPU memory, produced on demand, within
//! a 60 fps frame budget.
//!
//! Gates 1 and 2 streamed camera levels correctly, but every level of every
//! rock sat packed in CPU memory (2.7 GB for 256 rocks), standing in for a
//! disk, and frames with many loads reached 29–34 ms. A world worth
//! streaming does not fit in CPU memory either.
//!
//! Method:
//! - A scene can give its camera levels through a `LevelSource`, called on
//!   worker threads as levels are wanted (the stand-in for reading a disk;
//!   here the rocks are generated from their seeds). Produced levels wait
//!   in a CPU cache with a byte cap, the least recently used dropped first.
//! - A streaming time budget (the user's decision, 2026-11-12): loads build
//!   at most `frame_ms` of structures a frame, at a calibrated cost per
//!   triangle (about 30 ns on the M4, M14 gate 2), so loading never pushes
//!   a frame past the 60 fps budget; deferred loads are reported.
//! - Everything else is gate 2's (paged pool, level chains, feedback).
//!
//! Scene: 64 × 64 = 4,096 unique rocks 6 m apart (a 384 m square) with gate
//! 1's levels (finest 65,536 triangles); lighting levels resident; camera
//! budget 8 × 2²⁰ triangles; half a pixel of error; 1280×720, the view
//! cache on, hardware camera rays. Three flights of 1,800 frames (gate 1's
//! paths over the larger field, about 13 m/s at 60 fps).
//!
//! Reference: there is no fully resident renderer at this size. At each
//! settled view, a renderer built with every rock at its *ideal* level for
//! that view (the coarsest within half a pixel at its nearest distance, if
//! in the view; its lighting level otherwise), all resident. Gates 1–2
//! measured the ideal levels against the finest (0.06–0.10× one ray's
//! coverage error); this gate measures streaming's convergence to them.
//!
//! Pass criterion, fixed before the run:
//! 1. **Scale:** the rocks' finest levels hold at least 30× the camera
//!    budget's triangles, and more bytes packed than the machine's memory;
//!    resident camera-level triangles never exceed the budget, and the CPU
//!    level cache never exceeds its cap (every frame).
//! 2. **Frame budget (60 fps):** on each flight, rendering as fast as it
//!    can, the mean frame and the 99th percentile at most 16.7 ms, the
//!    worst frame at most 33.3 ms.
//! 3. **Settled:** at each flight's last view, once streaming and the cache
//!    are quiet, the coverage error against the ideal-level reference at
//!    most 0.25× one ray's (gate 1's measure, 4×4 samples).
//!
//! Reported: loads and production (latency from request to ready, cache
//! hits), loads deferred by the time budget, the share of seen rocks
//! showing a coarser level than wanted in flight, memory (GPU allocations,
//! the CPU cache), and the settled colour difference.
//!
//! `cargo run --release -p room3d --bin m14_gate3`
//!
//! `M14_DEV=1` runs a development check (16 × 16 rocks, short flights),
//! whose numbers are not the gate's.

fn main() {
    unimplemented!("M14 gate 3: stub committed with its criteria before the implementation");
}
