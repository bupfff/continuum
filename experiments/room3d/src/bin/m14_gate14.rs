//! M14 gate 14: several bounces with streamed lighting.
//!
//! M10 gate 25 carried light past its first bounce (`CacheOptions::bounces`;
//! the showcase room 6.54% from Cycles at 12 bounces with 8, against 45.36%
//! with one), with bounce rays through hardware ray queries, on scenes held
//! wholly in memory. M14 gate 10 flew 36,864 unique rocks with a sky and one
//! bounce within 60 fps, lighting levels streamed and direct light
//! certified. Untried: both together. With several bounces a bounce sample
//! also gathers the bounce light of the pages its rays hit, and pages whose
//! hits hold too few bounces ask for those pages with samples of their own:
//! more pages, on objects whose lighting levels may not be resident.
//!
//! Method: gate 10's binary with `bounces: 8` (M10 gate 25's setting); no
//! other change.
//!
//! Scenes: gate 10's (a) 9,216 rocks beside every lighting level resident,
//! and (b) 36,864 rocks; a uniform sky and bounce light, 8 bounces.
//!
//! Pass criterion, fixed before the run (gate 10's first three):
//! 1. **Frame budget** (b): each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//! 2. **Certified** (a): sun and sky pages shown, in flight and settled,
//!    identical (1e-5) to every lighting level resident; at least 100,000
//!    sun and 30,000 sky pages.
//! 3. **Bounce converges** (a): settled images within 1% of the
//!    all-resident renderer's (both with 8 bounces).
//!
//! Reported: frames to settle (gate 10 with one bounce: 8,000, 8,000 and
//! 6,026 here); the settled images against one bounce's.
//!
//! `cargo run --release -p room3d --bin m14_gate14`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    unimplemented!("M14 gate 14: written before the implementation");
}
