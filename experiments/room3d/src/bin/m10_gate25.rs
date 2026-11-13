//! M10 gate 25: more than one bounce (Landmark 2).
//!
//! The cache gathers one bounce of light: a bounce sample sees the direct
//! light of the pages its rays hit. That is enough outdoors (on M6's
//! textured scene Cycles' 1 and 4 bounces differ by 0.65–0.93%), but in the
//! showcase's closed gallery, lit by one skylight panel, light past the first
//! bounce is two fifths of it: Cycles' mean radiance is 0.188 with one
//! bounce and 0.318 with twelve, and the cache's one bounce is 45% from the
//! latter (3.6% from Cycles' one bounce).
//!
//! Now (`CacheOptions::bounces`): a bounce sample also gathers the bounce
//! light of the pages it hits, and records how many bounces it holds, one
//! more than the fewest its hits hold. A page whose hits hold too few asks
//! for them with samples of their own (one bounce fewer, down to pages with
//! none) and re-bakes its samples until they do (a capped number of times).
//!
//! Pass criteria, fixed before the run:
//! 1. **Accuracy indoors:** the showcase room (its static surfaces, the
//!    skylight panel, no sky), 640×360, the cache settled at 4×4 sub-pixel
//!    offsets, against instanced Cycles at 12 bounces and 4,096 samples per
//!    pixel: with 8 bounces, relative RMSE at most 10% and at most 0.25× the
//!    one bounce's.
//! 2. **Settling:** with 8 bounces, every one of the 16 offsets settles (32
//!    frames requesting and baking nothing) within 2,000 frames.
//! 3. **Outdoors unharmed:** M6's textured scene by day, the same way,
//!    against Cycles at 12 bounces: with 8 bounces at most the one bounce's
//!    error plus 0.1 point.
//!
//! Reported: 2 and 4 bounces; frames and bounce samples to settle from empty.
//!
//! `cargo run --release -p room3d --bin m10_gate25`

fn main() {
    println!("M10 gate 25: stub (criteria in the doc comment); not implemented yet");
}
