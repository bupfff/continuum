//! M14 gate 9: the frame budget with a sky and bounce light at scale.
//!
//! Gate 8 streamed lighting levels with a sky and bounce light, direct
//! light certified and bounce converging, but on 36,864 rocks its flights'
//! 99th percentile was 20–25 ms. The cache sizes each frame's bake from a
//! model of GPU time (the camera passes and the bake, fit as latency plus
//! cost per unit) aimed at `frame_target_ms`, 20% over and under on
//! alternate frames so the fit can tell latency from cost. With a sky and
//! bounce under streaming there is always baking left, so frames sit at the
//! target, and the swing above it, GPU work the model does not see
//! (structures built, uploads) and the CPU's streaming push them past
//! 16.7 ms.
//!
//! Method:
//! - a controller on the measured frame period caps the bake budget: while
//!   frames take longer than `CacheOptions::frame_budget_ms`, the cap
//!   shrinks in proportion; while they are well under it, it grows by 5% a
//!   frame, never past the model's budget;
//! - the swing for the fit goes only down (100% and 80% of the budget).
//!
//! Scenes, flights and options: gate 8's ((a) 9,216 rocks beside every
//! lighting level resident; (b) 36,864 rocks), a uniform sky and bounce
//! light, the cache's frame target back at its default (14 ms), the frame
//! budget 15 ms.
//!
//! Pass criterion, fixed before the implementation:
//! 1. **Frame budget** (b): each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//! 2. **Certified** (a), as gate 8: every page shown in flight (every 60
//!    frames) and settled at each flight's last view identical (1e-5) in its
//!    sun channel, and with its sky baked in its sky channel, to every
//!    lighting level resident; at least 100,000 sun and 30,000 sky pages.
//! 3. **Bounce converges** (a), as gate 8: each flight's settled image
//!    within 1% (mean absolute difference over the image's mean) of the
//!    all-resident renderer's.
//!
//! Reported: the bake budget's cap, pages baked a frame, frames over 16.7
//! ms.
//!
//! `cargo run --release -p room3d --bin m14_gate9`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    println!("M14 gate 9: not implemented yet");
}
