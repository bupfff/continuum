//! M14 gate 10: bounce rays through hardware ray queries, and the frame
//! budget with a sky and bounce at scale.
//!
//! Gate 9 held flights with a sky and bounce on 36,864 rocks at 12.8–13.1
//! ms on average, but their 99th percentile at 19.8–21.6 ms. Measured
//! since (`stream_dev` `FLIGHTS9`, gate 9's flights): with the cache's frame
//! budget at 11 ms the bake falls to its floor (11–22 pages a frame) and
//! frames still take 9–10 ms; turning off the levels bounce samples want
//! (they keep lighting loads at 64 a frame: 150,000–190,000 levels produced
//! a flight against gate 7's 30,000) gave 99th percentiles of 14.4–16.0 ms.
//! Bounce samples trace 192 rays each through the lighting hierarchy, in
//! compute, over every object, and need every lighting level they pass.
//!
//! Method: where camera rays use hardware ray queries, bounce samples trace
//! their rays through the same top-level structure (camera levels,
//! clusters of proxies): fast, and they never need a lighting level, so
//! they want none. A hit is shaded as before (its page, its albedo, its
//! emission). The cache's frame budget may be lowered from 15 ms; the value
//! is fixed and disclosed before the run.
//!
//! Scenes: gate 9's (a) 9,216 rocks beside every lighting level resident,
//! and (b) 36,864 rocks; a uniform sky and bounce light. And M10 gate 22's
//! night scene (bounce light from four lights) against its Cycles reference.
//!
//! Pass criterion, fixed before the implementation:
//! 1. **Frame budget** (b): each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//! 2. **Certified** (a), as gate 9: sun and sky pages shown, in flight and
//!    settled, identical (1e-5) to every lighting level resident; at least
//!    100,000 sun and 30,000 sky pages.
//! 3. **Bounce converges** (a), as gate 9: settled images within 1% of the
//!    all-resident renderer's.
//! 4. **Bounce accuracy**: M10 gate 22's scene within 5% of Cycles over
//!    surfaces (its bar) with bounce rays through hardware ray queries.
//!
//! `cargo run --release -p room3d --bin m14_gate10`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    println!("M14 gate 10: not implemented yet");
}
