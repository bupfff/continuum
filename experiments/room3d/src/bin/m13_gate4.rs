//! M13 gate 4: sub-pixel geometry, near surfaces resolved by hardware rays
//! (coverage only).
//!
//! M13 gate 3 passed accuracy in both views (0.264× and 0.231× one ray's
//! error) with directional coverage maps and 4×4 sub-pixel bins, but took
//! 178–707 ms a frame. Development after it (`m13-gate3.md`) found walking
//! the triangles of instances larger than 8 pixels, per pixel, was 75–180
//! ms of about 200 at 10,000 boulders; the maps and the object walk about
//! 25 ms. Those large instances are near, resolved surfaces, where one ray
//! is right except at their edges.
//!
//! Method (profile stage 16):
//! 1. A hardware ray through each pixel's top-left corner (one per pixel,
//!    `hw_m13_corners`) records the object it hits and whether that object
//!    is large (more than 8 pixels across, or without a coverage map).
//! 2. Each pixel (`m13_coverage_near`) whose four corners all hit the same
//!    large object is wholly covered by it. Every other pixel, and the last
//!    row and column, runs gate 3's walk (its development changes
//!    included: nodes over full bins skipped, far object nodes no taller
//!    than a bin splatted, map addressing hoisted).
//!
//! Scene, views and reference are gate 3's: the million-boulder field
//! without its ground (in the reference and the test alike), standing at
//! 1.7 m and 150 m up, 1280×720; per pixel, the share of 16×16 rays through
//! its square that hit a boulder. One ray: the central ray alone.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.5× one ray's (gate 3's bar).
//! 2. **Speed:** both passes at most 50 ms a frame at 1280×720 (median of 10
//!    frames each waited for). An intermediate bar: M13's goal stays a
//!    10 ms coverage pass, for which the far field's walk must still get
//!    cheaper.
//!
//! Reported: the share of pixels resolved by their corners; object and
//! triangle nodes per pixel.
//!
//! `cargo run --release -p room3d --bin m13_gate4`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the passes alone),
//! whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 4: written before the implementation");
}
