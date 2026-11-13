//! M4 gate 12: pages requested three levels finer: less error while
//! flying (Landmark 2), quicker warm-up.
//!
//! The error while flying, measured against settled references at the same
//! cameras (development, `flight_error_dev`): on M9's flight over the
//! million-boulder field with sun and sky, the image averaged 24–26% from
//! settled (relative RMSE). Pixels showed coarser levels than they wanted
//! (0.2% found their wanted level): the ground's boulder shadows smeared
//! into blotches, boulders unshadowed. A pixel showing a page coarser than
//! it wants requests the page one level finer, so a region walks from its
//! coarsest page to its wanted one in ten rounds or more, each a bake and a
//! readback two frames late, baking every level in between.
//!
//! Now a pixel requests up to three levels finer (`CacheOptions::
//! refine_levels`, 3 by default): fewer rounds, fewer pages in between.
//!
//! Pass criteria, fixed before the run:
//! 1. **Speed in flight** (run first): three pairs of M9's flight (600
//!    frames, standing height to 150 m, 1280×720, sun and sky, default
//!    options), one level then three: the three levels' mean frame at most
//!    1.05× the one level's (over the three flights each).
//! 2. **Error while flying:** the same flight at a fixed bake budget (11,000
//!    units a frame, so the comparison does not depend on the machine's
//!    heat), every 60th frame's image against a renderer settled at that
//!    camera: the mean relative RMSE with three levels at most 0.9× that
//!    with one. Reported: the largest, and both with default options.
//! 3. **Warm-up and result:** at M5 gate 13's four viewpoints with the fixed
//!    budget, from empty: frames until warm (16 quiet frames) with three
//!    levels at most 1.0× those with one, and the warm images within 0.01%
//!    (relative RMSE) of each other.
//!
//! `cargo run --release -p room3d --bin m4_gate12`

fn main() {
    println!("M4 gate 12: stub (criteria in the doc comment); not implemented yet");
}
