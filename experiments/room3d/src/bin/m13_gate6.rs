//! M13 gate 6: hierarchy nodes holding coverage, as view-dependent coverage
//! pages (coverage only).
//!
//! The design is DESIGN §3.4, "The far field for sub-pixel coverage", part
//! D, on top of gate 5's pass (A, B, C, E). At the standing view's horizon
//! a pixel sees boulders from about 540 m to the field's edge, each about a
//! pixel across, and walks up to thousands of nodes reading their maps.
//!
//! Method:
//! - A node's page is its coverage seen from the bake eye at one texel per
//!   bin: 512 8-bit texels in a shape from 32×16 to 128×4 (or transposed);
//!   each texel the share of 4×4 hardware rays from the bake eye through its
//!   square, clipped to the node's box, that meet an instance in the node's
//!   item range. A header holds the bake eye, the page's plane (through the
//!   node's centre, facing the eye), its axes and texel sizes.
//! - During gate 5's tile walk, a node whose projected box fits a page shape
//!   at one texel per bin is an entry if its page is resident and valid;
//!   otherwise it is descended, and requested if it fits (deduplicated on
//!   the GPU). Pixels read a page per bin where the bin's ray crosses its
//!   plane, bilinearly, combined with other entries as independent
//!   occluders. Instances keep their shared maps; nodes holding movers never
//!   get pages.
//! - A page is valid while `2 f |ΔE⊥| L / D²` (the largest shift of its
//!   contents, in bins, since its bake) is at most 1/4 and its texels are
//!   0.7–1.4 bins wide; past that it is not read until re-baked. Pages bake
//!   within a time budget each frame, in a pool of 32,768.
//!
//! Scene, views and reference are gates 4 and 5's (the million-boulder
//! field without its ground, standing at 1.7 m and 150 m up looking 45°
//! down, 1280×720, the reference 16×16 rays per pixel, one ray the central
//! one).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy, settled:** with every wanted page baked at the view,
//!    relative RMSE of per-pixel coverage from the reference at most 0.25×
//!    one ray's standing and 0.15× from 150 m (gate 5's bars).
//! 2. **Speed, settled:** at most 20 ms a frame in both views with the pages
//!    resident (median of 10 frames each waited for), timed first.
//! 3. **Memory:** the pages wanted in each view within the pool (32,768
//!    pages, at most 32 MB with headers).
//! 4. **Reuse in motion:** a 10 s walk at 1.5 m/s at standing height along
//!    the view, pages re-baked within a 2 ms budget a frame from a cold
//!    pool: at 10 frames evenly spread after the first second, coverage
//!    within 0.3× the error of that frame's one-ray image from its own 16×16
//!    reference.
//!
//! Reported: pages wanted, baked and read per view; entries per horizon
//! pixel; rays a frame for bakes; frames on the walk that read no stale page
//! and the largest bound read.
//!
//! `cargo run --release -p room3d --bin m13_gate6`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the passes alone),
//! whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 6: written before the implementation");
}
