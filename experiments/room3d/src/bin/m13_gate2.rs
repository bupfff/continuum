//! M13 gate 2: sub-pixel geometry, a scalar per-node test (coverage only).
//!
//! M13 gate 1 failed: prefiltering each pixel's cone at the pixel halved
//! one ray's coverage error from 150 m (0.38×), but each node visit
//! rasterized the node into the cone's 8×8 bins, about 9 ns a visit across
//! the GPU: 1.2–2 s a frame. Its standing view also measured badly: the
//! ground hid buried boulder parts from the reference's rays but not from
//! the cone walk (`m13-gate1.md`).
//!
//! Method (profile stage 12 of the pixel-cone entry point): each pixel walks
//! the object hierarchy, then each object's triangle hierarchy, testing
//! boxes in screen space (the 8 corners projected):
//! - a box whose projection misses the pixel is skipped;
//! - a box within one pixel (its projection's longer side) adds the share of
//!   the pixel its projected box covers times its prefiltered opacity seen
//!   from the eye (object nodes 1 − exp(−front area / silhouette), triangle
//!   nodes front area / silhouette), with no bins;
//! - a larger box is opened; a triangle larger than a pixel adds the exact
//!   area of the pixel it covers (front faces only).
//! Within an object, contributions add (its front faces do not overlap) up
//! to 1; objects combine as independent occluders. The walk stops once the
//! pixel is covered. Boxes crossing the eye's near plane are opened down to
//! triangles, and triangles crossing it are skipped.
//!
//! Scene: the million-boulder field with its ground left out (for the
//! reference and the test alike: the measure is the boulders' coverage,
//! occlusion by resolved surfaces being later work), M7 gate 1's two
//! viewpoints (standing at 1.7 m; 150 m up, 45° down), 1280×720.
//!
//! Reference: per pixel, the share of 16×16 rays through its square (a grid
//! of sub-pixel offsets) that hit a boulder. One ray: the central ray alone.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.5× one ray's.
//! 2. **Speed:** the coverage pass at most 10 ms a frame at 1280×720 (median
//!    of 10 frames each waited for), leaving room for shading within a
//!    60 Hz frame.
//! Reported: nodes and triangles per pixel.
//!
//! `cargo run --release -p room3d --bin m13_gate2`

fn main() {
    unimplemented!("M13 gate 2: written before the implementation");
}
