//! M13 gate 1: sub-pixel geometry through prefiltered pixel cones
//! (coverage only).
//!
//! One ray per pixel sees one surface per pixel, so wherever a pixel holds
//! many (the boulder field from afar) the image is noise: 23.49% from the
//! area-integrated image from 150 m (M7 gate 1). M7 gate 1's pixel cones
//! resolved every pixel to an eighth of a pixel and cost hundreds of
//! hierarchy nodes per pixel (10–87 s a frame on the field). M13's plan
//! (DESIGN §3.4, §3.8): a pixel's cone stops at hierarchy nodes about its
//! own size and reads their prefiltered summaries, so cost follows the
//! screen, not the scene. This gate tests the geometric half alone: how
//! much of each pixel is covered by geometry. Appearance (color and
//! light) is the next gate's.
//!
//! Method: each pixel's cone (its square seen from the eye, 8×8 bins) walks
//! the object hierarchy and the triangle hierarchies as light cones do, but
//! with the prefilter threshold at the pixel (8 bins): object nodes and
//! triangle nodes that project within a pixel splat their prefiltered
//! coverage (front area over silhouette; objects as independent
//! occluders), larger ones are descended. Coverage combines across meshes as
//! independent occluders. The ground is left out: the measure is how much
//! of each pixel the boulders cover.
//!
//! Scene: the million-boulder field (M4), M7 gate 1's two viewpoints
//! (standing at 1.7 m; 150 m up, looking 45° down), 1280×720.
//!
//! Reference: per pixel, the share of 16×16 rays through its square (a grid
//! of sub-pixel offsets) that hit a boulder. One ray: the same with the
//! pixel's central ray alone (0 or 1).
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of the boulders' per-pixel coverage from
//!    the reference at most 0.5× one ray's.
//! 2. **Speed:** the coverage pass at most 16.7 ms a frame at 1280×720
//!    (median of 10 frames each waited for, hardware or compute as the
//!    kernel chooses; the pass alone, no shading).
//! Reported: work per pixel (object nodes, meshes entered, triangle nodes),
//! and the same with M7 gate 1's threshold (one bin) for contrast.
//!
//! `cargo run --release -p room3d --bin m13_gate1`

fn main() {
    unimplemented!("M13 gate 1: written before the implementation");
}
