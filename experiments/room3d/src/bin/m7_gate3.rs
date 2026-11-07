//! M7 gate 3: anti-aliasing at silhouettes only.
//!
//! Pixel cones (M7 gate 1) resolved coverage in every pixel and cost 48× a
//! frame. Most pixels lie inside one surface, where one ray and the box
//! texture filter (gate 2) already approximate the area integral; the
//! aliasing is at edges.
//!
//! Method: in the cached shading pass, a pixel whose four neighbours'
//! visibility differs from its own (another object, or a depth 5% away)
//! is an edge. An edge pixel averages four rays through the centers of its
//! 2×2 quarters (compute rays through the scene's hierarchy), each shaded
//! from the cache; other pixels keep their one ray.
//!
//! Scenes, 1280×720, cache settled (32 quiet frames after the call):
//! M6's textured scene, and the million-boulder field from 150 m up (M7
//! gate 1's view), sun only.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Quality:** in both views, relative RMSE from the area-integrated
//!    cached image (16×16 sub-pixel offsets, each settled, textures for
//!    1/16 of a pixel) at most 0.75× that of one ray per pixel.
//! 2. **Speed:** the warm frame at most 1.5× one ray's in both views
//!    (median of 30 frames each waited for).
//!
//! Reported: the share of pixels found to be edges.
//!
//! `cargo run --release -p room3d --bin m7_gate3`

fn main() {
    unimplemented!("M7 gate 3: not yet implemented");
}
