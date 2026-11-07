//! M7 gate 2: texture filtering for the area integral.
//!
//! M7 gate 1 found that on the textured scene, one ray per pixel is 7.11%
//! from the area-integrated image, and pixel cones did not help: the error
//! is texture filtering, not silhouettes. Each pixel samples its texture at
//! one mip chosen for its footprint's longest axis, stretched by the viewing
//! angle, which blurs across the shorter axis (isotropic trilinear).
//!
//! Method: the pixel's footprint on the surface from ray differentials (the
//! camera ray's change per pixel, projected onto the triangle's plane),
//! mapped to texture space through the triangle's texture coordinates.
//! The texture is read with four bilinear taps at the centers of the
//! footprint's 2×2 quarters, each at the mip of a quarter's longest axis:
//! a box filter over the pixel, anisotropic where the footprint is.
//!
//! Scene: the textured scene of M6 (sun and sky), its camera, 1280×720,
//! per-pixel lighting (mask cones, no cache), one ray per pixel.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Quality:** relative RMSE from the 16×16 area-integrated reference
//!    (M7 gate 1's: 256 frames over sub-pixel offsets, textures sampled for
//!    1/16 of a pixel) at most 0.8× that of the isotropic filter, measured
//!    in the same run. Pixels where a reference sample sees a light, and
//!    their neighbours, are set apart (as in M7 gate 1).
//! 2. **Speed:** the warm cached frame on the same scene at most 1.15× the
//!    isotropic filter's (median of 30 frames, each waited for).
//!
//! `cargo run --release -p room3d --bin m7_gate2`

fn main() {
    unimplemented!("M7 gate 2: not yet implemented");
}
