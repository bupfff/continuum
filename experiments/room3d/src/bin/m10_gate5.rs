//! M10 gate 5: normal maps.
//!
//! glTF content carries surface detail in tangent-space normal maps; without
//! them, tiles and bricks shade flat.
//!
//! Method (in the cached path):
//! - `InstancedScene::geometry_normal_maps`: per geometry, a texture holding
//!   a tangent-space normal map (linear; OpenGL's convention, as glTF
//!   specifies: green toward the image's top), uploaded to its own linear
//!   layered texture with mip levels.
//! - The tangent frame per triangle, from its texture coordinates'
//!   derivatives (exact for planar texture mappings; MikkTSpace differs on
//!   curved meshes), orthogonalized against the interpolated normal.
//! - The mapped normal replaces the interpolated one for light from lights
//!   (the cached shadowed share times the unshadowed light at the mapped
//!   normal), specular and reflections. The cached sky and bounce light stay
//!   at the interpolated normal.
//!
//! Scene: the textured scene of M6 (sun and sky), its camera, 1280×720,
//! bounce light on, Lambertian. Normal maps, generated to match the color
//! textures: the ground's stone tiles beveled toward their mortar lines,
//! and the block's bricks likewise (bevels 12% of a tile's width, sloped
//! 30°).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 4%
//!    relative RMSE of instanced Cycles with the same normal maps (Normal
//!    Map node, tangent space, one bounce, 4,096 samples per pixel), and at
//!    most 0.6× the error of the same scene rendered ignoring the normal
//!    maps against that reference.
//! 2. **Speed:** the warm frame at most 1.15× the scene's without normal
//!    maps (median of 30 frames each waited for, bounce light on in both).
//!
//! Reported, not gated: Cycles with against without the normal maps (the
//! effect's size).
//!
//! Details fixed before the first run: as M10 gate 2 (settled: 32 quiet
//! frames after the call; 4×4 supersampling over sub-pixel offsets, each
//! settled; Cycles at 4,096 samples per pixel, its one-pixel box filter).
//!
//! `cargo run --release -p room3d --bin m10_gate5`

fn main() {
    unimplemented!("M10 gate 5: not yet implemented");
}
