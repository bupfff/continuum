//! M10 gate 6: alpha cutouts.
//!
//! Foliage, fences and chains in glTF are cards whose base color alpha
//! (alphaMode MASK) cuts their shape. Rendered opaque, they are solid
//! rectangles that cast rectangular shadows.
//!
//! Method: the cut becomes geometry, so every path stays exact with no
//! change: camera rays, light and sky cones, the cache and bounce light.
//! - `InstancedScene::geometry_alpha_cutoff`: per geometry, a cutoff for
//!   its base color texture's alpha.
//! - Before upload, each masked triangle is replaced by the opaque part of
//!   its texture: the contour of the bilinear alpha field at the cutoff
//!   (marching squares on texel centers, crossings interpolated linearly),
//!   at the mip where the triangle spans at most 128 texels, clipped to the
//!   triangle in texture space and mapped back through its barycentric
//!   coordinates (positions, normals, texture coordinates).
//! - Runs of fully opaque cells merge into rectangles.
//!
//! Scene: the textured scene of M6 (sun and sky), its camera, 1280×720,
//! bounce light on. Added: shrubs made of crossed double-sided leaf cards
//! and a lattice fence panel, both alpha-masked (cutoff 0.5) with textures
//! generated here.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles masking the same cards by the same
//!    alpha (a Transparent BSDF where alpha is below the cutoff; one
//!    bounce, 4,096 samples per pixel), and at most 0.5× the error of the
//!    same scene with the cards opaque against that reference.
//! 2. **Speed:** the warm frame at most 1.5× the opaque cards' (median of
//!    30 frames each waited for, bounce light on in both).
//!
//! Reported, not gated: triangles before and after cutting; the cut's CPU
//! time.
//!
//! Details fixed before the first run: as M10 gate 2 (settled: 32 quiet
//! frames after the call; 4×4 supersampling over sub-pixel offsets, each
//! settled; Cycles at 4,096 samples per pixel, its one-pixel box filter).
//!
//! `cargo run --release -p room3d --bin m10_gate6`

fn main() {
    unimplemented!("M10 gate 6: not yet implemented");
}
