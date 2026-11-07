//! M10 gate 2: metallic and rough materials, and glossy reflections.
//!
//! Everything so far is Lambertian. glTF's metallic-roughness model is what
//! content uses; metals and polished surfaces reflect their surroundings.
//!
//! Method (in the cached path):
//! - **Materials:** per geometry, metallic and roughness (glTF's model:
//!   F0 = 0.04 for dielectrics, the base color for metals; GGX microfacets,
//!   α = roughness²). Geometries without one stay Lambertian.
//! - **Light from lights:** GGX specular evaluated toward each light's
//!   nearest point to the reflection direction (a representative point,
//!   Karis 2013, the lobe widened by the light's angular size), times the
//!   cached shadowed share (the diffuse visibility stands in for the
//!   specular's).
//! - **Glossy reflections:** one ray per pixel along the reflection
//!   direction. The hit's light is read from the cache at the mip matching
//!   the lobe's width there (a roughness cone, prefiltered by the cache's
//!   mips rather than sampled, so no noise). A miss sees the sky. Weighted
//!   by the split-sum's directional albedo (Karis's analytic fit).
//! - **Diffuse:** the base color times (1 − metallic), less what the
//!   specular layer reflects.
//!
//! Scene: the textured scene of M6, its camera, 1280×720, sun and sky,
//! bounce light on. Materials: the ground dielectric, roughness 0.6; the
//! boulders metallic, roughness 0.3; the brick block dielectric, roughness
//! 0.4.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 6%
//!    relative RMSE of instanced Cycles (Principled BSDF, GGX, one bounce,
//!    4,096 samples per pixel), and at most 0.6× the error of the same
//!    scene rendered Lambertian against that reference.
//! 2. **Speed:** the warm frame at most 2× the Lambertian scene's (median
//!    of 30 frames each waited for, bounce light on in both).
//!
//! Reported, not gated: the specular share of the image; the error with
//! materials but without reflection rays.
//!
//! `cargo run --release -p room3d --bin m10_gate2`

fn main() {
    unimplemented!("M10 gate 2: not yet implemented");
}
