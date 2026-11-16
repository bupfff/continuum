//! M16 gate 4: surface lobes (a dielectric's reflectance and a clear coat).
//!
//! Method (DESIGN §3.4, "Transmission", surface lobes):
//! `InstancedScene::geometry_specular` (index, factor, tint; glTF's
//! `KHR_materials_ior` and `KHR_materials_specular`) sets the specular
//! layer's reflectance at normal and grazing incidence;
//! `InstancedScene::geometry_clearcoat` (glTF's `KHR_materials_clearcoat`)
//! adds a coat (index 1.5): its own specular from each light and a
//! reflection ray, over what lies under it dimmed by its directional
//! albedo.
//!
//! Scene: M6's textured scene with the ground rough (0.6), the boulders a
//! rough dielectric (0.6) under a smooth clear coat (factor 1, roughness
//! 0.05) and the brick block of index 2.0 with its reflectance tinted
//! (1, 0.75, 0.5), roughness 0.3 (`Textured::with_lobes`). 1280×720, bounce
//! light on, cache settled (32 quiet frames). Ours: 4×4 sub-pixel offsets,
//! each settled, textures for a quarter pixel. Cycles: 4,096 samples, one
//! bounce, the Principled BSDF's IOR, Specular IOR Level (half the factor),
//! Specular Tint, Coat Weight and Roughness (Coat IOR 1.5).
//!
//! The bars were set with development numbers known (DESIGN §3.4: 0.58×
//! over the changed pixels, 1.18× the frame, at 640×360).
//!
//! A view that does not settle within 20,000 frames is reported and the
//! gate goes on.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, five rounds of 30 warm
//!    frames each (each waited for), the scene without the lobes then with
//!    them; the median over rounds of the ratio of their medians at most
//!    1.25×.
//! 2. **Accuracy,** every channel of both images clamped at 4× the
//!    reference's mean (the coat's sun highlights cover a few pixels at the
//!    sun's radiance): relative RMSE from Cycles at most 6%, and over the
//!    pixels the lobes change (against the scene without them, by more
//!    than 1% of the image's mean) at most 0.65× the error of ignoring
//!    them.
//!
//! Reported: the unclamped errors; the means over the changed pixels;
//! pixels above the clamp.
//!
//! Development mode (`M16_DEV=1`): 640×360, Cycles at 256 samples.

fn main() {
    todo!("M16 gate 4: stub committed before the run")
}
