//! M16 gate 3: solid glass (refraction through volumes).
//!
//! Method (DESIGN §3.4, "Transmission", volumes): `InstancedScene::geometry_volumes`
//! (glTF's `KHR_materials_volume` with `KHR_materials_ior`). A camera ray
//! meeting solid glass refracts in by the hit's interpolated normal, crosses
//! it dimmed by Beer's law, reflects back at total internal reflection (up
//! to three times), refracts out and goes on; what lies behind is lit from
//! where it left. Shadows pass two surfaces at the cone's angle (Fresnel
//! twice), untinted by the attenuation.
//!
//! Scene: M6's textured scene with a clear glass sphere (radius 0.45 m,
//! index 1.5) and a glass block tinted by its attenuation (to 0.5, 0.8, 0.6
//! over 0.3 m), both smooth, transmission 1, white
//! (`Textured::with_glass_solids`). 1280×720, bounce light on, cache settled
//! (32 quiet frames). Ours: 4×4 sub-pixel offsets, each settled, textures
//! for a quarter pixel. Cycles: 4,096 samples, one diffuse bounce, the
//! Principled BSDF with its transmission weight and index, a Volume
//! Absorption for the attenuation, shadow rays meeting a Transparent BSDF
//! at each surface, no caustics.
//!
//! A view that does not settle within 20,000 frames is reported and the
//! gate goes on.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, five rounds of 30 warm
//!    frames each (each waited for), the glass opaque then solid; the
//!    median over rounds of the ratio of their medians at most 1.3×.
//! 2. **Accuracy,** every channel of both images clamped at 4× the
//!    reference's mean (the sun's reflection in the smooth sphere covers a
//!    few pixels at its full radiance, where its sub-pixel place decides
//!    the error): relative RMSE from Cycles at most 12%, and over the pixels
//!    transmission changes (against the glass opaque, by more than 1% of
//!    the image's mean) at most 0.5× the error of the same glass drawn thin
//!    (no volumes).
//!
//! Reported: the unclamped errors; the glass opaque against Cycles; the
//! means over the changed pixels; pixels above the clamp.
//!
//! Development mode (`M16_DEV=1`): 640×360, Cycles at 256 samples.

fn main() {
    todo!("M16 gate 3: stub committed before the run")
}
