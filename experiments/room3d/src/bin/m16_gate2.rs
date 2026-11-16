//! M16 gate 2: thin transmissive glass at speed.
//!
//! Gate 1 failed on speed only (1.389× the frame with the panes opaque
//! against 1.3×; accuracy passed, 5.09% from Cycles): the extra-samples
//! pass shaded each pane pixel's own transmissive hit again to learn what
//! it passes. Now the shading pass keeps that in the colour's unused fourth
//! word (8 bits a channel) and the extra-samples pass shades only the
//! layers behind (development at 1280×720, a game running on the machine:
//! 1.097×).
//!
//! Method (DESIGN §3.4, "Transmission"): `InstancedScene::geometry_transmission`
//! (glTF's `KHR_materials_transmission` without a volume). Of the light a
//! surface's specular layer does not reflect, the factor passes straight
//! through tinted by the base colour, the rest is diffuse; the specular
//! layer keeps its full reflection. Camera rays see through it (alpha
//! blending's layered records); cones weigh its triangles by what passes
//! at their angle (Fresnel, Schlick's), untinted.
//!
//! Scene: M6's textured scene with M10 gate 16's two panes as smooth thin
//! glass (`Textured::with_transmission`: blue-tinted, transmission 0.9;
//! green, 0.7), double-sided for Continuum and one-sided for Cycles (as
//! gate 16). 1280×720, bounce light on, cache settled (32 quiet frames).
//! Ours: 4×4 sub-pixel offsets, each settled, textures for a quarter pixel.
//! Cycles: 4,096 samples, one bounce, its reference material (the opaque
//! Principled BSDF mixed by one less the factor with its specular layer
//! plus a Transparent BSDF passing the base colour times one less
//! Schlick's Fresnel reflectance).
//!
//! Criteria as gate 1's, unchanged. A view that does not settle within
//! 20,000 frames is reported and the gate goes on (in development one
//! 1280×720 run of `transmission_dev` did not settle once and did the
//! next).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, five rounds of 30 warm
//!    frames each (each waited for), the panes opaque then transmissive;
//!    the median over rounds of the ratio of their medians at most 1.3×.
//! 2. **Accuracy:** relative RMSE from Cycles at most 7%, and over the
//!    pixels transmission changes (against the panes opaque, by more than
//!    1% of the image's mean) at most 0.5× the error of the panes drawn as
//!    alpha blending with opacity one less the factor.
//!
//! Reported: the panes opaque against Cycles; the means over the changed
//! pixels.
//!
//! Development mode (`M16_DEV=1`): 640×360, Cycles at 256 samples.
//!
//! Implementation notes: gate 1's binary with the settling report.

fn main() {
    todo!("M16 gate 2: stub committed before the run")
}
