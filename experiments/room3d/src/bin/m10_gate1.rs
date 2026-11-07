//! M10 gate 1: one bounce of light in the view-dependent cache.
//!
//! The instanced renderer lights with direct light only; one bounce ran
//! only in the lightmap path (M2 gate 6). Outdoors, the light the ground
//! and objects reflect fills shadows; without it they are too dark.
//!
//! Method:
//! - **Samples per page.** Each page holds a 3×3 grid of indirect samples
//!   (at its texels 0, 4 and 8 along each axis), read bilinearly by
//!   shading. Bounce light varies slowly; a sample costs hundreds of rays.
//! - **A hemicube of bins.** A sample casts one ray through the center of
//!   each of 192 bins of a hemicube around its normal (M2 gate 6's), each
//!   weighted by its projected solid angle.
//! - **Direct light at the hits from the cache.** Each hit's outgoing light
//!   is its albedo (textured) over π times its direct light, read from the
//!   cache's page at the hit at a mip matched to the bin's footprint (each
//!   light's ratio times its unshadowed light there, and the sky's channel).
//!   A hit with no page resident requests one and counts as missing; the
//!   page's samples re-bake once the requested pages have arrived.
//! - **Per light and the sky, in RGB,** per unit radiance, so re-coloring a
//!   light still bakes nothing.
//!
//! Scene: the textured scene of M6 (`Textured::new()`, sun and sky), its
//! camera, 1280×720, cache defaults with bounce light on.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4 (16 frames
//!    over sub-pixel offsets), within 5% relative RMSE of instanced Cycles
//!    with one diffuse bounce, and at most half the error of the same image
//!    without bounce light against that reference.
//! 2. **Speed:** the warm frame with bounce light at most 1.25× the warm
//!    frame without (median of 30 frames each waited for); the view settles
//!    in at most 3× the frames.
//! 3. **Free re-coloring:** re-coloring the sun on a settled view bakes
//!    nothing over the next 30 frames.
//!
//! Reported, not gated: the bounce's share of the image; samples per page
//! re-baked for missing hits; the error of the direct-only image against
//! direct-only Cycles, as a control.
//!
//! `cargo run --release -p room3d --bin m10_gate1`

fn main() {
    unimplemented!("M10 gate 1: not yet implemented");
}
