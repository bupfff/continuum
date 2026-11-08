//! M10 gate 23: emissive faces as area lights.
//!
//! Emissive geometry lights its surroundings through bounce light (M10
//! gate 12), whose 3×3 samples a page cannot resolve an emitter's sharp
//! shadows or its light's fall-off near it: 7.46% from Cycles against a 5%
//! bound. Rectangular lights through mask cones are exact.
//!
//! Method (`InstancedScene::emitters_to_lights`): each geometry with
//! uniform emission (no emissive texture) whose triangles all pair into
//! planar rectangles, at most a given number, becomes one one-sided
//! rectangular light per rectangle and instance, facing out of the
//! rectangle's front, at the emission's radiance, 1 mm in front of it. The
//! geometry stays (it still occludes, from behind and around) with its
//! emission set to zero, so its light is not counted twice: the lights show
//! it to the camera, light the scene directly, and bounce (any number of
//! lights, M10 gate 22).
//!
//! Scene: M10 gate 12's (M6's textured scene at night with a glowing box
//! panel, 1.2 × 0.08 × 0.9 m, emission (3, 1.86, 0.9): six faces, so six
//! lights besides the two night lights), its camera, 1280×720, bounce light
//! on. The comparison is gate 12's method: the panel's emission through
//! bounce light. Cycles renders the original scene (the emissive box), as
//! in gate 12.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) with emission through
//!    bounce and then with the faces as lights; the median over rounds of
//!    the ratio at most 1.5× (six lights more to shade).
//! 2. **Accuracy over surfaces:** the settled cached image, supersampled
//!    4×4, within 5% relative RMSE of instanced Cycles (one bounce, 4,096
//!    samples per pixel) and at most 0.5× the error of emission through
//!    bounce, both over the pixels not set apart: where Cycles exceeds 1% of
//!    the brightest light's radiance in a channel (the night lights' disks
//!    and the panel's faces), and their eight neighbours (gate 12's rule).
//!
//! `cargo run --release -p room3d --bin m10_gate23`
//!
//! `M10_DEV=1` saves the settled image alone (no Cycles): a development
//! check, whose numbers are not the gate's.

fn main() {
    unimplemented!("M10 gate 23: written before the implementation");
}
