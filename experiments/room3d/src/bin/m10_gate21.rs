//! M10 gate 21: bounce light with more than three lights.
//!
//! Bounce light keeps a channel per light so that re-coloring is free, and
//! it had room for three lights (with the sky and emission): scenes with
//! more lights could not have bounce light at all.
//!
//! Method: lights past the third add their bounce into the emission
//! channel, which holds absolute radiance, at their radiance when baked. The
//! first three keep their own channels and re-color free. Re-coloring a
//! later light clears every page's bounce samples, which shading asks for
//! again and the cache re-bakes over the following frames (pages show no
//! bounce light until then, never the old color).
//!
//! Scene (`Textured::with_lamps`): M6's textured scene at night (M5 gate
//! 8's point and spot lights, the sky at 2%), with three more point lights:
//! a dim blue one (the third light) and, as the fourth and fifth, the
//! brightest two: warm before the brick block and green among the
//! boulders, each 1.5 m up. 1280×720, bounce light on. The comparison is
//! the same scene with bounce light off.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) with bounce off and then
//!    on; the median over rounds of the ratio at most 1.15× (M10 gate 1
//!    measured 1.07×).
//! 2. **Accuracy over surfaces:** the settled cached image, supersampled
//!    4×4, within 5% relative RMSE of instanced Cycles (one bounce, 4,096
//!    samples per pixel) and at most 0.5× the error with bounce off, both
//!    over the pixels not set apart: where Cycles exceeds 1% of the
//!    brightest light's radiance in a channel (the lamps' disks), and their
//!    eight neighbours (M10 gate 12's rule).
//! 3. **Re-coloring a later light:** the fourth light re-colored (warm to
//!    cool, the same power), then settled: within 1% relative RMSE over
//!    surfaces of a fresh renderer built with the new color and settled
//!    (one sample per pixel each).
//!
//! `cargo run --release -p room3d --bin m10_gate21`
//!
//! `M10_DEV=1` saves the settled image alone (no Cycles): a development
//! check, whose numbers are not the gate's.

fn main() {
    unimplemented!("M10 gate 21: written before the implementation");
}
