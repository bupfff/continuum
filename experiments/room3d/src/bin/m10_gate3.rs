//! M10 gate 3: bounce light and reflections without request cascades.
//!
//! M10 gates 1 and 2 left the cache restless. Pages requested only for
//! light at bounce or reflection hits were baked with bounce samples of
//! their own, whose rays requested more pages, and so on through the scene.
//! Pages seen only in reflections were evicted and requested again. With
//! materials, a view took 4,266 frames to settle against 621; after the
//! supersampling, requests resumed and failed gate 1's re-coloring check.
//!
//! The fix: requests from bounce and reflection lookups are secondary.
//! They are planned after pixels' requests and baked without bounce
//! samples, and the fill clears what a previous page in the slot left. A
//! page shown by a pixel without samples asks for them, and the CPU
//! schedules them. Pages read through reflections are marked used.
//!
//! Disclosed: the fix and a development check (`M10_DEV=1` of gates 1 and
//! 2: settled in 602 and 585 frames) came before this stub was written.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Re-coloring:** gate 1's check again, on its scene and procedure
//!    (settle, supersample 4×4, settle, re-color the sun): no page baked
//!    over the next 30 frames.
//! 2. **Settling with materials:** gate 2's scene (materials, bounce light)
//!    settles in at most 1.5× the frames of the same scene, Lambertian,
//!    without bounce light.
//! 3. **No loss of accuracy:** against the same Cycles references,
//!    gate 1's image within 5% (it was 1.79%) and gate 2's within 0.5
//!    points of its 8.94%.
//!
//! `cargo run --release -p room3d --bin m10_gate3`

fn main() {
    unimplemented!("M10 gate 3: not yet implemented");
}
