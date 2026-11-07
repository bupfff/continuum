//! M9 gate 2: a moving light through the error ledger (the paradigm's third
//! rule, on the GPU, in 3D).
//!
//! When a light moves, every cached texel's channel for it may change, so
//! exact invalidation re-bakes every page every frame. The ledger instead
//! certifies a bound on how far each page can be from a re-bake, and
//! re-bakes only pages whose bound exceeds ε.
//!
//! Method:
//! - Texels store the light's shadowed share of its unshadowed light
//!   (ratios, M5 gate 7); shading multiplies by the unshadowed light at the
//!   pixel, computed each frame for the light where it is now. Only the
//!   ratio can be stale.
//! - **A guard band.** For a light marked as moving, each texel also traces
//!   a cone through the light's window enlarged three times about its
//!   center. A page is *clear* if no texel's enlarged cone meets anything,
//!   *dark* if every texel's is fully covered. Each page keeps, per moving
//!   light, the frame it was baked (the light's pose then is kept in a
//!   ring), the smallest angle the light can shift within every texel's
//!   enlarged window (the guard), whether it is clear or dark, and the
//!   smallest unshadowed light at its texels.
//! - **The bound.** Each frame, per page read in the frame before: the
//!   light's shift δ as seen from the page (each corner's displacement over
//!   its distance to the page's box, conservatively). A clear or dark page
//!   whose δ stays within its guard cannot change: bound 0. Otherwise the
//!   window gained or lost at most the band of width δ around its edge, so
//!   the ratio moved by at most 2 (2 δ P + π δ²) / U, with P the window's
//!   perimeter in radians and U the page's smallest unshadowed light.
//! - Pages whose bound exceeds ε = 1% are re-baked for that light in the
//!   frame (whole pages, appended to the bake); what does not fit is marked
//!   invalid, so no pixel shows a ratio more than ε from a re-bake.
//!
//! Scene: the million-boulder field with M5 gate 6's three colored lights,
//! no sky, 1280×720, hardware camera rays, cache defaults, prefilter 1 bin.
//! The sun drifts at 0.25° per second (a 24-minute day, flatland gate 7's
//! "realistic" speed; 1/240° per frame at 60 frames per second), turning
//! about the vertical axis through the field's center. The other lights
//! stay. Flight: M5 gate 8's path, 600 frames, from an empty cache.
//!
//! Control, before anything else: the sun-only warm frame at the standing
//! viewpoint (static scene) at most 3.3 ms, or the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Soundness:** over the first 300 frames of the flight, after every
//!    10th frame, for every page shading may read, the largest change of a
//!    texel's sun ratio when re-baked from scratch is at most the page's
//!    certified bound plus 1e-5 (rounding). Zero violations in all 30
//!    checks.
//! 2. **Quality:** in the same checks, no texel's sun ratio is more than
//!    ε = 0.01 from its re-baked value.
//! 3. **Work:** in the timed flights, pages re-baked for the sun's motion
//!    average at most 25% of the pages read in the frame before (exact
//!    invalidation re-bakes all of them).
//! 4. **Speed** (measured first, cool): three flights with the sun drifting,
//!    alternating with three with it still. Each drifting flight has a mean
//!    frame time at most 16.7 ms and at most 1.25× the still flights' mean,
//!    a 99th percentile at most 33.3 ms, and fallback at most 10%.
//!
//! Reported, not gated: the share of pages clear, dark and mixed; bounds'
//! distribution against actual changes; the same with the lamp moving
//! (0.5 m per second in a circle) instead of the sun.
//!
//! `cargo run --release -p room3d --bin m9_gate2`

fn main() {
    unimplemented!("M9 gate 2: not yet implemented");
}
