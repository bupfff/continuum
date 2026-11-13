//! M9 gate 6: a moving sun at real speed, its shadows' edges held within a
//! certified bound (Landmark 2).
//!
//! Gate 4 failed on speed (1.6× the still flights at a real sun's speed):
//! texels near shadow edges ("mixed") re-baked whenever the light moved at
//! all, because a moving light's ratio estimate wavers by a few thousandths
//! as the light cone's bins realign with the light, which no bound on the
//! light's motion covers.
//!
//! Now a moving rectangular light's cones keep their bins on a lattice fixed
//! in the frame of the page's version (the light's pose at its last whole
//! bake): the window is the lattice's 8×8 bins holding the light's
//! footprint, and each bin counts by the footprint's area in it. A bin's
//! coverage then no longer depends on where the light is, and the ratio
//! moves by at most the footprint's change: with every point of the light
//! moving at most δ (as seen from the page), at most 1.5 κ (P δ + π δ²) / Ω,
//! for the footprint's perimeter P and solid angle Ω and κ the spread of the
//! bins' weights. Mixed texels re-bake only once that bound passes ε (a
//! hashed share of it, a half to all), so at a real sun's speed about every
//! 6–12 frames instead of every frame; no snapping.
//!
//! Scene, flight and lights as gates 2 and 4 (the million-boulder field,
//! three coloured rect lights 1 km away, the warm "sun" moving); 1280×720.
//!
//! Pass criteria, fixed before the first run:
//! 1. **Speed at a real sun's speed** (1/14,400° per frame), run first: three
//!    flights alternating with three still ones, each moving flight's mean
//!    at most 16.7 ms and at most 1.25× the still flights' mean, its 99th
//!    percentile at most 33.3 ms, fallback at most 10% (gate 4's criterion
//!    2).
//! 2. **Soundness:** 30 checks over the flight, 10 each at a real sun's
//!    speed, 8× it, and gate 2's speed (60× it); each re-bakes every page the
//!    cache may show, in its own lattice: no page beyond its certified bound
//!    (with 1e-5 for rounding), none beyond ε (0.01).
//! 3. **Accuracy at rest:** after the real-sun flight, the sun held at its
//!    last pose and the view settled, the image within 1% (relative RMSE) of
//!    a renderer built with the sun fixed there (the standard light cones).
//!
//! Reported: one flight at gate 2's speed; mixed pages held within their
//! bound against re-baked, per frame.
//!
//! `cargo run --release -p room3d --bin m9_gate6`

fn main() {
    println!("M9 gate 6: stub (criteria in the doc comment); not implemented yet");
}
