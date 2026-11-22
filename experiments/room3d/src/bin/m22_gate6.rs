//! M22 gate 6: bounce light past the first bounce by default (item 5,
//! "bounce by default and quicker to settle"; the basics scorecard's worst
//! row, DESIGN §3.8).
//!
//! The cache's one bounce leaves a closed room 41% dark (46.55% from Cycles
//! at 12 bounces); M10 gate 25's exact bounces fix it (4 bounces: 7.74%) but
//! their cascade never lets Sponza settle. The candidate: **two exact
//! bounces** (`CacheOptions::bounces` 2) **with the orders past them
//! estimated** (`CacheOptions::bounce_rest`): the pages a bounce sample's
//! rays hit hold one-bounce samples of their own, each scaled by 1 / (1 −
//! ρ), ρ its closure-weighted albedo (progressive radiosity's ambient term,
//! per sample; Cohen et al. 1988). Hits whose samples hold a bounce ask for
//! no more where bounce light is under `bounce_tail` of their light.
//!
//! Measured exactly as the basics scorecard measures (`room3d::scorecard`):
//! its seven scenes, each with the defaults (one bounce) and the candidate.
//!
//! Pass criteria, fixed before the run:
//! 1. **Speed** (run first; per scene, one bounce then the candidate): the
//!    settled frame's median at most 1.10× one bounce's, and the moving
//!    frames' median at most 1.15×, in every scene.
//! 2. **Settling:** every scene that settles with one bounce within 120 s
//!    settles with the candidate too, within 2× its time.
//! 3. **Indoors:** the room and the movers each at most 10% from Cycles
//!    (clamped relative RMSE; one bounce: 46.55%, 44.46%).
//! 4. **Elsewhere:** the field, the helmet, Fox and BrainStem within 0.2
//!    points of one bounce's error, Sponza within 2.0 points.
//!
//! **Disclosed (before the run):** developed with the scorecard on five of
//! the scenes (not Fox and BrainStem), the candidate against one bounce:
//! room 7.68% (46.55%), movers 5.99% (44.46%), field 1.33% (1.68%), helmet
//! 4.46% (4.46%), Sponza 31.32% (29.94%); Sponza's settled frame 22.2 ms
//! (22.4), settled at 56.5 s (35.0 s; 33–50 s across runs), the room's 2.2
//! s (1.7). Alternatives tried: 4 exact bounces (room 7.74%, Sponza never
//! settled, 34–40 ms frames), the estimate on one bounce alone (room 20.31%,
//! Sponza 33.47%).
//!
//! `M22_REHEARSE=1`: the room and Sponza only, every criterion measured the
//! same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate6`

fn main() {
    println!("M22 gate 6: stub (criteria in the doc comment).");
}
