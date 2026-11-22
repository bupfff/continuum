//! M22 gate 8: bounce light past the first bounce by default where no sky
//! lights the scene (item 5; the basics scorecard's worst row).
//!
//! Gates 6 and 7 tried two exact bounces with the orders past them estimated
//! (`bounces` 2, `bounce_rest`) everywhere: no cost a frame (gate 7, side
//! by side), the closed room 46.55% → 7.68% from Cycles, but Sponza settling
//! 1.54–2.02× slower (gate 7 failed at 2.02× against 2×). After gate 7
//! (development, rehearsal mode): reading the second level's bounce light
//! two mips coarser took the room to 16.91% and Sponza's settling stayed
//! 1.51×; the deepest level's hits one mip coarser, 10.83% and 1.56×.
//!
//! The candidate: `CacheOptions::bounces` 0 by default, meaning two (with
//! the rest estimated) where no sky or environment lights the scene and one
//! under a sky. Without a sky, every light but the lights' is bounce light
//! (the room's one bounce is 41% dark), and no sky has to be baked at the
//! pages bounce rays hit. Under a sky, much of the indirect light comes
//! from it directly (the field: 1.68% → 1.31% with two bounces), and the
//! extra level is what slowed Sponza. `bounce_rest` now acts only below
//! the first of several levels.
//!
//! Measured as gate 7 (`room3d::scorecard::speed_pair`, side by side,
//! warm-up capped at 120 s), against one bounce.
//!
//! Pass criteria, fixed before the run (gate 7's):
//! 1. **Speed** (run first): in every scene that settles with one bounce,
//!    the candidate's settled-frame median at most 1.10× one bounce's and
//!    its moving median at most 1.15×; in a scene that settles with
//!    neither, its frames within 1.10× or under the 14 ms target.
//! 2. **Settling:** every scene that settles with one bounce within 120 s
//!    settles with the candidate too, within 2× its time.
//! 3. **Indoors:** the room and the movers each at most 10% from Cycles
//!    (one bounce: 46.55%, 44.46%).
//! 4. **Elsewhere:** the field, the helmet, Fox and BrainStem within 0.2
//!    points of one bounce's error, Sponza within 2.0 points.
//!
//! **Disclosed (before the run):** gate 7's numbers for the room (7.68%,
//! settling 1.00×) and the movers (5.99%, 0.98×), whose configuration this
//! repeats; the sky-lit scenes now run the same configuration on both sides.
//! Also changed since gate 7 (development, both sides alike): anti-
//! aliasing's edge samples trace no mirror ray on dielectrics in the edge
//! shading pass too.
//!
//! `M22_REHEARSE=1`: the room and Sponza only, every criterion measured the
//! same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate8`

fn main() {
    println!("M22 gate 8: stub (criteria in the doc comment).");
}
