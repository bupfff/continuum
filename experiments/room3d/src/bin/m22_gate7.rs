//! M22 gate 7: gate 6's candidate (two exact bounces with the orders past
//! them estimated, `CacheOptions::bounces` 2 and `bounce_rest`) by default,
//! its speed measured side by side.
//!
//! Gate 6 failed on speed alone: settled frames 1.184× on Sponza and 1.225×
//! on the field, each measured after the one-bounce renderer, on a warmer
//! machine. After gate 6 (development, `gltf_dev`): Sponza settled with two
//! bounces bakes nothing (no request, page or bounce sample in 1,500
//! frames), yet its frames swing 22.4–27 ms; with one bounce, later, 25–31
//! ms: the machine's heat, not bounce work. The field never settles with
//! either, so its frames are bounded by the time budget's 14 ms target.
//!
//! Measured as the basics scorecard measures, with `room3d::scorecard::
//! speed_pair`: per scene, both renderers alive, their frames alternated
//! (warm-up and settling count each one's own frames' time; settled frames
//! in six alternated blocks of ten; the motion frame by frame), warm-up
//! capped at 120 s.
//!
//! Pass criteria, fixed before the run:
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
//! **Disclosed (before the run):** gate 6's numbers (accuracy is measured
//! at a fixed bake budget, so 3 and 4 repeat it: room 7.68%, movers 5.99%,
//! field 1.32%, helmet 4.46%, Fox 1.21%, BrainStem 3.11%, Sponza 31.32%
//! against 29.94%); the side-by-side measure was written after gate 6
//! failed and is first run in this gate's rehearsal.
//!
//! `M22_REHEARSE=1`: the room and Sponza only, every criterion measured the
//! same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate7`

fn main() {
    println!("M22 gate 7: stub (criteria in the doc comment).");
}
