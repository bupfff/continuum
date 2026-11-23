//! M22 gate 10: coarse texels average their area, measured on what they
//! hold (basics item 1; gate 9's candidate, gate 9 parked unrun:
//! `docs/results/m22-gate9.md`).
//!
//! A texel stores its light at one point, its centre placed on the surface;
//! where it covers much of a curved surface that point stands for it badly
//! (each of the field's boulders is one chart: its top texel lies on its
//! side facing partly down, and a boulder shown at that mip is unlit, the
//! viewer's dark blocks, 2026-11-22). Candidate (`CacheOptions::
//! area_texels`): texels in the top three mips of their chart take each
//! static light's ratio from hardware rays at 16 points over their area (4
//! a point, weighted by each point's unshadowed light) and their sky from
//! the sky's 192 rays spread over the same points.
//!
//! Gate 9 compared against Cycles at four bounces, where our one bounce
//! (which never settles on the field) leaves the image 3–12% short; this
//! gate compares what texels hold: direct light and the sky.
//!
//! Pass criteria, fixed before the run:
//! 1. **From above** (`altitude_dev`'s view: the scorecard's field 40 m
//!    higher, 15° down), bounce light off here and in Cycles: at most 0.85×
//!    the current's error from Cycles, and the mean's shortfall from Cycles'
//!    at most 0.8× the current's.
//! 2. **Elsewhere:** the scorecard's seven scenes against Cycles (as the
//!    scorecard measures, bounce on) each within 0.2 points of the current
//!    (Sponza 0.5), or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair`; run first): in
//!    every scene the settled frame's median at most 1.05× the current's,
//!    the moving frames' at most 1.10×, and every scene that settles now
//!    settling, within 1.25× its time.
//!
//! **Disclosed (before the run):** developed on the view from above and
//! the field's start view (`viewer_repro`). From above, without bounce:
//! current 13.83% (mean 3.8% short), candidate 10.42% (2.9% short); with
//! bounce against four-bounce Cycles 14.23% and 11.32%. Averaging the top
//! five mips instead of three: 10.82% with bounce, the mean unchanged.
//!
//! `M22_REHEARSE=1`: the view from above, Sponza and the field only, every
//! criterion measured the same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate10`

fn main() {
    eprintln!("M22 gate 10: stub (criteria above); not implemented yet.");
}
