//! M22 gate 9: coarse texels average their area (basics item 1, DESIGN
//! §3.8).
//!
//! A texel stores its light at one point: its centre, placed on the
//! surface. Where a texel covers much of a curved surface that point stands
//! for it badly: each of the field's boulders is one chart, so its top mip
//! is one texel, placed halfway up its side facing partly down, and a
//! boulder shown at that mip (far away, or near while its finer pages wait)
//! is unlit (the viewer's dark blocks and darker horizon, 2026-11-22). From
//! 40 m above the scorecard's field, 14.23% from Cycles, 5.6% darker.
//!
//! Candidate (`CacheOptions::area_texels`): texels in the top three mips of
//! their chart take their light from 16 points spread over the texel's
//! cell of the chart (each placed as a texel's centre is; points off the
//! surface skipped). Each light's ratio is the visible share over the
//! points that face it, weighted by each point's unshadowed light (4
//! hardware rays a point toward the light), replacing the cone's; the sky's
//! 192 rays are spread over the same points. Hardware ray queries only (the
//! compute path keeps point samples).
//!
//! Measured exactly as the basics scorecard measures (`room3d::scorecard`),
//! the current defaults against the candidate:
//!
//! Pass criteria, fixed before the implementation:
//! 1. **From above** (`altitude_dev`'s view: the scorecard's field 40 m
//!    higher, 15° down): at most 0.80× the current's error from Cycles,
//!    and the mean within 3% of Cycles'.
//! 2. **Elsewhere:** the scorecard's seven scenes against Cycles each
//!    within 0.2 points of the current (Sponza 0.5), or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair`; run first): in
//!    every scene the settled frame's median at most 1.05× the current's,
//!    the moving frames' at most 1.10×, and every scene that settles now
//!    settling, within 1.25× its time.
//!
//! `M22_REHEARSE=1`: the view from above, Sponza and the field only, every
//! criterion measured the same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate9`

fn main() {
    eprintln!("M22 gate 9: stub (criteria above); not implemented yet.");
}
