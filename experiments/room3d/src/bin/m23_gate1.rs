//! M23 gate 1: far bounce hits lit by rays (basics item 2, "the field's
//! bounce light settling"; DESIGN §3.4 "M23 as built").
//!
//! A bounce sample's ray reads the page of whatever it hits, so a ground
//! sample on the million-boulder field asks for the pages of thousands of
//! distant boulders; nearly every sample stays incomplete and the field
//! never settles (187,000 pages resident after 1,200 frames without the
//! frame cap; `docs/results/viewer-glance-fixes.md`).
//!
//! Candidate (`CacheOptions::far_rays`): a hit whose footprint asks for its
//! chart's top mip (the whole chart one texel) reads no page. Its direct
//! light is the unshadowed light at the hit times one hardware shadow ray
//! toward each light; its sky one ray from the hit, cosine-distributed about
//! its normal (π times the open share); albedo and emission as before. It
//! never requests a page nor counts as missing. One bounce only (`bounces`
//! one, as outdoors by default); hardware ray queries only.
//!
//! Measured exactly as the basics scorecard measures (`room3d::scorecard`),
//! the current defaults against the candidate.
//!
//! Pass criteria, fixed before the implementation:
//! 1. **The field settles:** within 120 s in the speed measurement (the
//!    current never does), its pixels without a page in motion no more
//!    than the current's plus 0.1 points.
//! 2. **Against Cycles:** the scorecard's seven scenes each within 0.2
//!    points of the current (Sponza 0.5), or better; and the field from
//!    above (`field_from_above(40, 15)`, bounce on, Cycles at four bounces)
//!    within 0.5 points, or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair`; run first): in
//!    every scene the settled frame's median at most 1.05× the current's,
//!    the moving frames' at most 1.10×, and every scene that settles now
//!    settling, within 1.25× its time.
//!
//! `M23_REHEARSE=1`: the field, Sponza and the view from above only, every
//! criterion measured the same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m23_gate1`

fn main() {
    eprintln!("M23 gate 1: stub (criteria above); not implemented yet.");
}
