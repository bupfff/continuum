//! M18 gate 1: fog with light shafts.
//!
//! Method (DESIGN §3.4, "Fog with light shafts"): `CacheOptions::fog`. The
//! first light's visibility in the air comes from a world-space grid
//! around the eye (128 × 128 × 32 cells to the fog's top, snapped to its
//! cells), each cell traced toward it by a hardware ray and kept while the
//! eye stays in its cells; per frame, froxel columns (8 × 8 pixels, 64
//! slices) integrate the fog front to back (each slice's optical depth
//! exact, the light gathered at its middle: the first light's through its
//! visibility, the others' unshadowed, the sky's), and each pixel takes
//! what the fog lets through to its hit plus the light gathered. Surfaces
//! get each light dimmed by the fog.
//!
//! Scene: M6's textured scene with M10 gate 16's panes as thin glass
//! (`Textured::with_transmission`: double-sided for Continuum, one-sided
//! for Cycles), sun and sky, bounce light off. Views: (a) its camera in an
//! even layer (0.05 per metre, top 6 m, albedo 0.9, anisotropy 0.3); (b)
//! from beside the brick block (eye (6, 4, 1.6) toward (−3, 9.5, 0.6)) in a
//! layer falling with height (0.08 at the ground, 0.3 per metre, top 6 m,
//! albedo 0.9, anisotropy 0.3). 1280×720, cache settled. Ours: 4×4
//! sub-pixel offsets, each settled. Cycles: 4,096 samples, one bounce, a
//! box 500 m across from 1 m below the ground to the top holding Volume
//! Scatter and Volume Absorption at the fog's density with its albedo as
//! colour, no volume bounces.
//!
//! The bars were set with development numbers known (DESIGN §3.4: 0.218×
//! and 0.151× the error without fog, 0.80 ms at 1280×720).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost** (first): on view (a), fog adds at most 1.0 ms to the warm
//!    frame (medians of five interleaved rounds of 30 frames, each waited
//!    for).
//! 2. **Accuracy,** in each view: relative RMSE from Cycles at most 12%,
//!    and at most 0.35× that of the image without fog.
//!
//! Reported: the image's mean against Cycles' with and without fog; views
//! that did not settle.
//!
//! Development mode (`M18_DEV=1`): 640×360, Cycles at 256 samples.

fn main() {
    todo!("M18 gate 1: stub committed before the run")
}
