//! M4 gate 4: a visibility buffer (splitting camera rays from shading).
//!
//! M4 gate 3 left a high floor: 72 ms at 1080p for 1,000 boulders, mostly
//! camera rays. One large shader did both the ray cast and the mask cones, and
//! the cone state (accumulator, coverage bins, clip polygons) costs registers,
//! so few rays run at once. Now a lean pass finds each pixel's nearest hit and
//! writes it to a visibility buffer; a second pass shades from it.
//!
//! Pass criterion, fixed before the first run (all at 1920×1080, best of 5,
//! split vs single pass in the same session):
//! 1. Same output: on the 1,000,000-boulder field (per-pixel prefiltered cones)
//!    and on the showcase (cached render, direct and indirect light), every
//!    pixel of the split render is within 1e-5 of peak of the single pass.
//! 2. The 1,000,000-boulder frame is at least 1.3× faster split.
//! 3. The showcase's cached render is no more than 1.05× slower split.

fn main() {}
