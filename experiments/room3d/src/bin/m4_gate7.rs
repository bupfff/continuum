//! M4 gate 7: cones that pay only for what they see.
//!
//! M4 gate 6's corrected diagnostic found the field's frame dominated by
//! per-pixel sun cones, at a cost nearly independent of what the cone contains:
//! every cone summed 64 bin weights (4 edge integrals each, with `atan2`) even
//! when nothing covered it. Now a cone's visible light is the whole window's
//! weight (one 4-edge integral) minus the weights of the bins with coverage,
//! the same quantity rearranged, so an open cone costs 4 integrals, not 256.
//! And a node is projected for splatting only after a cheap angular-size test
//! says it could be small enough.
//!
//! Pass criterion, fixed before the first run:
//! 1. Same output: the showcase's cached render and per-pixel render (1920×1080)
//!    and the 1,000,000-boulder field (1280×720, from 150 m up) are within 1e-4
//!    of peak of the pre-change kernel at every pixel (the sum's rounding
//!    changes).
//! 2. Playable: the 1,000,000-boulder field at 1280×720 renders in at most 33.3
//!    ms (30 fps) from each viewpoint of gate 6 (standing; 10, 40, 150 m up),
//!    with prefiltered cones and the faster of compute or hardware rays (best
//!    of 5).
//!
//! Also reported: whether each viewpoint reaches 16.7 ms, and the bake times.

fn main() {}
