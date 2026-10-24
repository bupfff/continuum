//! M4 gate 3: nearest-first ray traversal (scale of amount, for camera rays).
//!
//! M4 gate 2 failed its scaling criterion (3.04× for 1000× the triangles), and
//! a diagnostic showed camera rays, not cones, set the cost. Rays walked both
//! hierarchies (objects, and each object's triangles) in a fixed child order.
//! Now, at every interior node, a ray tests both children's boxes and visits
//! the one it enters first, pushing the other only if it may still hold a
//! nearer hit.
//!
//! Scene and camera: gate 2's boulder field, per-pixel mask cones (prefiltered,
//! 1 bin) at 1920×1080.
//!
//! Pass criterion, fixed before the first run:
//! 1. Same output: with 1,000,000 boulders, every pixel is within 1e-5 of peak
//!    of the render with the fixed child order (the nearest hit and its tie rule
//!    do not depend on traversal order).
//! 2. Scale: 1,000,000 boulders take at most 2× the frame of 1,000 boulders
//!    (best of 5).
//! 3. Speed: the 1,000,000-boulder frame is at least 1.5× faster than with the
//!    fixed child order (best of 5).

fn main() {}
