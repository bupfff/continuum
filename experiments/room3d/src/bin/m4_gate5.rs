//! M4 gate 5: traversal stacks in workgroup memory.
//!
//! M4 gate 4 showed that ray traversal itself, not register pressure from the
//! cone code, limits camera rays. Every traversal (rays and cones, through
//! object and triangle hierarchies) uses a 48-entry stack in a per-thread array
//! indexed at runtime, which Apple GPUs tend to keep in slow memory. Now each
//! invocation's stack is a slice of a workgroup array (64 invocations × 48
//! entries per workgroup).
//!
//! Pass criterion, fixed before the first run (1920×1080, best of 5, both
//! variants in the same session):
//! 1. Same output: the 1,000,000-boulder field (per-pixel prefiltered cones)
//!    and the showcase's cached render match the per-thread-stack versions
//!    within 1e-5 of peak at every pixel.
//! 2. The 1,000,000-boulder frame is at least 1.3× faster.
//!
//! Also reported: the showcase's cached render, M2 gate 3's cone-heavy
//! per-pixel render, and the showcase bake.

fn main() {}
