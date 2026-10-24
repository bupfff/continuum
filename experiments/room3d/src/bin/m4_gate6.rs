//! M4 gate 6: hardware ray queries for camera rays.
//!
//! Flying above the boulder field was unplayable: from 150 m up, a 1080p frame
//! took about 500 ms, nearly all of it camera rays through the compute
//! hierarchy (gates 3–5 could not lower that floor). The M4's ray-tracing
//! hardware is reachable through wgpu's (experimental) ray queries. Camera rays
//! now go through a hardware visibility pass where supported: one bottom-level
//! structure per geometry, one top-level instance per object (with its
//! transform and index), writing the same visibility buffer as the compute
//! pass. Shading (mask cones, caches) is unchanged and stays in compute; the
//! compute hierarchy remains the fallback.
//!
//! Scene: the 1,000,000-boulder field, per-pixel prefiltered mask cones, best
//! of 5. Viewpoints: standing; 10, 40 and 150 m up looking 0°, 15°, 30° and
//! 45° down.
//!
//! Pass criterion, fixed before the first run:
//! 1. Agreement: at 1920×1080 from the 150 m viewpoint, the hardware render is
//!    within 0.5% relative RMSE of the compute-ray render, and at least 99.5%
//!    of pixels are within 1e-3 of peak (intersection arithmetic differs).
//! 2. Speed: at 1920×1080 from 150 m up, the frame is at least 4× faster than
//!    with compute rays.
//! 3. Playable: at 1280×720, every viewpoint renders in at most 16.7 ms.

fn main() {}
