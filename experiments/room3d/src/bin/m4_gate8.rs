//! M4 gate 8: a view-dependent lighting cache (real-time at scale).
//!
//! M4 gate 7 showed per-pixel cones every frame cannot run a million-object
//! field in real time. Static lighting only needs computing once per visible
//! patch of surface. The cache keeps pages of 8×8 texels (plus a one-texel
//! border) on each object's charts, at power-of-two resolutions chosen per
//! pixel (a texel about two pixels across). The shading pass reports the pages
//! it needed; a CPU page manager allocates them (coarse first) and evicts pages
//! unseen for a while; new pages are baked with mask cones within a per-frame
//! budget; pixels without their page use a coarser resident page or,
//! failing that, unshadowed light.
//!
//! Scene: the 1,000,000-boulder field at 1280×720, prefiltered cones, the
//! faster of compute or hardware camera rays.
//!
//! Pass criterion, fixed before the first run:
//! 1. Quality: warm (camera still until no page is missing), each of gate 6's
//!    four viewpoints (standing; 10, 40, 150 m up) is within 3% relative RMSE
//!    of per-pixel exact cones with the same rays.
//! 2. Speed: warm, each viewpoint renders in at most 16.7 ms (visibility,
//!    cached shading and the request readback; nothing left to bake).
//! 3. Flight: along a 600-frame scripted flight from standing height up to 150
//!    m, with a bake budget per frame, the mean frame takes at most 16.7 ms
//!    and on average at most 10% of pixels per frame fall back to unshadowed
//!    light.
//!
//! Reported: memory, the worst frame, frames until warm.

fn main() {}
