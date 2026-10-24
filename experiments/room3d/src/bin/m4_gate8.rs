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
//!
//! Details fixed before the first run (added with the cache's first commit,
//! before any measurement):
//! - Cache settings: base texel 5 mm (texels at mip m are 5 mm × 2^m × the
//!   object's scale), mip chosen so a texel spans about 2 pixels (log2
//!   rounded), one request per 4×4-pixel tile per frame, bake budget 20,000
//!   cones per frame (all phases), pool of 2^19 pages.
//! - Warm: a viewpoint is warm once 16 consecutive frames (every pixel of every
//!   tile sampled once) report no missing page. Warm-up uses the same budget;
//!   frames until warm are counted from an empty cache.
//! - Speed statistic: the median of 30 warm frames, wall clock from the
//!   frame's start until its request readback is mapped (the fastest is also
//!   reported).
//! - Reference: per-pixel exact cones (prefilter 0) through `render_instanced`
//!   with the same ray mode as the cached render; relative RMSE = RMSE / mean
//!   of the reference over all pixels and channels.
//! - Flight: frame f of 600 at t = f / 599 has the eye 300·t m ahead of the
//!   standing eye along the view's flat direction, at height 1.7 + 148.3·t m,
//!   looking 45·t degrees down, from an empty cache. A pixel falls back when
//!   no page of any mip is resident for it.

fn main() {}
