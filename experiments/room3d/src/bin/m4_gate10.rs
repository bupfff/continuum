//! M4 gate 10: smooth flight through the lighting cache.
//!
//! M4 gate 9 rendered warm views of the million-boulder field in 2.9–4.7 ms at
//! 720p, but its flight was bound by the CPU page manager: each frame it
//! re-derived a backlog of about 50,000 wanted pages by walking from each of
//! up to 32,000 request tiles to coarser pages, and spikes reached 34–92 ms.
//!
//! Changes (the cache's pages, mips, baking and shading are unchanged):
//! - Progressive requests from the shader, which already walks from the wanted
//!   mip to the finest resident one: a sample pixel requests the page one mip
//!   finer than the one it found, or, with none resident, the coarsest page.
//!   Priority: pixels with no page first, then the largest gap between found
//!   and wanted mip. The CPU no longer walks.
//! - Requests deduplicated on the GPU: a per-frame set of key fingerprints
//!   (atomic compare-exchange), and a compacted list, so the CPU sees each
//!   page once per frame.
//!
//! Scene and settings: gate 9's (1,000,000 boulders, 1280×720, prefiltered
//! cones, base texel 5 mm, 2 px per texel, 4×4 request tiles, budget 8,000
//! cones per frame, pool of 2^19 pages), hardware or compute camera rays,
//! whichever had the lower total of warm frame times.
//!
//! Pass criterion, fixed before the first run:
//! 1. Same images: warm, each viewpoint's relative RMSE against per-pixel
//!    exact cones is within 0.2 points of gate 8's (14.49, 25.87, 30.88,
//!    38.80% with hardware rays; 14.49, 25.86, 30.88, 38.80% with compute).
//! 2. Speed: warm, each viewpoint takes at most 16.7 ms per frame (wall time
//!    of 60 frames ending with a GPU wait, divided by 60).
//! 3. Flight: gate 8's 600-frame flight from an empty cache, run three times
//!    in a row; in every run, the mean frame is at most 16.7 ms, the 99th
//!    percentile frame at most 33.3 ms, and the mean fallback at most 10%.
//!
//! Reported: frames until warm, the worst frame, CPU planning and GPU wait
//! per frame, and the fraction of pixel samples in flight that found their
//! wanted mip.

fn main() {}
