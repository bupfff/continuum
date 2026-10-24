//! M4 gate 9: the lighting cache at speed.
//!
//! M4 gate 8's cache cut warm shading to 2–3 ms, but frames stayed at 20–43
//! ms. Its diagnostics found three costs outside the idea itself:
//! - wgpu-core spent 15–17 ms of CPU per `queue.submit` walking one TLAS
//!   dependency per instance (a million). A patched copy keeps one per
//!   distinct BLAS (`third_party/`). Measured before this gate was written:
//!   hardware visibility at 720p then takes 2.2–3.2 ms.
//! - Each frame waited for the GPU, then planned the next bake on the CPU
//!   (up to 30 ms in busy frames), so CPU and GPU never overlapped.
//! - Baking 20,000 cones took 12–23 ms of GPU time.
//!
//! Changes (all engineering; the cache's design is unchanged):
//! - the wgpu-core patch;
//! - pipelined requests: frame N plans its bake from frame N−2's requests,
//!   read back asynchronously, so the CPU plans while the GPU renders; slot-use
//!   readbacks for eviction are asynchronous too;
//! - a faster page manager: a cheap hash for its maps, and walks to coarser
//!   pages that stop at a page another request already listed;
//! - a bake budget of 8,000 cones per frame (gate 8 measured 2,500–40,000
//!   cones at 3.5–26.5 ms; fallback in gate 8's flight was 1.8% at 20,000).
//!
//! Scene: gate 8's (1,000,000 boulders, 1280×720, prefiltered cones, base
//! texel 5 mm, 2 px per texel, 4×4 request tiles, pool of 2^19 pages), the
//! faster of compute or hardware camera rays (lower total of warm times).
//!
//! Pass criterion, fixed before the first run:
//! 1. Same images: warm, each viewpoint's relative RMSE against per-pixel
//!    exact cones is within 0.2 points of gate 8's (14.49, 25.87, 30.88 and
//!    38.80% with hardware rays; 14.49, 25.86, 30.88, 38.80% with compute).
//! 2. Speed: warm, each viewpoint (standing; 10, 40, 150 m up) takes at most
//!    16.7 ms per frame: wall time of 60 consecutive frames, ending with a
//!    wait for the GPU, divided by 60.
//! 3. Flight: gate 8's 600-frame flight from an empty cache. The mean frame
//!    (wall time of all 600 frames ending with a GPU wait, divided by 600) is
//!    at most 16.7 ms; the 99th percentile of per-frame wall time (one call
//!    of `frame`) is at most 33.3 ms; and on average at most 10% of pixels
//!    fall back to unshadowed light.
//!
//! Warm: as gate 8 (16 consecutive frames with no request and nothing
//! baked). Reported: frames until warm, worst frame, memory, CPU planning
//! time, and both the cached and the per-pixel exact image against a 4×4
//! supersampled exact render (the area-integrated image).

fn main() {}
