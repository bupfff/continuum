//! M4 gate 11: baking to a frame-time target.
//!
//! M4 gate 10 took the page manager's walk off the CPU (planning 1.9 ms per
//! frame), which left its flight bound by two things:
//! - GPU work per frame. Rendering plus a fixed 8,000 cones took 11.5–18.6 ms
//!   per 100-frame stretch on a hot machine. The cost per cone varies with
//!   what cones see and with the machine's heat, so a fixed count cannot
//!   hold a frame time.
//! - Eviction rounds. Every 32 frames, up to 32,768 pages were freed at once
//!   (12–26 ms of CPU), which made 12 of the 17 frames over 33 ms in flight 1.
//!
//! Changes (pages, mips, requests, baking and shading are unchanged):
//! - The bake budget follows the frame time: the frame budget of the quality
//!   contract (DESIGN §3.3). Each frame measures its period P (between
//!   successive frames) and its wait for the GPU.
//!   - If the CPU waited more than 1 ms (GPU-bound) and P exceeds the target
//!     T = 14 ms, the budget scales by T / P.
//!   - If P is under T, it grows by 10%.
//!   - Otherwise it holds.
//!   - Bounds: 1,000 to 20,000 cones; start: 8,000.
//! - Eviction is spread over frames: when slot use arrives (every 32 frames),
//!   pages unused for 64 frames are queued, least recently used first, and
//!   at most 2,048 are freed per frame.
//!
//! Scene and settings: gate 10's (1,000,000 boulders, 1280×720, prefiltered
//! cones, base texel 5 mm, 2 px per texel, 4×4 request tiles, pool of 2^19
//! pages), with the faster of compute or hardware camera rays.
//!
//! Pass criterion, fixed before the first run (as gate 10's):
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
//! per frame, the bake budget over the flight, and the fraction of pixel
//! samples in flight that found their wanted mip.

fn main() {}
