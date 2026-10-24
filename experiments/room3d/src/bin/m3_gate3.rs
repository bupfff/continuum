//! M3 gate 3: the showcase in real time, against Cycles.
//!
//! The showcase scene with its moving objects, direct light cached in texels
//! with exact incremental re-bakes, and one bounce of indirect light from probe
//! G-buffers, measured against an independent path tracer.
//!
//! Pass criterion, fixed before the first run:
//! 1. Exact incremental updates: on every frame of the camera path, lightmap
//!    texels equal a full re-bake and probes equal a from-scratch trace, within
//!    1e-5 of peak.
//! 2. Speed: the mean frame (movers, re-bake, gather, and the 1920×1080 render
//!    with direct and indirect light, each submitted and waited on) over the
//!    300-frame camera path takes at most 16.7 ms, warm, on the M4 Air. A
//!    sustained number (after 60 s of continuous rendering) is reported.
//! 3. Quality: the static showcase frame (movers removed) at 960×540 is within
//!    5% relative RMSE (linear RGB) of a Cycles render of the same scene:
//!    Lambertian albedos, the same one-sided rectangular emitter, direct light
//!    plus one diffuse bounce, pixel-center sampling, 4096 samples per pixel.
//!    The reference's own noise is reported.

fn main() {}
