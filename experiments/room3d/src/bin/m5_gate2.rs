//! M5 gate 2: sky light against Cycles, with the reference fixed.
//!
//! M5 gate 1 failed (sun and sky 5.12% from Cycles; sky alone 2.74%), but
//! its diagnosis found two faults in the reference, not the sky:
//! - Cycles integrates each pixel's area (Blender 5.2 applies at least a
//!   one-pixel box filter whatever width is asked), while our renders took one
//!   sample per pixel center, so every silhouette counted as error.
//! - Cycles loses 1.9% of a 0.5° quad sun 3 km away, but matches the analytic
//!   irradiance at 1 km and closer.
//!
//! This gate repeats gate 1 with both fixed, and nothing else changed in the
//! renderer: our renders take 4×4 samples per pixel (rendered at 4× the
//! resolution and box filtered), and the outdoor scene's sun is 1 km away
//! (`Outdoor::with_sun_at(1000.0)`, same angular size, radiance and
//! direction). The sky's radiance follows from the sun's irradiance as before.
//!
//! Scene: `room3d::outdoor` at 640×360, Lambertian, direct light only, compute
//! rays. Reference: Cycles (Blender 5.2), direct light only, world color = sky
//! radiance, one-pixel box filter, 4096 samples per pixel, two seeds (noise
//! reported as the RMSE between them / √2).
//!
//! Pass criterion, fixed before the first run:
//! 1. Sky only (sun off): exact cones, 4×4 per pixel, within 3% relative RMSE
//!    of Cycles (seed 1).
//! 2. Sun and sky: exact cones and prefiltered cones (1 bin), 4×4 per pixel,
//!    each within 3% of Cycles (seed 1).
//!
//! Reported: render times, Cycles' noise, the same comparisons at one sample
//! per pixel, and per-surface ratios (sun and sky) in surface interiors.

fn main() {}
