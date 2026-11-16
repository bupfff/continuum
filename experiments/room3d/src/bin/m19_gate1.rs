//! M19 gate 1: bloom and upscaling.
//!
//! Method (DESIGN §3.4, "Bloom and upscaling"): bloom (`Bloom`, the
//! default: strength 0.04, radius 2% of the image's height) mixes into the
//! display (1 − strength) of each pixel plus strength of the image at a
//! quarter of its size (4 × 4 means) blurred by a Gaussian across and down
//! (weights renormalized at the edges), sampled bilinearly, in linear light;
//! upscaling (`Upscale::CatmullRom`) draws an image rendered at a lower
//! resolution with Catmull and Rom's bicubic clamped to its four nearest
//! pixels' range.
//!
//! Scene: M6's textured scene with foliage, its camera, sun and sky, cache
//! settled, drawn to a 1280×720 `Rgba32Float` target (linear light; no
//! auto-exposure, exposure 1). Timings: medians of 40 warm frames, each
//! waited for, interleaved in five rounds where two are compared.
//!
//! The bars were set with development numbers known (DESIGN §3.4: Catmull-
//! Rom 0.87× nearest's error, +0.29 ms, the frame 0.50× native; bloom
//! 0.25 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Bloom's image:** the drawn target within 1e-3 (relative RMSE) of
//!    bloom computed on the CPU from `read_colors`, and the image's total
//!    light within 1% of the unbloomed image's.
//! 2. **Bloom's cost:** at most 0.3 ms added to the native frame.
//! 3. **Upscaling:** rendered at 640×360 (scale 2), Catmull-Rom's image at
//!    most 0.9× nearest's error (relative RMSE) against the scene rendered
//!    at 1280×720; its frame at most 0.35 ms more than nearest's and at
//!    most 0.55× the native frame.
//!
//! Reported: bilinear's error and frame.
//!
//! Development mode (`M19_DEV=1`): fewer frames timed (10 a round).

fn main() {
    todo!("M19 gate 1: stub committed before the run")
}
