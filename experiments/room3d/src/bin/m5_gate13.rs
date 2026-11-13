//! M5 gate 13: the sky through hardware rays: warm in a few hundred frames
//! (Landmark 2), as accurate.
//!
//! A view with a sky warms slowly: with the library's defaults (a bake
//! budget that follows GPU time, capped by the frame period), the
//! million-boulder field's view from 10 m up took 12,058 frames to warm in
//! development, because a sky sample (six cube faces of mask cones) costs
//! about 10 µs of GPU time, 17 sun texels' worth.
//!
//! Now, where camera rays use hardware ray queries, a sky sample traces the
//! 192 directions of a hemicube around the texel's normal (one ray through
//! each bin's center, the first hit ending it) and adds the projected solid
//! angle of the open bins above the horizon: about 0.7 µs, counted as one
//! sun texel (`CacheOptions::sky_rays`, `sky_ray_cost`). Movers, streamed
//! levels and alpha-blended objects keep cones.
//!
//! Pass criteria, fixed before the first run:
//! 1. **Speed in flight** (run first): three pairs of flights over the field
//!    with sun and sky (M9's flight, 600 frames from standing height to 150
//!    m, 1280×720, default options), cones then rays: the rays' mean frame
//!    at most 1.05× the cones' (over the three flights each) and at most
//!    16.7 ms. Reported: 99th percentiles, fallback.
//! 2. **Warm-up:** from an empty cache with default options, at each of M5
//!    gate 9's four viewpoints (standing, 10, 40 and 150 m up; 1280×720),
//!    the view warms (16 frames in a row requesting and baking nothing) in
//!    at most 600 frames. Reported: the cones' frames with the same options.
//! 3. **Accuracy against Cycles:** at each of the four viewpoints, the field
//!    with its sun at 1 km, sun and sky, 640×360, the cache settled at 4×4
//!    sub-pixel offsets against instanced Cycles at 1,024 samples per pixel:
//!    the rays' relative RMSE at most 1.05× the cones', and at most 3% at the
//!    standing viewpoint (where the sky's estimate, not the cache's
//!    resolution at distance, decides). Reported: sky only.
//!
//! `cargo run --release -p room3d --bin m5_gate13`

fn main() {
    println!("M5 gate 13: stub (criteria in the doc comment); not implemented yet");
}
