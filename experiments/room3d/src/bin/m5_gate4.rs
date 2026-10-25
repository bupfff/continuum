//! M5 gate 4: sky light through the view-dependent cache, at scale.
//!
//! M5 gate 3 matched Cycles on the million-boulder field with sky light, but
//! per-pixel sky cones cost about 100–200 µs per pixel (six cones, most of it
//! in finding coverage). This gate bakes the sky into the view-dependent
//! lighting cache instead, and makes sky cones cheaper:
//! - Cache pages hold two channels per texel: irradiance per unit sun radiance
//!   and per unit sky radiance; shading applies the radiances. Pixels with no
//!   page use unshadowed sun and an unoccluded upper-hemisphere sky.
//! - Sky cones prefilter 8 times as coarsely as the light's cones
//!   (`set_sky_prefilter(8.0)`): their bins are 11° wide. Measured before this
//!   gate on the field at 320×180: 7.7× cheaper, 0.97% from factor 1.
//! - Each face's sky light is its whole horizon-clipped window weight minus
//!   the weights of covered bins (identical to 8e-8).
//!
//! Scene: the million-boulder field (sun 3 km for criteria 2–4, as gates
//! 8–11; 1 km for criterion 1, as gate 3), sky from `Field::sky`, prefilter 1
//! bin with object-level prefiltering, cache defaults (frame target 14 ms),
//! hardware camera rays, gate 8's four viewpoints at 1280×720.
//!
//! Pass criterion, fixed before the first run:
//! 1. Accuracy at the new sky prefilter: per-pixel sky only, 4×4 per pixel,
//!    within 3% of Cycles (instanced, 1024 spp), as gate 3's criterion 1.
//! 2. The sun channel unchanged: warm cached images with no sky within 0.2
//!    points of gate 8's relative RMSE against per-pixel exact cones.
//! 3. Sky in the cache: at each viewpoint, the warm cached sky-only image is
//!    within 5% relative RMSE of the per-pixel sky-only render (1 sample per
//!    pixel, same prefiltering).
//! 4. Speed with sun and sky: warm, each viewpoint at most 16.7 ms per frame;
//!    gate 8's flight from an empty cache, three runs, each with mean frame at
//!    most 16.7 ms, 99th percentile at most 33.3 ms, fallback at most 10%.
//!
//! Reported: frames until warm (sun and sky), bake budget, samples at their
//! wanted mip, and the cost of a sky texel.

fn main() {}
