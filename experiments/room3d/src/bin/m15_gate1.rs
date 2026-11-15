//! M15 gate 1: anti-aliasing on the default path, in motion.
//!
//! Method (DESIGN §3.4, "Anti-aliasing on the default path"):
//! `CacheOptions::anti_aliasing` with hardware camera rays. Edge pixels (a
//! neighbour, left, right, up or down, sees another object, light or the
//! sky, or a point more than half a pixel footprint off the pixel's own
//! triangle's plane) trace the outer four samples of a fixed pattern of
//! eight, then the inner four where those did not all see the pixel's own
//! surface, within a ray budget of half a ray a pixel (the default). A
//! sample on the pixel's own surface takes its colour; any other is shaded
//! from the cache.
//!
//! Views: M6's textured scene with foliage (`Textured::new().with_foliage()`,
//! its camera, sun and sky), and the million-boulder field from standing
//! height (its ground camera, sun only). 1280×720, release build, cache
//! settled (32 quiet frames).
//!
//! The reference: the cached image at 16×16 sub-pixel offsets, each
//! settled, textures read for 1/16 of a pixel (as M7's gates). Geometric
//! edge pixels: those whose 256 reference samples see more than one object
//! or light, or one object at distances more than 5% apart. Errors are
//! relative RMSE (over the image's mean) over the pixels named.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Cost** (timed first): anti-aliasing adds at most 3.3 ms (a fifth of
//!    a 60 fps frame) to one ray's warm frame: medians of five interleaved
//!    rounds of 20 frames each, each frame waited for.
//! 2. **Edges:** the anti-aliased image's error over geometric edge pixels
//!    at most 0.5× one ray's.
//! 3. **Elsewhere:** its error over the other pixels at most one ray's.
//! 4. **Motion:** the camera turns a quarter pixel a frame for six frames,
//!    each frame settled, each with its own reference. Over the pixels that
//!    are geometric edges in any frame, the RMS of the difference between
//!    each frame-to-frame change and the reference's at most 0.5× one
//!    ray's.
//!
//! Reported, not gated: the frames' ratio; edge pixels listed, samples
//! traced and shaded; M7's silhouette rays (`edge_rays`, four quarters)
//! on the same measures; the whole image's error; each method's
//! frame-to-frame change over the reference's.
//!
//! Development mode (`M15_DEV=1`): 640×360, 100,000 boulders, 4×4
//! reference offsets, three frames of motion.

fn main() {
    todo!("M15 gate 1: stub committed before the run")
}
