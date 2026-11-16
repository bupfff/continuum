//! M20 gate 1: depth of field.
//!
//! Method (DESIGN §3.4, "Depth of field"): `DepthOfField`, a thin lens
//! gathered each frame from the pinhole image and its depths: levels of
//! (colour, signed radius) merged toward the sharper side, highlights'
//! excess splatted exactly from lists per tile, a gather per pixel of a
//! centre tap, the pixels within 3.5 px and rings of 16 taps from coarser
//! levels in three groups (in front, its own surface, behind), every fourth
//! pixel gathered and the rest interpolated where the blur is wide and
//! smooth.
//!
//! Views (640×360, aperture radius 0.1 m, the largest blur 4% of the
//! height): (a) M6's textured scene with sun and sky, focused on the brick
//! block (9.1 m; the ground in front blurred); (b) the same focused at 5 m
//! (the distance blurred); (c) not used in development: the scene at night
//! with the glowing panel, foliage and the night lights (the point light
//! seen), focused at 3 m. Ours and the pinhole at 4×4 sub-pixel offsets,
//! each settled.
//!
//! References: (1) the lens sampled outright in our renderer: 16 × 16
//! aperture points by the concentric map, for each the eye moved to it and
//! the image plane sheared through the pixel offset so the plane in focus
//! stays put, settled, sub-pixel offsets cycling through 4 × 4; (2) Cycles'
//! thin lens (f-stop the focal length over twice the radius), 4,096
//! samples, one bounce.
//!
//! The bars were set with development numbers known (DESIGN §3.4: from the
//! sampled lens 0.129× and 0.134× the pinhole's error in views (b) and
//! (a); from Cycles 1.055× and 1.033× the sampled lens's error; +1.77 and
//! +1.71 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost:** depth of field adds at most 2.0 ms to the warm frame at
//!    1280×720 in views (a) and (b) (medians of 40 frames, five interleaved
//!    rounds; timed first).
//! 2. **Against the sampled lens:** in each view, ours within 0.2× the
//!    pinhole's error (relative RMSE).
//! 3. **Against Cycles:** in each view, ours at most 1.12× the sampled
//!    lens's error.
//!
//! Reported: each view's errors, means, highlights; the cost in view (c).
//!
//! Development mode (`M20_DEV=1`): 160×90, 2 × 2 aperture points, Cycles
//! at 16 samples, 10 frames a round: it checks the binary runs.

fn main() {
    unimplemented!("M20 gate 1: written after this stub is committed");
}
