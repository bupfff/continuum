//! M21 gate 1: motion blur.
//!
//! Method (DESIGN §3.4, "Motion blur"): `MotionBlur`, a shutter centred on
//! the frame: each pixel's segment from its hit point's motion since the
//! previous camera and movers' transforms; a gather along the tile's
//! fastest motion and the pixel's own, a tap covering the pixel when the
//! pixel lies on its segment, in three groups by depth.
//!
//! Views (640×360, the textured scene with sun and sky, shutter 0.5, the
//! longest smear 10% of the height): (a) the camera trucking sideways 0.4 m
//! a frame; (b) turning 2° a frame; (c) six boulders crossing at 0.5 m a
//! frame, the camera still; (d), not used in development: with foliage,
//! dollying forward 0.5 m a frame while turning 1°. Ours and the still
//! image at 4×4 sub-pixel offsets, each settled.
//!
//! References: (1) the shutter sampled outright in our renderer: 64
//! instants evenly across it, the camera and movers placed at each,
//! settled, sub-pixel offsets cycling through 4 × 4; (2) Cycles' motion
//! blur (the camera and moving instances keyed before, now and as far
//! again, linearly), 4,096 samples, one bounce.
//!
//! The bars were set with development numbers known (DESIGN §3.4: from the
//! sampled shutter 0.072×, 0.189× and 0.271× the still image's error; from
//! Cycles 0.998×, 1.029× and 1.025× the sampled shutter's error; means
//! within 0.06%; +1.44, +1.30 and +0.54 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost:** motion blur adds at most 1.7 ms to the warm frame at
//!    1280×720 in views (a) and (b), and at most 0.8 ms in view (c)
//!    (medians of 40 frames, five interleaved rounds; timed first).
//! 2. **Against the sampled shutter:** in each view, ours within 0.35× the
//!    still image's error (relative RMSE).
//! 3. **Against Cycles:** in each view, ours at most 1.12× the sampled
//!    shutter's error.
//! 4. **Light:** in each view, the mean of ours within 1% of the sampled
//!    shutter's.
//!
//! Reported: each view's errors and means against Cycles; the cost in view
//! (d).
//!
//! Development mode (`M21_DEV=1`): 160×90, 4 instants, Cycles at 16
//! samples, 10 frames a round: it checks the binary runs.

fn main() {
    unimplemented!("M21 gate 1: written after this stub is committed");
}
