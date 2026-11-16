//! M20 gate 2: depth of field, with seen lights and highlights fixed.
//!
//! Method (DESIGN §3.4, "Depth of field", with "For gate 2"): gate 1's
//! gather, and: a pixel that sees a light takes the light's depth (a bright
//! pixel beside one, its radius); highlights are pixels 4× brighter than
//! their tile's mean and 8× brighter than its background, in focus or not,
//! their excess over the background added as light (no coverage) by every
//! pixel their disc covers; a disc's soft edge no wider than the disc.
//!
//! Views (640×360, aperture radius 0.1 m, the largest blur 4% of the
//! height): gate 1's (a) M6's textured scene focused on the brick block
//! (9.1 m), (b) focused at 5 m, (c) at night with the glowing panel, foliage
//! and the night lights (the point light seen) focused at 3 m; and (d), not
//! used in development: M10 gate 2's materials (the boulders metallic,
//! roughness 0.3) under the sun and sky, focused at 3 m. Ours and the
//! pinhole at 4×4 sub-pixel offsets, each settled.
//!
//! References: (1) the lens sampled outright in our renderer: 16 × 16
//! aperture points by the concentric map, for each the eye moved to it and
//! the image plane sheared through the pixel offset so the plane in focus
//! stays put, settled, sub-pixel offsets cycling through 4 × 4; (2) Cycles'
//! thin lens (f-stop the focal length over twice the radius), 4,096
//! samples, one bounce.
//!
//! The bars were set with development numbers known (DESIGN §3.4: from the
//! sampled lens 0.134×, 0.129× and 0.025× the pinhole's error in views (a)
//! to (c); from Cycles 1.025×, 1.053× and 0.472× the sampled lens's error;
//! the image's mean within 0.5% of the sampled lens's; +1.78, +1.91 and
//! +2.49 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost:** depth of field adds at most 2.2 ms to the warm frame at
//!    1280×720 in views (a) and (b), and at most 2.8 ms in view (c)
//!    (medians of 40 frames, five interleaved rounds; timed first).
//! 2. **Against the sampled lens:** in each view, ours within 0.2× the
//!    pinhole's error (relative RMSE).
//! 3. **Against Cycles:** in each view, ours at most 1.12× the sampled
//!    lens's error.
//! 4. **Light:** in each view, the mean of ours within 1% of the sampled
//!    lens's.
//!
//! Reported: each view's errors, means against Cycles, highlights; the cost
//! in view (d).
//!
//! Development mode (`M20_DEV=1`): 160×90, 2 × 2 aperture points, Cycles
//! at 16 samples, 10 frames a round: it checks the binary runs.

fn main() {
    unimplemented!("M20 gate 2: written after this stub is committed");
}
