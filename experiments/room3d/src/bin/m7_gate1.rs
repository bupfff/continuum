//! M7 gate 1: pixel cones for primary visibility (anti-aliasing).
//!
//! Today each pixel casts one ray and shows the surface at its center. The
//! area-integrated image (what Cycles renders) averages everything inside
//! the pixel's square. One ray per pixel is 13–24% from the supersampled
//! image on the boulder field (M4 gate 8), the largest visible quality gap,
//! and distant boulders smaller than a pixel flicker as the camera moves.
//!
//! Method: each pixel traces a mask cone through its square (8×8 bins of
//! equal image-plane area), front to back through the object and triangle
//! hierarchies, splatting prefiltered coverage below a bin (as light cones
//! do). Each object it meets becomes a fragment: its coverage of the pixel,
//! a binary mask of its covered bins, and its depth. Fragments are sorted by
//! depth and composited by mask (nearer fragments hide farther ones bin by
//! bin); up to four are shaded, each at a ray through its visible bins, and
//! averaged by visible coverage. What no fragment covers shows the sky.
//!
//! Pass criterion, fixed before the first run:
//! 1. Quality: in each of three views at 1280×720 (the textured scene of M6
//!    from its camera; the million-boulder field from M4 gate 8's standing
//!    and 150 m viewpoints), the pixel-cone image's relative RMSE from the
//!    area-integrated reference is at most 0.35× the one-ray image's.
//! 2. Stability: from the field's 150 m viewpoint, moving the camera half a
//!    pixel sideways changes the pixel-cone image (relative RMS of the
//!    difference) by at most 1.5× as much as it changes the reference.
//! 3. Speed: with the lighting cache, warm, on the textured scene at
//!    1280×720, a frame with pixel cones takes at most 16.7 ms (median of 30
//!    warm frames). Measured first, on a cool machine, with the one-ray warm
//!    frame (M6 gate 1: 1.53 ms) as the control.
//!
//! Reported: the one-ray errors and changes; the field's warm frames with
//! pixel cones; fragments per pixel; coverage left to the sky.
//!
//! Details fixed before the first run:
//! - Lighting is per pixel (mask cones, no cache) in every image of
//!   criteria 1 and 2, so only primary visibility differs: the textured
//!   scene's sun and sky, the field's sun (M4 gate 8's light).
//! - Reference: the one-ray renderer at 16×16 samples per pixel (rendered at
//!   16 times the resolution), box filtered. Relative RMSE = RMSE / mean of
//!   the reference over all pixels and channels.
//! - Pixels where a sample of the reference sees a light, and their eight
//!   neighbours, are set apart in criterion 1 and 2 (as in M5 gate 6).
//! - Warm: 16 consecutive frames with no page missing; the camera still.
//! - Timing: release build, `ps` checked for stray processes first.

fn main() {
    unimplemented!("M7 gate 1 is pre-registered; the pixel cones come next");
}
