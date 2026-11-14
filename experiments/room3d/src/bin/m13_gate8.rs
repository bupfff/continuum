//! M13 gate 8: the coverage pass in 10 ms (coverage only).
//!
//! After M13 gates 5–7 the settled pass takes 16.1 ms standing and 17.2
//! ms from 150 m up at full size, the pixels' pass 72–76% of it; from
//! 150 m the instances' set-up alone is 5.0 ms, computed again by every
//! subgroup that touches an instance (`docs/results/m13-10ms-profile.md`).
//! The design (DESIGN §3.4, "The 10 ms bar") moves per-instance and
//! per-page work into passes that run once a frame.
//!
//! Method, on gate 7's pass (stage 18):
//! - Instance records: one invocation per object writes, for each small
//!   instance in front of the eye and inside the view, its projected
//!   centre and radius, its maps' level, two directions' layers and
//!   weights, and the affine map from screen position to each direction's
//!   map coordinates; the walk's items read the record.
//! - Page records: one invocation per page in use decides its validity
//!   from the eye (and its refresh) and the homography from screen
//!   position to its texels; page reads use the record.
//! - The other passes trimmed where free.
//!
//! Scene, views and reference are gates 4–6's: the million-boulder field
//! without its ground, standing at 1.7 m and 150 m up looking 45° down,
//! 1280×720; per pixel the share of 16×16 rays through its square that
//! hit a boulder; one ray the central one. Pages as gate 7 (settled at 512
//! bakes a frame).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed:** settled, at most 10 ms a frame standing and from 150 m up
//!    (median of 10 frames each waited for), timed first.
//! 2. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.25× one ray's standing and 0.15× from 150 m (gate 6's
//!    bars; gate 6 measured 0.212× and 0.135×).
//!
//! Reported: the settled frame 20 m up looking 10° down (gate 7's flight
//! view); each pass's time alone in a development profile after the gate.
//!
//! `cargo run --release -p room3d --bin m13_gate8`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders; `M13_DEV=1M` at
//! full size), whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 8: written before the implementation");
}
