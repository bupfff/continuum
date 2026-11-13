//! M4 gate 13: the error while flying, reported each frame (Landmark 2).
//!
//! M4 gate 12 measured the error while flying (20.8% from settled, relative
//! RMSE, on M9's flight at a fixed budget) against renderers settled at the
//! same cameras; nothing in the renderer reports it. The design (DESIGN
//! §3.3, "The error while flying, reported"): each frame `N` probe pixels,
//! on a low-discrepancy pattern that moves each frame, are shaded as shown
//! and again with each light's share and the sky's computed at the pixel's
//! own surface point (the bake's light cone and sky rays) instead of read
//! from pages. Over the probes, `sqrt(3 · mean d) / mean s` estimates the
//! image's relative RMSE (`d` the squared difference summed over channels,
//! `s` the exact value summed over channels), and a one-sided 95% bound
//! adds 1.645 standard errors of `d`'s mean. `CacheStats` carries both.
//!
//! Scene and flight are gate 12's: M9's flight (600 frames from standing
//! height to 150 m, pitching down 45°) over the million-boulder field with
//! sun and sky (sky through rays), 1280×720, 512 probes.
//!
//! Pass criteria, fixed before the run:
//! 1. **Cost** (run first): three pairs of the flight with default options,
//!    probes off then on, interleaved: the mean frame with probes at most
//!    1.05× without (over the three flights each).
//! 2. **Calibration:** the flight at gate 12's fixed budget (11,000 units a
//!    frame); every 60th frame, the reported bound at least the error
//!    measured against a renderer settled at that camera (gate 12's method)
//!    in at least 9 of the 10 checks.
//! 3. **Tightness:** the reported estimate's mean over the 10 checks within
//!    0.8–1.3× the measured mean.
//!
//! Reported: each check's estimate, bound and measured error; the estimate
//! at each settled reference (the cache's own floor against exact lighting);
//! the probe pass's GPU time.
//!
//! `cargo run --release -p room3d --bin m4_gate13`
//!
//! `M4_DEV=1`: a development check (100,000 boulders, a 200-frame flight);
//! not the gate's numbers.

fn main() {
    unimplemented!("M4 gate 13: written before the implementation");
}
