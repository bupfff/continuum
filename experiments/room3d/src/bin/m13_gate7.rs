//! M13 gate 7: coverage pages that keep up in flight (coverage only).
//!
//! M13 gate 6 passed with view-dependent coverage pages for far hierarchy
//! nodes (settled: 16.12 ms standing with 10,224 pages). In motion, a page
//! is valid while its contents shift by at most a quarter of a bin; pages
//! baked together expire together, and bakes cost about 0.087 ms a page at
//! 20 a frame (one workgroup lane per texel, 16 rays each). After the run
//! (development): a walk from a settled start needs 1.5 bakes a frame, but
//! a flight at 10 m/s 20 m up asks for about 2,000 a frame after 15 m and
//! frames reach 41 ms (`docs/results/m13-gate6.md`).
//!
//! Method, on gate 6's pass:
//! - Bakes with one invocation per ray (8,192 a page), texels' shares
//!   counted in workgroup memory, so a frame's few pages fill the GPU.
//! - A page's bake eye is led along the camera's motion (the last frame's
//!   eye step): by half the steps after which its contents would shift a
//!   quarter of a bin, so it is valid at once (a shift of an eighth of a
//!   bin, half the bound) and stays valid about twice as long along the
//!   path. The validity test is gate 6's, from the current eye: accuracy
//!   never depends on the lead.
//!
//! Scene: the million-boulder field without its ground, 1280×720, gate 6's
//! pages (512 texels, nodes of at least 8 instances, a pool of 32,768).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Flight, speed:** 20 m up, looking 10° down along the standing
//!    view's heading, at 10 m/s for 10 s (600 frames at 60 frames a
//!    second), from a settled start: the 95th percentile frame at most 1.2×
//!    the settled frame at the start (median of 10, each waited for).
//! 2. **Bake budget:** the flight's bakes a frame (a count) cost at most
//!    2 ms a frame, measured from a cold pool at the flight's start by
//!    interleaved frames with and without them (medians of 10).
//! 3. **Flight, accuracy:** at 10 frames evenly spread after the first
//!    second, coverage within 0.3× the error of that frame's one-ray image
//!    from its own 16×16 reference.
//! 4. **Fill:** gate 6's walk (1.7 m, 1.5 m/s along the standing view for
//!    10 s) from a cold pool at the same budget: the median frame at most
//!    1.25× the settled standing frame (gate 6: 34.0 ms against 16.1).
//!
//! Reported: bakes and requests a frame, pages read, the largest shift
//! read; frame times along both paths.
//!
//! `cargo run --release -p room3d --bin m13_gate7`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders; `M13_DEV=1M` at
//! full size), whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 7: written before the implementation");
}
