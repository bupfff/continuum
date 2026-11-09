//! M13 gate 3: sub-pixel geometry through directional coverage maps
//! (coverage only).
//!
//! M13 gate 2 failed: a scalar test per node was accurate off the horizon
//! (0.26–0.29× one ray's error) but lost correlation within a pixel at the
//! horizon (one row: 0.632 covered, ours 1.0), and took 257–281 ms a frame.
//! Development after it found the object walk alone takes 10.7–14.3 ms;
//! walking each small instance's triangle hierarchy per pixel was the cost
//! (`m13-gate2.md`).
//!
//! Method (profile stage 15, entry point `m13_coverage_maps`):
//! - **Sub-pixel bins:** each pixel keeps 4×4 bins of coverage; contributions
//!   combine per bin as independent occluders, and the pixel's coverage is
//!   the bins' mean.
//! - **The object walk** as gate 2's (screen-space boxes, culled by the
//!   pixel frustum, nearer child first); an object node within a pixel adds
//!   its prefiltered opacity spread over its projected box, per bin.
//! - **Small instances read a map:** each geometry has a directional
//!   coverage map, built once on the CPU: its opacity seen from 256
//!   directions (an octahedral grid of 16×16), each over its bounding
//!   sphere at 16×16 texels (4×4 samples a texel) with mip levels down to
//!   one texel. An instance at most 8 pixels across reads, per bin, the
//!   maps of the four directions nearest the view (blended bilinearly) at
//!   the level matching the bin's footprint.
//! - **Larger instances** walk their triangle hierarchy as in gate 2, per
//!   bin: nodes within a pixel splat by their projected box, triangles
//!   cover the bins whose centres they contain (front faces).
//!
//! Scene, views and reference are gate 2's: the million-boulder field
//! without its ground (in the reference and the test alike), standing at
//! 1.7 m and 150 m up, 1280×720; per pixel, the share of 16×16 rays through
//! its square that hit a boulder. One ray: the central ray alone.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.5× one ray's.
//! 2. **Speed:** the coverage pass at most 10 ms a frame at 1280×720 (median
//!    of 10 frames each waited for).
//!
//! Reported: the maps' build time and size; object and triangle nodes per
//! pixel.
//!
//! `cargo run --release -p room3d --bin m13_gate3`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the pass alone),
//! whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 3: written before the implementation");
}
