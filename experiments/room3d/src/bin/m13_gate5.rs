//! M13 gate 5: the coverage pass with exact culling, the bins' rays only
//! near large objects, one walk per tile and lighter map reads (coverage
//! only).
//!
//! M13 gate 4 passed accuracy (0.218× one ray's coverage error standing,
//! 0.132× from 150 m) but took 122–135 ms against a 50 ms bar. Profiled at
//! full size after it (`docs/results/m13-far-field-profile.md`): from 150 m
//! (104 ms with the packet walk) the bins' rays take about 51 ms, the maps'
//! reads about 24 and a root-to-instance walk in every pixel about 27;
//! standing (122 ms) the sky's pixels walked boxes straddling the eye's
//! plane for nothing, and the horizon walks thousands of nodes. The design
//! (DESIGN §3.4, "The far field for sub-pixel coverage") splits the work in
//! five parts; this gate takes A, B, C and E, leaving the horizon's far
//! field (D, coverage pages for hierarchy nodes) to gate 6.
//!
//! Method (gate 4's corner rays first):
//! - A. A box is also rejected when the pixel's or tile's frustum lies
//!   wholly on the eye's side of one of its faces (exact).
//! - B. A pixel traces its 16 bin rays only if one of the 16 corners of its
//!   3×3 block of pixels hits a large object (more than 8 pixels across, or
//!   without a coverage map) and its own four corners do not all hit the
//!   same one.
//! - C. Each 8×8 tile walks the object hierarchy once against its frustum,
//!   nearest first, listing its entries (instances read from their maps,
//!   object nodes splatted as in gate 4) in workgroup memory in chunks;
//!   after each chunk every pixel adds the entries over its open bins, and
//!   the walk ends when every pixel's bins are full or the hierarchy is
//!   exhausted.
//! - E. The instances' maps as 8-bit texels in a texture array (a layer per
//!   direction), one hardware-filtered sample per bin and direction.
//!
//! Scene, views and reference are gate 4's: the million-boulder field
//! without its ground (in the reference and the test alike), standing at
//! 1.7 m and 150 m up looking 45° down, 1280×720; per pixel, the share of
//! 16×16 rays through its square that hit a boulder. One ray: the central
//! ray alone.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.25× one ray's standing and 0.15× from 150 m (gate 4: 0.218×
//!    and 0.132×; a stated loss of about 15% for B and E).
//! 2. **Speed:** at most 25 ms a frame from 150 m and 50 ms standing at
//!    1280×720 (median of 10 frames each waited for), timed first. The
//!    standing bar leaves the horizon's far field to gate 6; M13's goal
//!    stays a 10 ms coverage pass.
//!
//! Reported: the share of pixels tracing bin rays and resolved by their
//! corners; entries per tile and per pixel; each part's time in a
//! development profile after the gate.
//!
//! `cargo run --release -p room3d --bin m13_gate5`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the passes alone),
//! whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 5: written before the implementation");
}
