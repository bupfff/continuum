//! M10 gate 18: vertex colors' speed (gate 17 re-measured).
//!
//! Gate 17 passed accuracy (1.29% from Cycles, 0.02× the error with the
//! colors ignored) and failed speed: 3.49 ms against 3.05 ms (1.15×, limit
//! 1.1×). Development runs after it (`vcolor_dev`, disclosed in
//! `m10-gate17.md`, seen before this gate was written) measured 0.998–1.009×
//! in four tries, including gate 17's own order: its colored warm frames
//! met a transient. The kernel is unchanged since gate 17, so its accuracy
//! result stands; this gate measures speed alone, with a method that a
//! single transient cannot decide.
//!
//! Scene, renderers and settling are gate 17's (M6's textured scene with
//! `Textured::with_vertex_colors`, 1280×720, bounce light on; the
//! comparison is the same scene with the colors ignored).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed:** both renderers settled, then five rounds, each the median
//!    of 30 warm frames (each waited for) of the colors ignored and then of
//!    the colored scene; the median over rounds of the colored-to-ignored
//!    ratio at most 1.1×.
//!
//! `cargo run --release -p room3d --bin m10_gate18`

fn main() {
    unimplemented!("M10 gate 18: written before the run");
}
