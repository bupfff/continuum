//! M10 gate 4: free re-coloring with bounce light, with a corrected settle.
//!
//! Gates 1 and 3 failed their re-coloring check (794 and 522 pages baked in
//! the 30 frames after re-coloring). The cause was the check's procedure,
//! not the renderer. Their `settle` looked at the last 32 frames already
//! read back, which, right after another settle, were that settle's quiet
//! frames, so it returned after one frame. The new view's requests arrived
//! later, inside the re-coloring window. (Each sub-pixel offset of their
//! supersampling was likewise rendered one frame after the switch.)
//!
//! Here `settle` waits for 32 quiet frames rendered after it is called.
//! Renderer as in gate 3 (secondary requests for bounce and reflection hits).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Re-coloring:** gate 1's scene and procedure (settle, supersample
//!    4×4, settle, re-color the sun), with the corrected settle: no page
//!    baked over the next 30 frames.
//!
//! Reported, not gated: gate 1's and gate 2's images against their Cycles
//! references, supersampled with the corrected settle.
//!
//! `cargo run --release -p room3d --bin m10_gate4`

fn main() {
    unimplemented!("M10 gate 4: not yet implemented");
}
