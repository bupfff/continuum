//! M10 gate 19: alpha blending's speed.
//!
//! M10 gate 16 passed accuracy (4.71% from Cycles) and failed speed: 1.33×
//! the frame with the panes opaque, against 1.3×. Development runs before
//! this gate (`blend_dev`, disclosed; each camera pass submitted alone):
//! blending adds two full-screen passes, `hw_edges` (0.96 ms alone) and
//! `vc_shade_edges` (0.90 ms alone). With their work removed they still
//! cost about 0.25 ms each in the warm frame (3.77 against 3.24 ms):
//! launching a workgroup per 8×8 pixels of these large kernels, most of
//! which have nothing to do. The layer traces themselves cost about
//! 0.13 ms, and the list's global atomic nothing measurable (a
//! per-workgroup append did not change the frame). CPU encoding is 0.12 ms
//! against 0.09 ms.
//!
//! Method:
//! - **Layers in the visibility pass.** `hw_visibility_blend` traces on
//!   through blended surfaces where the camera ray's hit is blended, and
//!   lists the pixel, as `hw_edges` did. With blending and no silhouette
//!   rays, the `hw_edges` pass is gone; with both, it lists silhouettes
//!   only.
//! - **Shading only the list.** A one-thread pass sizes `vc_shade_edges`'s
//!   dispatch from the list's count (dispatched indirectly), instead of a
//!   workgroup per 8×8 pixels.
//! Images are unchanged in principle: the same samples and weights.
//!
//! Scene and comparison are gate 16's (`Textured::with_glass(true)`; the
//! panes opaque; 1280×720, bounce light on).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) of the panes opaque and
//!    then blended; the median over rounds of the blended-to-opaque ratio
//!    at most 1.3× (gate 16's bound, gate 18's method).
//! 2. **Accuracy:** gate 16's criterion 1, unchanged: within 5% of Cycles,
//!    and over the pixels blending changes, at most 0.5× the error of the
//!    panes opaque.
//!
//! `cargo run --release -p room3d --bin m10_gate19`

fn main() {
    unimplemented!("M10 gate 19: written before the implementation");
}
