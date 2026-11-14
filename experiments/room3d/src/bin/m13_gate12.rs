//! M13 gate 12 (stub): the sub-pixel path in the renderer, lit by its cache.
//!
//! Gates 5–11 ran sub-pixel coverage and colour as a diagnostic profile
//! stage with its own lighting model (one sun, a constant ambient); gate 11
//! met colour's bar there (0.335× and 0.386× one ray's error, 13.85 and
//! 14.69 ms settled; `docs/results/m13-gate11.md`). DESIGN §3.4, "Into the
//! renderer", brings it into the renderer as an option whose colour comes
//! from the renderer's own lighting (the cache: every light's area, the
//! sky, bounce light), not a second model.
//!
//! Method (to be developed; DESIGN §3.4, parts A–D):
//! - A. `RendererOptions` gains the sub-pixel camera path for instanced
//!   scenes in place of one ray a pixel; the lighting cache bakes as now.
//! - B. Large hits (corners, bin rays, samples of pixels the corners
//!   resolved) are shaded as the renderer's shade pass shades its hit.
//! - C. Small instances (map entries): direct light per light from the
//!   maps' normals and a visibility per instance and light; sky and bounce
//!   from the cache's coarsest level for the instance (or its node's).
//! - D. Coverage pages bake colour from the same shading.
//!
//! Scene: the million-boulder field with its ground, its lights and a sky,
//! 1280×720; gate 11's views (standing at 1.7 m, 150 m up looking 45° down).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Colour:** relative RMSE of the RGB image from a reference at most
//!    0.5× one ray's, in both views. The reference: per pixel the mean of
//!    16×16 samples, each one ray shaded by the renderer's own shading with
//!    the cache settled; one ray: the centre sample, shaded the same way.
//! 2. **Speed:** settled (the cache settled, every wanted page baked), at
//!    most 16.7 ms a frame in both views (median of 10 frames each waited
//!    for), timed first.
//!
//! Reported: the frame with one ray a pixel and the cache (the renderer's
//! present path) in the same views; the 20 m view settled.
//!
//! `cargo run --release -p room3d --bin m13_gate12`

fn main() {
    println!(
        "M13 gate 12 is a stub: the method is not implemented yet (DESIGN §3.4, \"Into the renderer\")."
    );
}
