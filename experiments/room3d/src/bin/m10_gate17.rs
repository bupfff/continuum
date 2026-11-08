//! M10 gate 17: vertex colors.
//!
//! glTF meshes often carry a color per vertex (`COLOR_0`) multiplying the
//! base color, with or without a texture. Until now the importer dropped it.
//!
//! Method (`InstancedScene::geometry_colors`): per geometry, a linear RGB
//! color per triangle corner. It rides in the triangle buffer's unused w
//! (no new binding: the kernel is at Metal's 29 buffers per stage), as an
//! exact integer: 1 + an sRGB byte per channel (0: none, white). Shading
//! decodes the corners to linear, interpolates them barycentrically, and
//! multiplies the base color (and its texture) by the result, in the
//! cache's shading, reflections and bounce samples. Alpha cutouts carry
//! colors into the triangles they cut. glTF import reads `COLOR_0`'s RGB
//! (alpha ignored).
//!
//! Scene (`Textured::with_vertex_colors`): M6's textured scene, its camera,
//! sun and sky, 1280×720, bounce light on. The boulders get a smooth color
//! field over their corners, multiplying their lichen texture; the brick
//! block loses its texture and gets a color per corner (untextured path).
//! The comparison is the same scene with the colors ignored.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles (one bounce, 4,096 samples per
//!    pixel; the colors as a float corner attribute multiplying the base
//!    color) and at most 0.5× the error with the colors ignored.
//! 2. **Speed** (first): the warm frame at most 1.1× the colors ignored
//!    (median of 30 frames each waited for).
//!
//! Details as M10 gate 15 (settled: 32 quiet frames after the call; 4×4
//! supersampling over sub-pixel offsets, each settled; Cycles' one-pixel box
//! filter).
//!
//! `cargo run --release -p room3d --bin m10_gate17`
//!
//! `M10_DEV=1` saves the settled image alone (no Cycles): a development
//! check, whose numbers are not the gate's.

fn main() {
    unimplemented!("M10 gate 17: written before the implementation");
}
