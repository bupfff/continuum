//! M10 gate 20: texture wrap modes (glTF samplers).
//!
//! Textures have always repeated. glTF samplers also clamp to the edge and
//! mirror, per axis.
//!
//! Method (`InstancedScene::texture_wrap`, per texture and axis: repeat,
//! clamp or mirror):
//! - Each object's modes for its four texture slots (base color, emissive,
//!   normal map, metallic-roughness) ride in a word per object appended to
//!   the object-texture buffer (no new binding).
//! - Every texture read applies them to the coordinates before sampling
//!   with the repeating sampler: clamp keeps them half a texel (at the
//!   coarser of the two mip levels read) inside the edges, so filtering
//!   never reaches the far edge; mirror folds them into [0, 1] first
//!   (repeating every two tiles) and then clamps the same way, which
//!   repeats the edge texel at each fold, as glTF's mirrored repeat does.
//!   The box filter's taps, opacity, normal maps, metallic-roughness and
//!   emission all go through it.
//! - Alpha cutouts (cut on the CPU) index texels with the same modes.
//! - glTF import reads each texture's sampler (`wrapS`, `wrapT`); save
//!   writes them.
//!
//! Scene: M6's textured scene, its camera, sun and sky, 1280×720, bounce
//! light on, with the ground's tiles clamped (its coordinates run from −7.5
//! to 7.5, so most of the ground shows the edge texels), the boulders'
//! lichen and the block's bricks mirrored. The comparison is the same scene
//! with every texture repeating.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (first): both renderers settled, then five rounds, each the
//!    median of 30 warm frames (each waited for) of the repeating scene and
//!    then the wrapped one; the median over rounds of the ratio at most
//!    1.1× (gate 18's method).
//! 2. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles (one bounce, 4,096 samples per
//!    pixel; each image texture's extension EXTEND or MIRROR) and at most
//!    0.5× the error of the repeating scene against that reference.
//!
//! `cargo run --release -p room3d --bin m10_gate20`

fn main() {
    unimplemented!("M10 gate 20: written before the implementation");
}
