//! M12 gate 3: glTF skins and morph targets, posed as Blender poses them.
//!
//! The user's order (2026-11-15): animated characters imported and played
//! before anti-aliasing. M12 gates 1–2 render geometry an application poses
//! (`RendererOptions::deforming`, `Renderer::set_vertices`). This gate
//! brings glTF's own deformation:
//! - **Read:** skins (joints, inverse bind matrices), each vertex's joints
//!   and weights (`JOINTS_0`, `WEIGHTS_0`), morph targets' position and
//!   normal deltas, and the mesh's default weights. Kept per triangle
//!   corner, in the order and with the degenerate triangles left out as
//!   the geometry built from that primitive.
//! - **Animation:** scale and morph-weight channels as well as translation
//!   and rotation.
//! - **Placed:** a node with a skin, or a mesh with morph targets, gets a
//!   geometry of its own (poses differ per node). A skinned mesh sits at the
//!   world origin with its skinned rest pose as its shape (glTF ignores the
//!   skinned node's own transform); a morphing mesh without a skin keeps
//!   its node's transform.
//! - **Posed:** `Animations::deform(animations, time)` gives each deforming
//!   geometry's triangles and normals at a time: the targets' deltas by the
//!   weights, then linear blend skinning by the joints' world matrices times
//!   their inverse bind matrices. `Imported::deforming` lists the geometries
//!   for `RendererOptions::deforming`.
//!
//! Test file: `experiments/room3d/cycles/make_skinned_test.py`, made in
//! Blender 5.2 (no download): a tube 2 m tall of 32 segments by 24 rings,
//! skinned to three bones with automatic weights, with a shape key; a
//! 48-frame animation (24 fps) bends the middle bone 60° and back, twists
//! the top bone, and drives the shape key 0 → 1 → 0.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Posed as Blender poses it:** at 0, 0.25, 0.5, 0.75, 1.0, 1.25,
//!    1.5 and 1.9 s (both animations), every triangle corner of Continuum's
//!    pose lies within 0.2 mm (1e-4 of the tube's height) of a vertex of
//!    Blender's own import of the same file evaluated at that time
//!    (`compare_skinned_pose.py`: its glTF importer, armature and shape
//!    keys), and the mean is below 0.05 mm.
//! 2. **Played:** the imported tube rendered through `Renderer` with the
//!    cache, posed every frame from `Animations::deform` over the
//!    animation (48 frames at 24 fps, three loops), settles to an image
//!    within 1% relative RMSE of a renderer built with the final pose as
//!    its rest pose (the lag caught up), and the pose shows (at least 5%
//!    from the rest pose's image).
//!
//! Reported: the import's and a pose's CPU time; the tube's corner count.
//!
//! `cargo run --release -p room3d --bin m12_gate3`

fn main() {
    println!("M12 gate 3: stub (criteria above); the implementation comes next.");
}
