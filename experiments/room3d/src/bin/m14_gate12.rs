//! M14 gate 12: coarser levels shade with the finest's normals.
//!
//! A camera level is chosen so its surface lies within half a pixel of the
//! finest (M14 gate 1), but a coarser rock level drops the finer octaves of
//! its displacement, and with them their tilt: its vertex normals are its
//! own smoother surface's. Settled images of the rock field streamed were
//! 1.3–1.6% from those with every finest level resident in gate 1 (4×4
//! supersampled), 1.9–2.5% today.
//!
//! Now (`Detail::finest_normals`, on by default) a rock's camera levels
//! coarser than the finest carry the finest surface's normals at their
//! vertices (from the displacement's derivatives, once per grid vertex), so
//! they shade with the dropped octaves' tilt down to their vertex spacing.
//!
//! Pass criteria, fixed before the run:
//! 1. **Shading against the finest:** M14 gate 1's field (256 unique rocks
//!    6 m apart, sun only), a renderer streaming camera levels within a
//!    2.6-million-triangle budget against one with every finest level
//!    resident, each settled at the three flights' last views and averaged
//!    over 4×4 sub-pixel offsets (1280×720): the colour difference (relative
//!    RMSE) with the finest's normals at most 0.75× that with the levels'
//!    own, at every view.
//! 2. **Production:** producing a rock's coarser camera levels (levels 1 and
//!    2 of the shapes, 64 rocks) takes at most 1.5× as long.
//! 3. **The finest unchanged:** the resident renderers' settled images (finest
//!    levels only) identical with and without.
//!
//! `cargo run --release -p room3d --bin m14_gate12`

fn main() {
    println!("M14 gate 12: stub (criteria in the doc comment); not implemented yet");
}
