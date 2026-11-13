//! M14 gate 15: coarser levels for meshes that come without them
//! (DESIGN §3.8 item 2.3).
//!
//! Camera levels so far come only from callers: the rocks' levels are
//! generated from their analytic shapes (M14 gates 1–13; since gate 13 they
//! carry the finest surface's normals at their vertices). An imported mesh
//! has one level. `continuum::simplify::levels` makes coarser levels of any
//! surface by half-edge collapses ordered by quadric error: each keeps a
//! subset of the finest's vertices with their normals, chart and texture
//! coordinates, attribute seams and borders in place, and states its
//! largest distance to the finest, measured both ways from sample points
//! (vertices, edge midpoints, centroids).
//!
//! Test: each rock's finest level (65,536 triangles) treated as an imported
//! mesh. Its camera levels 1 and 2 are replaced by simplified ones of the
//! same triangle counts (4,096 and 16,384), stating their own errors; the
//! lighting level and the finest stay.
//!
//! Scene and renderers are gate 13's (256 unique rocks, sun only, a
//! 2.6-million-triangle streaming budget against every finest level
//! resident, fixed bake budget, the three flights' last views averaged over
//! 4×4 sub-pixel offsets, a first pass discarded).
//!
//! Pass criteria, fixed before the implementation:
//! 1. **Shading against the finest:** the colour difference (relative RMSE)
//!    streamed against resident with simplified camera levels at most 1.25×
//!    that with the analytic levels carrying the finest's normals (gate 13:
//!    1.25–1.48%), at each of the three views.
//! 2. **Stated errors hold:** for every simplified level of every rock, the
//!    largest distance to the finest measured from 64 random points per
//!    triangle both ways at most 1.1× its stated error.
//!
//! Reported: simplification time per rock; the simplified levels' stated
//! errors against the analytic levels'; with the lighting level simplified
//! too (to 1,024 triangles), its streamed image against the analytic
//! resident one.
//!
//! `M14_DEV=1`: a development check on 36 rocks; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m14_gate15`

fn main() {
    unimplemented!("M14 gate 15: written before the implementation");
}
