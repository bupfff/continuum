//! M4 gate 1: prefiltered cones (scale of detail).
//!
//! M3 gate 1 found that a cone's cost grows with the triangles really inside
//! it. Now a cone that reaches a node of a mesh's triangle hierarchy whose
//! projected size is at most one bin stops descending and splats the node: its
//! projected box, weighted by the fraction of that box the node's front-facing
//! triangles cover from the cone's direction (sampled at 26 directions per
//! node, summed bottom-up). Cone cost then follows bins, not triangles.
//!
//! Scene: the M2 render scene plus the sphere of M3 gate 1 at four
//! tessellations: 200×100, 400×200, 800×400 and 1600×800 segments (about 40k,
//! 160k, 640k and 2.56M triangles). Per-pixel mask cones at 960×540.
//!
//! Pass criterion, fixed before the first run: (1) at every tessellation, the
//! prefiltered render is within 1% relative RMSE (RMSE over the mean, linear
//! RGB) of the exact render (no prefiltering); (2) the prefiltered render at
//! 2.56M triangles takes at most 1.5× the prefiltered render at 40k triangles
//! (best of 5, warm).

fn main() {}
