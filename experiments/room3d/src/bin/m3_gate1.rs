//! M3 gate 1: per-mesh triangle hierarchies on the GPU.
//!
//! Until now the GPU hierarchy stopped at meshes: a ray or cone that reached a
//! mesh tested every one of its triangles, which is fine for 12-triangle boxes
//! and hopeless for real meshes. Each mesh now gets its own triangle hierarchy
//! (two levels: meshes, then triangles).
//!
//! Scene: the M2 render scene plus a finely tessellated sphere (200 × 100
//! segments, about 40,000 triangles; center (0.3, 0.72, 0.45), radius 0.08, as
//! in `with_sphere`). Per-pixel mask-cone render (no cache) at 960×540 with the
//! mesh hierarchy on.
//!
//! Pass criterion, fixed before the first run: (1) with per-mesh triangle
//! hierarchies, every pixel is within 1e-5 of peak brightness of the render
//! that tests each touched mesh's triangles one by one; (2) it is at least 10×
//! faster (best of 5, warm).

fn main() {}
