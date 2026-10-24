//! M3 gate 2: general meshes in the lighting cache.
//!
//! The M2 lighting cache put texels on box faces. It now takes any mesh with a
//! chart layout (each triangle has coordinates in a rectangular chart; texels
//! and probes are placed through those coordinates) and smooth shading normals.
//!
//! Scene: the showcase (static meshes only, movers removed) at 1920×1080 from
//! the first key of its camera path.
//!
//! Pass criterion, fixed before the first run: (1) the cached render (direct
//! light only) is within 2% relative RMSE of the per-pixel mask-cone render of
//! the same frame, which shades with the same interpolated normals; (2) the
//! cached render takes at most 16.7 ms (best of 5, warm). Bake time and texel
//! count are reported.

fn main() {}
