//! M5 gate 1: sky light through mask cones, against Cycles.
//!
//! Outdoor scenes have had black shadows: the only light was the sun. A
//! uniform sky (constant radiance over the whole sphere of directions, like
//! Cycles' world color) now lights every receiver through six mask cones, one
//! per cube face around the receiver:
//! - Each face is a 90° pyramid with 8×8 bins, its far plane beyond the
//!   scene. Faces wholly below the receiver's horizon are skipped.
//! - Bins straddling the horizon have their projected solid angle clipped to
//!   it (Lambert's edge integrals over the clipped polygon).
//! - Within a mesh, coverage adds as for the sun. Across meshes, sky cones
//!   combine coverage as independent occluders, 1 − Π(1 − c), because their
//!   11° bins often hold several partial occluders. Sun cones keep the maximum.
//! - Camera rays that miss see the sky's radiance.
//! - This gate covers per-pixel cones only (the lighting caches come later).
//!
//! Scene ("outdoor", `room3d::outdoor`): a 400 m ground, 60 boulders of the
//! field's geometry within 30 m of the camera (scales 0.6–2.0, some
//! touching), a wall and a pillar; the field's sun (0.5° wide, 30° up); a
//! blue-white sky whose unoccluded irradiance is about a quarter of the
//! sun's. Lambertian, direct light only. 640×360.
//!
//! Reference: Cycles (Blender 5.2), direct light only (no bounces), world
//! color = sky radiance, 4096 samples per pixel, two seeds; its own noise is
//! reported as the RMSE between the seeds divided by √2.
//!
//! Pass criterion, fixed before the first run:
//! 1. Sky only (sun off): per-pixel exact cones within 3% relative RMSE of
//!    Cycles (seed 1).
//! 2. Sun and sky: per-pixel exact cones within 3%, and prefiltered cones
//!    (1 bin) within 3%, of Cycles (seed 1).
//!
//! Reported: render times, Cycles' noise, and as a diagnostic the same renders
//! with sky cones combining coverage by the maximum.

fn main() {}
