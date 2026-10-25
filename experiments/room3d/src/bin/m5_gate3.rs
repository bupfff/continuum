//! M5 gate 3: object-level prefiltered coverage for wide cones, against
//! Cycles on the million-boulder field.
//!
//! M5 gate 2 matched Cycles to 0.7–1.2% with sky light on a 63-object scene.
//! On the million-boulder field, a sky cone from the ground looks across
//! hundreds of thousands of boulders, and walking the object hierarchy to each
//! one is unaffordable. Triangle hierarchies already prefilter (M4 gate 1);
//! this extends it to objects:
//! - Each object-hierarchy node stores directional coverage for the 26 cube
//!   directions: the summed projected front area of its objects. Each
//!   object's own comes from its geometry's root coverage, rotated and scaled.
//! - A cone meeting a node that projects to at most the prefilter threshold
//!   (in bins), with every corner between the cone's near and far planes,
//!   splats the node's hull instead of descending. Opacity is 1 − exp(−front
//!   area / silhouette area): independent occluders, so overlap saturates
//!   instead of exceeding one.
//! - It combines across meshes like any mesh (max for sun cones; as
//!   independent occluders for sky cones).
//!
//! Scene: the 1,000,000-boulder field from its standing viewpoint, 640×360,
//! the gate 2 sky (radiance from the sun's irradiance as in
//! `room3d::outdoor`), the field's sun moved to 1 km (same angular size and
//! direction). Lambertian, direct light only, compute rays, prefilter 1 bin.
//!
//! Reference: Cycles (Blender 5.2) with the boulders instanced through
//! geometry nodes (one mesh, a million instances, per-instance albedo),
//! direct light only, one-pixel box filter, 1024 samples per pixel, two seeds.
//!
//! Pass criterion, fixed before the first run:
//! 1. Sky only (sun off): per-pixel cones, 4×4 per pixel, with object-level
//!    prefiltering, within 3% relative RMSE of Cycles (seed 1).
//! 2. Sun and sky: the same, within 3% of Cycles (seed 1).
//! 3. Cost: the sky-only render, 1 sample per pixel, takes at most a tenth of
//!    the time without object-level prefiltering (or, if that render takes
//!    longer than 10 minutes, it is stopped and counted as more than 10×).
//! 4. No regression: gate 8's four sun-only viewpoints (1280×720, 1 sample
//!    per pixel, prefiltered, sun 3 km away) differ from the renders without
//!    object-level prefiltering by at most 0.5% relative RMSE.
//!
//! Reported: render times, Cycles' noise and time, node splats per cone.

fn main() {}
