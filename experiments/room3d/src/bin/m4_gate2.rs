//! M4 gate 2: ten billion instanced triangles in a large world (scale of
//! amount and size).
//!
//! Objects reference shared geometry with a rigid transform and a uniform
//! scale; rays and cones move into each object's space, where prefiltered
//! cones (gate 1) apply. Object positions are kept in f64 on the CPU and
//! uploaded relative to a render origin near the camera.
//!
//! Scene: a ground plane with a field of instances of one detailed procedural
//! object (about 10,000 triangles) on a jittered 2 m grid with random rotation
//! and scale, lit by a sun-sized (0.5°) one-sided rectangular light. Per-pixel
//! mask cones at 1920×1080 from a camera at standing height looking over the
//! field.
//!
//! Pass criterion, fixed before the first run:
//! 1. Correctness: with 64 instances and exact cones, the instanced render
//!    matches the same scene with every instance's triangles flattened into
//!    world space: relative RMSE at most 0.1%, and at least 99.9% of pixels
//!    within 1e-4 of peak.
//! 2. Scale: with 1,000,000 instances (about 10^10 triangles), the frame takes
//!    at most 2× the frame with 1,000 instances (about 10^7 triangles), same
//!    camera, best of 5.
//! 3. Large world: the 1,000,000-instance field moved 64 km from the world
//!    origin renders within 0.1% relative RMSE of the field at the origin, from
//!    the same relative camera.
//!
//! Frame times and GPU memory are reported.

fn main() {}
