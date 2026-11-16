//! Fog with light shafts (M18): a participating medium in a layer, its
//! single scattering of the lights and the sky.

use bytemuck::{Pod, Zeroable};
use continuum::glam::Vec3;

/// A fog layer (M18): a medium whose extinction is `density` per unit length
/// at height 0, falling off as `exp(-height_falloff · z)` above it (0: even),
/// none above `top`. It scatters `albedo` of what it removes (Henyey and
/// Greenstein's phase with `anisotropy`: 0 even, toward 1 forward). Camera
/// rays through it lose light to it and gain the light it scatters toward
/// them: the first light's (the sun's) with its shadows (light shafts),
/// the other lights' without, and the sky's; surfaces get the lights dimmed
/// by the fog on the way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    /// Extinction per unit length at height 0.
    pub density: f32,
    /// How fast the density falls with height (per unit length; 0: even).
    pub height_falloff: f32,
    /// The layer's top (no fog above).
    pub top: f32,
    /// The share of what it removes that it scatters (its colour).
    pub albedo: Vec3,
    /// Henyey and Greenstein's g.
    pub anisotropy: f32,
    /// How far camera rays gather fog (0: until 1% of light is left at the
    /// ground's density, at most 2,000 units).
    pub distance: f32,
    /// Half the width of the box of the sun's visibility around the eye
    /// (0: 64 units), 128 cells across and 32 up to `top`.
    pub shaft_extent: f32,
}

impl Default for Fog {
    fn default() -> Self {
        Fog {
            density: 0.02,
            height_falloff: 0.0,
            top: 20.0,
            albedo: Vec3::splat(0.9),
            anisotropy: 0.3,
            distance: 0.0,
            shaft_extent: 0.0,
        }
    }
}

/// The sun's visibility grid's cells across and up.
pub(crate) const GRID: [u32; 3] = [128, 128, 32];
/// Froxels per column (depth slices) and pixels per froxel's side.
pub(crate) const SLICES: u32 = 64;
pub(crate) const TILE: u32 = 8;

impl Fog {
    /// How far camera rays gather fog.
    pub(crate) fn range(&self) -> f32 {
        if self.distance > 0.0 {
            self.distance
        } else {
            (4.6 / self.density.max(1e-6)).min(2000.0)
        }
    }

    pub(crate) fn extent(&self) -> f32 {
        if self.shaft_extent > 0.0 {
            self.shaft_extent
        } else {
            64.0
        }
    }
}

/// What the fog passes read (binding 112).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FogParams {
    /// The visibility grid's corner (xyz).
    pub origin: [f32; 4],
    /// Its cells' size.
    pub cell: [f32; 4],
    /// Its cells across, across, up.
    pub dims: [u32; 4],
    /// Froxels: nearest and farthest depth, slices.
    pub froxel: [f32; 4],
    /// Froxels across, down, deep; pixels per froxel's side.
    pub fdims: [u32; 4],
    /// Extinction at 0, falloff, top, anisotropy.
    pub medium: [f32; 4],
    /// Albedo.
    pub albedo: [f32; 4],
    /// The sky's radiance the fog scatters (its upper half's mean, times
    /// one half for an even phase over the upper hemisphere).
    pub ambient: [f32; 4],
}

impl FogParams {
    /// The parameters for an eye at `eye` and an image `width × height`:
    /// the visibility grid snapped to its cells around the eye, from below
    /// the eye (or 0) to the top.
    pub(crate) fn new(fog: &Fog, eye: Vec3, sky: Vec3, width: u32, height: u32) -> FogParams {
        let h = fog.extent();
        let cell_xy = 2.0 * h / GRID[0] as f32;
        let snap = |v: f32| (v / cell_xy).floor() * cell_xy;
        let lo_z = (eye.z - h * 0.25).min(0.0);
        let cell_z = ((fog.top - lo_z) / GRID[2] as f32).max(1e-3);
        FogParams {
            origin: [snap(eye.x) - h, snap(eye.y) - h, lo_z, 0.0],
            cell: [cell_xy, cell_xy, cell_z, 0.0],
            dims: [GRID[0], GRID[1], GRID[2], 0],
            froxel: [0.05, fog.range(), SLICES as f32, 0.0],
            fdims: [width.div_ceil(TILE), height.div_ceil(TILE), SLICES, TILE],
            medium: [
                fog.density,
                fog.height_falloff,
                fog.top,
                fog.anisotropy.clamp(-0.95, 0.95),
            ],
            albedo: [fog.albedo.x, fog.albedo.y, fog.albedo.z, 0.0],
            ambient: [sky.x * 0.5, sky.y * 0.5, sky.z * 0.5, 0.0],
        }
    }
}
