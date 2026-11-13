//! GPU backend for Continuum (wgpu compute; Metal, Vulkan, DX12).
//!
//! **Start with [`Renderer`]:** an [`InstancedScene`] (geometry placed many
//! times, at f64 positions), its [`Light`]s (rectangles, spheres and spots,
//! with RGB radiance) and an optional uniform sky, rendered each frame into a
//! texture. `continuum_gltf::load` makes scenes from glTF files.
//!
//! How it lights: from each surface point, a *mask cone* toward each light
//! finds what covers the light, as exact coverage of 8×8 bins, with distant
//! geometry prefiltered. The view-dependent lighting cache keeps that work
//! across frames on the surfaces in view, coarse first; pixels shade from it
//! with their own normals. No frame blends in earlier frames.
//!
//! One kernel family underneath: mask cones from receivers to a light, a port
//! of `continuum::cone3` (see `mask_cones.wgsl`), used by [`Renderer`] and by
//! the older lightmap path: [`LiveRenderer`] renders meshes that move, with
//! exact soft shadows from one rectangular light and one bounce of indirect
//! light, from a world-space [`Lightmap`] (no instancing, no sky).

#![warn(missing_docs)]

use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use wgpu::util::DeviceExt;

pub use wgpu;

type Tri = [Vec3; 3];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightUniform {
    corner: [f32; 4],
    edge_u: [f32; 4],
    edge_v: [f32; 4],
    counts: [u32; 4],
    /// `[prefilter threshold in bins (0: exact), 1 for nearest-first ray
    /// traversal, 0, 0]`
    params: [f32; 4],
    /// Uniform sky radiance (linear RGB; zero: no sky).
    sky: [f32; 4],
    /// `[first object-hierarchy node's coverage in blas_cov (NONE: no
    /// object-level prefiltering), sky prefilter bits, 1 with an
    /// environment (M5 gate 11), 0]`
    objects: [u32; 4],
    /// The environment's radiance above the horizon as nine spherical
    /// harmonics coefficients (RGB in xyz; M5 gate 11).
    sky_sh: [[f32; 4]; 9],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReceiverGpu {
    point: [f32; 4],
    normal: [f32; 4],
}

/// The GPU kernel: compiled pipelines on a device, shared by every renderer
/// built on it. Make one with [`MaskConeKernel::new`] (its own device) or
/// [`MaskConeKernel::with_device`] (yours), and hand it to
/// [`Renderer::new`].
pub struct MaskConeKernel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The kernel's shader, its pipelines built the first time each is used
    /// (a renderer builds only the passes it runs).
    module: wgpu::ShaderModule,
    /// The shader's source, for the hardware-ray module.
    source: String,
    /// Whether the device has ray queries (for hardware camera rays), and
    /// that module, built when first needed.
    ray_queries: bool,
    hw_module: std::cell::OnceCell<wgpu::ShaderModule>,
    /// Pipeline-overridable constants: workgroup stacks, work counters.
    workgroup_stacks: bool,
    count_work: bool,
    pipelines: std::cell::RefCell<std::collections::HashMap<&'static str, wgpu::ComputePipeline>>,
    hardware_rays: std::cell::Cell<bool>,
    /// MikkTSpace tangents for normal maps (M10 gate 13; else each
    /// triangle's own frame).
    mikktspace: std::cell::Cell<bool>,
    /// The environment's radiance as nine spherical harmonics coefficients,
    /// if set (M5 gate 11).
    environment: std::cell::Cell<Option<[[f32; 4]; 9]>>,
    /// The environment's image for what rays see of it (M5 gate 12): sRGB
    /// over its peak radiance, and that peak.
    environment_image: std::cell::RefCell<Option<(Texture, f32)>>,
    pixel_cones: std::cell::Cell<bool>,
    prefilter: std::cell::Cell<f32>,
    sky: std::cell::Cell<Vec3>,
    sky_max: std::cell::Cell<bool>,
    sky_prefilter: std::cell::Cell<f32>,
    sky_horizon_cull: std::cell::Cell<bool>,
    sky_object_frame: std::cell::Cell<bool>,
    object_prefilter: std::cell::Cell<bool>,
    nearest_first: std::cell::Cell<bool>,
    pub(crate) profile_stage: std::cell::Cell<u32>,
    /// M13 gate 3: the coverage maps' build time and bytes, as last built.
    m13_maps_info: std::cell::RefCell<(std::time::Duration, usize)>,
    split: std::cell::Cell<bool>,
    /// The GPU's name, as the adapter reports it.
    pub adapter_name: String,
}

/// Everything a frame needs besides the pipeline.
struct Frame<'a> {
    camera: Camera,
    width: u32,
    height: u32,
    light: RectLight,
    meshes: &'a [Vec<Tri>],
    albedo: &'a [Vec3],
    radiance: f32,
    hierarchy: Hierarchy,
}

mod coverage_maps;
mod cutout;
mod hierarchy;
mod hw;
mod instanced;
mod lightmap;
mod live;
mod renderer;
mod stream;
mod tangents;
mod vcache;

pub use cutout::cut_out;
pub use lightmap::Lightmap;
pub use live::{LiveRenderer, MAX_MOVING, MAX_SWEPT, PROBE_RESOLUTION};
pub use renderer::{Renderer, RendererOptions};
#[doc(hidden)]
pub use stream::lighting_package_bytes;
pub use stream::{Level, LevelSource, Levels, StreamOptions, StreamStats};
pub use vcache::{CacheOptions, CacheStats};

// Experiment-only: used by the gates in `experiments/`, not part of the
// library's surface.
#[doc(hidden)]
pub use hierarchy::Hierarchy;
#[doc(hidden)]
pub use instanced::{CacheCheck, FrameProfile, InstancedRenderer, LedgerCheck, SkyCheck};
#[doc(hidden)]
pub use lightmap::{BLOCK, BLOCK_PLANES, hemicube_weights};
#[doc(hidden)]
pub use vcache::CachePage;

use hierarchy::{NodeGpu, Placement, flatten, flatten_instanced, mesh_hierarchy, object_hierarchy};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    params: [f32; 4],
}

/// An environment's radiance over directions (M5 gate 11): an
/// equirectangular image in Blender's convention, rows from the top (the
/// zenith) down, column `u` at azimuth `atan2(y, x) = π − 2π u`, linear RGB.
#[derive(Clone, Debug, Default)]
pub struct Environment {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height` linear RGB radiances, row by row from the top.
    pub rgb: Vec<[f32; 3]>,
}

impl Environment {
    /// An environment from a function of direction (unit, z up), sampled at
    /// each pixel's centre.
    pub fn from_fn(width: u32, height: u32, f: impl Fn(Vec3) -> Vec3) -> Environment {
        let mut rgb = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                rgb.push(f(Self::direction(width, height, x, y)).to_array());
            }
        }
        Environment { width, height, rgb }
    }

    /// Its image over its peak radiance as sRGB bytes, and the peak (for
    /// what rays see of it: limited dynamic range; M5 gate 12).
    fn texture(&self) -> (Texture, f32) {
        let peak = self
            .rgb
            .iter()
            .flat_map(|p| p.iter().copied())
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let encode = |c: f32| {
            let c = (c / peak).clamp(0.0, 1.0);
            let s = if c <= 0.003_130_8 {
                12.92 * c
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0).round() as u8
        };
        let rgba = self
            .rgb
            .iter()
            .flat_map(|p| [encode(p[0]), encode(p[1]), encode(p[2]), 255])
            .collect();
        (
            Texture {
                width: self.width,
                height: self.height,
                rgba,
            },
            peak,
        )
    }

    /// The unit direction through pixel `(x, y)`'s centre.
    pub fn direction(width: u32, height: u32, x: u32, y: u32) -> Vec3 {
        let u = (x as f32 + 0.5) / width as f32;
        let v = 1.0 - (y as f32 + 0.5) / height as f32;
        let phi = std::f32::consts::PI - std::f32::consts::TAU * u;
        let e = (v - 0.5) * std::f32::consts::PI;
        Vec3::new(e.cos() * phi.cos(), e.cos() * phi.sin(), e.sin())
    }

    /// Its radiance above the horizon as nine real spherical harmonics
    /// coefficients (RGB), and its mean radiance there (over solid angle).
    fn sh9(&self) -> ([[f32; 4]; 9], Vec3) {
        let mut c = [[0.0f64; 3]; 9];
        let (mut sum, mut area) = (Vec3::ZERO.as_dvec3(), 0.0f64);
        let (w, h) = (self.width.max(1), self.height.max(1));
        let cell = std::f64::consts::TAU / w as f64 * std::f64::consts::PI / h as f64;
        for y in 0..h {
            for x in 0..w {
                let d = Self::direction(w, h, x, y);
                if d.z <= 0.0 {
                    continue;
                }
                let solid = cell * (d.z.asin().cos() as f64);
                let l = Vec3::from_array(self.rgb[(y * w + x) as usize]).as_dvec3();
                for (k, b) in sh_basis(d).iter().enumerate() {
                    for ch in 0..3 {
                        c[k][ch] += l[ch] * *b as f64 * solid;
                    }
                }
                sum += l * solid;
                area += solid;
            }
        }
        let sh = c.map(|k| [k[0] as f32, k[1] as f32, k[2] as f32, 0.0]);
        (sh, (sum / area.max(1e-12)).as_vec3())
    }
}

/// The nine real spherical harmonics at unit direction `d` (bands 0–2).
fn sh_basis(d: Vec3) -> [f32; 9] {
    [
        0.282_095,
        0.488_603 * d.y,
        0.488_603 * d.z,
        0.488_603 * d.x,
        1.092_548 * d.x * d.y,
        1.092_548 * d.y * d.z,
        0.315_392 * (3.0 * d.z * d.z - 1.0),
        1.092_548 * d.x * d.z,
        0.546_274 * (d.x * d.x - d.y * d.y),
    ]
}

/// Compile-time choices for a [`MaskConeKernel`] (experiments only).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelOptions {
    /// Traversal stacks in workgroup memory, or in per-thread arrays (the
    /// default: M4 gate 5 found workgroup stacks 2.6× slower, as their 12 KB per
    /// workgroup limits how many workgroups run at once).
    pub workgroup_stacks: bool,
    /// Diagnostics: the view cache's bake counts its work (nodes tested and
    /// splatted, triangles rasterized) into [`CacheStats::work`]. Off by
    /// default, when the counting compiles away.
    pub count_work: bool,
    /// Each invocation walks hierarchies alone, as before M5 gate 5, instead
    /// of with its subgroup (packet traversal, used where the device has
    /// subgroups, except in the view cache's bakes, which always walk
    /// alone). Results are identical; only speed differs.
    pub without_packets: bool,
}

/// A pinhole camera looking from `eye` toward `target`, with `+Z` up.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Where the camera is, relative to the render origin.
    pub eye: Vec3,
    /// A point it looks at, relative to the render origin.
    pub target: Vec3,
    /// The full vertical field of view, in degrees.
    pub vertical_fov_degrees: f32,
}

/// A rectangular light: a corner and two perpendicular edges.
#[derive(Clone, Copy, Debug)]
pub struct RectLight {
    /// One corner, relative to the render origin.
    pub corner: Vec3,
    /// The edge from `corner` to the next corner.
    pub edge_u: Vec3,
    /// The other edge from `corner`, perpendicular to `edge_u`.
    pub edge_v: Vec3,
    /// One-sided: emits only toward `edge_u × edge_v`, and camera rays see its
    /// front as a surface of the light's radiance. Otherwise (the M2 scenes) it
    /// emits to both sides and is invisible to the camera.
    pub one_sided: bool,
}

/// A light: what emits, and its RGB radiance (linear). Make one with
/// [`Light::rect`], [`Light::sun`], [`Light::point`] or [`Light::spot`].
#[derive(Clone, Copy, Debug)]
pub struct Light {
    /// Its shape and position.
    pub emitter: Emitter,
    /// Radiance over the emitter's surface (linear RGB), in the scene's
    /// units of light.
    pub radiance: Vec3,
}

/// What emits a light.
#[derive(Clone, Copy, Debug)]
pub enum Emitter {
    /// A rectangle: an area light, or a sun far away.
    Rect(RectLight),
    /// A sphere: a point light with a size, or a spot.
    Sphere(SphereLight),
}

/// A spherical light (M5): uniform radiance over its surface, seen from
/// outside as a disk, optionally narrowed to a spot.
#[derive(Clone, Copy, Debug)]
pub struct SphereLight {
    /// The center, relative to the render origin.
    pub center: Vec3,
    /// The sphere's radius: larger lights cast softer shadows.
    pub radius: f32,
    /// A spot's cone, or `None` for light in every direction.
    pub spot: Option<Spot>,
}

/// A spot's cone, as glTF's (`KHR_lights_punctual`): full radiance toward
/// directions within `inner` of `axis`, none beyond `outer` (half-angles in
/// radians), and between them the square of a ramp in the cosine. The
/// falloff applies to the direction from the light's center.
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    /// The direction the spot shines.
    pub axis: Vec3,
    /// Full radiance within this half-angle of `axis` (radians).
    pub inner: f32,
    /// No light beyond this half-angle of `axis` (radians).
    pub outer: f32,
}

impl Light {
    /// A white rectangular light of `radiance`.
    pub fn white(rect: RectLight, radiance: f32) -> Light {
        Light {
            emitter: Emitter::Rect(rect),
            radiance: Vec3::splat(radiance),
        }
    }

    /// A rectangular light.
    pub fn rect(rect: RectLight, radiance: Vec3) -> Light {
        Light {
            emitter: Emitter::Rect(rect),
            radiance,
        }
    }

    /// A point light of `radius` at `center` with radiant intensity
    /// `intensity` (per steradian) in every direction: radiance
    /// `intensity / (π radius²)` over a sphere.
    pub fn point(center: Vec3, radius: f32, intensity: Vec3) -> Light {
        Light {
            emitter: Emitter::Sphere(SphereLight {
                center,
                radius,
                spot: None,
            }),
            radiance: intensity / (std::f32::consts::PI * radius * radius),
        }
    }

    /// [`Light::point`] narrowed to a spot.
    pub fn spot(center: Vec3, radius: f32, intensity: Vec3, spot: Spot) -> Light {
        Light {
            emitter: Emitter::Sphere(SphereLight {
                center,
                radius,
                spot: Some(spot),
            }),
            ..Light::point(center, radius, intensity)
        }
    }

    /// The rectangle, for a rectangular light.
    pub fn as_rect(&self) -> Option<RectLight> {
        match self.emitter {
            Emitter::Rect(r) => Some(r),
            Emitter::Sphere(_) => None,
        }
    }

    /// A square sun 0.5° across, `distance` from `center` in the direction
    /// `toward`, one-sided and facing `center`, whose radiance gives
    /// `irradiance` on a surface facing it. `center` is relative to the
    /// render origin, like every light.
    pub fn sun(center: Vec3, toward: DVec3, distance: f64, irradiance: Vec3) -> Light {
        let toward = toward.normalize();
        let size = 2.0 * distance * 0.25f64.to_radians().tan();
        let w = -toward;
        let side = if w.cross(DVec3::Z).length() > 1e-6 {
            DVec3::Z.cross(w)
        } else {
            DVec3::X
        };
        let a = side.normalize() * size;
        let b = w.cross(a).normalize() * size;
        let middle = center.as_dvec3() + toward * distance;
        let solid = (size / distance) * (size / distance);
        Light::rect(
            RectLight {
                corner: (middle - (a + b) * 0.5).as_vec3(),
                edge_u: a.as_vec3(),
                edge_v: b.as_vec3(),
                one_sided: true,
            },
            irradiance / solid as f32,
        )
    }
}

/// A light as the GPU reads it (`LightRect` in the shader): for a
/// rectangle, its corner (w: 1 if one-sided) and edges; for a sphere, its
/// center (w: 2), radius and spot; and radiance (RGB).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LightGpu {
    corner: [f32; 4],
    edge_u: [f32; 4],
    edge_v: [f32; 4],
    radiance: [f32; 4],
}

impl From<&Light> for LightGpu {
    fn from(l: &Light) -> Self {
        let v4 = |v: Vec3, w: f32| [v.x, v.y, v.z, w];
        match l.emitter {
            Emitter::Rect(rect) => LightGpu {
                corner: v4(rect.corner, rect.one_sided as u32 as f32),
                edge_u: v4(rect.edge_u, 0.0),
                edge_v: v4(rect.edge_v, 0.0),
                radiance: v4(l.radiance, 0.0),
            },
            Emitter::Sphere(sphere) => {
                // A spot's falloff ramp, as glTF's reference: its scale and
                // offset in the cosine to the axis.
                let (axis, scale, offset, on) = match sphere.spot {
                    Some(s) => {
                        let scale = 1.0 / (s.inner.cos() - s.outer.cos()).max(1e-3);
                        (s.axis.normalize(), scale, -s.outer.cos() * scale, 1.0)
                    }
                    None => (Vec3::Z, 0.0, 1.0, 0.0),
                };
                LightGpu {
                    corner: v4(sphere.center, 2.0),
                    edge_u: [sphere.radius, scale, offset, 0.0],
                    edge_v: v4(axis, on),
                    radiance: v4(l.radiance, 0.0),
                }
            }
        }
    }
}

/// The most lights a scene can have.
pub const MAX_LIGHTS: usize = 16;

impl RectLight {
    #[allow(clippy::too_many_arguments)]
    fn uniform(
        &self,
        meshes: usize,
        debug_mode: u32,
        hierarchy: Hierarchy,
        prefilter: f32,
        nearest_first: bool,
        profile_stage: u32,
        (sky, sky_max, sky_prefilter, sky_horizon_cull, sky_object_frame): (
            Vec3,
            bool,
            f32,
            bool,
            bool,
        ),
        objects: u32,
    ) -> LightUniform {
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        LightUniform {
            corner: v4(self.corner),
            edge_u: v4(self.edge_u),
            edge_v: v4(self.edge_v),
            counts: [
                meshes as u32,
                debug_mode,
                hierarchy as u32,
                self.one_sided as u32,
            ],
            params: [
                prefilter,
                nearest_first as u32 as f32,
                profile_stage as f32,
                (sky_max as u32 | (!sky_horizon_cull as u32) << 1 | (sky_object_frame as u32) << 2)
                    as f32,
            ],
            sky: v4(sky),
            objects: [objects, sky_prefilter.to_bits(), 0, 0],
            sky_sh: [[0.0; 4]; 9],
        }
    }
}

/// Bindings the `render` entry point uses (render_frame filters the scene's
/// buffers to these when a lightmap provides more).
const RENDER_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 6, 7, 8, 9, 10, 24, 25, 26, 29, 30, 52, 55, 56, 57, 58,
];

/// The same list, for other modules.
pub(crate) const RECT_LIGHT_BINDINGS: &[u32] = RENDER_BINDINGS;

/// Which of the scene's bindings a pipeline uses.
type KeepBinding = Box<dyn Fn(u32) -> bool>;

/// Bindings only per-pixel shading and the view cache read (lights, base
/// color textures): the lightmap paths, which bind every scene buffer, leave
/// them out.
pub(crate) const PER_PIXEL_ONLY: &[u32] = &[52, 55, 56, 57, 58];

/// The kernel's compute entry points (`mask_cones.wgsl`).
const ENTRY_POINTS: &[&str] = &[
    "main",
    "render",
    "render_cached",
    "mark_blocks",
    "rebake",
    "probe_build",
    "probe_gather",
    "bake_static",
    "visibility",
    "shade",
    "shade_cached_pass",
    "vc_update",
    "vc_bake",
    "vc_bake_sky",
    "vc_self_bake",
    "vc_fill",
    "mark",
    "vc_shade",
    "render_cones",
    "vc_shade_cones",
    "vc_stale",
    "vc_args",
    "vc_stale_texels",
    "vc_args_bake",
    "vc_ledger_reset",
    "vc_ledger",
    "vc_bake_indirect",
    "vc_shade_edges",
    "vc_args_edges",
    "m13_coverage",
    "m13_coverage_maps",
];

/// The hardware-ray module's entry points (`hw_visibility.wgsl`).
const HW_ENTRY_POINTS: &[&str] = &[
    "hw_visibility",
    "hw_visibility_movers",
    "hw_visibility_blend",
    "hw_edges",
    "hw_m13_corners",
    "hw_m13_near",
];

/// Bindings the `visibility` pass uses.
const VISIBILITY_BINDINGS: &[u32] = &[0, 2, 3, 6, 9, 10, 25, 26, 30, 31, 52];

/// Bindings the hardware `hw_visibility` pass uses (32: the acceleration structure).
pub(crate) const HW_VISIBILITY_BINDINGS: &[u32] = &[2, 3, 6, 30, 31, 32, 52];
/// `hw_visibility_movers` adds the movers' acceleration structure (59).
pub(crate) const HW_VISIBILITY_MOVERS_BINDINGS: &[u32] = &[2, 3, 6, 30, 31, 32, 52, 59];
/// `hw_edges` (M7 gate 4): silhouette rays through ray queries.
pub(crate) const HW_EDGES_BINDINGS: &[u32] = &[2, 3, 6, 30, 31, 32, 41, 42, 55, 56, 57, 58, 76, 81];
/// `m13_coverage` (M13): the scalar coverage walk alone.
pub(crate) const M13_COVERAGE_BINDINGS: &[u32] = &[0, 2, 3, 6, 7, 9, 10, 25, 26, 29, 30];
/// `m13_coverage_maps` (M13 gate 3): with the coverage maps (82).
pub(crate) const M13_MAPS_BINDINGS: &[u32] = &[0, 2, 3, 6, 7, 9, 10, 25, 26, 29, 30, 82];
/// M13 gate 4: the corner rays (`hw_m13_corners`), and the coverage pass
/// after them (`m13_coverage_near`).
pub(crate) const HW_M13_CORNERS_BINDINGS: &[u32] = &[2, 3, 6, 30, 31, 32, 82];
pub(crate) const M13_NEAR_BINDINGS: &[u32] = &[0, 2, 3, 6, 7, 9, 10, 25, 26, 29, 30, 31, 32, 82];
/// `hw_visibility_blend` (M10 gate 19): visibility, listing the layers
/// behind alpha-blended hits.
pub(crate) const HW_VISIBILITY_BLEND_BINDINGS: &[u32] =
    &[2, 3, 6, 30, 31, 32, 42, 52, 55, 56, 57, 58, 76, 81];
/// `vc_args_edges` (M10 gate 19): `vc_shade_edges`'s dispatch from the
/// list's count.
pub(crate) const VC_ARGS_EDGES_BINDINGS: &[u32] = &[42, 66, 81];

/// Bindings of the lighting cache's passes.
pub(crate) const VC_UPDATE_BINDINGS: &[u32] = &[36, 41, 44];
pub(crate) const VC_BAKE_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 9, 10, 23, 24, 25, 26, 29, 30, 33, 34, 54, 37, 38, 41, 42, 43, 52, 62, 64, 68, 69,
    70, 75, 83,
];
pub(crate) const VC_BAKE_SKY_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 9, 10, 23, 24, 25, 26, 29, 30, 33, 34, 54, 38, 41, 42, 46, 47, 49, 51, 62, 64, 75,
    83,
];
pub(crate) const VC_SELF_BAKE_BINDINGS: &[u32] =
    &[0, 2, 3, 5, 23, 24, 25, 26, 29, 30, 33, 34, 54, 49, 50, 51];
pub(crate) const VC_FILL_BINDINGS: &[u32] = &[
    3, 30, 33, 37, 38, 40, 41, 46, 47, 48, 61, 62, 63, 64, 73, 77,
];
pub(crate) const VC_SHADE_BINDINGS: &[u32] = &[
    79, 0, 2, 3, 6, 7, 8, 9, 10, 23, 24, 25, 26, 30, 31, 33, 35, 36, 37, 39, 40, 41, 42, 45, 52,
    55, 56, 57, 58, 62, 73, 76,
];
/// `vc_shade_edges` (M7 gate 4): the cache's shading of listed edge
/// pixels.
pub(crate) const VC_SHADE_EDGES_BINDINGS: &[u32] = &[
    79, 0, 2, 3, 6, 7, 8, 9, 10, 23, 24, 25, 26, 30, 33, 35, 36, 37, 39, 40, 41, 42, 45, 52, 55,
    56, 57, 58, 62, 73, 76, 81,
];
/// Bounce light's samples (M10 gate 1).
pub(crate) const VC_BAKE_INDIRECT_BINDINGS: &[u32] = &[
    0, 2, 3, 8, 9, 10, 23, 24, 25, 26, 30, 33, 34, 35, 36, 37, 38, 39, 41, 42, 45, 52, 54, 55, 56,
    57, 58, 64, 73, 74, 75,
];
/// The movers' stale passes (M9) and the dispatch sizes after each.
pub(crate) const VC_STALE_BINDINGS: &[u32] = &[
    3, 30, 33, 38, 40, 41, 43, 46, 48, 52, 60, 61, 62, 63, 64, 65, 67, 77, 78,
];
pub(crate) const VC_ARGS_BINDINGS: &[u32] = &[64, 66, 67];
pub(crate) const VC_STALE_TEXELS_BINDINGS: &[u32] = &[
    2, 3, 23, 24, 30, 33, 34, 41, 43, 52, 54, 60, 61, 62, 63, 64, 65, 67,
];
pub(crate) const VC_ARGS_BAKE_BINDINGS: &[u32] = &[41, 46, 64, 66];
/// Moving lights' ledger (M9 gate 2): fresh records for planned pages, and
/// the per-frame check.
pub(crate) const VC_LEDGER_RESET_BINDINGS: &[u32] = &[38, 41, 69, 72];
pub(crate) const VC_LEDGER_BINDINGS: &[u32] =
    &[3, 30, 38, 40, 41, 43, 52, 61, 62, 63, 64, 68, 69, 70, 72];
/// Bindings of `vc_shade_cones`: the cache's shading with pixel cones for
/// visibility (the scene's hierarchies instead of a visibility buffer).
pub(crate) const VC_SHADE_CONES_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 6, 7, 8, 9, 10, 23, 24, 25, 26, 29, 30, 33, 35, 36, 37, 39, 40, 41, 42, 45, 52, 55,
    56, 57, 58, 62,
];
/// The compute `visibility` pass's bindings, for other modules.
pub(crate) const COMPUTE_VISIBILITY_BINDINGS: &[u32] = VISIBILITY_BINDINGS;

/// What Continuum asks of a device: the adapter's limits, and ray queries (an
/// experimental wgpu feature, used only for camera rays, with the compute path
/// as fallback) where the adapter supports them.
pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let ray_query = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    // Subgroups for packet traversal, and timestamps for the view cache's
    // bake budget, where available (M5 gate 5).
    let subgroups =
        adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY);
    wgpu::DeviceDescriptor {
        label: Some("continuum"),
        required_features: subgroups
            | if ray_query {
                wgpu::Features::EXPERIMENTAL_RAY_QUERY
            } else {
                wgpu::Features::empty()
            },
        required_limits: adapter.limits(),
        experimental_features: if ray_query {
            // SAFETY: wgpu's experimental ray queries are opted into knowingly;
            // they are used for camera rays only, and can be turned off
            // (`set_hardware_rays`).
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        },
        ..Default::default()
    }
}

/// The kernel's WGSL source as a device compiles it: with subgroup
/// operations for packet traversal where `packets` is set, and the workgroup
/// stack array shrunk unless `workgroup_stacks` (unused, it would still cost
/// workgroup memory; an override cannot remove it). The kernel's entry points
/// take the overrides `WORKGROUP_STACKS`, `COUNT_WORK` and `PACKETS`.
#[doc(hidden)]
pub fn kernel_source(packets: bool, workgroup_stacks: bool) -> String {
    // Line endings as written: a Windows checkout may have converted them,
    // and the patterns below match `\n`.
    let mut source = include_str!("mask_cones.wgsl").replace("\r\n", "\n");
    if packets {
        let solo = "fn packet_any(x: bool) -> bool {\n    return x;\n}";
        assert!(source.contains(solo), "packet_any not found");
        // Naga takes subgroup built-ins without `enable subgroups;` (which
        // it rejects in wgpu 30) once the device has the feature.
        source = source.replace(
            solo,
            "fn packet_any(x: bool) -> bool {\n    if (PACKETS) { return subgroupAny(x); }\n    return x;\n}",
        );
        let first = "fn packet_first(x: bool) -> bool {\n    return x;\n}";
        assert!(source.contains(first), "packet_first not found");
        source = source.replace(
            first,
            "fn packet_first(x: bool) -> bool {\n    if (PACKETS) { return subgroupBroadcastFirst(select(0u, 1u, x)) == 1u; }\n    return x;\n}",
        );
    }
    if !workgroup_stacks {
        source = source.replace(
            "var<workgroup> wg_stacks: array<u32, 3072>;",
            "var<workgroup> wg_stacks: array<u32, 1>;",
        );
    }
    source
}

impl MaskConeKernel {
    /// A kernel on its own device: the high-performance adapter, unless
    /// wgpu's environment variables choose another (`WGPU_BACKEND`, such as
    /// `vulkan` or `dx12`; `WGPU_ADAPTER_NAME`, a part of the adapter's name;
    /// `WGPU_POWER_PREF`; `WGPU_DX12_COMPILER`). `None` if no GPU adapter is
    /// available.
    ///
    /// # Panics
    ///
    /// If `WGPU_ADAPTER_NAME` is set and no adapter's name contains it.
    pub fn new() -> Option<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            match pollster::block_on(wgpu::util::initialize_adapter_from_env(&instance, None)) {
                Ok(adapter) => adapter,
                Err(_) => pollster::block_on(
                    instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::from_env()
                            .unwrap_or(wgpu::PowerPreference::HighPerformance),
                        ..Default::default()
                    }),
                )
                .ok()?,
            };
        let adapter_name = adapter.get_info().name;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&device_descriptor(&adapter))).ok()?;
        Some(Self::with_device(device, queue, adapter_name))
    }

    /// A kernel on an existing device (for example one that also drives a window).
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue, adapter_name: String) -> Self {
        Self::with_options(device, queue, adapter_name, KernelOptions::default())
    }

    /// [`Self::new`] with options.
    #[doc(hidden)]
    pub fn new_with(options: KernelOptions) -> Option<Self> {
        let kernel = Self::new()?;
        Some(Self::with_options(
            kernel.device,
            kernel.queue,
            kernel.adapter_name,
            options,
        ))
    }

    /// [`Self::with_device`] with options.
    #[doc(hidden)]
    pub fn with_options(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        options: KernelOptions,
    ) -> Self {
        let packets =
            !options.without_packets && device.features().contains(wgpu::Features::SUBGROUP);
        let source = kernel_source(packets, options.workgroup_stacks);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mask_cones"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let ray_queries = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        MaskConeKernel {
            device,
            queue,
            module,
            source,
            ray_queries,
            hw_module: std::cell::OnceCell::new(),
            workgroup_stacks: options.workgroup_stacks,
            count_work: options.count_work,
            pipelines: Default::default(),
            hardware_rays: std::cell::Cell::new(ray_queries),
            mikktspace: std::cell::Cell::new(true),
            environment: std::cell::Cell::new(None),
            environment_image: std::cell::RefCell::new(None),
            pixel_cones: std::cell::Cell::new(false),
            prefilter: std::cell::Cell::new(0.0),
            sky: std::cell::Cell::new(Vec3::ZERO),
            sky_max: std::cell::Cell::new(false),
            sky_prefilter: std::cell::Cell::new(1.0),
            sky_horizon_cull: std::cell::Cell::new(true),
            sky_object_frame: std::cell::Cell::new(false),
            object_prefilter: std::cell::Cell::new(true),
            nearest_first: std::cell::Cell::new(true),
            profile_stage: std::cell::Cell::new(0),
            m13_maps_info: std::cell::RefCell::new((std::time::Duration::ZERO, 0)),
            split: std::cell::Cell::new(false),
            adapter_name,
        }
    }

    /// The pipeline of entry point `name`, built on first use. Entry points
    /// named `hw_*` come from the hardware-ray module.
    pub(crate) fn entry(&self, name: &'static str) -> wgpu::ComputePipeline {
        if let Some(p) = self.pipelines.borrow().get(name) {
            return p.clone();
        }
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        // The view cache's bakes walk alone: their texels can be far apart,
        // and packets made small bakes slower (M5 gate 5).
        let solo = matches!(name, "vc_bake" | "vc_bake_sky" | "vc_self_bake");
        let constants = [
            ("WORKGROUP_STACKS", flag(self.workgroup_stacks)),
            ("COUNT_WORK", flag(self.count_work)),
            ("PACKETS", flag(!solo)),
        ];
        let module = if name.starts_with("hw_") {
            assert!(self.ray_queries, "no ray queries on this device");
            self.hw_module.get_or_init(|| {
                let hw_source = format!(
                    "enable wgpu_ray_query;\n{}\n{}",
                    self.source,
                    include_str!("hw_visibility.wgsl")
                );
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("hw_visibility"),
                        source: wgpu::ShaderSource::Wgsl(hw_source.into()),
                    })
            })
        } else {
            &self.module
        };
        // CONTINUUM_PIPELINE_TIMES: print how long each pipeline takes to
        // build (shader compilers differ by orders of magnitude).
        let start = Instant::now();
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            });
        if std::env::var_os("CONTINUUM_PIPELINE_TIMES").is_some() {
            eprintln!("pipeline {name}: {:.2} s", start.elapsed().as_secs_f64());
        }
        self.pipelines.borrow_mut().insert(name, pipeline.clone());
        pipeline
    }

    /// Builds every pipeline now (they are otherwise built on first use):
    /// for timing, and to check that each compiles on the device.
    #[doc(hidden)]
    pub fn build_all_pipelines(&self) {
        for name in ENTRY_POINTS {
            self.entry(name);
        }
        if self.ray_queries {
            for name in HW_ENTRY_POINTS {
                self.entry(name);
            }
        }
    }

    /// The device the kernel runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue the kernel submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Prefiltered cones: a cone stops descending a mesh's triangle hierarchy at
    /// nodes that project to at most `bins` bins and splats their precomputed
    /// coverage instead. 0 (the default) is exact. Applies to work set up after
    /// the call, including [`LiveRenderer`]s created after it.
    #[doc(hidden)]
    pub fn set_prefilter(&self, bins: f32) {
        self.prefilter.set(bins.max(0.0));
    }

    /// A uniform sky of radiance `rgb` (linear) over the whole sphere of
    /// directions, lighting every receiver through six cube-face mask cones
    /// (M5 gate 1); camera rays that miss see it. Zero (the default) turns it
    /// off. Per-pixel rendering and the view-dependent cache
    /// ([`InstancedRenderer::with_cache`]) support it; the world-space
    /// lightmaps ([`Lightmap`], [`LiveRenderer`]) refuse a sky so far. Applies
    /// to work set up after the call.
    #[doc(hidden)]
    pub fn set_sky(&self, rgb: Vec3) {
        self.sky.set(rgb.max(Vec3::ZERO));
    }

    /// Sky light from an environment (M5 gate 11): radiance varying with
    /// direction above the horizon (below it, the scene's own geometry is
    /// taken to be the ground), or None for [`Self::set_sky`]'s uniform sky.
    /// The view cache keeps each texel's sky visibility for a uniform sky,
    /// and shading scales it by the environment's irradiance at the pixel's
    /// normal over the open sky's (as lights' ratios): exact for an
    /// unoccluded sky, approximate where occlusion depends on direction.
    /// Camera and reflection rays that miss see the environment through its
    /// nine spherical harmonics (smooth). Sets the uniform sky to the
    /// environment's mean radiance above the horizon, which bounce light
    /// and per-pixel rendering use. Read when a renderer is made.
    pub fn set_environment(&self, environment: Option<&Environment>) {
        match environment {
            Some(e) => {
                let (sh, mean) = e.sh9();
                self.environment.set(Some(sh));
                self.set_sky(mean);
                *self.environment_image.borrow_mut() = Some(e.texture());
            }
            None => {
                self.environment.set(None);
                *self.environment_image.borrow_mut() = None;
            }
        }
    }

    /// The scene's base color textures, and the environment's image after
    /// them if there is one (M5 gate 12), with its layer (NONE: none).
    pub(crate) fn textures_with_environment(&self, textures: &[Texture]) -> (Vec<Texture>, u32) {
        let mut all = textures.to_vec();
        match &*self.environment_image.borrow() {
            Some((t, _)) => {
                all.push(t.clone());
                (all, textures.len() as u32)
            }
            None => (all, hierarchy::NONE),
        }
    }

    /// The light uniform's environment fields (M5 gate 11).
    /// The light uniform's environment fields (M5 gates 11–12): its
    /// harmonics, and with `layer` (the base color array's layer of its
    /// image, NONE: none) the image's layer and peak.
    fn environment_uniform(&self, u: &mut LightUniform, layer: u32) {
        u.objects[3] = hierarchy::NONE;
        if let Some(sh) = self.environment.get() {
            u.sky_sh = sh;
            u.objects[2] = 1;
            if let Some((_, peak)) = &*self.environment_image.borrow() {
                u.objects[3] = layer;
                u.sky_sh[0][3] = *peak;
            }
        }
    }

    /// The sky set with [`Self::set_sky`].
    #[doc(hidden)]
    pub fn sky(&self) -> Vec3 {
        self.sky.get()
    }

    /// Object-level prefiltering for instanced scenes (M5 gate 3, on by
    /// default): with prefiltering on, cones splat small object-hierarchy
    /// nodes (whole groups of objects) instead of descending into them.
    /// Applies to work set up after the call.
    #[doc(hidden)]
    pub fn set_object_prefilter(&self, on: bool) {
        self.object_prefilter.set(on);
    }

    /// Sky cones prefilter `factor` times as coarsely as the light's cones
    /// (default 1): their bins are 11° wide, so coarser splats cost little
    /// accuracy and much time. Applies to work set up after the call.
    #[doc(hidden)]
    pub fn set_sky_prefilter(&self, factor: f32) {
        self.sky_prefilter.set(factor.max(0.0));
    }

    /// Sky cones skip geometry wholly below the receiver's tangent plane (on
    /// by default, M5 gate 5). It can only block directions below the
    /// horizon, which carry no light to the receiver; walking it only made
    /// the binned coverage of bins the horizon crosses darker. Applies to work
    /// set up after the call.
    #[doc(hidden)]
    pub fn set_sky_horizon_cull(&self, on: bool) {
        self.sky_horizon_cull.set(on);
    }

    /// Sky cones turn their cube faces with the object they start on
    /// (default off): per pixel, and in the view cache for texels without
    /// own-occlusion masks (those with masks always do; see
    /// `CacheOptions::self_spacing`). A uniform sky's light does not depend
    /// on the cube's orientation, but binned coverage does: on the boulder
    /// field, turning the cube moves the sky 1.9–4.2% from Cycles (M5 gate 5).
    /// Applies to work set up after the call.
    #[doc(hidden)]
    pub fn set_sky_object_frame(&self, on: bool) {
        self.sky_object_frame.set(on);
    }

    /// Diagnostic: sky cones combine coverage across meshes by the maximum,
    /// like sun cones, instead of as independent occluders.
    #[doc(hidden)]
    pub fn set_sky_combination_max(&self, on: bool) {
        self.sky_max.set(on);
    }

    /// The prefilter set with [`Self::set_prefilter`].
    #[doc(hidden)]
    pub fn prefilter(&self) -> f32 {
        self.prefilter.get()
    }

    /// Ray traversal order: nearest child first (the default), or a fixed child
    /// order (as before M4 gate 3). Results are the same; only speed differs.
    #[doc(hidden)]
    pub fn set_nearest_first(&self, on: bool) {
        self.nearest_first.set(on);
    }

    /// Whether this kernel can cast camera rays with hardware ray queries.
    pub fn has_hardware_rays(&self) -> bool {
        self.ray_queries
    }

    /// Camera rays for instanced scenes through hardware ray queries (the
    /// default where available) or the compute hierarchy. Applies to
    /// renderers created after the call.
    pub fn set_hardware_rays(&self, on: bool) {
        self.hardware_rays.set(on && self.has_hardware_rays());
    }

    /// Experiments: MikkTSpace tangents for normal maps (on), or each
    /// triangle's own frame (M10 gate 13). Read when a scene is uploaded.
    #[doc(hidden)]
    pub fn set_mikktspace(&self, on: bool) {
        self.mikktspace.set(on);
    }

    /// Debugging: cut cones short to time their stages (0: off, 1: no cone,
    /// 2: setup and weights only, 3: plus the object hierarchy, 4: plus triangle
    /// hierarchies but no rasterization). Results are wrong unless 0.
    #[doc(hidden)]
    pub fn set_profile_stage(&self, stage: u32) {
        self.profile_stage.set(stage);
    }

    /// Diagnostics (M13 gate 3): the coverage maps' build time and size in
    /// bytes, as last built (profile stage 15).
    #[doc(hidden)]
    pub fn m13_maps_info(&self) -> (std::time::Duration, usize) {
        *self.m13_maps_info.borrow()
    }

    /// Primary visibility through pixel cones (M7) instead of one camera
    /// ray per pixel, in instanced renderers created after the call.
    #[doc(hidden)]
    pub fn set_pixel_cones(&self, on: bool) {
        self.pixel_cones.set(on);
    }

    pub(crate) fn uses_hardware_rays(&self) -> bool {
        self.hardware_rays.get() && self.ray_queries
    }

    /// Camera frames as a visibility pass then a shading pass, or as one pass
    /// (the default: M4 gate 4 found the split slower). Results are the same;
    /// only speed differs. Applies to [`LiveRenderer`]s created after the call.
    #[doc(hidden)]
    pub fn set_visibility_buffer(&self, on: bool) {
        self.split.set(on);
    }

    /// Irradiance per unit radiance at each receiver `(point, normal)`, and the
    /// fastest of `runs` GPU executions (submit to completion, excluding uploads
    /// and readback). Every receiver tests every mesh.
    #[doc(hidden)]
    pub fn irradiance(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        runs: usize,
    ) -> (Vec<f32>, Duration) {
        self.irradiance_with(receivers, light, meshes, false, runs)
    }

    /// [`Self::irradiance`], optionally culling through the mesh hierarchy.
    #[doc(hidden)]
    pub fn irradiance_with(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        hierarchy: impl Into<Hierarchy>,
        runs: usize,
    ) -> (Vec<f32>, Duration) {
        let (values, _, best) = self.run(receivers, light, meshes, runs, 0, hierarchy.into());
        (values, best)
    }

    /// Renders a `width × height` view: each pixel's ray finds the nearest
    /// triangle, and a mask cone gives that point's direct irradiance from the
    /// light. Returns linear RGB (albedo per mesh × radiance × irradiance / π)
    /// and the fastest of `runs` GPU executions.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        camera: Camera,
        width: u32,
        height: u32,
        light: RectLight,
        meshes: &[Vec<Tri>],
        albedo: &[Vec3],
        radiance: f32,
        hierarchy: impl Into<Hierarchy>,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        let frame = Frame {
            camera,
            width,
            height,
            light,
            meshes,
            albedo,
            radiance,
            hierarchy: hierarchy.into(),
        };
        self.render_frame(false, &frame, None, runs)
    }

    /// Like [`Self::render`] (mask cones at every pixel, with the hierarchy),
    /// but shading with the lightmap's interpolated normals: the reference for
    /// [`Self::render_cached`] on smooth meshes. The lightmap's texels are unused.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_per_pixel(
        &self,
        camera: Camera,
        width: u32,
        height: u32,
        light: RectLight,
        meshes: &[Vec<Tri>],
        albedo: &[Vec3],
        radiance: f32,
        lightmap: &Lightmap,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        let frame = Frame {
            camera,
            width,
            height,
            light,
            meshes,
            albedo,
            radiance,
            hierarchy: Hierarchy::Triangles,
        };
        self.render_frame(false, &frame, Some(lightmap), runs)
    }

    /// Like [`Self::render`] (with the hierarchy), but static faces read their
    /// irradiance from a baked [`Lightmap`]; only dynamic geometry gets a cone
    /// per pixel.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_cached(
        &self,
        camera: Camera,
        width: u32,
        height: u32,
        light: RectLight,
        meshes: &[Vec<Tri>],
        albedo: &[Vec3],
        radiance: f32,
        lightmap: &Lightmap,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        assert!(
            self.sky() == Vec3::ZERO,
            "the lighting caches do not support a sky yet"
        );
        let frame = Frame {
            camera,
            width,
            height,
            light,
            meshes,
            albedo,
            radiance,
            hierarchy: Hierarchy::Triangles,
        };
        self.render_frame(true, &frame, Some(lightmap), runs)
    }

    /// Per-pixel mask cones over an instanced scene: shared geometry placed many
    /// times. Instance positions (f64) are uploaded relative to `origin`; the
    /// camera and light are given relative to `origin` too. Returns linear RGB,
    /// the fastest of `runs` executions, and the scene's GPU memory in bytes.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_instanced(
        &self,
        camera: Camera,
        width: u32,
        height: u32,
        light: RectLight,
        scene: &InstancedScene,
        origin: DVec3,
        radiance: f32,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration, u64) {
        self.render_instanced_lights(
            camera,
            width,
            height,
            &[Light::white(light, radiance)],
            scene,
            origin,
            runs,
        )
    }

    /// [`Self::render_instanced`] lit by several lights, each with RGB
    /// radiance (at most [`MAX_LIGHTS`]), and the sky if set.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_instanced_lights(
        &self,
        camera: Camera,
        width: u32,
        height: u32,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration, u64) {
        let (buffers, _, _) = self.instanced_buffers(lights, scene, origin, &[], 0, None);
        let bytes = buffers.iter().map(|(_, b)| b.size()).sum();
        let cam = camera_uniform(camera, width, height, lights[0].radiance.x);
        let hw_scene = self
            .uses_hardware_rays()
            .then(|| self.hw_scene(scene, origin, &[], 0));
        let (pixels, time) = self.render_buffers_with(
            false,
            cam,
            width,
            height,
            buffers,
            hw_scene.as_ref().map(|h| &h.tlas),
            &self.scene_textures(&scene.textures, &scene.linear_textures()),
            runs,
        );
        (pixels, time, bytes)
    }

    /// The lights' buffer (binding 52), writable for changing radiance.
    pub(crate) fn lights_buffer(&self, lights: &[Light]) -> wgpu::Buffer {
        assert!(
            (1..=MAX_LIGHTS).contains(&lights.len()),
            "between 1 and {MAX_LIGHTS} lights"
        );
        let gpu: Vec<LightGpu> = lights.iter().map(LightGpu::from).collect();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lights"),
                contents: bytemuck::cast_slice(&gpu),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Scene buffers for an instanced scene, by binding (see `render_instanced`).
    /// The first light also fills the light uniform, which the paths without
    /// light lists read.
    #[allow(clippy::type_complexity)]
    pub(crate) fn instanced_buffers(
        &self,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
        movable: &[usize],
        pool: u32,
        lighting: Option<&stream::LightingLayout>,
    ) -> (
        Vec<(u32, wgpu::Buffer)>,
        Option<hierarchy::MoverTree>,
        Option<stream::LitNodes>,
    ) {
        // Paths without the list read the first light as a rectangle; a
        // sphere stands there as a tiny square at its center.
        let light = lights[0].as_rect().unwrap_or_else(|| {
            let c = match lights[0].emitter {
                Emitter::Sphere(s) => s.center,
                Emitter::Rect(r) => r.corner,
            };
            RectLight {
                corner: c,
                edge_u: Vec3::X * 1e-3,
                edge_v: Vec3::Y * 1e-3,
                one_sided: false,
            }
        });
        // Lighting levels streamed (M14 gates 4–5): the streamed geometries
        // (the scene's last, from `skip`) leave every buffer for packages
        // loaded into slots; each keeps its record.
        let skip = lighting.map_or(usize::MAX, |l| l.first);
        let geometries: Vec<Vec<Tri>> = scene
            .geometries
            .iter()
            .enumerate()
            .map(|(g, s)| {
                if g >= skip {
                    Vec::new()
                } else {
                    s.triangles.clone()
                }
            })
            .collect();
        let placements: Vec<Placement> = scene
            .instances
            .iter()
            .map(|i| Placement {
                geometry: i.geometry,
                rotation: i.rotation,
                scale: i.scale,
                translation: (i.position - origin).as_vec3(),
                clear: scene.blend_clear(i.geometry),
            })
            .collect();
        let (mut flat, xforms, mut bounds, geometry_ranges) =
            flatten_instanced(&geometries, &placements);
        if let Some(lay) = lighting {
            for (b, p) in bounds.iter_mut().zip(&placements) {
                if p.geometry >= skip {
                    *b = p.world_bounds(lay.records[p.geometry - skip].bounds);
                }
            }
        }
        // Vertex colors (M10 gate 17) in the triangles' unused w: 1 plus an
        // sRGB byte per channel, an exact integer (0: none).
        let mut first = 0;
        for (g, geometry) in scene.geometries.iter().enumerate().take(skip) {
            if let Some(Some(colors)) = scene.geometry_colors.get(g)
                && colors.len() == geometry.triangles.len()
            {
                for (t, corners) in colors.iter().enumerate() {
                    for (k, c) in corners.iter().enumerate() {
                        flat.tris[3 * (first + t) + k][3] = color_code(*c);
                    }
                }
            }
            first += geometry.triangles.len();
        }
        let (nodes, items, movers) = if movable.is_empty() {
            let (nodes, items) = object_hierarchy(&bounds, |_| true);
            (nodes, items, None)
        } else {
            let movable: Vec<u32> = movable.iter().map(|&m| m as u32).collect();
            let (nodes, items, tree) = hierarchy::object_hierarchy_with_movers(&bounds, &movable);
            (nodes, items, Some(tree))
        };
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        // Per triangle corner, its shading normal, and in w its MikkTSpace
        // tangent's code where the geometry has a normal map (M10 gate 13;
        // 0: none).
        let mut normals: Vec<[f32; 4]> = Vec::new();
        for (g, geometry) in scene.geometries.iter().enumerate().take(skip) {
            let mapped = scene
                .geometry_normal_maps
                .get(g)
                .copied()
                .flatten()
                .is_some()
                && geometry.tex.len() == geometry.triangles.len()
                && geometry.normals.len() == geometry.triangles.len();
            let tangents = (self.mikktspace.get() && mapped)
                .then(|| tangents::mikktspace(geometry))
                .flatten();
            for (t, corners) in geometry.normals.iter().enumerate() {
                for (k, n) in corners.iter().enumerate() {
                    let w = tangents
                        .as_ref()
                        .map_or(0.0, |ts| tangents::encode(ts[t][k]));
                    normals.push([n.x, n.y, n.z, w]);
                }
            }
        }
        normals.push([0.0; 4]);
        let albedo: Vec<[f32; 4]> = scene.instances.iter().map(|i| v4(i.albedo)).collect();
        // Per object, its texture layer (NONE: none), then per object its
        // emission as two words of half floats (for bounce samples, which
        // cannot bind the materials buffer: M10 gate 11), then per object its
        // wrap modes; per geometry triangle, its corners' texture coordinates
        // (zero without).
        let mut object_textures: Vec<u32> = scene
            .instances
            .iter()
            .map(|i| {
                let has = scene.geometries[i.geometry].tex.len()
                    == scene.geometries[i.geometry].triangles.len();
                match scene.geometry_textures.get(i.geometry).copied().flatten() {
                    Some(t) if has => t,
                    _ => hierarchy::NONE,
                }
            })
            .collect();
        // A geometry's emissive texture's layer in the base color array, plus
        // one (0: none; M10 gate 14).
        let emissive_of = |i: &Instance| {
            let g = &scene.geometries[i.geometry];
            scene
                .geometry_emissive_textures
                .get(i.geometry)
                .copied()
                .flatten()
                .filter(|_| g.tex.len() == g.triangles.len())
                .map_or(0, |t| t + 1)
        };
        for i in &scene.instances {
            let e = scene
                .geometry_emission
                .get(i.geometry)
                .copied()
                .unwrap_or(Vec3::ZERO)
                .to_array()
                .map(half_bits);
            object_textures.extend([e[0] | e[1] << 16, e[2] | emissive_of(i) << 16]);
        }
        // Per object, its texture slots' wrap modes (M10 gate 20): four bits
        // a slot (two for u, then two for v), base color, emissive, normal
        // map, then metallic-roughness.
        for i in &scene.instances {
            let g = i.geometry;
            let slots = [
                scene.geometry_textures.get(g),
                scene.geometry_emissive_textures.get(g),
                scene.geometry_normal_maps.get(g),
                scene.geometry_metallic_roughness.get(g),
            ];
            let mut word = 0u32;
            for (k, slot) in slots.into_iter().enumerate() {
                if let Some(&Some(t)) = slot
                    && let Some(w) = scene.texture_wrap.get(t as usize)
                {
                    word |= (w[0].code() | w[1].code() << 2) << (4 * k);
                }
            }
            object_textures.push(word);
        }
        // Per object, its material: metallic and roughness (roughness below
        // zero: Lambertian), and its emission as half floats.
        let linear = scene.linear_textures();
        let layer_of = |maps: &[Option<u32>], i: &Instance| {
            let g = &scene.geometries[i.geometry];
            maps.get(i.geometry)
                .copied()
                .flatten()
                .filter(|_| g.tex.len() == g.triangles.len())
                .and_then(|k| linear.iter().position(|&m| m == k))
                .map_or(0, |p| p as u32 + 1)
        };
        let materials: Vec<[u32; 4]> = scene
            .instances
            .iter()
            .map(|i| -> [u32; 8] {
                // Its normal map's and metallic-roughness texture's
                // layers plus one (0: none).
                let layer = layer_of(&scene.geometry_normal_maps, i);
                let mr = layer_of(&scene.geometry_metallic_roughness, i);
                let [metallic, roughness] =
                    match scene.geometry_materials.get(i.geometry).copied().flatten() {
                        Some(m) => [m.metallic, m.roughness.clamp(0.0, 1.0)],
                        None => [0.0, -1.0],
                    };
                let e = scene
                    .geometry_emission
                    .get(i.geometry)
                    .copied()
                    .unwrap_or(Vec3::ZERO)
                    .to_array()
                    .map(half_bits);
                [
                    metallic.to_bits(),
                    roughness.to_bits(),
                    e[0] | e[1] << 16,
                    e[2] | layer << 16,
                    mr,
                    emissive_of(i),
                    // Alpha blending (M10 gate 16): the opacity factor, and 1.
                    scene
                        .geometry_blend
                        .get(i.geometry)
                        .copied()
                        .flatten()
                        .map_or(0, |o| o.clamp(0.0, 1.0).to_bits()),
                    scene
                        .geometry_blend
                        .get(i.geometry)
                        .copied()
                        .flatten()
                        .is_some() as u32,
                ]
            })
            .chain(std::iter::once([0, (-1.0f32).to_bits(), 0, 0, 0, 0, 0, 0]))
            .flat_map(|[a, b, c, d, mr, glow, blend, blended]| {
                [[a, b, c, d], [mr, glow, blend, blended]]
            })
            .collect();
        let mut tex: Vec<[f32; 2]> = Vec::new();
        for g in scene.geometries.iter().take(skip) {
            if g.tex.len() == g.triangles.len() {
                tex.extend(g.tex.iter().flatten().map(|v| v.to_array()));
            } else {
                tex.extend(std::iter::repeat_n([0.0; 2], 3 * g.triangles.len()));
            }
        }
        if tex.is_empty() {
            tex.push([0.0; 2]);
        }
        // Lighting levels streamed (M14 gates 4–5): a stub per streamed
        // geometry (its root and coverage, from its record) follows the
        // resident nodes, then the slots.
        let mut lit_parts = None;
        if let Some(lay) = lighting {
            let cs = hierarchy::COVERAGE_SAMPLES;
            let t_s = lay.static_tris as usize;
            assert_eq!(
                geometry_ranges[skip][0] as usize, t_s,
                "streamed geometries come last"
            );
            let slot_total = (lay.slots * lay.slot_tris) as usize;
            flat.tris.truncate(3 * t_s);
            flat.tris.resize(3 * (t_s + slot_total), [0.0; 4]);
            normals.truncate(3 * t_s);
            normals.resize(3 * (t_s + slot_total), [0.0; 4]);
            tex.truncate(3 * t_s);
            tex.resize(3 * (t_s + slot_total), [0.0; 2]);
            flat.blas_items.truncate(t_s);
            flat.blas_items.resize(t_s + slot_total, 0);
            // Each stub (its hierarchy's top levels, M14 gate 8) in turn.
            let mut stub_roots = Vec::with_capacity(lay.records.len());
            for r in lay.records.iter() {
                let root = flat.blas.len();
                stub_roots.push(root as u32);
                flat.blas
                    .extend(r.stub.iter().map(|n| n.rebased(root as i64, 0)));
                flat.blas_cov.extend_from_slice(&r.stub_cov);
            }
            let slot_base = flat.blas.len();
            let all_nodes = slot_base + (lay.slots * lay.slot_nodes) as usize;
            flat.blas.resize(all_nodes, NodeGpu::zeroed());
            flat.blas_cov.resize(all_nodes * cs, 0.0);
            for (k, i) in scene.instances.iter().enumerate() {
                if i.geometry >= skip {
                    let w = flat.ranges[k][3];
                    flat.ranges[k] = [hierarchy::NONE, 0, stub_roots[i.geometry - skip], w];
                }
            }
            lit_parts = Some(stream::LitNodes {
                stub_roots,
                slot_base: slot_base as u32,
                slot_nodes: lay.slot_nodes,
            });
        }
        // Object-level prefiltering: the object hierarchy's coverage follows
        // the triangle hierarchies' in the same buffer.
        let objects = if self.object_prefilter.get() && self.prefilter.get() > 0.0 {
            let roots: Vec<u32> = flat.ranges.iter().map(|r| r[2]).collect();
            let cov =
                hierarchy::object_coverage(&nodes, &items, &flat.blas_cov, &roots, &placements);
            let base = (flat.blas_cov.len() / hierarchy::COVERAGE_SAMPLES) as u32;
            flat.blas_cov.extend(cov);
            base
        } else {
            hierarchy::NONE
        };
        // Camera levels' pool (M14 gate 1): after the lighting level's
        // triangles in every per-triangle buffer the camera's shading reads.
        if pool > 0 {
            // The clusters' member lists follow the objects' ranges (M14
            // gate 6; as `hw::proxy_clusters` orders them), then the page
            // table.
            let members: Vec<u32> = if self.uses_hardware_rays() {
                hw::proxy_clusters(scene, origin).concat()
            } else {
                Vec::new()
            };
            let words = members.len().div_ceil(4);
            let at = flat.ranges.len();
            flat.ranges.extend(std::iter::repeat_n([0u32; 4], words));
            for (w, &k) in members.iter().enumerate() {
                flat.ranges[at + w / 4][w % 4] = k;
            }
            let entries = stream::table_entries(scene) as usize;
            flat.ranges
                .extend(std::iter::repeat_n([0u32; 4], entries.div_ceil(4)));
            let lit: usize = lighting.map_or_else(
                || scene.geometries.iter().map(|g| g.triangles.len()).sum(),
                |l| l.lit_tris() as usize,
            );
            let all = 3 * (lit + pool as usize);
            flat.tris.resize(all, [0.0; 4]);
            normals.resize(all, [0.0; 4]);
            if stream::textured(scene) {
                tex.resize(all, [0.0; 2]);
            }
        }
        let mut uniform = light.uniform(
            scene.instances.len(),
            0,
            Hierarchy::Triangles,
            self.prefilter.get(),
            self.nearest_first.get(),
            self.profile_stage.get(),
            (
                self.sky.get(),
                self.sky_max.get(),
                self.sky_prefilter.get(),
                self.sky_horizon_cull.get(),
                self.sky_object_frame.get(),
            ),
            objects,
        );
        self.environment_uniform(&mut uniform, scene.textures.len() as u32);
        // Writable for movers (the hierarchy's movers subtree, transforms).
        let storage = |label: &str, contents: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
        };
        // With a pool, camera levels' bottom-level structures build straight
        // from the triangles' buffer (M14 gate 1).
        let tris = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tris"),
                contents: bytemuck::cast_slice(&flat.tris),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | if pool > 0 {
                        wgpu::BufferUsages::BLAS_INPUT
                    } else {
                        wgpu::BufferUsages::empty()
                    },
            });
        let buffers = vec![
            (
                0,
                uniform_buffer(&self.device, "light", bytemuck::bytes_of(&uniform)),
            ),
            (2, tris),
            (3, storage("objects", bytemuck::cast_slice(&flat.ranges))),
            (5, storage("debug", &[0; 256])),
            (8, storage("albedo", bytemuck::cast_slice(&albedo))),
            (9, storage("object nodes", bytemuck::cast_slice(&nodes))),
            (10, storage("object items", bytemuck::cast_slice(&items))),
            (24, storage("normals", bytemuck::cast_slice(&normals))),
            (
                25,
                storage("triangle nodes", bytemuck::cast_slice(&flat.blas)),
            ),
            (
                26,
                storage("triangle items", bytemuck::cast_slice(&flat.blas_items)),
            ),
            (
                29,
                storage("node coverage", bytemuck::cast_slice(&flat.blas_cov)),
            ),
            (30, storage("transforms", bytemuck::cast_slice(&xforms))),
            (52, self.lights_buffer(lights)),
            (
                57,
                storage("object textures", bytemuck::cast_slice(&object_textures)),
            ),
            (
                58,
                storage("texture coordinates", bytemuck::cast_slice(&tex)),
            ),
            (76, storage("materials", bytemuck::cast_slice(&materials))),
        ];
        let mut buffers = buffers;
        // M13 gate 3 (profile stage 15): each placed geometry's directional
        // coverage maps.
        if matches!(self.profile_stage.get(), 15 | 16) {
            let start = std::time::Instant::now();
            let tris: Vec<(&[continuum::mesh::Tri], Option<u32>)> = scene
                .geometries
                .iter()
                .enumerate()
                .map(|(g, s)| {
                    let root = scene
                        .instances
                        .iter()
                        .position(|i| i.geometry == g)
                        .map(|i| flat.ranges[i][2])
                        .filter(|&r| r != hierarchy::NONE);
                    (s.triangles.as_slice(), root)
                })
                .collect();
            let maps = coverage_maps::build(&tris);
            *self.m13_maps_info.borrow_mut() = (start.elapsed(), maps.len() * 4);
            buffers.push((82, storage("coverage maps", bytemuck::cast_slice(&maps))));
        }
        (buffers, movers, lit_parts)
    }

    /// A scene's textures, resampled into a layered texture with mip levels
    /// (a 1×1 white layer without any), and its normal maps (textures
    /// `normal_maps`, M10 gate 5) into a linear one (a flat 1×1 layer
    /// without any).
    pub(crate) fn scene_textures(
        &self,
        textures: &[Texture],
        normal_maps: &[u32],
    ) -> SceneTextures {
        let (view, sampler) = self.layered(textures, false);
        let maps: Vec<Texture> = normal_maps
            .iter()
            .map(|&k| textures[k as usize].clone())
            .collect();
        let (normals, _) = self.layered(&maps, true);
        SceneTextures {
            view,
            sampler,
            normals,
        }
    }

    /// Textures resampled into one layered texture with mip levels (sRGB, or
    /// `linear`); a 1×1 layer without any (white, or a flat normal).
    fn layered(&self, textures: &[Texture], linear: bool) -> (wgpu::TextureView, wgpu::Sampler) {
        let size = if textures.is_empty() { 1 } else { TEXTURE_SIZE };
        let levels = size.ilog2() + 1;
        let layers = textures.len().max(1) as u32;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("base color"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if linear {
                wgpu::TextureFormat::Rgba8Unorm
            } else {
                wgpu::TextureFormat::Rgba8UnormSrgb
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white = Texture {
            width: 1,
            height: 1,
            rgba: if linear {
                vec![128, 128, 255, 255]
            } else {
                vec![255; 4]
            },
        };
        let sources: Vec<&Texture> = if textures.is_empty() {
            vec![&white]
        } else {
            textures.iter().collect()
        };
        for (layer, t) in sources.iter().enumerate() {
            for (level, pixels) in texture_levels(t, size, linear).iter().enumerate() {
                let side = size >> level;
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * side),
                        rows_per_image: Some(side),
                    },
                    wgpu::Extent3d {
                        width: side,
                        height: side,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("base color"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        (view, sampler)
    }

    /// Hardware acceleration structures for an instanced scene.
    pub(crate) fn hw_scene(
        &self,
        scene: &InstancedScene,
        origin: DVec3,
        movable: &[usize],
        dynamic: u32,
    ) -> hw::HwScene {
        let geometries: Vec<Vec<Tri>> = scene
            .geometries
            .iter()
            .map(|g| g.triangles.clone())
            .collect();
        let placements: Vec<Placement> = scene
            .instances
            .iter()
            .map(|i| Placement {
                geometry: i.geometry,
                rotation: i.rotation,
                scale: i.scale,
                translation: (i.position - origin).as_vec3(),
                clear: scene.blend_clear(i.geometry),
            })
            .collect();
        // Geometries with a proxy level are never seen by camera rays
        // themselves (M14 gate 4): no structure. Their objects show their
        // proxies through clusters, and finer levels through `dynamic`
        // slots (M14 gate 6; none when nothing streams).
        let skip: Vec<bool> = (0..scene.geometries.len())
            .map(|g| hw::proxied(scene, g))
            .collect();
        let clusters = if dynamic > 0 {
            hw::proxy_clusters(scene, origin)
        } else {
            Vec::new()
        };
        let proxies: Vec<Option<Vec<Tri>>> = (0..scene.geometries.len())
            .map(|g| {
                scene
                    .geometry_levels
                    .get(g)
                    .and_then(|l| l.as_ref())
                    .and_then(|l| l.proxy.as_ref())
                    .map(|p| p.surface.triangles.clone())
            })
            .collect();
        hw::build(
            &self.device,
            &self.queue,
            &geometries,
            &placements,
            movable,
            &skip,
            clusters,
            &proxies,
            dynamic,
        )
    }

    /// Bakes a lightmap for [`LiveRenderer::new`]: one mask cone per texel
    /// toward `light`, through the meshes' hierarchy.
    /// Returns the bake time.
    pub fn bake(&self, lightmap: &mut Lightmap, light: RectLight, meshes: &[Vec<Tri>]) -> Duration {
        assert!(
            self.sky() == Vec3::ZERO,
            "the lighting caches do not support a sky yet"
        );
        let (values, time) = self.irradiance_with(&lightmap.receivers, light, meshes, true, 1);
        lightmap.irradiance = values;
        time
    }

    fn render_frame(
        &self,
        cached: bool,
        frame: &Frame,
        lightmap: Option<&Lightmap>,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        let (width, height) = (frame.width, frame.height);
        let cam = camera_uniform(frame.camera, width, height, frame.radiance);
        let buffers = self.scene_buffers(frame, lightmap);
        self.render_buffers(cached, cam, width, height, buffers, runs)
    }

    /// Runs a render pipeline over prepared scene buffers; returns linear RGB
    /// and the fastest of `runs` executions.
    fn render_buffers(
        &self,
        cached: bool,
        cam: CameraUniform,
        width: u32,
        height: u32,
        buffers: Vec<(u32, wgpu::Buffer)>,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        let textures = self.scene_textures(&[], &[]);
        self.render_buffers_with(cached, cam, width, height, buffers, None, &textures, runs)
    }

    /// [`Self::render_buffers`], with camera rays through hardware ray queries
    /// when `tlas` is given.
    #[allow(clippy::too_many_arguments)]
    fn render_buffers_with(
        &self,
        cached: bool,
        cam: CameraUniform,
        width: u32,
        height: u32,
        mut buffers: Vec<(u32, wgpu::Buffer)>,
        tlas: Option<&wgpu::Tlas>,
        textures: &SceneTextures,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        buffers.push((
            6,
            uniform_buffer(&self.device, "camera", bytemuck::bytes_of(&cam)),
        ));
        let size = (width * height) as u64 * 16;
        let storage = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let colors_buf = storage(
            "colors",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = storage(
            "staging",
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let vis_buf = storage("visibility", size, wgpu::BufferUsages::STORAGE);
        buffers.push((7, colors_buf.clone()));
        buffers.push((31, vis_buf));
        // Single pass, or a visibility pass then a shading pass; each pipeline
        // gets exactly the bindings it uses.
        let split = self.split.get();
        let hw = tlas.is_some() && self.ray_queries;
        let passes: Vec<(wgpu::ComputePipeline, KeepBinding)> = match (cached, split || hw) {
            (false, true) if hw => vec![
                (
                    self.entry("hw_visibility"),
                    Box::new(|b| HW_VISIBILITY_BINDINGS.contains(&b)),
                ),
                (
                    self.entry("shade"),
                    Box::new(|b| b == 31 || RENDER_BINDINGS.contains(&b)),
                ),
            ],
            (false, false) => vec![(
                self.entry("render"),
                Box::new(|b| RENDER_BINDINGS.contains(&b)),
            )],
            (true, false) => vec![(
                self.entry("render_cached"),
                Box::new(|b| b != 31 && !PER_PIXEL_ONLY.contains(&b)),
            )],
            (false, true) => vec![
                (
                    self.entry("visibility"),
                    Box::new(|b| VISIBILITY_BINDINGS.contains(&b)),
                ),
                (
                    self.entry("shade"),
                    Box::new(|b| b == 31 || RENDER_BINDINGS.contains(&b)),
                ),
            ],
            (true, true) => vec![
                (
                    self.entry("visibility"),
                    Box::new(|b| VISIBILITY_BINDINGS.contains(&b)),
                ),
                (
                    self.entry("shade_cached_pass"),
                    Box::new(|b| !PER_PIXEL_ONLY.contains(&b)),
                ),
            ],
        };
        let groups: Vec<(&wgpu::ComputePipeline, wgpu::BindGroup)> = passes
            .iter()
            .map(|(pipeline, keep)| {
                let mut entries: Vec<wgpu::BindGroupEntry> = buffers
                    .iter()
                    .filter(|(binding, _)| keep(*binding))
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();
                if let Some(tlas) = tlas
                    && keep(32)
                {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 32,
                        resource: tlas.as_binding(),
                    });
                }
                if keep(55) {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 55,
                        resource: wgpu::BindingResource::TextureView(&textures.view),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 56,
                        resource: wgpu::BindingResource::Sampler(&textures.sampler),
                    });
                }
                if keep(79) {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 79,
                        resource: wgpu::BindingResource::TextureView(&textures.normals),
                    });
                }
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("render"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                (pipeline, group)
            })
            .collect();
        let mut best = Duration::MAX;
        for _ in 0..runs.max(1) {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for (pipeline, group) in &groups {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }
            let start = Instant::now();
            self.queue.submit([encoder.finish()]);
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
            best = best.min(start.elapsed());
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&colors_buf, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let rgba =
            bytemuck::cast_slice::<u8, [f32; 4]>(&slice.get_mapped_range().expect("mapped colors"))
                .to_vec();
        staging.unmap();
        (rgba.iter().map(|c| [c[0], c[1], c[2]]).collect(), best)
    }

    /// Static scene buffers for the render kernels, by binding: light (0),
    /// triangles (2), mesh ranges (3), debug (5), albedo (8), hierarchies (9,
    /// 10, 25, 26), shading normals (24: the lightmap's, else flat), and with a
    /// lightmap its charts, texels, probe irradiance and chart coordinates (11-13,
    /// 21, 23).
    fn scene_buffers(
        &self,
        frame: &Frame,
        lightmap: Option<&Lightmap>,
    ) -> Vec<(u32, wgpu::Buffer)> {
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let mut uniform = frame.light.uniform(
            frame.meshes.len(),
            0,
            frame.hierarchy,
            self.prefilter.get(),
            self.nearest_first.get(),
            self.profile_stage.get(),
            (
                self.sky.get(),
                self.sky_max.get(),
                self.sky_prefilter.get(),
                self.sky_horizon_cull.get(),
                self.sky_object_frame.get(),
            ),
            hierarchy::NONE,
        );
        self.environment_uniform(&mut uniform, hierarchy::NONE);
        let flat = flatten(frame.meshes);
        let (nodes, items) = mesh_hierarchy(frame.meshes);
        let albedo4: Vec<[f32; 4]> = frame.albedo.iter().map(|&a| v4(a)).collect();
        let storage = |label: &str, contents: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
        };
        let mut buffers = vec![
            (
                0,
                uniform_buffer(&self.device, "light", bytemuck::bytes_of(&uniform)),
            ),
            (2, storage("tris", bytemuck::cast_slice(&flat.tris))),
            (3, storage("meshes", bytemuck::cast_slice(&flat.ranges))),
            (
                25,
                storage("triangle nodes", bytemuck::cast_slice(&flat.blas)),
            ),
            (
                26,
                storage("triangle items", bytemuck::cast_slice(&flat.blas_items)),
            ),
            (
                29,
                storage("node coverage", bytemuck::cast_slice(&flat.blas_cov)),
            ),
            // No transforms: every object's triangles are in world space.
            (30, storage("transforms", &[0; 64])),
            (5, storage("debug", &[0; 256])),
            (8, storage("albedo", bytemuck::cast_slice(&albedo4))),
            (9, storage("nodes", bytemuck::cast_slice(&nodes))),
            (10, storage("items", bytemuck::cast_slice(&items))),
        ];
        let flat_normals: Vec<[f32; 4]>;
        let normals: &[[f32; 4]] = match lightmap {
            Some(lm) if !lm.tri_normals.is_empty() => &lm.tri_normals,
            _ => {
                flat_normals = frame
                    .meshes
                    .iter()
                    .flatten()
                    .flat_map(|t| [v4((t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero()); 3])
                    .chain(std::iter::once([0.0; 4]))
                    .collect();
                &flat_normals
            }
        };
        buffers.push((24, storage("normals", bytemuck::cast_slice(normals))));
        buffers.push((
            52,
            self.lights_buffer(&[Light::white(frame.light, frame.radiance)]),
        ));
        // No textures: every object NONE.
        let none = vec![hierarchy::NONE; frame.meshes.len().max(1)];
        buffers.push((57, storage("object textures", bytemuck::cast_slice(&none))));
        buffers.push((58, storage("texture coordinates", &[0; 8])));
        if let Some(lm) = lightmap {
            let texels: &[f32] = if lm.irradiance.is_empty() {
                &[0.0]
            } else {
                &lm.irradiance
            };
            let charts: &[lightmap::ChartGpu] = if lm.charts.is_empty() {
                &[lightmap::ChartGpu::zeroed()]
            } else {
                &lm.charts
            };
            let tri_uv: &[[f32; 2]] = if lm.tri_uv.is_empty() {
                &[[0.0; 2]]
            } else {
                &lm.tri_uv
            };
            let indirect: &[[f32; 4]] =
                if lm.indirect.len() == lm.probes.len() && !lm.probes.is_empty() {
                    &lm.indirect
                } else if lm.probes.is_empty() {
                    &[[0.0; 4]]
                } else {
                    // Probes without values yet: all invalid, so lookups add nothing.
                    &vec![[0.0; 4]; lm.probes.len()]
                };
            buffers.push((
                11,
                storage("tri_chart", bytemuck::cast_slice(&lm.tri_chart)),
            ));
            buffers.push((12, storage("charts", bytemuck::cast_slice(charts))));
            buffers.push((23, storage("tri_uv", bytemuck::cast_slice(tri_uv))));
            buffers.push((13, storage("texels", bytemuck::cast_slice(texels))));
            buffers.push((
                21,
                storage("probe irradiance", bytemuck::cast_slice(indirect)),
            ));
        }
        buffers
    }

    /// Per-bin coverage (64 per receiver) as the kernel computed it, for debugging.
    #[doc(hidden)]
    pub fn coverage(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
    ) -> Vec<[f32; 64]> {
        let (_, debug, _) = self.run(receivers, light, meshes, 1, 1, Hierarchy::None);
        debug.chunks(64).map(|c| c.try_into().unwrap()).collect()
    }

    /// Clipped polygons as the kernel drew them for the first receiver: each
    /// entry is `[n, x0, y0, x1, y1, ...]` (21 floats), for debugging.
    #[doc(hidden)]
    pub fn polygons(
        &self,
        receiver: (Vec3, Vec3),
        light: RectLight,
        meshes: &[Vec<Tri>],
    ) -> Vec<Vec<f32>> {
        let (_, debug, _) = self.run(&[receiver], light, meshes, 1, 3, Hierarchy::None);
        debug
            .as_chunks::<21>()
            .0
            .iter()
            .take_while(|c| c[0] > 0.0)
            .map(|c| c[..1 + 2 * c[0] as usize].to_vec())
            .collect()
    }

    /// The last mesh's raw accumulation buffer (first 64 cells), for debugging.
    #[doc(hidden)]
    pub fn accumulator(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
    ) -> Vec<[f32; 64]> {
        let (_, debug, _) = self.run(receivers, light, meshes, 1, 2, Hierarchy::None);
        debug.chunks(64).map(|c| c.try_into().unwrap()).collect()
    }

    fn run(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        runs: usize,
        debug_mode: u32,
        hierarchy: Hierarchy,
    ) -> (Vec<f32>, Vec<f32>, Duration) {
        let main = self.entry("main");
        let debug = debug_mode != 0;
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let mut uniform = light.uniform(
            meshes.len(),
            debug_mode,
            hierarchy,
            self.prefilter.get(),
            self.nearest_first.get(),
            self.profile_stage.get(),
            (
                self.sky.get(),
                self.sky_max.get(),
                self.sky_prefilter.get(),
                self.sky_horizon_cull.get(),
                self.sky_object_frame.get(),
            ),
            hierarchy::NONE,
        );
        self.environment_uniform(&mut uniform, hierarchy::NONE);
        let receivers_gpu: Vec<ReceiverGpu> = receivers
            .iter()
            .map(|&(p, n)| ReceiverGpu {
                point: v4(p),
                normal: v4(n),
            })
            .collect();
        let flat = flatten(meshes);
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let light_buf = init(
            "light",
            bytemuck::bytes_of(&uniform),
            wgpu::BufferUsages::UNIFORM,
        );
        let receivers_buf = init(
            "receivers",
            bytemuck::cast_slice(&receivers_gpu),
            wgpu::BufferUsages::STORAGE,
        );
        let tris_buf = init(
            "tris",
            bytemuck::cast_slice(&flat.tris),
            wgpu::BufferUsages::STORAGE,
        );
        let meshes_buf = init(
            "meshes",
            bytemuck::cast_slice(&flat.ranges),
            wgpu::BufferUsages::STORAGE,
        );
        let blas_buf = init(
            "triangle nodes",
            bytemuck::cast_slice(&flat.blas),
            wgpu::BufferUsages::STORAGE,
        );
        let blas_items_buf = init(
            "triangle items",
            bytemuck::cast_slice(&flat.blas_items),
            wgpu::BufferUsages::STORAGE,
        );
        let blas_cov_buf = init(
            "node coverage",
            bytemuck::cast_slice(&flat.blas_cov),
            wgpu::BufferUsages::STORAGE,
        );
        let xforms_buf = init("transforms", &[0; 64], wgpu::BufferUsages::STORAGE);
        let (nodes, items) = mesh_hierarchy(meshes);
        let nodes_buf = init(
            "nodes",
            bytemuck::cast_slice(&nodes),
            wgpu::BufferUsages::STORAGE,
        );
        let items_buf = init(
            "items",
            bytemuck::cast_slice(&items),
            wgpu::BufferUsages::STORAGE,
        );
        let size = (receivers.len() * 4) as u64;
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let debug_size = if debug { (size * 64).max(65536) } else { 256 };
        let debug_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug"),
            size: debug_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: size + debug_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask_cones"),
            layout: &main.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: receivers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tris_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meshes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: debug_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: items_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: blas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: blas_items_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: blas_cov_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: xforms_buf.as_entire_binding(),
                },
            ],
        });
        let groups = receivers.len().div_ceil(64) as u32;
        let mut best = Duration::MAX;
        for _ in 0..runs.max(1) {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&main);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            let start = Instant::now();
            self.queue.submit([encoder.finish()]);
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
            best = best.min(start.elapsed());
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, size);
        encoder.copy_buffer_to_buffer(&debug_buf, 0, &staging, size, debug_size);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let all =
            bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range().expect("mapped result"))
                .to_vec();
        staging.unmap();
        let (values, debug_values) = all.split_at(receivers.len());
        (
            values.to_vec(),
            if debug {
                debug_values.to_vec()
            } else {
                Vec::new()
            },
            best,
        )
    }
}

fn camera_uniform(camera: Camera, width: u32, height: u32, radiance: f32) -> CameraUniform {
    let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
    let forward = (camera.target - camera.eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let up = right.cross(forward);
    CameraUniform {
        eye: v4(camera.eye),
        forward: v4(forward),
        right: v4(right),
        up: v4(up),
        params: [
            width as f32,
            height as f32,
            (camera.vertical_fov_degrees.to_radians() * 0.5).tan(),
            radiance,
        ],
    }
}

/// Shared geometry placed many times (instancing), in a world that may be far
/// larger than f32 positions can resolve: positions are f64 and are uploaded
/// relative to a render origin.
#[derive(Clone, Default)]
pub struct InstancedScene {
    /// Unique geometries, each with per-vertex shading normals.
    pub geometries: Vec<Surface>,
    /// Placements of the geometries.
    pub instances: Vec<Instance>,
    /// Base color textures (M6), multiplying instances' albedo where their
    /// geometry has one and texture coordinates (`Surface::tex`).
    pub textures: Vec<Texture>,
    /// Each geometry's texture, if any (empty: none).
    pub geometry_textures: Vec<Option<u32>>,
    /// Each geometry's material, if any (empty or `None`: Lambertian). Used
    /// by the cache's shading (M10 gate 2).
    pub geometry_materials: Vec<Option<Material>>,
    /// Each geometry's emitted radiance (linear RGB, added to what it
    /// reflects; empty: none). Shown by the cache's shading, directly and in
    /// reflections. With bounce light (`CacheOptions::indirect` and
    /// `emission_bounce`, M10 gate 11) it lights other surfaces, one bounce
    /// deep; otherwise use a light for that.
    /// Stored as half floats (at most 65,504).
    pub geometry_emission: Vec<Vec3>,
    /// Each geometry's emissive texture, if any (empty: none; M10 gate 14):
    /// an index into `textures` (sRGB, glTF's emissive map) multiplying its
    /// emission, read through its texture coordinates.
    pub geometry_emissive_textures: Vec<Option<u32>>,
    /// Each geometry's opacity if it is alpha-blended (empty: none; M10
    /// gate 16): Some(factor), times its base color texture's alpha where it
    /// has one. With hardware camera rays, pixels see through blended
    /// surfaces (a few layers, composited front to back); shadows from them
    /// keep the factor's share of the light. glTF's BLEND materials.
    pub geometry_blend: Vec<Option<f32>>,
    /// Each geometry's normal map, if any (empty: none; M10 gate 5): an
    /// index into `textures` of a tangent-space normal map (linear, OpenGL's
    /// convention as glTF specifies: green toward the image's top), read
    /// through the geometry's texture coordinates. Used by the cache's
    /// shading for light from lights, specular and reflections; the cached
    /// sky and bounce light keep the interpolated normal.
    pub geometry_normal_maps: Vec<Option<u32>>,
    /// Each geometry's metallic-roughness texture, if any (empty: none): an
    /// index into `textures` (linear; glTF's layout, roughness in green and
    /// metallic in blue), multiplying its material's factors. Ignored for a
    /// geometry without a material.
    pub geometry_metallic_roughness: Vec<Option<u32>>,
    /// Each geometry's alpha cutoff, if any (empty: none; M10 gate 6): where
    /// its base color texture's alpha is below it, the geometry is cut
    /// away, as geometry, before upload ([`cut_out`]), so camera rays,
    /// shadows, sky, the cache and bounce light all see the cut. Ignored
    /// for a geometry without a texture.
    pub geometry_alpha_cutoff: Vec<Option<f32>>,
    /// Each geometry's vertex colors, if any (empty: none; M10 gate 17): a
    /// linear RGB color per triangle corner (as `Surface::tex`), multiplying
    /// its base color and texture, interpolated across each triangle. Kept
    /// at 8 bits per channel (sRGB-encoded), so within `[0, 1]`. glTF's
    /// `COLOR_0`. Ignored unless it has one entry per triangle.
    pub geometry_colors: Vec<Option<Vec<[Vec3; 3]>>>,
    /// Each texture's wrap modes along u and v, as glTF samplers give them
    /// (empty, or past its end: repeat; M10 gate 20).
    pub texture_wrap: Vec<[Wrap; 2]>,
    /// Each geometry's finer camera levels, if any (empty: none; M14 gate
    /// 1): camera rays see the finest one resident, streamed by need within
    /// a budget ([`StreamOptions`], with hardware camera rays); lighting
    /// keeps the geometry itself.
    pub geometry_levels: Vec<Option<Levels>>,
    /// Gives the camera levels whose surfaces are left empty in
    /// `geometry_levels`, on demand (M14 gate 3).
    pub level_source: Option<std::sync::Arc<dyn LevelSource>>,
}

/// How a texture continues past its edges (glTF sampler wrap modes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Wrap {
    /// Tiles (glTF `REPEAT`, the default).
    #[default]
    Repeat,
    /// Holds the edge texels (glTF `CLAMP_TO_EDGE`).
    Clamp,
    /// Tiles mirrored every other time (glTF `MIRRORED_REPEAT`).
    Mirror,
}

impl Wrap {
    /// Its two-bit code in the kernel's wrap words.
    fn code(self) -> u32 {
        match self {
            Wrap::Repeat => 0,
            Wrap::Clamp => 1,
            Wrap::Mirror => 2,
        }
    }

    /// Texel index `i` of a row or column of `n` texels under this mode.
    pub(crate) fn texel(self, i: i64, n: i64) -> i64 {
        match self {
            Wrap::Repeat => i.rem_euclid(n),
            Wrap::Clamp => i.clamp(0, n - 1),
            Wrap::Mirror => {
                let k = i.rem_euclid(2 * n);
                if k < n { k } else { 2 * n - 1 - k }
            }
        }
    }
}

impl InstancedScene {
    /// One less geometry `g`'s opacity if it is alpha-blended (0: opaque;
    /// M10 gate 16), as transforms carry it for shadows.
    pub(crate) fn blend_clear(&self, g: usize) -> f32 {
        self.geometry_blend
            .get(g)
            .copied()
            .flatten()
            .map_or(0.0, |o| 1.0 - o.clamp(0.0, 1.0))
    }
    /// The scene with alpha blending replaced by its fallback (M10 gate 16):
    /// a textured blended geometry cut at an alpha of 0.5 over its opacity
    /// factor, an untextured one opaque. For renderers without blending
    /// (no hardware camera rays, movers, or no cache).
    pub(crate) fn blend_as_mask(&self) -> InstancedScene {
        let mut s = self.clone();
        let n = s.geometries.len();
        s.geometry_alpha_cutoff.resize(n, None);
        for g in 0..n {
            if let Some(o) = s.geometry_blend.get(g).copied().flatten()
                && s.geometry_textures.get(g).copied().flatten().is_some()
                && s.geometry_alpha_cutoff[g].is_none()
            {
                s.geometry_alpha_cutoff[g] = Some(0.5 / o.max(1e-6));
            }
        }
        s.geometry_blend.clear();
        s
    }

    /// Emissive faces made area lights (M10 gate 23). Each geometry with
    /// uniform emission (no emissive texture, not blended) whose triangles
    /// all pair into planar rectangles, at most `max_faces` of them, becomes
    /// one one-sided rectangular light per rectangle and instance: facing out
    /// of the rectangle's front (its triangles' winding), at the emission's
    /// radiance, 1 mm in front of it. Its emission is set to zero, so its
    /// light is not counted twice: the lights show it to the camera and
    /// light the scene, with exact shadows, and it still occludes. Returns
    /// the lights, relative to `origin`, to add to the scene's (the
    /// renderer takes at most [`MAX_LIGHTS`] in all).
    pub fn emitters_to_lights(&mut self, origin: DVec3, max_faces: usize) -> Vec<Light> {
        let mut lights = Vec::new();
        for g in 0..self.geometries.len() {
            let e = self.geometry_emission.get(g).copied().unwrap_or(Vec3::ZERO);
            let textured = self
                .geometry_emissive_textures
                .get(g)
                .copied()
                .flatten()
                .is_some();
            let blended = self.geometry_blend.get(g).copied().flatten().is_some();
            if e.max_element() <= 0.0 || textured || blended {
                continue;
            }
            let tris = &self.geometries[g].triangles;
            if tris.len() > 2 * max_faces {
                continue;
            }
            let Some(rects) = rectangles(tris) else {
                continue;
            };
            for i in self.instances.iter().filter(|i| i.geometry == g) {
                let to_world =
                    |p: Vec3| i.position - origin + (i.rotation * (p * i.scale)).as_dvec3();
                for &(c, u, v) in &rects {
                    let (wu, wv) = (i.rotation * u * i.scale, i.rotation * v * i.scale);
                    let normal = wu.cross(wv).normalize_or_zero();
                    let corner = to_world(c) + (normal * 1e-3).as_dvec3();
                    lights.push(Light::rect(
                        RectLight {
                            corner: corner.as_vec3(),
                            edge_u: wu,
                            edge_v: wv,
                            one_sided: true,
                        },
                        e,
                    ));
                }
            }
            self.geometry_emission[g] = Vec3::ZERO;
        }
        lights
    }

    /// This scene with its alpha cutouts made geometry ([`cut_out`]):
    /// borrowed when it has none.
    pub fn with_cutouts(&self) -> std::borrow::Cow<'_, InstancedScene> {
        let masked: Vec<(usize, f32, u32)> = (0..self.geometries.len())
            .filter_map(|g| {
                let cutoff = self.geometry_alpha_cutoff.get(g).copied().flatten()?;
                let texture = self.geometry_textures.get(g).copied().flatten()?;
                let s = &self.geometries[g];
                (s.tex.len() == s.triangles.len()).then_some((g, cutoff, texture))
            })
            .collect();
        if masked.is_empty() {
            return std::borrow::Cow::Borrowed(self);
        }
        let mut scene = self.clone();
        for (g, cutoff, texture) in masked {
            let colors = self.geometry_colors.get(g).and_then(|c| c.as_deref());
            let (surface, colors) = cutout::cut_out_colored(
                &self.geometries[g],
                colors,
                &self.textures[texture as usize],
                self.texture_wrap
                    .get(texture as usize)
                    .copied()
                    .unwrap_or_default(),
                cutoff,
            );
            scene.geometries[g] = surface;
            if let Some(colors) = colors {
                scene.geometry_colors[g] = Some(colors);
            }
        }
        scene.geometry_alpha_cutoff.clear();
        std::borrow::Cow::Owned(scene)
    }

    /// The textures read linearly (normal maps and metallic-roughness
    /// textures), each once, in order.
    pub(crate) fn linear_textures(&self) -> Vec<u32> {
        let mut used: Vec<u32> = (0..self.geometries.len())
            .filter(|&g| self.geometries[g].tex.len() == self.geometries[g].triangles.len())
            .flat_map(|g| {
                [
                    &self.geometry_normal_maps,
                    &self.geometry_metallic_roughness,
                ]
                .map(|maps| maps.get(g).copied().flatten())
            })
            .flatten()
            .collect();
        used.sort_unstable();
        used.dedup();
        used
    }
}

/// Triangles paired into planar rectangles, each as a corner and two
/// perpendicular edges whose cross product points out of the triangles'
/// front; None unless every triangle pairs (M10 gate 23).
fn rectangles(tris: &[continuum::mesh::Tri]) -> Option<Vec<(Vec3, Vec3, Vec3)>> {
    if tris.is_empty() || !tris.len().is_multiple_of(2) {
        return None;
    }
    let normal = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]);
    let mut used = vec![false; tris.len()];
    let mut out = Vec::with_capacity(tris.len() / 2);
    for i in 0..tris.len() {
        if used[i] {
            continue;
        }
        let a = tris[i];
        let size = (a[1] - a[0]).length().max((a[2] - a[0]).length());
        let eps = 1e-5 * size.max(1e-6);
        let same = |p: Vec3, q: Vec3| (p - q).length() <= eps;
        let mut found = None;
        for j in i + 1..tris.len() {
            if used[j]
                || normal(&a)
                    .normalize_or_zero()
                    .dot(normal(&tris[j]).normalize_or_zero())
                    < 0.9999
            {
                continue;
            }
            let b = tris[j];
            let shared: Vec<Vec3> = a
                .iter()
                .copied()
                .filter(|&p| b.iter().any(|&q| same(p, q)))
                .collect();
            if shared.len() != 2 {
                continue;
            }
            let c = a
                .iter()
                .copied()
                .find(|&p| !shared.iter().any(|&s| same(p, s)))?;
            let d = b
                .iter()
                .copied()
                .find(|&q| !shared.iter().any(|&s| same(q, s)))?;
            let (mut u, mut v) = (shared[0] - c, shared[1] - c);
            if u.dot(v).abs() > 1e-4 * u.length() * v.length() || !same(d, c + u + v) {
                continue;
            }
            if u.cross(v).dot(normal(&a)) < 0.0 {
                std::mem::swap(&mut u, &mut v);
            }
            found = Some((j, (c, u, v)));
            break;
        }
        let (j, rect) = found?;
        used[i] = true;
        used[j] = true;
        out.push(rect);
    }
    Some(out)
}

/// A vertex color's code (M10 gate 17): 1 plus its sRGB bytes, red lowest,
/// as an exact integer in an f32 (at most 2^24).
pub(crate) fn color_code(c: Vec3) -> f32 {
    let byte = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        let s = if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u32
    };
    (1 + byte(c.x) + (byte(c.y) << 8) + (byte(c.z) << 16)) as f32
}

/// `x` (clamped to 0..=65,504) as half-float bits, truncated; values below
/// the smallest normal half are zero.
fn half_bits(x: f32) -> u32 {
    let x = x.clamp(0.0, 65_504.0);
    if x < 6.104e-5 {
        return 0;
    }
    let b = x.to_bits();
    let exponent = ((b >> 23) & 0xff) as i32 - 127 + 15;
    (exponent as u32) << 10 | (b >> 13) & 0x3ff
}

/// glTF's metallic-roughness model (M10 gate 2): a dielectric (metallic 0,
/// specular reflectance 0.04) or a metal (metallic 1, reflecting its base
/// color), with GGX microfacets of α = roughness².
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// 0 for a dielectric, 1 for a metal (between: a blend).
    pub metallic: f32,
    /// Perceptual roughness, 0 (a mirror) to 1.
    pub roughness: f32,
}

/// A base color texture: sRGB-encoded RGBA8, rows from the top (as glTF
/// images), sampled with repeat.
#[derive(Clone, Debug)]
pub struct Texture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height` pixels of four bytes each.
    pub rgba: Vec<u8>,
}

/// Textures are resampled into layers of this size, with mip levels.
pub const TEXTURE_SIZE: u32 = 1024;

/// A scene's textures as the GPU samples them (bindings 55 and 56).
pub(crate) struct SceneTextures {
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
    /// Normal maps (linear), M10 gate 5.
    pub(crate) normals: wgpu::TextureView,
}

/// One placement of a geometry: `world = scale · rotation · p + position`.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    /// Index into [`InstancedScene::geometries`].
    pub geometry: usize,
    /// Orthonormal.
    pub rotation: Mat3,
    /// Uniform scale.
    pub scale: f32,
    /// World position (f64).
    pub position: DVec3,
    /// Diffuse reflectance (linear RGB), multiplying the geometry's texture
    /// where it has one.
    pub albedo: Vec3,
}

/// A texture resampled to `size × size` (bilinear, repeating), then its mip
/// levels down to 1×1, each averaging 2×2 texels in linear light. Returns
/// sRGB-encoded RGBA8 per level.
fn texture_levels(t: &Texture, size: u32, linear: bool) -> Vec<Vec<u8>> {
    let to_linear = |c: u8| {
        let c = c as f32 / 255.0;
        if linear {
            c
        } else if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_srgb = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if linear {
            c
        } else if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    let (w, h) = (t.width.max(1) as usize, t.height.max(1) as usize);
    let texel = |x: usize, y: usize| {
        let i = 4 * ((y % h) * w + (x % w));
        [
            to_linear(t.rgba[i]),
            to_linear(t.rgba[i + 1]),
            to_linear(t.rgba[i + 2]),
            t.rgba[i + 3] as f32 / 255.0,
        ]
    };
    let side = size as usize;
    let mut level: Vec<[f32; 4]> = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            // Texel centers mapped to the source, bilinear.
            let u = (x as f32 + 0.5) / side as f32 * w as f32 - 0.5;
            let v = (y as f32 + 0.5) / side as f32 * h as f32 - 0.5;
            let (x0, y0) = (u.floor(), v.floor());
            let (fx, fy) = (u - x0, v - y0);
            let (xi, yi) = (
                (x0 as isize).rem_euclid(w as isize) as usize,
                (y0 as isize).rem_euclid(h as isize) as usize,
            );
            let mut c = [0.0f32; 4];
            for (dx, dy, wt) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let s = texel(xi + dx, yi + dy);
                for k in 0..4 {
                    c[k] += s[k] * wt;
                }
            }
            level.push(c);
        }
    }
    let encode = |level: &[[f32; 4]]| -> Vec<u8> {
        level
            .iter()
            .flat_map(|c| {
                [
                    to_srgb(c[0]),
                    to_srgb(c[1]),
                    to_srgb(c[2]),
                    (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect()
    };
    let mut out = vec![encode(&level)];
    let mut side = side;
    while side > 1 {
        let half = side / 2;
        let mut next = Vec::with_capacity(half * half);
        for y in 0..half {
            for x in 0..half {
                let mut c = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let s = level[(2 * y + dy) * side + 2 * x + dx];
                    for k in 0..4 {
                        c[k] += s[k] * 0.25;
                    }
                }
                next.push(c);
            }
        }
        level = next;
        side = half;
        out.push(encode(&level));
    }
    out
}

fn uniform_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

#[cfg(test)]
mod emitter_tests {
    use super::*;

    /// A glowing box becomes six one-sided lights, one per face, facing
    /// out, covering its surface, placed by the instance, and the box's
    /// emission is set to zero (M10 gate 23).
    #[test]
    fn a_glowing_box_becomes_six_outward_lights() {
        let (lo, hi) = (Vec3::new(-0.6, -0.04, 0.0), Vec3::new(0.6, 0.04, 0.9));
        let glow = Vec3::new(3.0, 1.86, 0.9);
        let mut scene = InstancedScene {
            geometries: vec![Surface::cuboid(lo, hi)],
            instances: vec![Instance {
                geometry: 0,
                rotation: Mat3::from_rotation_z(0.3),
                scale: 2.0,
                position: DVec3::new(10.0, 5.0, 1.0),
                albedo: Vec3::splat(0.8),
            }],
            geometry_emission: vec![glow],
            ..Default::default()
        };
        let origin = DVec3::new(9.0, 4.0, 0.0);
        let lights = scene.emitters_to_lights(origin, 6);
        assert_eq!(lights.len(), 6);
        assert_eq!(scene.geometry_emission[0], Vec3::ZERO);
        let size = (hi - lo) * 2.0;
        let area = 2.0 * (size.x * size.y + size.y * size.z + size.z * size.x);
        let i = &scene.instances[0];
        let center = (i.position - origin).as_vec3() + i.rotation * ((lo + hi) * 0.5 * 2.0);
        let mut total = 0.0;
        for l in &lights {
            let Emitter::Rect(r) = l.emitter else {
                panic!("a rectangle")
            };
            assert!(r.one_sided);
            assert_eq!(l.radiance, glow);
            assert!(r.edge_u.dot(r.edge_v).abs() < 1e-4);
            let n = r.edge_u.cross(r.edge_v);
            total += n.length();
            let middle = r.corner + 0.5 * (r.edge_u + r.edge_v);
            assert!(n.dot(middle - center) > 0.0, "a light facing in");
            // 1 mm outside its face: its plane's distance from the centre
            // is half the box's extent along its normal, plus 1 mm.
            let normal = n.normalize();
            let half = 0.5 * (i.rotation.transpose() * normal).abs().dot(size);
            assert!(((middle - center).dot(normal) - half - 1e-3).abs() < 1e-4);
        }
        assert!((total - area).abs() < 1e-4 * area, "{total} against {area}");
        // A box with more faces than allowed is left as it is.
        let mut kept = InstancedScene {
            geometry_emission: vec![glow],
            ..scene.clone()
        };
        assert!(kept.emitters_to_lights(origin, 5).is_empty());
        assert_eq!(kept.geometry_emission[0], glow);
    }
}
