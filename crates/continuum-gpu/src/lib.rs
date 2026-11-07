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
    /// object-level prefiltering), 0, 0, 0]`
    objects: [u32; 4],
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
    pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::ComputePipeline,
    cached_pipeline: wgpu::ComputePipeline,
    mark_pipeline: wgpu::ComputePipeline,
    rebake_pipeline: wgpu::ComputePipeline,
    probe_build_pipeline: wgpu::ComputePipeline,
    probe_gather_pipeline: wgpu::ComputePipeline,
    bake_static_pipeline: wgpu::ComputePipeline,
    visibility_pipeline: wgpu::ComputePipeline,
    /// Camera rays through hardware ray queries, if the device has them.
    hw_visibility_pipeline: Option<wgpu::ComputePipeline>,
    hardware_rays: std::cell::Cell<bool>,
    shade_pipeline: wgpu::ComputePipeline,
    shade_cached_pipeline: wgpu::ComputePipeline,
    /// The view-dependent lighting cache's passes.
    vc_update_pipeline: wgpu::ComputePipeline,
    vc_bake_pipeline: wgpu::ComputePipeline,
    vc_bake_sky_pipeline: wgpu::ComputePipeline,
    vc_self_bake_pipeline: wgpu::ComputePipeline,
    vc_fill_pipeline: wgpu::ComputePipeline,
    /// An empty pass that timestamps the end of the camera passes.
    mark_pipeline_empty: wgpu::ComputePipeline,
    vc_shade_pipeline: wgpu::ComputePipeline,
    render_cones_pipeline: wgpu::ComputePipeline,
    vc_shade_cones_pipeline: wgpu::ComputePipeline,
    pixel_cones: std::cell::Cell<bool>,
    prefilter: std::cell::Cell<f32>,
    sky: std::cell::Cell<Vec3>,
    sky_max: std::cell::Cell<bool>,
    sky_prefilter: std::cell::Cell<f32>,
    sky_horizon_cull: std::cell::Cell<bool>,
    sky_object_frame: std::cell::Cell<bool>,
    object_prefilter: std::cell::Cell<bool>,
    nearest_first: std::cell::Cell<bool>,
    profile_stage: std::cell::Cell<u32>,
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

mod hierarchy;
mod hw;
mod instanced;
mod lightmap;
mod live;
mod renderer;
mod vcache;

pub use lightmap::Lightmap;
pub use live::{LiveRenderer, MAX_MOVING, MAX_SWEPT, PROBE_RESOLUTION};
pub use renderer::{Renderer, RendererOptions};
pub use vcache::{CacheOptions, CacheStats};

// Experiment-only: used by the gates in `experiments/`, not part of the
// library's surface.
#[doc(hidden)]
pub use hierarchy::Hierarchy;
#[doc(hidden)]
pub use instanced::{FrameProfile, InstancedRenderer};
#[doc(hidden)]
pub use lightmap::{BLOCK, BLOCK_PLANES, hemicube_weights};
#[doc(hidden)]
pub use vcache::CachePage;

use hierarchy::{Placement, flatten, flatten_instanced, mesh_hierarchy, object_hierarchy};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    params: [f32; 4],
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

/// Bindings the `visibility` pass uses.
const VISIBILITY_BINDINGS: &[u32] = &[0, 2, 3, 6, 9, 10, 25, 26, 30, 31, 52];

/// Bindings the hardware `hw_visibility` pass uses (32: the acceleration structure).
pub(crate) const HW_VISIBILITY_BINDINGS: &[u32] = &[3, 6, 31, 32, 52];

/// Bindings of the lighting cache's passes.
pub(crate) const VC_UPDATE_BINDINGS: &[u32] = &[36, 41, 44];
pub(crate) const VC_BAKE_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 9, 10, 23, 24, 25, 26, 29, 30, 33, 34, 54, 37, 38, 41, 42, 43, 52,
];
pub(crate) const VC_BAKE_SKY_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 9, 10, 23, 24, 25, 26, 29, 30, 33, 34, 54, 38, 41, 42, 46, 47, 49, 51,
];
pub(crate) const VC_SELF_BAKE_BINDINGS: &[u32] =
    &[0, 2, 3, 5, 23, 24, 25, 26, 29, 30, 33, 34, 54, 49, 50, 51];
pub(crate) const VC_FILL_BINDINGS: &[u32] = &[3, 30, 33, 37, 38, 40, 41, 46, 47, 48];
pub(crate) const VC_SHADE_BINDINGS: &[u32] = &[
    0, 2, 3, 6, 7, 8, 23, 24, 30, 31, 33, 35, 36, 37, 39, 40, 41, 42, 45, 52, 55, 56, 57, 58,
];
/// Bindings of `vc_shade_cones`: the cache's shading with pixel cones for
/// visibility (the scene's hierarchies instead of a visibility buffer).
pub(crate) const VC_SHADE_CONES_BINDINGS: &[u32] = &[
    0, 2, 3, 5, 6, 7, 8, 9, 10, 23, 24, 25, 26, 29, 30, 33, 35, 36, 37, 39, 40, 41, 42, 45, 52, 55,
    56, 57, 58,
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
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        let constants = [
            ("WORKGROUP_STACKS", flag(options.workgroup_stacks)),
            ("COUNT_WORK", flag(options.count_work)),
            ("PACKETS", 1.0),
        ];
        // The view cache's bakes walk alone: their texels can be far apart,
        // and packets made small bakes slower (M5 gate 5).
        let solo_constants = [constants[0], constants[1], ("PACKETS", 0.0)];
        let entry_with = |name: &str, constants: &[(&str, f64)]| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let entry = |name: &str| entry_with(name, &constants);
        let pipeline = entry("main");
        let render_pipeline = entry("render");
        let cached_pipeline = entry("render_cached");
        let mark_pipeline = entry("mark_blocks");
        let rebake_pipeline = entry("rebake");
        let probe_build_pipeline = entry("probe_build");
        let probe_gather_pipeline = entry("probe_gather");
        let bake_static_pipeline = entry("bake_static");
        let visibility_pipeline = entry("visibility");
        let shade_pipeline = entry("shade");
        let shade_cached_pipeline = entry("shade_cached_pass");
        let vc_update_pipeline = entry("vc_update");
        let vc_bake_pipeline = entry_with("vc_bake", &solo_constants);
        let vc_bake_sky_pipeline = entry_with("vc_bake_sky", &solo_constants);
        let vc_self_bake_pipeline = entry_with("vc_self_bake", &solo_constants);
        let vc_fill_pipeline = entry("vc_fill");
        let mark_pipeline_empty = entry("mark");
        let vc_shade_pipeline = entry("vc_shade");
        let render_cones_pipeline = entry("render_cones");
        let vc_shade_cones_pipeline = entry("vc_shade_cones");
        let hw_visibility_pipeline = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .then(|| {
                let hw_source = format!(
                    "enable wgpu_ray_query;\n{source}\n{}",
                    include_str!("hw_visibility.wgsl")
                );
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("hw_visibility"),
                    source: wgpu::ShaderSource::Wgsl(hw_source.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("hw_visibility"),
                    layout: None,
                    module: &module,
                    entry_point: Some("hw_visibility"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                })
            });
        MaskConeKernel {
            device,
            queue,
            pipeline,
            render_pipeline,
            cached_pipeline,
            mark_pipeline,
            rebake_pipeline,
            probe_build_pipeline,
            probe_gather_pipeline,
            bake_static_pipeline,
            visibility_pipeline,
            hardware_rays: std::cell::Cell::new(hw_visibility_pipeline.is_some()),
            hw_visibility_pipeline,
            shade_pipeline,
            shade_cached_pipeline,
            vc_update_pipeline,
            vc_bake_pipeline,
            vc_bake_sky_pipeline,
            vc_self_bake_pipeline,
            vc_fill_pipeline,
            mark_pipeline_empty,
            vc_shade_pipeline,
            render_cones_pipeline,
            vc_shade_cones_pipeline,
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
            split: std::cell::Cell::new(false),
            adapter_name,
        }
    }

    /// The device the kernel runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
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
        self.hw_visibility_pipeline.is_some()
    }

    /// Camera rays for instanced scenes through hardware ray queries (the
    /// default where available) or the compute hierarchy. Applies to
    /// renderers created after the call.
    pub fn set_hardware_rays(&self, on: bool) {
        self.hardware_rays.set(on && self.has_hardware_rays());
    }

    /// Debugging: cut cones short to time their stages (0: off, 1: no cone,
    /// 2: setup and weights only, 3: plus the object hierarchy, 4: plus triangle
    /// hierarchies but no rasterization). Results are wrong unless 0.
    #[doc(hidden)]
    pub fn set_profile_stage(&self, stage: u32) {
        self.profile_stage.set(stage);
    }

    /// Primary visibility through pixel cones (M7) instead of one camera
    /// ray per pixel, in instanced renderers created after the call.
    #[doc(hidden)]
    pub fn set_pixel_cones(&self, on: bool) {
        self.pixel_cones.set(on);
    }

    pub(crate) fn uses_hardware_rays(&self) -> bool {
        self.hardware_rays.get() && self.hw_visibility_pipeline.is_some()
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
        let buffers = self.instanced_buffers(lights, scene, origin);
        let bytes = buffers.iter().map(|(_, b)| b.size()).sum();
        let cam = camera_uniform(camera, width, height, lights[0].radiance.x);
        let hw_scene = self
            .uses_hardware_rays()
            .then(|| self.hw_scene(scene, origin));
        let (pixels, time) = self.render_buffers_with(
            false,
            cam,
            width,
            height,
            buffers,
            hw_scene.as_ref().map(|h| &h.tlas),
            &self.scene_textures(&scene.textures),
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
    pub(crate) fn instanced_buffers(
        &self,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
    ) -> Vec<(u32, wgpu::Buffer)> {
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
            })
            .collect();
        let (mut flat, xforms, bounds) = flatten_instanced(&geometries, &placements);
        let (nodes, items) = object_hierarchy(&bounds, |_| true);
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
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let normals: Vec<[f32; 4]> = scene
            .geometries
            .iter()
            .flat_map(|g| g.normals.iter().flatten().map(|&n| v4(n)))
            .chain(std::iter::once([0.0; 4]))
            .collect();
        let albedo: Vec<[f32; 4]> = scene.instances.iter().map(|i| v4(i.albedo)).collect();
        // Per object, its texture layer (NONE: none); per geometry triangle,
        // its corners' texture coordinates (zero without).
        let object_textures: Vec<u32> = scene
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
        let mut tex: Vec<[f32; 2]> = Vec::new();
        for g in &scene.geometries {
            if g.tex.len() == g.triangles.len() {
                tex.extend(g.tex.iter().flatten().map(|v| v.to_array()));
            } else {
                tex.extend(std::iter::repeat_n([0.0; 2], 3 * g.triangles.len()));
            }
        }
        if tex.is_empty() {
            tex.push([0.0; 2]);
        }
        let uniform = light.uniform(
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
        let storage = |label: &str, contents: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let buffers = vec![
            (
                0,
                uniform_buffer(&self.device, "light", bytemuck::bytes_of(&uniform)),
            ),
            (2, storage("tris", bytemuck::cast_slice(&flat.tris))),
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
        ];
        buffers
    }

    /// A scene's textures, resampled into a layered texture with mip levels
    /// (a 1×1 white layer without any).
    pub(crate) fn scene_textures(&self, textures: &[Texture]) -> SceneTextures {
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white = Texture {
            width: 1,
            height: 1,
            rgba: vec![255; 4],
        };
        let sources: Vec<&Texture> = if textures.is_empty() {
            vec![&white]
        } else {
            textures.iter().collect()
        };
        for (layer, t) in sources.iter().enumerate() {
            for (level, pixels) in texture_levels(t, size).iter().enumerate() {
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
        SceneTextures { view, sampler }
    }

    /// Hardware acceleration structures for an instanced scene.
    pub(crate) fn hw_scene(&self, scene: &InstancedScene, origin: DVec3) -> hw::HwScene {
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
            })
            .collect();
        hw::build(&self.device, &self.queue, &geometries, &placements)
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
        let textures = self.scene_textures(&[]);
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
        let hw = tlas.is_some() && self.hw_visibility_pipeline.is_some();
        let passes: Vec<(&wgpu::ComputePipeline, KeepBinding)> = match (cached, split || hw) {
            (false, true) if hw => vec![
                (
                    self.hw_visibility_pipeline.as_ref().unwrap(),
                    Box::new(|b| HW_VISIBILITY_BINDINGS.contains(&b)),
                ),
                (
                    &self.shade_pipeline,
                    Box::new(|b| b == 31 || RENDER_BINDINGS.contains(&b)),
                ),
            ],
            (false, false) => vec![(
                &self.render_pipeline,
                Box::new(|b| RENDER_BINDINGS.contains(&b)),
            )],
            (true, false) => vec![(
                &self.cached_pipeline,
                Box::new(|b| b != 31 && !PER_PIXEL_ONLY.contains(&b)),
            )],
            (false, true) => vec![
                (
                    &self.visibility_pipeline,
                    Box::new(|b| VISIBILITY_BINDINGS.contains(&b)),
                ),
                (
                    &self.shade_pipeline,
                    Box::new(|b| b == 31 || RENDER_BINDINGS.contains(&b)),
                ),
            ],
            (true, true) => vec![
                (
                    &self.visibility_pipeline,
                    Box::new(|b| VISIBILITY_BINDINGS.contains(&b)),
                ),
                (
                    &self.shade_cached_pipeline,
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
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("render"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                (*pipeline, group)
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
        let uniform = frame.light.uniform(
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
        let debug = debug_mode != 0;
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let uniform = light.uniform(
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
            layout: &self.pipeline.get_bind_group_layout(0),
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
                pass.set_pipeline(&self.pipeline);
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
fn texture_levels(t: &Texture, size: u32) -> Vec<Vec<u8>> {
    let to_linear = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_srgb = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 {
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
