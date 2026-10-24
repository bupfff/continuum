//! GPU backend for Continuum (wgpu compute; Metal, Vulkan, DX12).
//!
//! One kernel family: mask cones from receivers to a rectangular light, a port
//! of `continuum::cone3` (see `mask_cones.wgsl`), used for receivers, camera
//! renders, a world-space lighting cache ([`Lightmap`]), and live rendering with
//! moving meshes ([`LiveRenderer`]).

use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use continuum::glam::Vec3;
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
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReceiverGpu {
    point: [f32; 4],
    normal: [f32; 4],
}

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
mod lightmap;
mod live;

pub use hierarchy::Hierarchy;
pub use lightmap::{BLOCK, BLOCK_PLANES, Lightmap, hemicube_weights};
pub use live::{LiveRenderer, MAX_MOVING, MAX_SWEPT, PROBE_RESOLUTION};

use hierarchy::{flatten, mesh_hierarchy};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    params: [f32; 4],
}

/// A pinhole camera looking from `eye` toward `target`, with `+Z` up.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub vertical_fov_degrees: f32,
}

/// A rectangular light: a corner and two perpendicular edges.
#[derive(Clone, Copy, Debug)]
pub struct RectLight {
    pub corner: Vec3,
    pub edge_u: Vec3,
    pub edge_v: Vec3,
    /// One-sided: emits only toward `edge_u × edge_v`, and camera rays see its
    /// front as a surface of the light's radiance. Otherwise (the M2 scenes) it
    /// emits to both sides and is invisible to the camera.
    pub one_sided: bool,
}

impl RectLight {
    fn uniform(&self, meshes: usize, debug_mode: u32, hierarchy: Hierarchy) -> LightUniform {
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
        }
    }
}

/// Bindings the `render` entry point uses (render_frame filters the scene's
/// buffers to these when a lightmap provides more).
const RENDER_BINDINGS: &[u32] = &[0, 2, 3, 5, 6, 7, 8, 9, 10, 24, 25, 26];

impl MaskConeKernel {
    /// `None` if no GPU adapter is available.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("continuum"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        Some(Self::with_device(device, queue, adapter_name))
    }

    /// A kernel on an existing device (for example one that also drives a window).
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue, adapter_name: String) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mask_cones"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mask_cones.wgsl").into()),
        });
        let entry = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = entry("main");
        let render_pipeline = entry("render");
        let cached_pipeline = entry("render_cached");
        let mark_pipeline = entry("mark_blocks");
        let rebake_pipeline = entry("rebake");
        let probe_build_pipeline = entry("probe_build");
        let probe_gather_pipeline = entry("probe_gather");
        let bake_static_pipeline = entry("bake_static");
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
            adapter_name,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Irradiance per unit radiance at each receiver `(point, normal)`, and the
    /// fastest of `runs` GPU executions (submit to completion, excluding uploads
    /// and readback). Every receiver tests every mesh.
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
        self.render_frame(&self.render_pipeline, &frame, None, runs)
    }

    /// Like [`Self::render`] (mask cones at every pixel, with the hierarchy),
    /// but shading with the lightmap's interpolated normals: the reference for
    /// [`Self::render_cached`] on smooth meshes. The lightmap's texels are unused.
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
        self.render_frame(&self.render_pipeline, &frame, Some(lightmap), runs)
    }

    /// Like [`Self::render`] (with the hierarchy), but static faces read their
    /// irradiance from a baked [`Lightmap`]; only dynamic geometry gets a cone
    /// per pixel.
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
        self.render_frame(&self.cached_pipeline, &frame, Some(lightmap), runs)
    }

    /// Bakes a lightmap: one mask cone per texel (through the hierarchy).
    /// Returns the bake time.
    pub fn bake(&self, lightmap: &mut Lightmap, light: RectLight, meshes: &[Vec<Tri>]) -> Duration {
        let (values, time) = self.irradiance_with(&lightmap.receivers, light, meshes, true, 1);
        lightmap.irradiance = values;
        time
    }

    fn render_frame(
        &self,
        pipeline: &wgpu::ComputePipeline,
        frame: &Frame,
        lightmap: Option<&Lightmap>,
        runs: usize,
    ) -> (Vec<[f32; 3]>, Duration) {
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let (camera, width, height) = (frame.camera, frame.width, frame.height);
        let forward = (camera.target - camera.eye).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward);
        let cam = CameraUniform {
            eye: v4(camera.eye),
            forward: v4(forward),
            right: v4(right),
            up: v4(up),
            params: [
                width as f32,
                height as f32,
                (camera.vertical_fov_degrees.to_radians() * 0.5).tan(),
                frame.radiance,
            ],
        };
        let mut buffers = self.scene_buffers(frame, lightmap);
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
        let per_pixel = std::ptr::eq(pipeline, &self.render_pipeline);
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .filter(|(binding, _)| !per_pixel || RENDER_BINDINGS.contains(binding))
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: colors_buf.as_entire_binding(),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut best = Duration::MAX;
        for _ in 0..runs.max(1) {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
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
        let uniform = frame.light.uniform(frame.meshes.len(), 0, frame.hierarchy);
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
        let uniform = light.uniform(meshes.len(), debug_mode, hierarchy);
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

fn uniform_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
