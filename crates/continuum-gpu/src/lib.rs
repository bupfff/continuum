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
    hierarchy: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FaceGpu {
    origin: [f32; 4],
    u: [f32; 4],
    v: [f32; 4],
    /// `[first texel, nu, nv, blocks per row]`
    info: [u32; 4],
}

/// A world-space lighting cache: a texel grid on every face of every static box
/// mesh (triangles from `continuum::cone3::box_mesh`, two per face in order).
/// Each texel caches direct irradiance; dynamic meshes have no texels and are
/// shaded per pixel.
///
/// Texels are stored in 8×8 blocks of 64 consecutive entries (faces are padded
/// to whole blocks; padding texels have a zero normal and stay dark), so a block
/// can be re-baked as a unit when something moves near it.
pub struct Lightmap {
    faces: Vec<FaceGpu>,
    /// Face of each triangle (global index), or `u32::MAX` for dynamic geometry.
    tri_face: Vec<u32>,
    /// Texel centers nudged off their faces, with face normals: the receivers to bake.
    pub receivers: Vec<(Vec3, Vec3)>,
    /// Cached irradiance per texel (empty until baked).
    pub irradiance: Vec<f32>,
    /// Bounds of each block's real texels (their receiver points).
    pub blocks: Vec<(Vec3, Vec3)>,
}

/// Texels per block side; a block is `BLOCK × BLOCK` consecutive texels.
pub const BLOCK: u32 = 8;

impl Lightmap {
    /// Texels of about `texel_size` on every face of the meshes `is_static`
    /// selects (each must be a 12-triangle box mesh).
    pub fn boxes(
        meshes: &[Vec<Tri>],
        texel_size: f32,
        is_static: impl Fn(usize) -> bool,
    ) -> Lightmap {
        let mut lm = Lightmap {
            faces: Vec::new(),
            tri_face: Vec::new(),
            receivers: Vec::new(),
            irradiance: Vec::new(),
            blocks: Vec::new(),
        };
        for (m, mesh) in meshes.iter().enumerate() {
            if !is_static(m) {
                lm.tri_face
                    .extend(std::iter::repeat_n(u32::MAX, mesh.len()));
                continue;
            }
            assert_eq!(mesh.len(), 12, "lightmaps support box meshes only");
            for quad in mesh.chunks(2) {
                // box_mesh emits each face as [q0, q1, q2], [q0, q2, q3].
                let (q0, q1, q3) = (quad[0][0], quad[0][1], quad[1][2]);
                let (u, v) = (q1 - q0, q3 - q0);
                let nu = ((u.length() / texel_size).ceil() as u32).clamp(1, 1024);
                let nv = ((v.length() / texel_size).ceil() as u32).clamp(1, 1024);
                let (nbu, nbv) = (nu.div_ceil(BLOCK), nv.div_ceil(BLOCK));
                let normal = u.cross(v).normalize();
                let face = lm.faces.len() as u32;
                lm.faces.push(FaceGpu {
                    origin: [q0.x, q0.y, q0.z, 0.0],
                    u: [u.x, u.y, u.z, 0.0],
                    v: [v.x, v.y, v.z, 0.0],
                    info: [lm.receivers.len() as u32, nu, nv, nbu],
                });
                for bj in 0..nbv {
                    for bi in 0..nbu {
                        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                        for k in 0..BLOCK * BLOCK {
                            let (i, j) = (bi * BLOCK + k % BLOCK, bj * BLOCK + k / BLOCK);
                            if i < nu && j < nv {
                                let p = q0
                                    + u * ((i as f32 + 0.5) / nu as f32)
                                    + v * ((j as f32 + 0.5) / nv as f32)
                                    + normal * 1e-4;
                                (lo, hi) = (lo.min(p), hi.max(p));
                                lm.receivers.push((p, normal));
                            } else {
                                lm.receivers.push((q0, Vec3::ZERO));
                            }
                        }
                        lm.blocks.push((lo, hi));
                    }
                }
                lm.tri_face.extend([face, face]);
            }
        }
        lm
    }

    pub fn texels(&self) -> usize {
        self.receivers.len()
    }

    /// For each block, [`BLOCK_PLANES`] planes `[n.x, n.y, n.z, w]` (inside:
    /// `n · p <= w`) of a convex volume enclosing every one of its texels' cones
    /// toward `light`: the hull of the block's texels and the light, pushed out by
    /// a small margin. Unused planes are zero (they exclude nothing). A block whose
    /// texels all face away from the light never receives direct light and gets a
    /// plane that excludes everything.
    pub fn block_planes(&self, light: RectLight) -> Vec<[f32; 4]> {
        const MARGIN: f32 = 1e-3;
        let center = light.corner + (light.edge_u + light.edge_v) * 0.5;
        let light_corners = [
            light.corner,
            light.corner + light.edge_u,
            light.corner + light.edge_v,
            light.corner + light.edge_u + light.edge_v,
        ];
        let mut out = Vec::with_capacity(self.blocks.len() * BLOCK_PLANES);
        for (b, &(lo, hi)) in self.blocks.iter().enumerate() {
            let texels = &self.receivers[b * 64..(b + 1) * 64];
            let mut planes: Vec<[f32; 4]> = Vec::new();
            if !texels.iter().any(|&(p, n)| n.dot(center - p) > -1e-3) {
                planes.push([0.0, 0.0, 0.0, -1.0]);
            } else {
                let mut points: Vec<Vec3> = Vec::new();
                for k in 0..8 {
                    let p = Vec3::new(
                        if k & 1 == 0 { lo.x } else { hi.x },
                        if k & 2 == 0 { lo.y } else { hi.y },
                        if k & 4 == 0 { lo.z } else { hi.z },
                    );
                    if !points.contains(&p) {
                        points.push(p);
                    }
                }
                points.extend(light_corners);
                // Every plane through three points with all points on one side is
                // a face plane of the hull.
                for i in 0..points.len() {
                    for j in i + 1..points.len() {
                        for k in j + 1..points.len() {
                            let n = (points[j] - points[i]).cross(points[k] - points[i]);
                            if n.length_squared() < 1e-14 {
                                continue;
                            }
                            let n = n.normalize();
                            let d = n.dot(points[i]);
                            let (mut below, mut above) = (0.0f32, 0.0f32);
                            for &p in &points {
                                (below, above) = (below.min(n.dot(p) - d), above.max(n.dot(p) - d));
                            }
                            let (n, d) = if above <= 1e-6 {
                                (n, d)
                            } else if below >= -1e-6 {
                                (-n, -d)
                            } else {
                                continue;
                            };
                            if planes.iter().any(|q| {
                                Vec3::new(q[0], q[1], q[2]).dot(n) > 1.0 - 1e-6
                                    && (q[3] - MARGIN - d).abs() < 1e-6
                            }) {
                                continue;
                            }
                            planes.push([n.x, n.y, n.z, d + MARGIN]);
                        }
                    }
                }
            }
            // Fewer planes enclose more, so dropping extras stays conservative.
            planes.resize(BLOCK_PLANES, [0.0; 4]);
            out.extend(planes);
        }
        out
    }
}

/// Planes per block in [`Lightmap::block_planes`].
pub const BLOCK_PLANES: usize = 16;

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
}

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
        MaskConeKernel {
            device,
            queue,
            pipeline,
            render_pipeline,
            cached_pipeline,
            mark_pipeline,
            rebake_pipeline,
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
        hierarchy: bool,
        runs: usize,
    ) -> (Vec<f32>, Duration) {
        let (values, _, best) = self.run(receivers, light, meshes, runs, 0, hierarchy);
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
        hierarchy: bool,
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
            hierarchy,
        };
        self.render_frame(&self.render_pipeline, &frame, None, runs)
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
            hierarchy: true,
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
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
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
    /// triangles (2), mesh ranges (3), debug (5), albedo (8), hierarchy (9, 10),
    /// and with a lightmap its faces and texels (11-13).
    fn scene_buffers(
        &self,
        frame: &Frame,
        lightmap: Option<&Lightmap>,
    ) -> Vec<(u32, wgpu::Buffer)> {
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let uniform = LightUniform {
            corner: v4(frame.light.corner),
            edge_u: v4(frame.light.edge_u),
            edge_v: v4(frame.light.edge_v),
            counts: [frame.meshes.len() as u32, 0, frame.hierarchy as u32, 0],
        };
        let (tris, ranges) = flatten(frame.meshes);
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
            (2, storage("tris", bytemuck::cast_slice(&tris))),
            (3, storage("meshes", bytemuck::cast_slice(&ranges))),
            (5, storage("debug", &[0; 256])),
            (8, storage("albedo", bytemuck::cast_slice(&albedo4))),
            (9, storage("nodes", bytemuck::cast_slice(&nodes))),
            (10, storage("items", bytemuck::cast_slice(&items))),
        ];
        if let Some(lm) = lightmap {
            let texels: &[f32] = if lm.irradiance.is_empty() {
                &[0.0]
            } else {
                &lm.irradiance
            };
            let faces: &[FaceGpu] = if lm.faces.is_empty() {
                &[FaceGpu::zeroed()]
            } else {
                &lm.faces
            };
            buffers.push((11, storage("tri_face", bytemuck::cast_slice(&lm.tri_face))));
            buffers.push((12, storage("faces", bytemuck::cast_slice(faces))));
            buffers.push((13, storage("texels", bytemuck::cast_slice(texels))));
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
        let (_, debug, _) = self.run(receivers, light, meshes, 1, 1, false);
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
        let (_, debug, _) = self.run(&[receiver], light, meshes, 1, 3, false);
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
        let (_, debug, _) = self.run(receivers, light, meshes, 1, 2, false);
        debug.chunks(64).map(|c| c.try_into().unwrap()).collect()
    }

    fn run(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        runs: usize,
        debug_mode: u32,
        hierarchy: bool,
    ) -> (Vec<f32>, Vec<f32>, Duration) {
        let debug = debug_mode != 0;
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let uniform = LightUniform {
            corner: v4(light.corner),
            edge_u: v4(light.edge_u),
            edge_v: v4(light.edge_v),
            counts: [meshes.len() as u32, debug_mode, hierarchy as u32, 0],
        };
        let receivers_gpu: Vec<ReceiverGpu> = receivers
            .iter()
            .map(|&(p, n)| ReceiverGpu {
                point: v4(p),
                normal: v4(n),
            })
            .collect();
        let (tris, ranges) = flatten(meshes);
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
            bytemuck::cast_slice(&tris),
            wgpu::BufferUsages::STORAGE,
        );
        let meshes_buf = init(
            "meshes",
            bytemuck::cast_slice(&ranges),
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

/// Triangles as `vec4` vertices (three per triangle) and per-mesh `[first, count]`
/// ranges. Bindings may not be empty, so empty inputs get unread padding.
fn flatten(meshes: &[Vec<Tri>]) -> (Vec<[f32; 4]>, Vec<[u32; 2]>) {
    let mut tris: Vec<[f32; 4]> = Vec::new();
    let mut ranges: Vec<[u32; 2]> = Vec::new();
    for mesh in meshes {
        ranges.push([(tris.len() / 3) as u32, mesh.len() as u32]);
        for t in mesh {
            tris.extend(t.iter().map(|v| [v.x, v.y, v.z, 0.0]));
        }
    }
    if tris.is_empty() {
        tris.extend([[0.0; 4]; 3]);
    }
    if ranges.is_empty() {
        ranges.push([0, 0]);
    }
    (tris, ranges)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NodeGpu {
    lo: [f32; 4],
    hi: [f32; 4],
    /// Leaf: `[first item, item count, 0, 0]`; interior: `[left child, 0, 0, 0]`
    /// with the right child at `left + 1`.
    info: [u32; 4],
}

/// A bounding volume hierarchy over meshes (median splits on the longest
/// centroid axis, at most two meshes per leaf), flattened for the GPU: nodes and
/// the mesh indices their leaves refer to.
fn mesh_hierarchy(meshes: &[Vec<Tri>]) -> (Vec<NodeGpu>, Vec<u32>) {
    let bounds: Vec<(Vec3, Vec3)> = meshes
        .iter()
        .map(|m| {
            m.iter().flatten().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(lo, hi), &p| (lo.min(p), hi.max(p)),
            )
        })
        .collect();
    let mut items: Vec<u32> = (0..meshes.len() as u32).collect();
    let mut nodes = vec![NodeGpu::zeroed()];
    fn split(
        nodes: &mut Vec<NodeGpu>,
        node: usize,
        items: &mut [u32],
        start: usize,
        bounds: &[(Vec3, Vec3)],
    ) {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let (mut clo, mut chi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &m in items.iter() {
            let (a, b) = bounds[m as usize];
            (lo, hi) = (lo.min(a), hi.max(b));
            let c = (a + b) * 0.5;
            (clo, chi) = (clo.min(c), chi.max(c));
        }
        nodes[node].lo = [lo.x, lo.y, lo.z, 0.0];
        nodes[node].hi = [hi.x, hi.y, hi.z, 0.0];
        if items.len() <= 2 {
            nodes[node].info = [start as u32, items.len() as u32, 0, 0];
            return;
        }
        let e = chi - clo;
        let axis = if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        };
        let mid = items.len() / 2;
        let key = |m: &u32| bounds[*m as usize].0[axis] + bounds[*m as usize].1[axis];
        items.select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)));
        let left = nodes.len();
        nodes.push(NodeGpu::zeroed());
        nodes.push(NodeGpu::zeroed());
        nodes[node].info = [left as u32, 0, 0, 0];
        let (l, r) = items.split_at_mut(mid);
        split(nodes, left, l, start, bounds);
        split(nodes, left + 1, r, start + mid, bounds);
    }
    if items.is_empty() {
        items.push(0);
        nodes[0].info = [0, 0, 0, 0];
    } else {
        split(&mut nodes, 0, &mut items, 0, &bounds);
    }
    (nodes, items)
}

fn uniform_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DisplayDims {
    size: [u32; 2],
    flags: [u32; 2],
}

/// Persistent GPU state for interactive rendering with a lighting cache. The
/// scene, its hierarchy and the baked lightmap stay on the GPU; a frame uploads
/// only the camera, runs the cached kernel, and draws into a surface texture.
///
/// Meshes without lightmap texels are dynamic: they are shaded per pixel and can
/// move ([`Self::update_meshes`]), which re-bakes on the GPU exactly the lightmap
/// blocks they can affect.
pub struct LiveRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute: wgpu::ComputePipeline,
    display: wgpu::RenderPipeline,
    mark: wgpu::ComputePipeline,
    rebake: wgpu::ComputePipeline,
    scene: Vec<(u32, wgpu::Buffer)>,
    camera_buf: wgpu::Buffer,
    dims_buf: wgpu::Buffer,
    colors_buf: wgpu::Buffer,
    compute_bind: wgpu::BindGroup,
    display_bind: wgpu::BindGroup,
    motion: Motion,
    width: u32,
    height: u32,
    radiance: f32,
    srgb: bool,
}

/// What [`LiveRenderer`] keeps to move dynamic meshes.
struct Motion {
    meshes: Vec<Vec<Tri>>,
    ranges: Vec<[u32; 2]>,
    bounds: Vec<(Vec3, Vec3)>,
    dynamic: Vec<bool>,
    blocks: u32,
    uniform: wgpu::Buffer,
    args: wgpu::Buffer,
    mark_bind: wgpu::BindGroup,
    rebake_bind: wgpu::BindGroup,
}

/// Swept boxes per update (each `[lo, hi]`) and the block count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionUniform {
    counts: [u32; 4],
    boxes: [[f32; 4]; 2 * MAX_SWEPT],
}

/// Changed meshes per [`LiveRenderer::update_meshes`] call.
pub const MAX_SWEPT: usize = 16;

fn mesh_bounds(mesh: &[Tri]) -> (Vec3, Vec3) {
    mesh.iter().flatten().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), &p| (lo.min(p), hi.max(p)),
    )
}

fn binding(scene: &[(u32, wgpu::Buffer)], b: u32) -> &wgpu::Buffer {
    &scene
        .iter()
        .find(|(k, _)| *k == b)
        .expect("scene binding")
        .1
}

fn bind_group(
    device: &wgpu::Device,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .map(|(b, buffer)| wgpu::BindGroupEntry {
            binding: *b,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

impl LiveRenderer {
    /// Uploads a scene and its baked lightmap. `target_format` is the surface's format.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        light: RectLight,
        meshes: &[Vec<Tri>],
        albedo: &[Vec3],
        radiance: f32,
        lightmap: &Lightmap,
    ) -> Self {
        let device = kernel.device.clone();
        let frame = Frame {
            camera: Camera {
                eye: Vec3::ZERO,
                target: Vec3::X,
                vertical_fov_degrees: 50.0,
            },
            width,
            height,
            light,
            meshes,
            albedo,
            radiance,
            hierarchy: true,
        };
        let scene = kernel.scene_buffers(&frame, Some(lightmap));
        let camera_buf = uniform_buffer(
            &device,
            "camera",
            &[0; std::mem::size_of::<CameraUniform>()],
        );
        let dims_buf = uniform_buffer(&device, "dims", &[0; std::mem::size_of::<DisplayDims>()]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display"),
            source: wgpu::ShaderSource::Wgsl(include_str!("display.wgsl").into()),
        });
        let display = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let (colors_buf, compute_bind, display_bind) = Self::targets(
            &device,
            &kernel.cached_pipeline,
            &display,
            &scene,
            &camera_buf,
            &dims_buf,
            width,
            height,
        );
        let motion = Self::motion(kernel, &scene, light, meshes, lightmap);
        LiveRenderer {
            device,
            queue: kernel.queue.clone(),
            compute: kernel.cached_pipeline.clone(),
            display,
            mark: kernel.mark_pipeline.clone(),
            rebake: kernel.rebake_pipeline.clone(),
            scene,
            camera_buf,
            dims_buf,
            colors_buf,
            compute_bind,
            display_bind,
            motion,
            width,
            height,
            radiance,
            srgb: target_format.is_srgb(),
        }
    }

    fn motion(
        kernel: &MaskConeKernel,
        scene: &[(u32, wgpu::Buffer)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        lightmap: &Lightmap,
    ) -> Motion {
        let device = &kernel.device;
        let (_, ranges) = flatten(meshes);
        let dynamic = meshes
            .iter()
            .zip(&ranges)
            .map(|(mesh, r)| mesh.is_empty() || lightmap.tri_face[r[0] as usize] == u32::MAX)
            .collect();
        let blocks = lightmap.blocks.len() as u32;
        assert!(
            blocks <= kernel.device.limits().max_compute_workgroups_per_dimension,
            "too many lightmap blocks for one re-bake dispatch"
        );
        let storage = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | usage,
            })
        };
        let receivers: Vec<ReceiverGpu> = lightmap
            .receivers
            .iter()
            .map(|&(p, n)| ReceiverGpu {
                point: [p.x, p.y, p.z, 0.0],
                normal: [n.x, n.y, n.z, 0.0],
            })
            .collect();
        let receivers = storage(
            "texel receivers",
            bytemuck::cast_slice(&receivers),
            wgpu::BufferUsages::empty(),
        );
        let planes = lightmap.block_planes(light);
        let planes = storage(
            "block planes",
            bytemuck::cast_slice(&planes),
            wgpu::BufferUsages::empty(),
        );
        let args = storage(
            "rebake args",
            bytemuck::cast_slice(&[0u32, 1, 1]),
            wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let dirty = storage(
            "dirty blocks",
            &vec![0; 4 * blocks.max(1) as usize],
            wgpu::BufferUsages::empty(),
        );
        let uniform = uniform_buffer(device, "motion", &[0; std::mem::size_of::<MotionUniform>()]);
        let mark_bind = bind_group(
            device,
            "mark",
            &kernel.mark_pipeline,
            &[(14, &uniform), (15, &planes), (16, &args), (17, &dirty)],
        );
        let s = |b| binding(scene, b);
        let rebake_bind = bind_group(
            device,
            "rebake",
            &kernel.rebake_pipeline,
            &[
                (0, s(0)),
                (1, &receivers),
                (2, s(2)),
                (3, s(3)),
                (4, s(13)),
                (5, s(5)),
                (9, s(9)),
                (10, s(10)),
                (17, &dirty),
            ],
        );
        Motion {
            meshes: meshes.to_vec(),
            ranges,
            bounds: meshes.iter().map(|m| mesh_bounds(m)).collect(),
            dynamic,
            blocks,
            uniform,
            args,
            mark_bind,
            rebake_bind,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn targets(
        device: &wgpu::Device,
        compute: &wgpu::ComputePipeline,
        display: &wgpu::RenderPipeline,
        scene: &[(u32, wgpu::Buffer)],
        camera_buf: &wgpu::Buffer,
        dims_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup) {
        let colors_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: (width as u64 * height as u64 * 16).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut entries: Vec<(u32, &wgpu::Buffer)> =
            scene.iter().map(|(b, buf)| (*b, buf)).collect();
        entries.push((6, camera_buf));
        entries.push((7, &colors_buf));
        let compute_bind = bind_group(device, "live compute", compute, &entries);
        let display_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("live display"),
            layout: &display.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: colors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });
        (colors_buf, compute_bind, display_bind)
    }

    /// Call when the surface size changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) == (self.width, self.height) || width == 0 || height == 0 {
            return;
        }
        (self.colors_buf, self.compute_bind, self.display_bind) = Self::targets(
            &self.device,
            &self.compute,
            &self.display,
            &self.scene,
            &self.camera_buf,
            &self.dims_buf,
            width,
            height,
        );
        (self.width, self.height) = (width, height);
    }

    /// Moves dynamic meshes: each change replaces a mesh's triangles (same count).
    /// Uploads them, rebuilds the mesh hierarchy, and re-bakes on the GPU exactly
    /// the lightmap blocks whose cone volumes a changed mesh's swept bounds (old ∪
    /// new) reach. Submitted at once; the next frame sees the result.
    pub fn update_meshes(&mut self, changes: &[(usize, Vec<Tri>)]) {
        let m = &mut self.motion;
        let mut boxes = Vec::new();
        for (mesh, tris) in changes {
            let mesh = *mesh;
            assert!(
                m.dynamic[mesh],
                "mesh {mesh} has lightmap texels and cannot move"
            );
            assert_eq!(
                tris.len(),
                m.meshes[mesh].len(),
                "a moved mesh keeps its triangle count"
            );
            if *tris == m.meshes[mesh] {
                continue;
            }
            let (old, new) = (m.bounds[mesh], mesh_bounds(tris));
            let (lo, hi) = (old.0.min(new.0), old.1.max(new.1));
            boxes.push([lo.x, lo.y, lo.z, 0.0]);
            boxes.push([hi.x, hi.y, hi.z, 0.0]);
            let vertices: Vec<[f32; 4]> = tris
                .iter()
                .flatten()
                .map(|v| [v.x, v.y, v.z, 0.0])
                .collect();
            self.queue.write_buffer(
                binding(&self.scene, 2),
                m.ranges[mesh][0] as u64 * 48,
                bytemuck::cast_slice(&vertices),
            );
            m.meshes[mesh] = tris.clone();
            m.bounds[mesh] = new;
        }
        if boxes.is_empty() {
            return;
        }
        assert!(
            boxes.len() <= 2 * MAX_SWEPT,
            "at most {MAX_SWEPT} meshes per update"
        );
        // Median splits depend only on the mesh count, so the layout is unchanged.
        let (nodes, items) = mesh_hierarchy(&m.meshes);
        self.queue
            .write_buffer(binding(&self.scene, 9), 0, bytemuck::cast_slice(&nodes));
        self.queue
            .write_buffer(binding(&self.scene, 10), 0, bytemuck::cast_slice(&items));
        let mut uniform = MotionUniform::zeroed();
        uniform.counts = [(boxes.len() / 2) as u32, m.blocks, 0, 0];
        uniform.boxes[..boxes.len()].copy_from_slice(&boxes);
        self.queue
            .write_buffer(&m.uniform, 0, bytemuck::bytes_of(&uniform));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&m.args, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.mark);
            pass.set_bind_group(0, &m.mark_bind, &[]);
            pass.dispatch_workgroups(m.blocks.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rebake);
            pass.set_bind_group(0, &m.rebake_bind, &[]);
            pass.dispatch_workgroups_indirect(&m.args, 0);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Renders one frame from `camera` into `target` and submits it.
    pub fn frame(&self, camera: Camera, vertical_fov_degrees: f32, target: &wgpu::TextureView) {
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let forward = (camera.target - camera.eye).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward);
        let cam = CameraUniform {
            eye: v4(camera.eye),
            forward: v4(forward),
            right: v4(right),
            up: v4(up),
            params: [
                self.width as f32,
                self.height as f32,
                (vertical_fov_degrees.to_radians() * 0.5).tan(),
                self.radiance,
            ],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [!self.srgb as u32, 0],
        };
        self.queue
            .write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.compute);
            pass.set_bind_group(0, &self.compute_bind, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.display);
            pass.set_bind_group(0, &self.display_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Blocks until submitted GPU work is done.
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
    }

    /// The lightmap's current texels, read back from the GPU.
    pub fn read_texels(&self) -> Vec<f32> {
        bytemuck::cast_slice(&self.read(binding(&self.scene, 13))).to_vec()
    }

    /// Blocks re-baked by the last [`Self::update_meshes`] that moved something.
    pub fn rebaked_blocks(&self) -> u32 {
        bytemuck::cast_slice::<u8, u32>(&self.read(&self.motion.args))[0]
    }

    /// Total lightmap blocks.
    pub fn blocks(&self) -> u32 {
        self.motion.blocks
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.wait();
        let bytes = slice.get_mapped_range().expect("mapped readback").to_vec();
        staging.unmap();
        bytes
    }
}
