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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FaceGpu {
    origin: [f32; 4],
    u: [f32; 4],
    v: [f32; 4],
    /// `[first texel, nu, nv, blocks per row]`
    info: [u32; 4],
    /// `[first probe, pu, pv, mesh]` (`pu = 0`: no probes)
    probes: [u32; 4],
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
    /// Indirect-light probes `(point, normal)`, row-major per face (see [`Self::with_probes`]).
    pub probes: Vec<(Vec3, Vec3)>,
    /// Indirect irradiance per probe per unit light radiance (RGB, then 1 if
    /// valid), for renders outside [`LiveRenderer`]; empty for none.
    pub indirect: Vec<[f32; 4]>,
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
            probes: Vec::new(),
            indirect: Vec::new(),
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
                    probes: [0, 0, 0, m as u32],
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

    /// Adds indirect-light probes about `spacing` apart on every face, at cell
    /// centers nudged off the face like texels.
    pub fn with_probes(mut self, spacing: f32) -> Self {
        for face in &mut self.faces {
            let q0 = Vec3::from_slice(&face.origin[..3]);
            let (u, v) = (
                Vec3::from_slice(&face.u[..3]),
                Vec3::from_slice(&face.v[..3]),
            );
            let pu = ((u.length() / spacing).ceil() as u32).clamp(1, 1024);
            let pv = ((v.length() / spacing).ceil() as u32).clamp(1, 1024);
            let normal = u.cross(v).normalize();
            face.probes = [self.probes.len() as u32, pu, pv, face.probes[3]];
            for j in 0..pv {
                for i in 0..pu {
                    let p = q0
                        + u * ((i as f32 + 0.5) / pu as f32)
                        + v * ((j as f32 + 0.5) / pv as f32)
                        + normal * 1e-4;
                    self.probes.push((p, normal));
                }
            }
        }
        self
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

/// Bins of a hemicube around a probe's normal with `r × r` bins on the top face
/// and `r × r/2` on each side face, in the kernel's order: the projected solid
/// angle of each (they sum to π).
pub fn hemicube_weights(r: u32) -> Vec<f32> {
    assert!(r.is_multiple_of(2), "hemicube resolution must be even");
    let g = r as f32;
    let at = |i: u32, j: u32, di: f32, dj: f32| {
        (
            -1.0 + 2.0 * (i as f32 + di) / g,
            -1.0 + 2.0 * (j as f32 + dj) / g,
            2.0 * (j as f32 + dj) / g,
        )
    };
    let mut weights = Vec::with_capacity(3 * (r * r) as usize);
    for bin in 0..r * r {
        let (i, j) = (bin % r, bin / r);
        let quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(di, dj)| {
            let (x, y, _) = at(i, j, di, dj);
            Vec3::new(x, y, 1.0)
        });
        weights.push(continuum::cone3::projected_solid_angle(Vec3::Z, &quad));
    }
    for side in 0..4 {
        for k in 0..r * r / 2 {
            let (i, j) = (k % r, k / r);
            let quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(di, dj)| {
                let (a, _, z) = at(i, j, di, dj);
                match side {
                    0 => Vec3::new(1.0, a, z),
                    1 => Vec3::new(-1.0, a, z),
                    2 => Vec3::new(a, 1.0, z),
                    _ => Vec3::new(a, -1.0, z),
                }
            });
            weights.push(continuum::cone3::projected_solid_angle(Vec3::Z, &quad));
        }
    }
    weights
}

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
        let probe_build_pipeline = entry("probe_build");
        let probe_gather_pipeline = entry("probe_gather");
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
            (25, storage("triangle nodes", bytemuck::cast_slice(&flat.blas))),
            (26, storage("triangle items", bytemuck::cast_slice(&flat.blas_items))),
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
            let indirect: &[[f32; 4]] =
                if lm.indirect.len() == lm.probes.len() && !lm.probes.is_empty() {
                    &lm.indirect
                } else if lm.probes.is_empty() {
                    &[[0.0; 4]]
                } else {
                    // Probes without values yet: all invalid, so lookups add nothing.
                    &vec![[0.0; 4]; lm.probes.len()]
                };
            buffers.push((11, storage("tri_face", bytemuck::cast_slice(&lm.tri_face))));
            buffers.push((12, storage("faces", bytemuck::cast_slice(faces))));
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

/// Meshes flattened for the GPU. Bindings may not be empty, so empty inputs
/// get unread padding.
struct Flat {
    /// Triangles as `vec4` vertices, three per triangle.
    tris: Vec<[f32; 4]>,
    /// Per mesh: `[first triangle, count, triangle hierarchy root (or NONE), 0]`.
    ranges: Vec<[u32; 4]>,
    /// Every mesh's triangle hierarchy: its root first, then its subtree.
    blas: Vec<NodeGpu>,
    /// Triangle indices referenced by `blas` leaves; mesh `m`'s are at the same
    /// offsets as its triangles.
    blas_items: Vec<u32>,
}

const NONE: u32 = u32::MAX;

fn flatten(meshes: &[Vec<Tri>]) -> Flat {
    let mut flat = Flat {
        tris: Vec::new(),
        ranges: Vec::new(),
        blas: Vec::new(),
        blas_items: Vec::new(),
    };
    for mesh in meshes {
        let first = (flat.tris.len() / 3) as u32;
        let root = if mesh.is_empty() {
            NONE
        } else {
            let root = flat.blas.len() as u32;
            let (nodes, items) = triangle_hierarchy(mesh, root, first);
            flat.blas.extend(nodes);
            flat.blas_items.extend(items);
            root
        };
        flat.ranges.push([first, mesh.len() as u32, root, 0]);
        for t in mesh {
            flat.tris.extend(t.iter().map(|v| [v.x, v.y, v.z, 0.0]));
        }
    }
    if flat.tris.is_empty() {
        flat.tris.extend([[0.0; 4]; 3]);
    }
    if flat.ranges.is_empty() {
        flat.ranges.push([0, 0, NONE, 0]);
    }
    if flat.blas.is_empty() {
        flat.blas.push(NodeGpu::zeroed());
        flat.blas_items.push(0);
    }
    flat
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

/// Median splits on the longest centroid axis over `items` (indices into
/// `bounds`), writing `nodes[node]` and its subtree. Leaves hold at most `leaf`
/// items, recorded as `[item_base + start, count]`; children as `node_base +
/// index`. The tree's shape depends only on the item count.
#[allow(clippy::too_many_arguments)]
fn split(
    nodes: &mut Vec<NodeGpu>,
    node: usize,
    items: &mut [u32],
    start: usize,
    bounds: &[(Vec3, Vec3)],
    leaf: usize,
    node_base: u32,
    item_base: u32,
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
    if items.len() <= leaf {
        nodes[node].info = [item_base + start as u32, items.len() as u32, 0, 0];
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
    nodes[node].info = [node_base + left as u32, 0, 0, 0];
    let (l, r) = items.split_at_mut(mid);
    split(nodes, left, l, start, bounds, leaf, node_base, item_base);
    split(nodes, left + 1, r, start + mid, bounds, leaf, node_base, item_base);
}

/// A bounding volume hierarchy over meshes (at most two meshes per leaf),
/// flattened for the GPU: nodes and the mesh indices their leaves refer to.
fn mesh_hierarchy(meshes: &[Vec<Tri>]) -> (Vec<NodeGpu>, Vec<u32>) {
    mesh_hierarchy_of(meshes, |_| true)
}

/// [`mesh_hierarchy`] over only the meshes `keep` selects (indices stay global).
fn mesh_hierarchy_of(
    meshes: &[Vec<Tri>],
    keep: impl Fn(usize) -> bool,
) -> (Vec<NodeGpu>, Vec<u32>) {
    let bounds: Vec<(Vec3, Vec3)> = meshes.iter().map(|m| padded(mesh_bounds(m))).collect();
    let mut items: Vec<u32> = (0..meshes.len() as u32)
        .filter(|&m| keep(m as usize))
        .collect();
    let mut nodes = vec![NodeGpu::zeroed()];
    if items.is_empty() {
        // A leaf holding mesh 0, which must then have no triangles: true when
        // there are no meshes (`flatten` pads an empty range). Callers never
        // traverse a hierarchy that filtered every mesh out.
        items.push(0);
        nodes[0].info = [0, 1, 0, 0];
    } else {
        split(&mut nodes, 0, &mut items, 0, &bounds, 2, 0, 0);
    }
    (nodes, items)
}

/// One mesh's triangle hierarchy (at most [`LEAF_TRIANGLES`] per leaf): nodes
/// numbered from `root`, and triangle indices numbered from `first`, stored at
/// item positions from `first`.
fn triangle_hierarchy(mesh: &[Tri], root: u32, first: u32) -> (Vec<NodeGpu>, Vec<u32>) {
    let bounds: Vec<(Vec3, Vec3)> = mesh.iter().map(|t| padded(mesh_bounds(&[*t]))).collect();
    let mut items: Vec<u32> = (0..mesh.len() as u32).collect();
    let mut nodes = vec![NodeGpu::zeroed()];
    split(&mut nodes, 0, &mut items, 0, &bounds, LEAF_TRIANGLES, root, first);
    (nodes, items.iter().map(|&t| first + t).collect())
}

/// Triangles per leaf of a mesh's triangle hierarchy. A box test costs about
/// three triangle tests, so small leaves only pay off for large meshes.
const LEAF_TRIANGLES: usize = 8;

/// How far rays and cones descend before testing triangles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hierarchy {
    /// Test every triangle of every mesh.
    None = 0,
    /// Cull meshes through the mesh hierarchy, then test each touched mesh's
    /// triangles one by one.
    Meshes = 1,
    /// Also cull triangles through each mesh's triangle hierarchy.
    Triangles = 2,
}

impl From<bool> for Hierarchy {
    fn from(on: bool) -> Self {
        if on { Hierarchy::Triangles } else { Hierarchy::None }
    }
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
///
/// If the lightmap has probes ([`Lightmap::with_probes`]), static surfaces also
/// get one bounce of indirect light: each probe caches a hemicube G-buffer of
/// static geometry once, and gathers bounce radiance from the lightmap after
/// every update, with moving meshes composited as black occluders.
pub struct LiveRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute: wgpu::ComputePipeline,
    display: wgpu::RenderPipeline,
    mark: wgpu::ComputePipeline,
    rebake: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    scene: Vec<(u32, wgpu::Buffer)>,
    camera_buf: wgpu::Buffer,
    dims_buf: wgpu::Buffer,
    colors_buf: wgpu::Buffer,
    compute_bind: wgpu::BindGroup,
    display_bind: wgpu::BindGroup,
    motion: Motion,
    indirect: Option<Indirect>,
    width: u32,
    height: u32,
    radiance: f32,
    srgb: bool,
}

/// Indirect-light probes on the GPU (see `probe_gather` in the kernel).
struct Indirect {
    probes: u32,
    points: wgpu::Buffer,
    gbuffer: wgpu::Buffer,
    params: wgpu::Buffer,
    gather_bind: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProbeParams {
    counts: [u32; 4],
    moving: [[u32; 4]; MAX_MOVING],
    bounds: [[f32; 4]; 2 * MAX_MOVING],
}

/// Dynamic meshes a [`LiveRenderer`] with probes supports.
pub const MAX_MOVING: usize = 16;

/// Hemicube resolution of live probes: `r × r` bins on the top face and
/// `r × r/2` on each side (768 bins per probe).
pub const PROBE_RESOLUTION: u32 = 16;

/// What [`LiveRenderer`] keeps to move dynamic meshes.
struct Motion {
    meshes: Vec<Vec<Tri>>,
    ranges: Vec<[u32; 4]>,
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

/// Bounds grown by a margin well above float rounding, so box tests that cull
/// rays or cones never reject a triangle the exact test would hit. (Box meshes lie
/// exactly on their bounds, so a ray grazing a face can otherwise pass the
/// triangle test and fail the box test, depending on how boxes are grouped.)
fn padded((lo, hi): (Vec3, Vec3)) -> (Vec3, Vec3) {
    let pad = 1e-5 * lo.abs().max(hi.abs()).max_element().max(1.0);
    (lo - pad, hi + pad)
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
            hierarchy: Hierarchy::Triangles,
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
        let indirect = (!lightmap.probes.is_empty())
            .then(|| Self::indirect(kernel, &scene, &motion, lightmap));
        LiveRenderer {
            device,
            queue: kernel.queue.clone(),
            compute: kernel.cached_pipeline.clone(),
            display,
            mark: kernel.mark_pipeline.clone(),
            rebake: kernel.rebake_pipeline.clone(),
            gather: kernel.probe_gather_pipeline.clone(),
            scene,
            camera_buf,
            dims_buf,
            colors_buf,
            compute_bind,
            display_bind,
            motion,
            indirect,
            width,
            height,
            radiance,
            srgb: target_format.is_srgb(),
        }
    }

    /// Uploads the probes, builds their G-buffers from static geometry, and
    /// runs the first gather.
    fn indirect(
        kernel: &MaskConeKernel,
        scene: &[(u32, wgpu::Buffer)],
        motion: &Motion,
        lightmap: &Lightmap,
    ) -> Indirect {
        let device = &kernel.device;
        let probes = lightmap.probes.len() as u32;
        assert!(
            probes <= device.limits().max_compute_workgroups_per_dimension,
            "too many probes for one gather dispatch"
        );
        let r = PROBE_RESOLUTION;
        let bins = 3 * r * r;
        let storage = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let points: Vec<ReceiverGpu> = lightmap
            .probes
            .iter()
            .map(|&(p, n)| ReceiverGpu {
                point: [p.x, p.y, p.z, 0.0],
                normal: [n.x, n.y, n.z, 0.0],
            })
            .collect();
        let points = storage("probe points", bytemuck::cast_slice(&points));
        let gbuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe G-buffers"),
            size: probes as u64 * bins as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let weights = storage("bin weights", bytemuck::cast_slice(&hemicube_weights(r)));
        let params = uniform_buffer(
            device,
            "probe params",
            bytemuck::bytes_of(&Self::probe_params(motion, probes, r, 0)),
        );
        // The G-buffer holds static geometry only: trace through a hierarchy of
        // the static meshes.
        let (nodes, items) = mesh_hierarchy_of(&motion.meshes, |m| !motion.dynamic[m]);
        let nodes = storage("static nodes", bytemuck::cast_slice(&nodes));
        let items = storage("static items", bytemuck::cast_slice(&items));
        let s = |b| binding(scene, b);
        let build_bind = bind_group(
            device,
            "probe build",
            &kernel.probe_build_pipeline,
            &[
                (0, s(0)),
                (2, s(2)),
                (3, s(3)),
                (9, &nodes),
                (10, &items),
                (25, s(25)),
                (26, s(26)),
                (11, s(11)),
                (18, &points),
                (19, &gbuffer),
                (22, &params),
            ],
        );
        let gather_bind =
            Self::gather_bind(kernel, scene, &points, &gbuffer, &weights, s(21), &params);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernel.probe_build_pipeline);
            pass.set_bind_group(0, &build_bind, &[]);
            pass.dispatch_workgroups(bins / 64, probes, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernel.probe_gather_pipeline);
            pass.set_bind_group(0, &gather_bind, &[]);
            pass.dispatch_workgroups(probes, 1, 1);
        }
        kernel.queue.submit([encoder.finish()]);
        Indirect {
            probes,
            points,
            gbuffer,
            params,
            gather_bind,
        }
    }

    fn gather_bind(
        kernel: &MaskConeKernel,
        scene: &[(u32, wgpu::Buffer)],
        points: &wgpu::Buffer,
        gbuffer: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        out: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let s = |b| binding(scene, b);
        bind_group(
            &kernel.device,
            "probe gather",
            &kernel.probe_gather_pipeline,
            &[
                (0, s(0)),
                (2, s(2)),
                (3, s(3)),
                (8, s(8)),
                (9, s(9)),
                (10, s(10)),
                (25, s(25)),
                (26, s(26)),
                (11, s(11)),
                (12, s(12)),
                (13, s(13)),
                (18, points),
                (19, gbuffer),
                (20, weights),
                (21, out),
                (22, params),
            ],
        )
    }

    /// Gather parameters: every dynamic mesh with its current bounds.
    fn probe_params(motion: &Motion, probes: u32, resolution: u32, mode: u32) -> ProbeParams {
        let mut params = ProbeParams::zeroed();
        let mut k = 0;
        if mode == 0 {
            for (m, mesh) in motion.meshes.iter().enumerate() {
                if motion.dynamic[m] && !mesh.is_empty() {
                    assert!(
                        k < MAX_MOVING,
                        "at most {MAX_MOVING} dynamic meshes with probes"
                    );
                    let (lo, hi) = padded(motion.bounds[m]);
                    params.moving[k] = [m as u32, 0, 0, 0];
                    params.bounds[2 * k] = [lo.x, lo.y, lo.z, 0.0];
                    params.bounds[2 * k + 1] = [hi.x, hi.y, hi.z, 0.0];
                    k += 1;
                }
            }
        }
        params.counts = [probes, resolution, mode, k as u32];
        params
    }

    fn motion(
        kernel: &MaskConeKernel,
        scene: &[(u32, wgpu::Buffer)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        lightmap: &Lightmap,
    ) -> Motion {
        let device = &kernel.device;
        let ranges = flatten(meshes).ranges;
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
                (25, s(25)),
                (26, s(26)),
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
            // Same triangle count, so the same hierarchy shape and offsets.
            let [first, _, root, _] = m.ranges[mesh];
            if root != NONE {
                let (nodes, items) = triangle_hierarchy(tris, root, first);
                self.queue.write_buffer(
                    binding(&self.scene, 25),
                    root as u64 * std::mem::size_of::<NodeGpu>() as u64,
                    bytemuck::cast_slice(&nodes),
                );
                self.queue.write_buffer(
                    binding(&self.scene, 26),
                    first as u64 * 4,
                    bytemuck::cast_slice(&items),
                );
            }
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
        if let Some(ind) = &self.indirect {
            let params = Self::probe_params(&self.motion, ind.probes, PROBE_RESOLUTION, 0);
            self.queue
                .write_buffer(&ind.params, 0, bytemuck::bytes_of(&params));
            self.encode_gather(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
    }

    fn encode_gather(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ind) = &self.indirect {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gather);
            pass.set_bind_group(0, &ind.gather_bind, &[]);
            pass.dispatch_workgroups(ind.probes, 1, 1);
        }
    }

    /// Runs the probe gather alone (it also runs in every [`Self::update_meshes`]
    /// that moves something); for timing.
    pub fn regather(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_gather(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    /// Probe count (0 without indirect light).
    pub fn probes(&self) -> u32 {
        self.indirect.as_ref().map_or(0, |i| i.probes)
    }

    /// The probes' current indirect irradiance (RGB, then 1 if valid).
    pub fn read_probes(&self) -> Vec<[f32; 4]> {
        let n = self.probes() as usize;
        bytemuck::cast_slice::<u8, [f32; 4]>(&self.read(binding(&self.scene, 21)))[..n].to_vec()
    }

    /// Indirect irradiance per probe recomputed from scratch, for verification:
    /// every bin of a `resolution` hemicube is traced through the current scene
    /// (static and moving meshes; moving ones black) instead of read from the
    /// cached G-buffer. With [`PROBE_RESOLUTION`] it must equal [`Self::read_probes`].
    pub fn probes_from_scratch(&self, kernel: &MaskConeKernel, resolution: u32) -> Vec<[f32; 4]> {
        let ind = self.indirect.as_ref().expect("no probes");
        let device = &self.device;
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bin weights"),
            contents: bytemuck::cast_slice(&hemicube_weights(resolution)),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = uniform_buffer(
            device,
            "probe params",
            bytemuck::bytes_of(&Self::probe_params(&self.motion, ind.probes, resolution, 1)),
        );
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe irradiance"),
            size: ind.probes as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = Self::gather_bind(
            kernel,
            &self.scene,
            &ind.points,
            &ind.gbuffer,
            &weights,
            &out,
            &params,
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gather);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(ind.probes, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        bytemuck::cast_slice::<u8, [f32; 4]>(&self.read(&out)).to_vec()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hemicube_weights_cover_the_hemisphere() {
        for r in [2, 8, 16, 64] {
            let w = hemicube_weights(r);
            assert_eq!(w.len(), 3 * (r * r) as usize);
            let total: f32 = w.iter().sum();
            assert!(
                (total - std::f32::consts::PI).abs() < 1e-4,
                "r = {r}: {total}"
            );
        }
    }
}
