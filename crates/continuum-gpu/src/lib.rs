//! GPU backend for Continuum (wgpu compute; Metal, Vulkan, DX12).
//!
//! For now one kernel: mask cones from receivers to a rectangular light, a port
//! of `continuum::cone3` (see `mask_cones.wgsl`). Every receiver tests every
//! triangle; hierarchy traversal on the GPU comes later.

use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use continuum::glam::Vec3;
use wgpu::util::DeviceExt;

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
    pub adapter_name: String,
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
        Some(MaskConeKernel {
            device,
            queue,
            pipeline,
            render_pipeline,
            adapter_name,
        })
    }

    /// Irradiance per unit radiance at each receiver `(point, normal)`, and the
    /// fastest of `runs` GPU executions (submit to completion, excluding uploads
    /// and readback).
    pub fn irradiance(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        runs: usize,
    ) -> (Vec<f32>, Duration) {
        let (values, _, best) = self.run(receivers, light, meshes, runs, 0);
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
                width as f32,
                height as f32,
                (camera.vertical_fov_degrees.to_radians() * 0.5).tan(),
                radiance,
            ],
        };
        let uniform = LightUniform {
            corner: v4(light.corner),
            edge_u: v4(light.edge_u),
            edge_v: v4(light.edge_v),
            counts: [meshes.len() as u32, 0, hierarchy as u32, 0],
        };
        let (tris, ranges) = flatten(meshes);
        let (nodes, items) = mesh_hierarchy(meshes);
        let albedo4: Vec<[f32; 4]> = albedo.iter().map(|&a| v4(a)).collect();
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
        let camera_buf = init(
            "camera",
            bytemuck::bytes_of(&cam),
            wgpu::BufferUsages::UNIFORM,
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
        let albedo_buf = init(
            "albedo",
            bytemuck::cast_slice(&albedo4),
            wgpu::BufferUsages::STORAGE,
        );
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
        let debug_buf = storage("debug", 256, wgpu::BufferUsages::STORAGE);
        let staging = storage(
            "staging",
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &self.render_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buf.as_entire_binding(),
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
                    binding: 5,
                    resource: debug_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: colors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: albedo_buf.as_entire_binding(),
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
        let mut best = Duration::MAX;
        for _ in 0..runs.max(1) {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.render_pipeline);
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

    /// Per-bin coverage (64 per receiver) as the kernel computed it, for debugging.
    pub fn coverage(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
    ) -> Vec<[f32; 64]> {
        let (_, debug, _) = self.run(receivers, light, meshes, 1, 1);
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
        let (_, debug, _) = self.run(&[receiver], light, meshes, 1, 3);
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
        let (_, debug, _) = self.run(receivers, light, meshes, 1, 2);
        debug.chunks(64).map(|c| c.try_into().unwrap()).collect()
    }

    fn run(
        &self,
        receivers: &[(Vec3, Vec3)],
        light: RectLight,
        meshes: &[Vec<Tri>],
        runs: usize,
        debug_mode: u32,
    ) -> (Vec<f32>, Vec<f32>, Duration) {
        let debug = debug_mode != 0;
        let v4 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let uniform = LightUniform {
            corner: v4(light.corner),
            edge_u: v4(light.edge_u),
            edge_v: v4(light.edge_v),
            counts: [meshes.len() as u32, debug_mode, 0, 0],
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
