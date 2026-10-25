//! Interactive rendering: the scene, its caches and moving meshes stay on the GPU.

use bytemuck::{Pod, Zeroable};
use continuum::glam::Vec3;
use wgpu::util::DeviceExt;

use crate::hierarchy::{
    COVERAGE_SAMPLES, NONE, NodeGpu, flatten, mesh_bounds, mesh_hierarchy, mesh_hierarchy_of,
    padded, triangle_hierarchy,
};
use crate::lightmap::{DYNAMIC, Lightmap, hemicube_weights};
use crate::{
    Camera, CameraUniform, Frame, Hierarchy, MaskConeKernel, ReceiverGpu, RectLight, Tri,
    uniform_buffer,
};
use continuum::mesh::Surface;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DisplayDims {
    pub size: [u32; 2],
    pub flags: [u32; 2],
}

/// The pass that tonemaps the colors buffer into a surface texture.
pub(crate) fn display_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("display"),
        source: wgpu::ShaderSource::Wgsl(include_str!("display.wgsl").into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    })
}

pub(crate) fn display_bind(
    device: &wgpu::Device,
    display: &wgpu::RenderPipeline,
    colors: &wgpu::Buffer,
    dims: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("display"),
        layout: &display.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dims.as_entire_binding(),
            },
        ],
    })
}

/// Records the display pass into `target`.
pub(crate) fn encode_display(
    encoder: &mut wgpu::CommandEncoder,
    display: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites>,
) {
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
        timestamp_writes,
        ..Default::default()
    });
    pass.set_pipeline(display);
    pass.set_bind_group(0, bind, &[]);
    pass.draw(0..3, 0..1);
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
    /// The camera passes: the cached render, or a visibility pass then cached shading.
    camera_passes: Vec<wgpu::ComputePipeline>,
    display: wgpu::RenderPipeline,
    mark: wgpu::ComputePipeline,
    rebake: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    scene: Vec<(u32, wgpu::Buffer)>,
    camera_buf: wgpu::Buffer,
    dims_buf: wgpu::Buffer,
    colors_buf: wgpu::Buffer,
    camera_binds: Vec<wgpu::BindGroup>,
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
    gather_bind: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProbeParams {
    counts: [u32; 4],
    /// `[probes on static charts, 0, 0, 0]`
    extra: [u32; 4],
    moving: [[u32; 4]; MAX_MOVING],
    bounds: [[f32; 4]; 2 * MAX_MOVING],
}

/// Dynamic meshes a [`LiveRenderer`] with probes supports.
pub const MAX_MOVING: usize = 16;

/// Hemicube resolution of live probes: `r × r` bins on the top face and
/// `r × r/2` on each side (768 bins per probe).
pub const PROBE_RESOLUTION: u32 = 16;

/// A moved mesh: its index, new triangles, and new normals if they changed.
type MeshChange<'a> = (usize, &'a [Tri], Option<&'a [[Vec3; 3]]>);

/// What [`LiveRenderer`] keeps to move dynamic meshes.
struct Motion {
    meshes: Vec<Vec<Tri>>,
    ranges: Vec<[u32; 4]>,
    bounds: Vec<(Vec3, Vec3)>,
    dynamic: Vec<bool>,
    blocks: u32,
    probes: u32,
    static_probes: u32,
    /// Probes on moving meshes (they follow index `static_probes` on): mesh,
    /// triangle in the mesh, barycentric coordinates.
    moving_probes: Vec<(usize, usize, Vec3)>,
    /// Current shading normals of each moving mesh.
    normals: Vec<Vec<[Vec3; 3]>>,
    uniform: wgpu::Buffer,
    args: wgpu::Buffer,
    /// The moving meshes and their bounds (`ProbeParams`, mode 0), read by the
    /// re-bake and the probe gather.
    params: wgpu::Buffer,
    /// A hierarchy of the static meshes only, for the static coverage cache and
    /// probe G-buffers.
    static_nodes: wgpu::Buffer,
    static_items: wgpu::Buffer,
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
        assert!(
            kernel.sky() == Vec3::ZERO,
            "the lighting caches do not support a sky yet"
        );
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
        let camera_passes = if kernel.split.get() {
            vec![
                kernel.visibility_pipeline.clone(),
                kernel.shade_cached_pipeline.clone(),
            ]
        } else {
            vec![kernel.cached_pipeline.clone()]
        };
        let (colors_buf, camera_binds, display_bind) = Self::targets(
            &device,
            &camera_passes,
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
            camera_passes,
            display,
            mark: kernel.mark_pipeline.clone(),
            rebake: kernel.rebake_pipeline.clone(),
            gather: kernel.probe_gather_pipeline.clone(),
            scene,
            camera_buf,
            dims_buf,
            colors_buf,
            camera_binds,
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
        let params = &motion.params;
        let storage = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        let static_probes = motion.static_probes;
        let gbuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe G-buffers"),
            size: static_probes.max(1) as u64 * bins as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let weights = storage("bin weights", bytemuck::cast_slice(&hemicube_weights(r)));
        // The G-buffer holds static geometry only: trace through a hierarchy of
        // the static meshes.
        let (nodes, items) = (&motion.static_nodes, &motion.static_items);
        let s = |b| binding(scene, b);
        let build_bind = bind_group(
            device,
            "probe build",
            &kernel.probe_build_pipeline,
            &[
                (0, s(0)),
                (2, s(2)),
                (3, s(3)),
                (9, nodes),
                (10, items),
                (25, s(25)),
                (26, s(26)),
                (30, s(30)),
                (11, s(11)),
                (18, &points),
                (19, &gbuffer),
                (22, params),
            ],
        );
        let gather_bind =
            Self::gather_bind(kernel, scene, &points, &gbuffer, &weights, s(21), params);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernel.probe_build_pipeline);
            pass.set_bind_group(0, &build_bind, &[]);
            pass.dispatch_workgroups(bins / 64, static_probes, 1);
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
                (30, s(30)),
                (11, s(11)),
                (12, s(12)),
                (13, s(13)),
                (23, s(23)),
                (18, points),
                (19, gbuffer),
                (20, weights),
                (21, out),
                (22, params),
            ],
        )
    }

    /// Re-bake and gather parameters: every dynamic mesh with its current bounds.
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
        params.extra = [motion.static_probes, 0, 0, 0];
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
        let dynamic: Vec<bool> = meshes
            .iter()
            .zip(&ranges)
            .map(|(mesh, r)| mesh.is_empty() || lightmap.tri_chart[r[0] as usize] & DYNAMIC != 0)
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
        let planes_cpu = lightmap.block_planes(light);
        let planes = storage(
            "block planes",
            bytemuck::cast_slice(&planes_cpu),
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
        // Static coverage cache: a slot per block that can receive direct light
        // (the others have a plane that excludes everything).
        let never = [0.0, 0.0, 0.0, -1.0];
        let mut slots = 0u32;
        let block_slot: Vec<u32> = (0..blocks as usize)
            .map(|b| {
                if lightmap.blocks.is_empty() || planes_cpu[b * crate::BLOCK_PLANES] == never {
                    NONE
                } else {
                    slots += 1;
                    slots - 1
                }
            })
            .collect();
        let cov_size = slots.max(1) as u64 * 64 * 64 * 4;
        assert!(
            cov_size <= device.limits().max_storage_buffer_binding_size,
            "static coverage cache ({cov_size} bytes) exceeds the device's storage buffer limit"
        );
        let static_cov = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("static coverage"),
            size: cov_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let block_slot = storage(
            "block slots",
            bytemuck::cast_slice(&[block_slot, vec![NONE]].concat()),
            wgpu::BufferUsages::empty(),
        );
        let (nodes, items) = mesh_hierarchy_of(meshes, |m| !dynamic[m]);
        let static_nodes = storage(
            "static nodes",
            bytemuck::cast_slice(&nodes),
            wgpu::BufferUsages::empty(),
        );
        let static_items = storage(
            "static items",
            bytemuck::cast_slice(&items),
            wgpu::BufferUsages::empty(),
        );
        let probes = lightmap.probes.len() as u32;
        let normals = meshes
            .iter()
            .zip(&ranges)
            .zip(&dynamic)
            .map(|((mesh, r), &d)| {
                if !d {
                    return Vec::new();
                }
                let first = 3 * r[0] as usize;
                lightmap.tri_normals[first..first + 3 * mesh.len()]
                    .chunks(3)
                    .map(|n| [0, 1, 2].map(|k| Vec3::from_slice(&n[k][..3])))
                    .collect()
            })
            .collect();
        let params = uniform_buffer(
            device,
            "dynamic meshes",
            &[0; std::mem::size_of::<ProbeParams>()],
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
                (25, s(25)),
                (26, s(26)),
                (30, s(30)),
                (29, s(29)),
                (17, &dirty),
                (22, &params),
                (27, &static_cov),
                (28, &block_slot),
            ],
        );
        let motion = Motion {
            meshes: meshes.to_vec(),
            ranges,
            bounds: meshes.iter().map(|m| mesh_bounds(m)).collect(),
            dynamic,
            blocks,
            probes,
            static_probes: lightmap.static_probes as u32,
            moving_probes: lightmap.moving_probes.clone(),
            normals,
            uniform,
            args,
            params,
            static_nodes,
            static_items,
            mark_bind,
            rebake_bind,
        };
        kernel.queue.write_buffer(
            &motion.params,
            0,
            bytemuck::bytes_of(&Self::probe_params(&motion, probes, PROBE_RESOLUTION, 0)),
        );
        // Fill the static coverage cache once (the texels themselves were baked
        // through the whole scene already).
        let bake_bind = bind_group(
            device,
            "bake static",
            &kernel.bake_static_pipeline,
            &[
                (0, s(0)),
                (1, &receivers),
                (2, s(2)),
                (3, s(3)),
                (5, s(5)),
                (9, &motion.static_nodes),
                (10, &motion.static_items),
                (25, s(25)),
                (26, s(26)),
                (30, s(30)),
                (29, s(29)),
                (27, &static_cov),
                (28, &block_slot),
            ],
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernel.bake_static_pipeline);
            pass.set_bind_group(0, &bake_bind, &[]);
            pass.dispatch_workgroups(blocks, 1, 1);
        }
        kernel.queue.submit([encoder.finish()]);
        motion
    }

    #[allow(clippy::too_many_arguments)]
    fn targets(
        device: &wgpu::Device,
        camera_passes: &[wgpu::ComputePipeline],
        display: &wgpu::RenderPipeline,
        scene: &[(u32, wgpu::Buffer)],
        camera_buf: &wgpu::Buffer,
        dims_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> (wgpu::Buffer, Vec<wgpu::BindGroup>, wgpu::BindGroup) {
        let size = (width as u64 * height as u64 * 16).max(16);
        let buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let colors_buf = buffer("colors");
        let vis_buf = buffer("visibility");
        let mut entries: Vec<(u32, &wgpu::Buffer)> = scene
            .iter()
            .filter(|(b, _)| !crate::PER_PIXEL_ONLY.contains(b))
            .map(|(b, buf)| (*b, buf))
            .collect();
        entries.push((6, camera_buf));
        entries.push((7, &colors_buf));
        let camera_binds = if camera_passes.len() == 2 {
            let visibility: Vec<(u32, &wgpu::Buffer)> = entries
                .iter()
                .copied()
                .chain([(31, &vis_buf)])
                .filter(|(b, _)| crate::VISIBILITY_BINDINGS.contains(b))
                .collect();
            entries.push((31, &vis_buf));
            vec![
                bind_group(device, "live visibility", &camera_passes[0], &visibility),
                bind_group(device, "live shading", &camera_passes[1], &entries),
            ]
        } else {
            vec![bind_group(
                device,
                "live compute",
                &camera_passes[0],
                &entries,
            )]
        };
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
        (colors_buf, camera_binds, display_bind)
    }

    /// Call when the surface size changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) == (self.width, self.height) || width == 0 || height == 0 {
            return;
        }
        (self.colors_buf, self.camera_binds, self.display_bind) = Self::targets(
            &self.device,
            &self.camera_passes,
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
        let changes: Vec<_> = changes
            .iter()
            .map(|(m, tris)| (*m, tris.as_slice(), None))
            .collect();
        self.update(&changes);
    }

    /// [`Self::update_meshes`] for meshes that may also rotate: uploads their
    /// shading normals too.
    pub fn update_surfaces(&mut self, changes: &[(usize, &Surface)]) {
        let changes: Vec<_> = changes
            .iter()
            .map(|(m, s)| (*m, s.triangles.as_slice(), Some(s.normals.as_slice())))
            .collect();
        self.update(&changes);
    }

    fn update(&mut self, changes: &[MeshChange]) {
        let m = &mut self.motion;
        let mut boxes = Vec::new();
        for &(mesh, tris, normals) in changes {
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
            if let Some(normals) = normals {
                m.normals[mesh] = normals.to_vec();
                let normals: Vec<[f32; 4]> = normals
                    .iter()
                    .flatten()
                    .map(|n| [n.x, n.y, n.z, 0.0])
                    .collect();
                self.queue.write_buffer(
                    binding(&self.scene, 24),
                    m.ranges[mesh][0] as u64 * 48,
                    bytemuck::cast_slice(&normals),
                );
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
                let (nodes, items, cov) = triangle_hierarchy(tris, root, first);
                self.queue.write_buffer(
                    binding(&self.scene, 25),
                    root as u64 * std::mem::size_of::<NodeGpu>() as u64,
                    bytemuck::cast_slice(&nodes),
                );
                self.queue.write_buffer(
                    binding(&self.scene, 29),
                    root as u64 * (COVERAGE_SAMPLES * 4) as u64,
                    bytemuck::cast_slice(&cov),
                );
                self.queue.write_buffer(
                    binding(&self.scene, 26),
                    first as u64 * 4,
                    bytemuck::cast_slice(&items),
                );
            }
            m.meshes[mesh] = tris.to_vec();
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
        let params = Self::probe_params(m, m.probes, PROBE_RESOLUTION, 0);
        self.queue
            .write_buffer(&m.params, 0, bytemuck::bytes_of(&params));
        // Probes on moving meshes follow them.
        if let Some(ind) = &self.indirect
            && !m.moving_probes.is_empty()
        {
            let points: Vec<ReceiverGpu> = m
                .moving_probes
                .iter()
                .map(|&(mesh, t, b)| {
                    let (tri, n) = (m.meshes[mesh][t], m.normals[mesh][t]);
                    let normal = (n[0] * b.x + n[1] * b.y + n[2] * b.z).normalize();
                    let p = tri[0] * b.x + tri[1] * b.y + tri[2] * b.z + normal * 1e-4;
                    ReceiverGpu {
                        point: [p.x, p.y, p.z, 0.0],
                        normal: [normal.x, normal.y, normal.z, 0.0],
                    }
                })
                .collect();
            self.queue.write_buffer(
                &ind.points,
                m.static_probes as u64 * std::mem::size_of::<ReceiverGpu>() as u64,
                bytemuck::cast_slice(&points),
            );
        }
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
        self.encode_gather(&mut encoder);
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
        for (pipeline, group) in self.camera_passes.iter().zip(&self.camera_binds) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
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
