//! Motion blur (M21): what a shutter open for part of the frame sees,
//! gathered each frame from the image, its depths and each pixel's motion
//! since the previous camera and transforms (`motion.wgsl`).

/// A shutter (M21): moving surfaces smear along their motion, as Blender's
/// Cycles renders motion blur with the shutter centred on the frame. The
/// motion is the camera's and the movers' since the poses given as the
/// previous ones ([`crate::Renderer`] gives the last frame's), taken as
/// constant. Off by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlur {
    /// How long the shutter is open, as a share of the time since the
    /// previous poses (Cycles' "shutter", 0.5 by default).
    pub shutter: f32,
    /// The longest smear as a share of the image's height (longer ones are
    /// clamped).
    pub max_blur: f32,
}

impl Default for MotionBlur {
    fn default() -> Self {
        MotionBlur {
            shutter: 0.5,
            max_blur: 0.1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionParams {
    p: [f32; 4],
    tiles: [u32; 4],
}

/// Motion blur's passes and buffers for one image size.
pub(crate) struct MotionPasses {
    prepare: wgpu::ComputePipeline,
    dilate: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// The previous camera (a `CameraUniform`).
    pub(crate) before: wgpu::Buffer,
    /// Per transform its mover's slot plus one; per slot the mover's
    /// previous rows.
    mover_of: wgpu::Buffer,
    pub(crate) moved: wgpu::Buffer,
    src: wgpu::Buffer,
    tiles: wgpu::Buffer,
    binds: Option<[wgpu::BindGroup; 3]>,
    size: (u32, u32),
}

impl MotionPasses {
    /// For a `size` image; `mover_of` maps each transform to its mover's slot
    /// plus one (0: none), for `movers` movers.
    pub(crate) fn new(
        device: &wgpu::Device,
        size: (u32, u32),
        mover_of: &[u32],
        movers: usize,
    ) -> MotionPasses {
        use wgpu::util::DeviceExt;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion"),
            source: wgpu::ShaderSource::Wgsl(include_str!("motion.wgsl").into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let storage = |label, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform = |label, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let tiles_n = size.0.div_ceil(16) as u64 * size.1.div_ceil(16) as u64;
        let mut lookup = mover_of.to_vec();
        if lookup.is_empty() {
            lookup.push(0);
        }
        MotionPasses {
            prepare: pipeline("motion_prepare"),
            dilate: pipeline("motion_dilate"),
            gather: pipeline("motion_gather"),
            params: uniform("motion params", std::mem::size_of::<MotionParams>() as u64),
            before: uniform("motion camera before", 80),
            mover_of: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("motion mover of"),
                contents: bytemuck::cast_slice(&lookup),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            moved: storage("motion moved", movers as u64 * 64),
            src: storage("motion source", size.0 as u64 * size.1 as u64 * 16),
            tiles: storage("motion tiles", 2 * tiles_n * 8),
            binds: None,
            size,
        }
    }

    pub(crate) fn fits(&self, size: (u32, u32)) -> bool {
        self.size == size
    }

    /// Binds the camera, colour, visibility, object and transform buffers.
    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        camera: &wgpu::Buffer,
        colors: &wgpu::Buffer,
        vis: &wgpu::Buffer,
        meshes: &wgpu::Buffer,
        xforms: &wgpu::Buffer,
    ) {
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, buf)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("motion"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        self.binds = Some([
            group(
                &self.prepare,
                &[
                    (0, camera),
                    (1, colors),
                    (2, vis),
                    (3, &self.params),
                    (4, &self.before),
                    (5, meshes),
                    (6, xforms),
                    (7, &self.mover_of),
                    (8, &self.moved),
                    (9, &self.src),
                    (10, &self.tiles),
                ],
            ),
            group(&self.dilate, &[(3, &self.params), (10, &self.tiles)]),
            group(
                &self.gather,
                &[
                    (0, camera),
                    (1, colors),
                    (3, &self.params),
                    (9, &self.src),
                    (10, &self.tiles),
                ],
            ),
        ]);
    }

    /// Records the passes; `movers_moved`: whether any mover's previous
    /// transform differs from its current.
    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        blur: &MotionBlur,
        movers_moved: bool,
    ) {
        let Some([prepare, dilate, gather]) = &self.binds else {
            return;
        };
        let (w, h) = self.size;
        let longest = (blur.max_blur * h as f32).max(1.0);
        let (tx, ty) = (w.div_ceil(16), h.div_ceil(16));
        let params = MotionParams {
            p: [
                blur.shutter.max(0.0),
                longest,
                movers_moved as u32 as f32,
                0.03,
            ],
            tiles: [tx, ty, (0.5 * longest / 16.0).ceil() as u32 + 1, 0],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("motion blur"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prepare);
        pass.set_bind_group(0, prepare, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        pass.set_pipeline(&self.dilate);
        pass.set_bind_group(0, dilate, &[]);
        pass.dispatch_workgroups(tx.div_ceil(8), ty.div_ceil(8), 1);
        pass.set_pipeline(&self.gather);
        pass.set_bind_group(0, gather, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }
}
