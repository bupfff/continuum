//! Interactive rendering of an instanced scene: shared geometry placed many
//! times, lit per pixel with mask cones (no caches), kept on the GPU between
//! frames.

use continuum::glam::DVec3;

use crate::live::{DisplayDims, display_bind, display_pipeline};
use crate::{
    Camera, CameraUniform, InstancedScene, MaskConeKernel, RECT_LIGHT_BINDINGS, RectLight,
    camera_uniform, uniform_buffer,
};

/// Persistent GPU state for flying through an [`InstancedScene`]: geometry,
/// hierarchies and transforms are uploaded once; a frame uploads the camera,
/// casts a ray per pixel and runs a mask cone at each hit.
pub struct InstancedRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass: wgpu::ComputePipeline,
    display: wgpu::RenderPipeline,
    scene: Vec<(u32, wgpu::Buffer)>,
    camera_buf: wgpu::Buffer,
    dims_buf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    display_bind: wgpu::BindGroup,
    /// The rendered image's size: the target's divided by `scale`.
    width: u32,
    height: u32,
    scale: u32,
    radiance: f32,
    srgb: bool,
}

impl InstancedRenderer {
    /// Uploads the scene with positions relative to `origin`; the light is given
    /// relative to `origin` too. Uses the kernel's current prefilter setting.
    /// Renders at `1 / scale` of the target's `width × height` and upscales.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        light: RectLight,
        scene: &InstancedScene,
        origin: DVec3,
        radiance: f32,
    ) -> Self {
        let device = kernel.device.clone();
        let scale = scale.max(1);
        let (width, height) = (target_width.div_ceil(scale), target_height.div_ceil(scale));
        let scene_buffers = kernel.instanced_buffers(light, scene, origin);
        let camera_buf = uniform_buffer(
            &device,
            "camera",
            &[0; std::mem::size_of::<CameraUniform>()],
        );
        let dims_buf = uniform_buffer(&device, "dims", &[0; std::mem::size_of::<DisplayDims>()]);
        let display = display_pipeline(&device, target_format);
        let pass = kernel.render_pipeline.clone();
        let (bind, display_bind) = Self::targets(
            &device,
            &pass,
            &display,
            &scene_buffers,
            &camera_buf,
            &dims_buf,
            width,
            height,
        );
        InstancedRenderer {
            device,
            queue: kernel.queue.clone(),
            pass,
            display,
            scene: scene_buffers,
            camera_buf,
            dims_buf,
            bind,
            display_bind,
            width,
            height,
            scale,
            radiance,
            srgb: target_format.is_srgb(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn targets(
        device: &wgpu::Device,
        pass: &wgpu::ComputePipeline,
        display: &wgpu::RenderPipeline,
        scene: &[(u32, wgpu::Buffer)],
        camera_buf: &wgpu::Buffer,
        dims_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let colors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: (width as u64 * height as u64 * 16).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut entries: Vec<wgpu::BindGroupEntry> = scene
            .iter()
            .filter(|(b, _)| RECT_LIGHT_BINDINGS.contains(b))
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: camera_buf.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: colors.as_entire_binding(),
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instanced render"),
            layout: &pass.get_bind_group_layout(0),
            entries: &entries,
        });
        (bind, display_bind(device, display, &colors, dims_buf))
    }

    /// Call when the surface size changes.
    pub fn resize(&mut self, target_width: u32, target_height: u32) {
        let (width, height) = (
            target_width.div_ceil(self.scale),
            target_height.div_ceil(self.scale),
        );
        if (width, height) == (self.width, self.height) || width == 0 || height == 0 {
            return;
        }
        (self.bind, self.display_bind) = Self::targets(
            &self.device,
            &self.pass,
            &self.display,
            &self.scene,
            &self.camera_buf,
            &self.dims_buf,
            width,
            height,
        );
        (self.width, self.height) = (width, height);
    }

    /// GPU memory held by the scene, in bytes.
    pub fn scene_bytes(&self) -> u64 {
        self.scene.iter().map(|(_, b)| b.size()).sum()
    }

    /// Renders one frame from `camera` (relative to the origin) into `target`.
    pub fn frame(&self, camera: Camera, target: &wgpu::TextureView) {
        let cam = camera_uniform(camera, self.width, self.height, self.radiance);
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [!self.srgb as u32, self.scale],
        };
        self.queue
            .write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pass);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        crate::live::encode_display(&mut encoder, &self.display, &self.display_bind, target);
        self.queue.submit([encoder.finish()]);
    }

    /// Blocks until submitted GPU work is done.
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
    }
}
