//! Interactive rendering of an instanced scene: shared geometry placed many
//! times, lit per pixel with mask cones (no caches), kept on the GPU between
//! frames.

use std::time::{Duration, Instant};

use continuum::glam::DVec3;

use crate::hw::HwScene;
use crate::live::{DisplayDims, display_bind, display_pipeline};
use crate::vcache::VirtualCache;
use crate::{
    COMPUTE_VISIBILITY_BINDINGS, CacheOptions, CacheStats, Camera, CameraUniform,
    HW_VISIBILITY_BINDINGS, InstancedScene, MaskConeKernel, RECT_LIGHT_BINDINGS, RectLight,
    VC_BAKE_BINDINGS, VC_FILL_BINDINGS, VC_SHADE_BINDINGS, VC_UPDATE_BINDINGS, camera_uniform,
    uniform_buffer,
};

/// Where a frame's time goes (see [`InstancedRenderer::frame_profiled`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameProfile {
    /// The cache's bake passes (table updates, texels, borders).
    pub bake: Duration,
    /// The visibility pass (or the single camera pass without a split).
    pub visibility: Duration,
    /// Of `visibility`, the time spent in `queue.submit` on the CPU.
    pub visibility_submit: Duration,
    /// The shading pass, if separate.
    pub shading: Duration,
    /// Display, request readback and the next frame's planning.
    pub readback: Duration,
}

/// The lighting cache's state and its bake passes (table updates, texels,
/// borders).
struct Cache {
    state: VirtualCache,
    passes: [wgpu::ComputePipeline; 3],
    binds: Vec<wgpu::BindGroup>,
}

/// Persistent GPU state for flying through an [`InstancedScene`]: geometry,
/// hierarchies and transforms are uploaded once; a frame uploads the camera,
/// casts a ray per pixel (through hardware ray queries where the kernel uses
/// them) and lights each hit, with a mask cone per pixel or, with a lighting
/// cache, from cached pages (baking the pages the last frame asked for first).
pub struct InstancedRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Camera passes with the bindings each uses: one pass, or a visibility
    /// pass and a shading pass.
    passes: Vec<(wgpu::ComputePipeline, Vec<u32>)>,
    hw: Option<HwScene>,
    display: wgpu::RenderPipeline,
    scene: Vec<(u32, wgpu::Buffer)>,
    camera_buf: wgpu::Buffer,
    dims_buf: wgpu::Buffer,
    binds: Vec<wgpu::BindGroup>,
    display_bind: wgpu::BindGroup,
    colors: wgpu::Buffer,
    cache: Option<Cache>,
    /// The rendered image's size: the target's divided by `scale`.
    width: u32,
    height: u32,
    scale: u32,
    radiance: f32,
    srgb: bool,
}

fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
    tlas: Option<&wgpu::Tlas>,
    keep: &[u32],
) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .filter(|(b, _)| keep.contains(b))
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: *b,
            resource: buf.as_entire_binding(),
        })
        .collect();
    if let Some(tlas) = tlas
        && keep.contains(&32)
    {
        entries.push(wgpu::BindGroupEntry {
            binding: 32,
            resource: tlas.as_binding(),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("instanced render"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

impl InstancedRenderer {
    /// Uploads the scene with positions relative to `origin`; the light is given
    /// relative to `origin` too. Uses the kernel's current prefilter setting.
    /// Renders at `1 / scale` of the target's `width × height` and upscales.
    /// Every pixel gets its own mask cone each frame.
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
        Self::build(
            kernel,
            target_format,
            target_width,
            target_height,
            scale,
            light,
            scene,
            origin,
            radiance,
            None,
        )
    }

    /// [`Self::new`] with a view-dependent lighting cache: pixels read cached
    /// direct light, and each frame bakes (within the budget) the pages the
    /// previous frame lacked. The light and geometry must stay still.
    #[allow(clippy::too_many_arguments)]
    pub fn with_cache(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        light: RectLight,
        scene: &InstancedScene,
        origin: DVec3,
        radiance: f32,
        options: CacheOptions,
    ) -> Self {
        Self::build(
            kernel,
            target_format,
            target_width,
            target_height,
            scale,
            light,
            scene,
            origin,
            radiance,
            Some(options),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        light: RectLight,
        scene: &InstancedScene,
        origin: DVec3,
        radiance: f32,
        cache: Option<CacheOptions>,
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
        let hw = kernel
            .uses_hardware_rays()
            .then(|| kernel.hw_scene(scene, origin));
        let visibility = match (&hw, &kernel.hw_visibility_pipeline) {
            (Some(_), Some(pipeline)) => Some((pipeline.clone(), HW_VISIBILITY_BINDINGS.to_vec())),
            _ => None,
        };
        let shade_bindings = |list: &[u32]| {
            let mut keep = list.to_vec();
            keep.push(31);
            keep
        };
        let passes = match (cache.is_some(), visibility) {
            (false, Some(v)) => vec![
                v,
                (
                    kernel.shade_pipeline.clone(),
                    shade_bindings(RECT_LIGHT_BINDINGS),
                ),
            ],
            (false, None) => vec![(kernel.render_pipeline.clone(), RECT_LIGHT_BINDINGS.to_vec())],
            (true, v) => vec![
                v.unwrap_or_else(|| {
                    (
                        kernel.visibility_pipeline.clone(),
                        COMPUTE_VISIBILITY_BINDINGS.to_vec(),
                    )
                }),
                (kernel.vc_shade_pipeline.clone(), VC_SHADE_BINDINGS.to_vec()),
            ],
        };
        let cache = cache.map(|options| Cache {
            state: VirtualCache::new(&device, &kernel.queue, scene, options, width, height),
            passes: [
                kernel.vc_update_pipeline.clone(),
                kernel.vc_bake_pipeline.clone(),
                kernel.vc_fill_pipeline.clone(),
            ],
            binds: Vec::new(),
        });
        let colors = Self::colors_buffer(&device, width, height);
        let mut renderer = InstancedRenderer {
            device: device.clone(),
            queue: kernel.queue.clone(),
            passes,
            hw,
            display_bind: display_bind(&device, &display, &colors, &dims_buf),
            display,
            scene: scene_buffers,
            camera_buf,
            dims_buf,
            binds: Vec::new(),
            colors,
            cache,
            width,
            height,
            scale,
            radiance,
            srgb: target_format.is_srgb(),
        };
        renderer.targets();
        renderer
    }

    fn colors_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: (width as u64 * height as u64 * 16).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// (Re)creates the per-resolution buffers and every bind group.
    fn targets(&mut self) {
        let device = &self.device;
        let (width, height) = (self.width, self.height);
        self.colors = Self::colors_buffer(device, width, height);
        let vis = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility"),
            size: (width as u64 * height as u64 * 16).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        if let Some(cache) = &mut self.cache {
            cache.state.resize(device, width, height);
        }
        let mut all: Vec<(u32, &wgpu::Buffer)> =
            self.scene.iter().map(|(b, buf)| (*b, buf)).collect();
        all.push((6, &self.camera_buf));
        all.push((7, &self.colors));
        all.push((31, &vis));
        if let Some(cache) = &self.cache {
            all.extend(cache.state.buffers.iter().map(|(b, buf)| (*b, buf)));
            all.push((41, cache.state.params()));
        }
        let tlas = self.hw.as_ref().map(|h| &h.tlas);
        self.binds = self
            .passes
            .iter()
            .map(|(pipeline, keep)| bind_group(device, pipeline, &all, tlas, keep))
            .collect();
        let cache_binds = self.cache.as_ref().map(|cache| {
            cache
                .passes
                .iter()
                .zip([VC_UPDATE_BINDINGS, VC_BAKE_BINDINGS, VC_FILL_BINDINGS])
                .map(|(pipeline, keep)| bind_group(device, pipeline, &all, None, keep))
                .collect()
        });
        if let (Some(cache), Some(binds)) = (&mut self.cache, cache_binds) {
            cache.binds = binds;
        }
        self.display_bind = display_bind(device, &self.display, &self.colors, &self.dims_buf);
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
        (self.width, self.height) = (width, height);
        self.targets();
    }

    /// Whether camera rays go through hardware ray queries.
    pub fn hardware_rays(&self) -> bool {
        self.hw.is_some()
    }

    /// GPU memory held by the scene, in bytes (not counting hardware
    /// acceleration structures or the lighting cache).
    pub fn scene_bytes(&self) -> u64 {
        self.scene.iter().map(|(_, b)| b.size()).sum()
    }

    /// GPU memory held by the lighting cache, in bytes (0 without one).
    pub fn cache_bytes(&self) -> u64 {
        self.cache.as_ref().map_or(0, |c| c.state.bytes())
    }

    /// What the lighting cache did in the last frame.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.state.stats())
    }

    /// Renders one frame from `camera` (relative to the origin) into `target`.
    /// With a lighting cache, this waits for the frame to finish (its page
    /// requests are read back).
    pub fn frame(&mut self, camera: Camera, target: &wgpu::TextureView) {
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
        if let Some(cache) = &mut self.cache {
            let (updates, texels, pages) = cache.state.upload(
                &self.queue,
                self.width,
                self.height,
                camera.vertical_fov_degrees,
            );
            for ((pipeline, group), count) in cache
                .passes
                .iter()
                .zip(&cache.binds)
                .zip([updates, texels, pages])
            {
                if count == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            }
            cache.state.encode_clear(&mut encoder);
        }
        for ((pipeline, _), group) in self.passes.iter().zip(&self.binds) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        if let Some(cache) = &self.cache {
            cache.state.encode_readback(&mut encoder);
        }
        crate::live::encode_display(&mut encoder, &self.display, &self.display_bind, target);
        self.queue.submit([encoder.finish()]);
        if let Some(cache) = &mut self.cache {
            cache.state.finish_frame(&self.device);
        }
    }

    /// [`Self::frame`] split into stages, each submitted and waited for on its
    /// own, for profiling. The stages' sum exceeds a normal frame by the extra
    /// synchronization.
    pub fn frame_profiled(&mut self, camera: Camera, target: &wgpu::TextureView) -> FrameProfile {
        let cam = camera_uniform(camera, self.width, self.height, self.radiance);
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [!self.srgb as u32, self.scale],
        };
        self.queue
            .write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        self.wait();
        let mut profile = FrameProfile::default();
        let start = Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some(cache) = &mut self.cache {
            let (updates, texels, pages) = cache.state.upload(
                &self.queue,
                self.width,
                self.height,
                camera.vertical_fov_degrees,
            );
            for ((pipeline, group), count) in cache
                .passes
                .iter()
                .zip(&cache.binds)
                .zip([updates, texels, pages])
            {
                if count == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            }
            cache.state.encode_clear(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
        self.wait();
        profile.bake = start.elapsed();
        for (k, ((pipeline, _), group)) in self.passes.iter().zip(&self.binds).enumerate() {
            let start = Instant::now();
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
            }
            let commands = encoder.finish();
            let submit = Instant::now();
            self.queue.submit([commands]);
            let submitted = submit.elapsed();
            self.wait();
            if k == 0 {
                profile.visibility = start.elapsed();
                profile.visibility_submit = submitted;
            } else {
                profile.shading += start.elapsed();
            }
        }
        let start = Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some(cache) = &self.cache {
            cache.state.encode_readback(&mut encoder);
        }
        crate::live::encode_display(&mut encoder, &self.display, &self.display_bind, target);
        self.queue.submit([encoder.finish()]);
        if let Some(cache) = &mut self.cache {
            cache.state.finish_frame(&self.device);
        }
        self.wait();
        profile.readback = start.elapsed();
        profile
    }

    /// The last frame's linear colors (waits for the GPU).
    pub fn read_colors(&self) -> Vec<[f32; 3]> {
        let size = self.colors.size();
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.colors, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        self.wait();
        let data = slice.get_mapped_range().expect("mapped colors");
        let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
        values[..(self.width * self.height) as usize]
            .iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }

    /// Blocks until submitted GPU work is done.
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
    }
}
