//! Interactive rendering of an instanced scene: shared geometry placed many
//! times, lit per pixel with mask cones (no caches), kept on the GPU between
//! frames.

use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};

use crate::hw::HwScene;
use crate::live::{DisplayDims, display_bind, display_pipeline};
use crate::vcache::VirtualCache;
use crate::{
    COMPUTE_VISIBILITY_BINDINGS, CacheOptions, CacheStats, Camera, CameraUniform,
    HW_VISIBILITY_BINDINGS, InstancedScene, Light, LightGpu, MaskConeKernel, RECT_LIGHT_BINDINGS,
    RectLight, SceneTextures, VC_BAKE_BINDINGS, VC_BAKE_SKY_BINDINGS, VC_FILL_BINDINGS,
    VC_SELF_BAKE_BINDINGS, VC_SHADE_BINDINGS, VC_UPDATE_BINDINGS, camera_uniform, uniform_buffer,
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
    /// Display and the copies for readback.
    pub readback: Duration,
    /// Planning the frame's bake on the CPU (from the requests of two frames
    /// before).
    pub planning: Duration,
}

impl Cache {
    /// The bake passes with `counts` invocations each (skipping empty ones),
    /// timed on the GPU where the device has timestamps: the first pass that
    /// runs writes query 0, the last query 1.
    fn encode_bake(&self, encoder: &mut wgpu::CommandEncoder, counts: [u32; 4]) {
        let ran: Vec<usize> = (0..counts.len()).filter(|&k| counts[k] > 0).collect();
        let set = self.state.timer.as_ref().map(|t| &t.set);
        for &k in &ran {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: timestamps(set, &ran, k, 0),
            });
            pass.set_pipeline(&self.passes[k]);
            pass.set_bind_group(0, &self.binds[k], &[]);
            pass.dispatch_workgroups(counts[k].div_ceil(64), 1, 1);
        }
    }
}

/// Timestamp writes for pass `k` of `ran` (the passes that run): the first
/// writes query `begin`, the last `begin + 1`.
fn timestamps<'a>(
    set: Option<&'a wgpu::QuerySet>,
    ran: &[usize],
    k: usize,
    begin: u32,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    let first = ran.first() == Some(&k);
    let last = ran.last() == Some(&k);
    set.filter(|_| first || last)
        .map(|query_set| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: first.then_some(begin),
            end_of_pass_write_index: last.then_some(begin + 1),
        })
}

/// The lighting cache's state and its bake passes (table updates, sun
/// texels, sky faces, sky interpolation and borders).
struct Cache {
    state: VirtualCache,
    passes: [wgpu::ComputePipeline; 4],
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
    /// An empty pass that timestamps the end of the camera passes.
    mark: wgpu::ComputePipeline,
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
    /// Lights in the scene's light buffer (binding 52).
    lights: usize,
    /// The scene's base color textures (bindings 55 and 56).
    textures: SceneTextures,
    srgb: bool,
}

fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
    tlas: Option<&wgpu::Tlas>,
    textures: Option<&SceneTextures>,
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
    if let Some(t) = textures
        && keep.contains(&55)
    {
        entries.push(wgpu::BindGroupEntry {
            binding: 55,
            resource: wgpu::BindingResource::TextureView(&t.view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 56,
            resource: wgpu::BindingResource::Sampler(&t.sampler),
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
            &[Light::white(light, radiance)],
            scene,
            origin,
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
            &[Light::white(light, radiance)],
            scene,
            origin,
            Some(options),
        )
    }

    /// [`Self::new`] lit by several lights, each with RGB radiance (at most
    /// [`crate::MAX_LIGHTS`]), and the kernel's sky.
    #[allow(clippy::too_many_arguments)]
    pub fn new_lights(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
    ) -> Self {
        Self::build(
            kernel,
            target_format,
            target_width,
            target_height,
            scale,
            lights,
            scene,
            origin,
            None,
        )
    }

    /// [`Self::with_cache`] lit by several lights. The cache keeps each
    /// light's irradiance per unit radiance in its own channel, so
    /// [`Self::set_light_radiance`] re-colors a light with no bake.
    #[allow(clippy::too_many_arguments)]
    pub fn with_cache_lights(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
        options: CacheOptions,
    ) -> Self {
        Self::build(
            kernel,
            target_format,
            target_width,
            target_height,
            scale,
            lights,
            scene,
            origin,
            Some(options),
        )
    }

    /// Changes light `index`'s RGB radiance from the next frame. Lighting is
    /// linear in emitted power: per-pixel frames shade with it, and the cache
    /// re-weights its stored channels, baking nothing.
    pub fn set_light_radiance(&mut self, index: usize, radiance: Vec3) {
        assert!(index < self.lights, "no light {index}");
        let buffer = &self
            .scene
            .iter()
            .find(|(b, _)| *b == 52)
            .expect("lights buffer")
            .1;
        let offset = (index * std::mem::size_of::<LightGpu>() + 48) as u64;
        self.queue.write_buffer(
            buffer,
            offset,
            bytemuck::cast_slice(&[radiance.x, radiance.y, radiance.z, 0.0]),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
        cache: Option<CacheOptions>,
    ) -> Self {
        let device = kernel.device.clone();
        let scale = scale.max(1);
        let (width, height) = (target_width.div_ceil(scale), target_height.div_ceil(scale));
        let scene_buffers = kernel.instanced_buffers(lights, scene, origin);
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
            state: VirtualCache::new(
                &device,
                &kernel.queue,
                scene,
                options,
                kernel.sky() != Vec3::ZERO,
                lights.len() as u32,
                width,
                height,
            ),
            passes: [
                kernel.vc_update_pipeline.clone(),
                kernel.vc_bake_pipeline.clone(),
                kernel.vc_bake_sky_pipeline.clone(),
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
            mark: kernel.mark_pipeline_empty.clone(),
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
            // Shading reads the lights' buffer; the camera's radiance is unused.
            radiance: lights[0].radiance.x,
            lights: lights.len(),
            textures: kernel.scene_textures(&scene.textures),
            srgb: target_format.is_srgb(),
        };
        renderer.targets();
        renderer.bake_self_masks(kernel);
        renderer
    }

    /// Bakes each geometry's own sky occlusion once (see
    /// `CacheOptions::self_spacing`), waiting for it.
    fn bake_self_masks(&mut self, kernel: &MaskConeKernel) {
        let Some(cache) = &mut self.cache else { return };
        let count = std::mem::take(&mut cache.state.self_texels);
        if count == 0 {
            return;
        }
        let all: Vec<(u32, &wgpu::Buffer)> = self
            .scene
            .iter()
            .map(|(b, buf)| (*b, buf))
            .chain(cache.state.buffers.iter().map(|(b, buf)| (*b, buf)))
            .collect();
        let pipeline = &kernel.vc_self_bake_pipeline;
        let group = bind_group(
            &self.device,
            pipeline,
            &all,
            None,
            None,
            VC_SELF_BAKE_BINDINGS,
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            let groups = (6 * count).div_ceil(64);
            // Within the per-dimension dispatch limit.
            pass.dispatch_workgroups(groups.min(65_535), groups.div_ceil(65_535), 1);
        }
        self.queue.submit([encoder.finish()]);
        self.wait();
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
            .map(|(pipeline, keep)| {
                bind_group(device, pipeline, &all, tlas, Some(&self.textures), keep)
            })
            .collect();
        let cache_binds = self.cache.as_ref().map(|cache| {
            cache
                .passes
                .iter()
                .zip([
                    VC_UPDATE_BINDINGS,
                    VC_BAKE_BINDINGS,
                    VC_BAKE_SKY_BINDINGS,
                    VC_FILL_BINDINGS,
                ])
                .map(|(pipeline, keep)| bind_group(device, pipeline, &all, None, None, keep))
                .collect()
        });
        if let (Some(cache), Some(binds)) = (&mut self.cache, cache_binds) {
            cache.binds = binds;
        }
        self.display_bind = display_bind(device, &self.display, &self.colors, &self.dims_buf);
    }

    /// Camera pass `k` (of `passes`), timed on the GPU where the cache has
    /// timestamps: the first pass writes query 2 as it begins.
    fn encode_camera_pass(&self, encoder: &mut wgpu::CommandEncoder, k: usize) {
        let set = self
            .cache
            .as_ref()
            .and_then(|c| c.state.timer.as_ref())
            .map(|t| &t.set);
        let (pipeline, _) = &self.passes[k];
        // Only the first writes (query 2); the display pass ends the span.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: set.filter(|_| k == 0).map(|query_set| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: None,
                }
            }),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.binds[k], &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// Where the cache has timestamps: an empty pass whose beginning writes
    /// query 3, ending the camera passes' span.
    fn encode_camera_end(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(timer) = self.cache.as_ref().and_then(|c| c.state.timer.as_ref()) else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("camera end"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &timer.set,
                beginning_of_pass_write_index: Some(3),
                end_of_pass_write_index: None,
            }),
        });
        pass.set_pipeline(&self.mark);
        pass.dispatch_workgroups(1, 1, 1);
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

    /// What the lighting cache did in the latest frame read back (two frames
    /// behind the latest frame rendered).
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.state.stats())
    }

    /// Renders one frame from `camera` (relative to the origin) into `target`.
    /// With a lighting cache, this first plans the frame's bake from the page
    /// requests of two frames before (waiting for that frame only).
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
        if let Some(cache) = &mut self.cache {
            cache.state.begin_frame(&self.device);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some(cache) = &mut self.cache {
            // Before the bake, which may count its work in the counters.
            cache.state.encode_clear(&mut encoder);
            let counts = cache.state.upload(
                &self.queue,
                self.width,
                self.height,
                camera.vertical_fov_degrees,
            );
            cache.encode_bake(&mut encoder, counts);
        }
        for k in 0..self.passes.len() {
            self.encode_camera_pass(&mut encoder, k);
        }
        self.encode_camera_end(&mut encoder);
        if let Some(cache) = &mut self.cache {
            cache.state.encode_readback(&mut encoder);
        }
        crate::live::encode_display(
            &mut encoder,
            &self.display,
            &self.display_bind,
            target,
            None,
        );
        if let Some(cache) = &self.cache {
            cache.state.encode_timestamps(&mut encoder);
        }
        let index = self.queue.submit([encoder.finish()]);
        if let Some(cache) = &mut self.cache {
            cache.state.after_submit(index);
        }
    }

    /// Waits for every submitted frame and reads back what the cache has not
    /// read yet (for reporting: [`Self::cache_log`] is then complete).
    pub fn finish(&mut self) {
        self.wait();
        if let Some(cache) = &mut self.cache {
            cache.state.drain(&self.device);
        }
    }

    /// What the lighting cache did in every frame so far. Shading counters
    /// arrive two frames late; call [`Self::finish`] first for all of them.
    pub fn cache_log(&self) -> &[CacheStats] {
        self.cache.as_ref().map_or(&[], |c| c.state.log())
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
        if let Some(cache) = &mut self.cache {
            cache.state.begin_frame(&self.device);
        }
        profile.planning = start.elapsed();
        let start = Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some(cache) = &mut self.cache {
            cache.state.encode_clear(&mut encoder);
            let counts = cache.state.upload(
                &self.queue,
                self.width,
                self.height,
                camera.vertical_fov_degrees,
            );
            cache.encode_bake(&mut encoder, counts);
        }
        self.queue.submit([encoder.finish()]);
        self.wait();
        profile.bake = start.elapsed();
        for k in 0..self.passes.len() {
            let start = Instant::now();
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.encode_camera_pass(&mut encoder, k);
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
        if let Some(cache) = &mut self.cache {
            // Split into submissions, its GPU timings would mislead the bake
            // budget.
            cache.state.untimed();
            cache.state.encode_readback(&mut encoder);
        }
        crate::live::encode_display(
            &mut encoder,
            &self.display,
            &self.display_bind,
            target,
            None,
        );
        if let Some(cache) = &self.cache {
            cache.state.encode_timestamps(&mut encoder);
        }
        let index = self.queue.submit([encoder.finish()]);
        if let Some(cache) = &mut self.cache {
            cache.state.after_submit(index);
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
