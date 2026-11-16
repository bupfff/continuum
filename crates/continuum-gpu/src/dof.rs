//! Depth of field (M20): a thin lens's blur gathered from the pinhole image
//! and its depths each frame (`dof.wgsl`).

/// A thin lens (M20): points off the plane in focus spread over discs, as
/// Blender's Cycles renders a camera with depth of field (a circular
/// aperture, no blades). Off by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthOfField {
    /// The distance to the plane in focus, along the view direction, in
    /// the scene's units.
    pub focus_distance: f32,
    /// The aperture's radius in the scene's units: a 50 mm lens at f/2 has
    /// 12.5 mm (Blender's f-stop is the focal length over twice this).
    pub aperture_radius: f32,
    /// The largest disc's radius as a share of the image's height (blur
    /// beyond it is clamped).
    pub max_blur: f32,
}

impl Default for DepthOfField {
    fn default() -> Self {
        DepthOfField {
            focus_distance: 10.0,
            aperture_radius: 0.0125,
            max_blur: 0.04,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofParams {
    lens: [f32; 4],
    tiles: [u32; 4],
    padded: [f32; 4],
    taps: [[f32; 4]; 16],
}

/// A ring's 16 taps for an inner radius of 1 (outer 2): equal areas on a
/// golden-angle spiral; x, y and their length.
fn ring_taps() -> [[f32; 4]; 16] {
    std::array::from_fn(|n| {
        let d = (1.0 + 3.0 * (n as f32 + 0.5) / 16.0).sqrt();
        let a = n as f32 * 2.399_963_2;
        [d * a.cos(), d * a.sin(), d, 0.0]
    })
}

const R_INNER: f32 = 3.5;

/// The rings a largest radius of `max_px` pixels needs (at least one).
fn rings_for(max_px: f32) -> u32 {
    ((max_px / R_INNER).max(1.0).log2().ceil() as u32).clamp(1, 8)
}

/// Depth of field's passes, textures and bind groups for one image size and
/// ring count.
pub(crate) struct DofPasses {
    prepare: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    dilate: wgpu::ComputePipeline,
    gather: [wgpu::ComputePipeline; 3],
    levels: wgpu::Texture,
    base: wgpu::Buffer,
    out: wgpu::Buffer,
    hl_count: wgpu::Buffer,
    hl: wgpu::Buffer,
    hl_tiles: wgpu::Buffer,
    bin: wgpu::ComputePipeline,
    tiles: wgpu::Buffer,
    params: wgpu::Buffer,
    binds: Option<DofBinds>,
    size: (u32, u32),
    padded: (u32, u32),
    rings: u32,
}

struct DofBinds {
    prepare: wgpu::BindGroup,
    bin: wgpu::BindGroup,
    down: Vec<wgpu::BindGroup>,
    dilate: wgpu::BindGroup,
    gather: [wgpu::BindGroup; 3],
}

impl DofPasses {
    /// For a `size` image whose largest radius is `max_px` pixels.
    pub(crate) fn new(device: &wgpu::Device, size: (u32, u32), max_px: f32) -> DofPasses {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dof.wgsl").into()),
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
        let rings = rings_for(max_px);
        // The prepare pass writes levels 1 to 4; levels below that come
        // from passes of their own.
        let mips = rings.max(4);
        let step = 1u32 << mips;
        let padded = (
            size.0.div_ceil(step).max(1) * step,
            size.1.div_ceil(step).max(1) * step,
        );
        // Levels 1 to `rings` (level 0 is `base`).
        let levels = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dof levels"),
            size: wgpu::Extent3d {
                width: padded.0 / 2,
                height: padded.1 / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let base = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof level 0"),
            size: padded.0 as u64 * padded.1 as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof out"),
            size: (size.0 as u64 * size.1 as u64 * 8).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tiles_n = size.0.div_ceil(16) as u64 * size.1.div_ceil(16) as u64;
        // Highlights: the list's length and each tile's; the list (4,096
        // entries of 32 bytes); 256 indices per tile.
        let hl_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof highlight counts"),
            size: (tiles_n + 1) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hl = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof highlights"),
            size: 4096 * 32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let hl_tiles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof highlight tiles"),
            size: (tiles_n * 256 * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tiles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof tiles"),
            size: (4 * tiles_n * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof params"),
            size: std::mem::size_of::<DofParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        DofPasses {
            prepare: pipeline("dof_prepare"),
            down: pipeline("dof_down"),
            dilate: pipeline("dof_dilate"),
            gather: [
                pipeline("dof_gather_coarse"),
                pipeline("dof_refine_half"),
                pipeline("dof_refine_full"),
            ],
            levels,
            base,
            out,
            hl_count,
            hl,
            hl_tiles,
            bin: pipeline("dof_bin"),
            tiles,
            params,
            binds: None,
            size,
            padded,
            rings,
        }
    }

    /// Whether these passes serve a `size` image with a largest radius of
    /// `max_px` pixels.
    pub(crate) fn fits(&self, size: (u32, u32), max_px: f32) -> bool {
        self.size == size && self.rings == rings_for(max_px)
    }

    /// Binds the camera, colour, visibility and light buffers (after they
    /// are made or remade; without a light list, an empty one).
    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        camera: &wgpu::Buffer,
        colors: &wgpu::Buffer,
        vis: &wgpu::Buffer,
        lights: Option<&wgpu::Buffer>,
    ) {
        let none = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof no lights"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let lights = lights.unwrap_or(&none);
        let level = |k: u32, count: Option<u32>| {
            self.levels.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: k,
                mip_level_count: count,
                ..Default::default()
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, r)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: r.clone(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let all = level(0, None);
        let first: Vec<wgpu::TextureView> = (0..4).map(|m| level(m, Some(1))).collect();
        let prepare = group(
            &self.prepare,
            &[
                (0, camera.as_entire_binding()),
                (1, colors.as_entire_binding()),
                (2, vis.as_entire_binding()),
                (3, self.params.as_entire_binding()),
                (4, self.tiles.as_entire_binding()),
                (8, self.base.as_entire_binding()),
                (10, wgpu::BindingResource::TextureView(&first[0])),
                (11, wgpu::BindingResource::TextureView(&first[1])),
                (12, wgpu::BindingResource::TextureView(&first[2])),
                (13, wgpu::BindingResource::TextureView(&first[3])),
                (14, self.hl_count.as_entire_binding()),
                (15, self.hl.as_entire_binding()),
                (17, lights.as_entire_binding()),
            ],
        );
        let bin = group(
            &self.bin,
            &[
                (3, self.params.as_entire_binding()),
                (14, self.hl_count.as_entire_binding()),
                (15, self.hl.as_entire_binding()),
                (16, self.hl_tiles.as_entire_binding()),
            ],
        );
        // Mip m holds level m + 1; levels 6 and below from the one above.
        let down = (4..self.levels.mip_level_count())
            .map(|m| {
                let (out, inp) = (level(m, Some(1)), level(m - 1, Some(1)));
                group(
                    &self.down,
                    &[
                        (5, wgpu::BindingResource::TextureView(&out)),
                        (6, wgpu::BindingResource::TextureView(&inp)),
                    ],
                )
            })
            .collect();
        let dilate = group(
            &self.dilate,
            &[
                (3, self.params.as_entire_binding()),
                (4, self.tiles.as_entire_binding()),
            ],
        );
        // The colours only for the last pass, which writes them.
        let gather_entries = |pipeline, last: bool| {
            let mut entries = vec![
                (0, camera.as_entire_binding()),
                (3, self.params.as_entire_binding()),
                (4, self.tiles.as_entire_binding()),
                (6, wgpu::BindingResource::TextureView(&all)),
                (7, wgpu::BindingResource::Sampler(&sampler)),
                (8, self.base.as_entire_binding()),
                (9, self.out.as_entire_binding()),
                (14, self.hl_count.as_entire_binding()),
                (15, self.hl.as_entire_binding()),
                (16, self.hl_tiles.as_entire_binding()),
            ];
            if last {
                entries.push((1, colors.as_entire_binding()));
            }
            group(pipeline, &entries)
        };
        self.binds = Some(DofBinds {
            prepare,
            bin,
            down,
            dilate,
            gather: std::array::from_fn(|k| gather_entries(&self.gather[k], k == 2)),
        });
    }

    /// Diagnostics: the last frame's highlights, and the tiles whose lists
    /// overflowed.
    pub(crate) fn highlights(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (u32, u32) {
        let size = self.hl_count.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof highlight readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.hl_count, 0, &staging, 0, size);
        queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let words: Vec<u32> =
            bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        (
            words[0],
            words[1..].iter().filter(|&&n| n > 256).count() as u32,
        )
    }

    /// Records the passes for a camera of `vertical_fov_degrees`.
    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        vertical_fov_degrees: f32,
        lens: &DepthOfField,
    ) {
        let Some(binds) = &self.binds else { return };
        let (w, h) = self.size;
        let f = h as f32 / (2.0 * (vertical_fov_degrees.to_radians() * 0.5).tan());
        let max_px = lens.max_blur * h as f32;
        let (tx, ty) = (w.div_ceil(16), h.div_ceil(16));
        let params = DofParams {
            lens: [
                f * lens.aperture_radius,
                1.0 / lens.focus_distance.max(1e-6),
                max_px,
                self.rings as f32,
            ],
            tiles: [tx, ty, (max_px / 16.0).ceil() as u32 + 1, 0],
            padded: [self.padded.0 as f32, self.padded.1 as f32, 0.0, 0.0],
            taps: ring_taps(),
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(&self.hl_count, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("depth of field"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prepare);
        pass.set_bind_group(0, &binds.prepare, &[]);
        pass.dispatch_workgroups(self.padded.0.div_ceil(16), self.padded.1.div_ceil(16), 1);
        pass.set_pipeline(&self.down);
        for (k, group) in binds.down.iter().enumerate() {
            let m = k as u32 + 4;
            let (lw, lh) = (self.padded.0 >> (m + 1), self.padded.1 >> (m + 1));
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(lw.div_ceil(8), lh.div_ceil(8), 1);
        }
        pass.set_pipeline(&self.bin);
        pass.set_bind_group(0, &binds.bin, &[]);
        pass.dispatch_workgroups(4096 / 64, 1, 1);
        pass.set_pipeline(&self.dilate);
        pass.set_bind_group(0, &binds.dilate, &[]);
        pass.dispatch_workgroups(tx.div_ceil(8), ty.div_ceil(8), 1);
        // Every fourth pixel, then the even ones, then the rest.
        for (k, step) in [4u32, 2, 1].into_iter().enumerate() {
            pass.set_pipeline(&self.gather[k]);
            pass.set_bind_group(0, &binds.gather[k], &[]);
            pass.dispatch_workgroups(w.div_ceil(8 * step), h.div_ceil(8 * step), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{R_INNER, rings_for};

    #[test]
    fn rings_cover_the_largest_radius() {
        for max_px in [0.5f32, 2.5, 3.0, 10.0, 28.8, 57.6] {
            let rings = rings_for(max_px);
            assert!(R_INNER * (1u32 << rings) as f32 >= max_px, "{max_px}");
            assert!(rings == 1 || (R_INNER * (1u32 << (rings - 1)) as f32) < max_px);
        }
    }
}
