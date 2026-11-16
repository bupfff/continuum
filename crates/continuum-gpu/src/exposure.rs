//! Display: tone mapping and auto-exposure (M17), bloom and upscaling
//! (M19).

use bytemuck::{Pod, Zeroable};

/// How the display maps the renderer's linear light to the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ToneMap {
    /// Khronos PBR Neutral (the default; Khronos's for glTF viewers): base
    /// colours shown as they are up to a highlight compression that
    /// desaturates toward white. As Blender's "Khronos PBR Neutral" view
    /// (M17 gate 1).
    #[default]
    PbrNeutral,
    /// An approximation of the original AgX (Benjamin Wrensch's fit of
    /// Troy Sobotka's AgX): a filmic curve over 16.5 stops. Not Blender
    /// 5.2's AgX, which differs on saturated colours.
    Agx,
    /// `c / (c + 0.6)`, as images before M17 were made.
    Reinhard,
}

impl ToneMap {
    pub(crate) fn code(self) -> f32 {
        match self {
            ToneMap::Reinhard => 0.0,
            ToneMap::PbrNeutral => 1.0,
            ToneMap::Agx => 2.0,
        }
    }
}

/// Auto-exposure (M17): each frame the display's exposure is the one that
/// shows the frame's middle luminances (the mean of log2 luminance between
/// two percentiles, at every other pixel in each direction) at `key`, approached over `seconds` (an exponential
/// adaptation, as an eye or a camera; only this one number carries from
/// frame to frame). It multiplies any exposure set by hand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoExposure {
    /// The luminance the trimmed mean is shown at (0.18: middle grey).
    pub key: f32,
    /// The adaptation's time constant (0: at once).
    pub seconds: f32,
    /// The percentiles bounding the mean (0.5 and 0.95: the brighter half
    /// less its brightest twentieth).
    pub percentiles: (f32, f32),
    /// The exposure's bounds, as factors.
    pub bounds: (f32, f32),
}

impl Default for AutoExposure {
    fn default() -> Self {
        AutoExposure {
            key: 0.18,
            seconds: 0.5,
            percentiles: (0.5, 0.95),
            bounds: (1.0 / 65536.0, 65536.0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeterParams {
    size: [u32; 2],
    pad: [u32; 2],
    p: [f32; 4],
    bounds: [f32; 4],
}

/// The auto-exposure meter's passes and buffers.
pub(crate) struct Meter {
    histogram: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    hist: wgpu::Buffer,
    pub(crate) state: wgpu::Buffer,
    params: wgpu::Buffer,
    binds: Option<[wgpu::BindGroup; 2]>,
}

impl Meter {
    pub(crate) fn new(device: &wgpu::Device) -> Meter {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("meter.wgsl").into()),
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
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        Meter {
            histogram: pipeline("meter_histogram"),
            resolve: pipeline("meter_resolve"),
            hist: buffer("meter histogram", 256, U::STORAGE),
            state: buffer("meter", 16, U::STORAGE | U::COPY_SRC | U::COPY_DST),
            params: buffer("meter params", 48, U::UNIFORM | U::COPY_DST),
            binds: None,
        }
    }

    /// Binds the colour buffer (after it is made or remade).
    pub(crate) fn bind(&mut self, device: &wgpu::Device, colors: &wgpu::Buffer) {
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, buf)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("meter"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        self.binds = Some([
            group(
                &self.histogram,
                &[(0, colors), (1, &self.hist), (3, &self.params)],
            ),
            group(
                &self.resolve,
                &[(1, &self.hist), (2, &self.state), (3, &self.params)],
            ),
        ]);
    }

    /// Records the meter for a `width × height` frame, adapting by `rate`
    /// (1 − exp(−dt / τ)).
    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: (u32, u32),
        auto: &AutoExposure,
        rate: f32,
    ) {
        let Some([h, r]) = &self.binds else { return };
        let params = MeterParams {
            size: [size.0, size.1],
            pad: [0; 2],
            p: [auto.key, rate, auto.percentiles.0, auto.percentiles.1],
            bounds: [auto.bounds.0, auto.bounds.1, 0.0, 0.0],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.histogram);
        pass.set_bind_group(0, h, &[]);
        // Every other pixel in each direction.
        pass.dispatch_workgroups(size.0.div_ceil(32), size.1.div_ceil(32), 1);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, r, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// Bloom (M19): a glare around bright light, energy kept. The display shows
/// `1 − strength` of each pixel plus `strength` of the image blurred by a
/// Gaussian of `radius` (a share of the image's height; blurred at a
/// quarter of its size), in linear light before the tone map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bloom {
    /// The share of light spread (0 to 1).
    pub strength: f32,
    /// The Gaussian's σ as a share of the image's height.
    pub radius: f32,
}

impl Default for Bloom {
    fn default() -> Self {
        Bloom {
            strength: 0.04,
            radius: 0.02,
        }
    }
}

/// How the display scales a rendered image up to its target (M19; with a
/// render scale above 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Upscale {
    /// Catmull and Rom's bicubic, clamped to its four nearest pixels' range
    /// (no ringing).
    #[default]
    CatmullRom,
    /// Bilinear.
    Bilinear,
    /// The nearest pixel (as before M19).
    Nearest,
}

impl Upscale {
    pub(crate) fn code(self) -> u32 {
        match self {
            Upscale::Nearest => 0,
            Upscale::Bilinear => 1,
            Upscale::CatmullRom => 2,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    size: [u32; 2],
    quarter: [u32; 2],
    p: [f32; 4],
}

/// Bloom's passes and buffers (the blurred quarter image in `out`).
pub(crate) struct BloomPasses {
    down: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    pub(crate) out: wgpu::Buffer,
    spare: wgpu::Buffer,
    params: [wgpu::Buffer; 2],
    binds: Option<[wgpu::BindGroup; 3]>,
    quarter: (u32, u32),
}

impl BloomPasses {
    pub(crate) fn new(device: &wgpu::Device, size: (u32, u32)) -> BloomPasses {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
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
        let quarter = (size.0.div_ceil(4).max(1), size.1.div_ceil(4).max(1));
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let texels = quarter.0 as u64 * quarter.1 as u64 * 16;
        let uniform = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom params"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        BloomPasses {
            down: pipeline("bloom_down"),
            blur: pipeline("bloom_blur"),
            out: buffer("bloom", texels),
            spare: buffer("bloom spare", texels),
            params: [uniform(), uniform()],
            binds: None,
            quarter,
        }
    }

    pub(crate) fn quarter(&self) -> (u32, u32) {
        self.quarter
    }

    /// Binds the colour buffer (after it is made or remade).
    pub(crate) fn bind(&mut self, device: &wgpu::Device, colors: &wgpu::Buffer) {
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, buf)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        self.binds = Some([
            // Down: colours into `out`; across: `out` into `spare`; down:
            // `spare` into `out`.
            group(
                &self.down,
                &[(0, colors), (2, &self.out), (3, &self.params[0])],
            ),
            group(
                &self.blur,
                &[(1, &self.out), (2, &self.spare), (3, &self.params[0])],
            ),
            group(
                &self.blur,
                &[(1, &self.spare), (2, &self.out), (3, &self.params[1])],
            ),
        ]);
    }

    /// Records bloom's passes for a `width × height` image.
    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: (u32, u32),
        bloom: &Bloom,
    ) {
        let Some([d, a, b]) = &self.binds else { return };
        let sigma = (bloom.radius * size.1 as f32 / 4.0).max(0.25);
        let taps = (3.0 * sigma).ceil();
        for (k, down) in [(0, 0.0f32), (1, 1.0)] {
            let params = BloomParams {
                size: [size.0, size.1],
                quarter: [self.quarter.0, self.quarter.1],
                p: [sigma, taps, down, 0.0],
            };
            queue.write_buffer(&self.params[k], 0, bytemuck::bytes_of(&params));
        }
        let groups = (self.quarter.0.div_ceil(8), self.quarter.1.div_ceil(8));
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.down);
        pass.set_bind_group(0, d, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
        pass.set_pipeline(&self.blur);
        pass.set_bind_group(0, a, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }
}
