//! Display: tone mapping and auto-exposure (M17).

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
