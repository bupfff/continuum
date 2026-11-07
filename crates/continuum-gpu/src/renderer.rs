//! The library's front door (v0.1): an instanced scene, its lights and an
//! optional sky, rendered into a texture every frame.
//!
//! ```no_run
//! use continuum_gpu::{Camera, Light, MaskConeKernel, Renderer, RendererOptions, wgpu};
//! # fn scene() -> continuum_gpu::InstancedScene { Default::default() }
//! # fn sun() -> Light { unimplemented!() }
//! // A kernel on its own device (or `MaskConeKernel::with_device` on yours,
//! // created with `continuum_gpu::device_descriptor`).
//! let kernel = MaskConeKernel::new().expect("a GPU");
//! let format = wgpu::TextureFormat::Rgba8UnormSrgb;
//! let mut renderer = Renderer::new(
//!     kernel,
//!     format,
//!     (1280, 720),
//!     &scene(),
//!     continuum::glam::DVec3::ZERO,
//!     &[sun()],
//!     RendererOptions::default(),
//! );
//! # let view: wgpu::TextureView = unimplemented!();
//! let camera = Camera {
//!     eye: continuum::glam::Vec3::new(0.0, -5.0, 1.7),
//!     target: continuum::glam::Vec3::ZERO,
//!     vertical_fov_degrees: 50.0,
//! };
//! renderer.render(camera, &view);
//! ```

use continuum::glam::{DVec3, Vec3};

use crate::{
    CacheOptions, CacheStats, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel,
};

/// How a [`Renderer`] renders.
#[derive(Clone, Debug)]
pub struct RendererOptions {
    /// Target pixels per rendered pixel along each axis (the image is
    /// upscaled to the target).
    pub scale: u32,
    /// Light through the view-dependent lighting cache (the default): pixels
    /// read cached light, and each frame bakes, within a time budget, the
    /// pages the view lacks, coarse first. Otherwise every pixel traces a
    /// mask cone per light every frame (exact for the frame, slow).
    pub cache: bool,
    /// A uniform sky (linear RGB radiance), lighting what it can see of it.
    /// With the cache, sky light converges slowly (M5 gate 5). Without it,
    /// every pixel traces sky cones every frame, which is far slower than
    /// lights alone.
    pub sky: Option<Vec3>,
    /// The cache's tuning.
    pub cache_options: CacheOptions,
    /// Anti-aliasing (experimental, M7): each pixel traces a mask cone
    /// through its square instead of one ray through its center, and shows
    /// the surfaces it meets in proportion to their visible coverage. Off:
    /// one ray per pixel (hardware ray queries where available).
    pub pixel_cones: bool,
}

impl Default for RendererOptions {
    fn default() -> Self {
        RendererOptions {
            scale: 1,
            cache: true,
            sky: None,
            cache_options: CacheOptions::default(),
            pixel_cones: false,
        }
    }
}

/// An instanced scene with its lights, rendered with mask cones: exact soft
/// shadows from every light's area, through a lighting cache kept across
/// frames. The scene and lights stay still; the camera moves freely, and
/// lights can change color and intensity at no cost
/// ([`Renderer::set_light_radiance`]).
pub struct Renderer {
    kernel: MaskConeKernel,
    inner: InstancedRenderer,
}

impl Renderer {
    /// Uploads `scene` and `lights` (at most [`crate::MAX_LIGHTS`]) for
    /// rendering into `width × height` textures of `format`.
    ///
    /// Instance positions are f64; everything is uploaded relative to
    /// `origin`, a point near where the camera will be, so worlds far larger
    /// than f32 can resolve render without jitter. Lights and cameras are
    /// given relative to `origin` too.
    pub fn new(
        kernel: MaskConeKernel,
        format: wgpu::TextureFormat,
        (width, height): (u32, u32),
        scene: &InstancedScene,
        origin: DVec3,
        lights: &[Light],
        options: RendererOptions,
    ) -> Renderer {
        // Prefiltered coverage at one bin, and sky cones eight times coarser
        // (M5 gates 3–4): the settings every measurement since uses.
        kernel.set_prefilter(1.0);
        kernel.set_sky_prefilter(8.0);
        kernel.set_sky(options.sky.unwrap_or(Vec3::ZERO));
        kernel.set_pixel_cones(options.pixel_cones);
        let inner = if options.cache {
            InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                width,
                height,
                options.scale,
                lights,
                scene,
                origin,
                options.cache_options,
            )
        } else {
            InstancedRenderer::new_lights(
                &kernel,
                format,
                width,
                height,
                options.scale,
                lights,
                scene,
                origin,
            )
        };
        Renderer { kernel, inner }
    }

    /// Renders a frame seen from `camera` into `target` (a texture of the
    /// format and size given). Returns without waiting for the GPU.
    pub fn render(&mut self, camera: Camera, target: &wgpu::TextureView) {
        self.inner.frame(camera, target);
    }

    /// Follows the target's new size.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.inner.resize(width, height);
    }

    /// Changes light `index`'s RGB radiance from the next frame. Lighting is
    /// linear in emitted power, so the cache re-weights what it holds and
    /// bakes nothing.
    pub fn set_light_radiance(&mut self, index: usize, radiance: Vec3) {
        self.inner.set_light_radiance(index, radiance);
    }

    /// The last frame the cache has reported on (`None` without the cache).
    pub fn stats(&self) -> Option<CacheStats> {
        self.inner.cache_stats()
    }

    /// Whether the cache baked nothing and was asked for nothing in the last
    /// `frames` frames it reported on: the view's lighting is complete.
    pub fn is_settled(&self, frames: usize) -> bool {
        let log = self.inner.cache_log();
        let latest = self.stats().map_or(0, |s| s.frame as usize).min(log.len());
        latest >= frames
            && log[latest - frames..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
    }

    /// Blocks until the GPU has finished everything submitted so far (the
    /// upload after [`Renderer::new`], or the frames rendered).
    pub fn wait(&self) {
        self.inner.wait();
    }

    /// Waits for the GPU and returns the last frame's linear RGB colors, row
    /// by row from the top.
    pub fn read_colors(&mut self) -> Vec<[f32; 3]> {
        self.inner.finish();
        self.inner.read_colors()
    }

    /// GPU memory the scene and the cache hold, in bytes.
    pub fn gpu_bytes(&self) -> u64 {
        self.inner.scene_bytes() + self.inner.cache_bytes()
    }

    /// Whether camera rays use hardware ray queries (otherwise compute).
    pub fn hardware_rays(&self) -> bool {
        self.inner.hardware_rays()
    }

    /// The kernel, for its device and adapter.
    pub fn kernel(&self) -> &MaskConeKernel {
        &self.kernel
    }
}
