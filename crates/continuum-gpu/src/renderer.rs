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

use continuum::glam::{DVec3, Mat3, Vec3};

use crate::{
    CacheOptions, CacheStats, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel,
    StreamOptions, StreamStats,
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
    /// A sky whose radiance varies with direction (M5 gate 12), in place of
    /// `sky`: an equirectangular image above the horizon. See
    /// [`crate::MaskConeKernel::set_environment`].
    pub environment: Option<crate::Environment>,
    /// The cache's tuning.
    pub cache_options: CacheOptions,
    /// Anti-aliasing (experimental, M7): each pixel traces a mask cone
    /// through its square instead of one ray through its center, and shows
    /// the surfaces it meets in proportion to their visible coverage. Off:
    /// one ray per pixel (hardware ray queries where available).
    pub pixel_cones: bool,
    /// Instances that may move ([`Renderer::set_transform`]). The cache
    /// re-bakes, within the frame, what a move makes stale (M9 gate 1), and
    /// never shows stale light. With the cache, movers and a sky do not mix
    /// yet. Empty (the default): a static scene.
    pub movable: Vec<usize>,
    /// Geometries whose vertices change ([`Renderer::set_vertices`]: skinned
    /// characters, morph targets, anything posed each frame; M12). Their
    /// instances move as movers do (they are made movable). Lighting near
    /// them lags their pose within `CacheOptions::deforming_budget` and
    /// catches up when they stop (0: exact every frame, about 10× the frame
    /// for a large object). Not with streaming or the sub-pixel path; under
    /// a sky, as movers (slow). Empty (the default): none.
    pub deforming: Vec<usize>,
    /// Lights that may move ([`Renderer::set_light`]; not spots). With the
    /// cache, an error ledger re-bakes only what may be more than
    /// `CacheOptions::ledger_epsilon` off (M9 gate 2). Not yet with movers
    /// or a sky in the cache.
    pub moving_lights: Vec<usize>,
    /// Camera levels streamed by need (M14 gate 1): geometries with
    /// `InstancedScene::geometry_levels` show camera rays their finest level
    /// resident, kept within the budget where the view needs the detail;
    /// lighting keeps each geometry itself. With hardware camera rays and
    /// no movers. None (the default): only the geometries themselves.
    pub streaming: Option<StreamOptions>,
    /// Sub-pixel detail (experimental, M13 gates 5–13): each pixel resolves
    /// what is smaller than it from precomputed coverage maps of instanced
    /// geometries and coverage pages of far hierarchy nodes, with rays at
    /// its corners and bins near large objects, and colours each part from
    /// the lighting cache. Thin and distant geometry keeps its coverage
    /// instead of shimmering. On a million instanced boulders at 1280×720 a
    /// settled frame costs about 15 ms more than one ray a pixel on an Apple
    /// M4 (`docs/results/m13-gate13.md`).
    ///
    /// Taken only with the cache, hardware ray queries, a static scene (no
    /// `movable`, no `moving_lights`), no `streaming` and at most 64
    /// geometries with instances; otherwise ignored. The cache then bakes a
    /// fixed amount a frame (`frame_target_ms` and `frame_budget_ms` 0: the
    /// path's own passes take most of a frame target), so views settle in
    /// hundreds of frames. Lights seen directly are drawn bin by bin behind
    /// what covers them (M13 gate 14). Not yet: small instances lit by the
    /// cache (they take the first light and the sky, with a shadow per
    /// instance).
    pub sub_pixel: bool,
}

impl Default for RendererOptions {
    fn default() -> Self {
        RendererOptions {
            scale: 1,
            cache: true,
            sky: None,
            environment: None,
            cache_options: CacheOptions::default(),
            pixel_cones: false,
            movable: Vec::new(),
            deforming: Vec::new(),
            moving_lights: Vec::new(),
            streaming: None,
            sub_pixel: false,
        }
    }
}

/// An instanced scene with its lights, rendered with mask cones: exact soft
/// shadows from every light's area, through a lighting cache kept across
/// frames. The camera moves freely; instances marked movable move
/// ([`Renderer::set_transform`]), and the cache re-bakes only what their
/// moves can change; lights stay put but can change color and intensity at
/// no cost ([`Renderer::set_light_radiance`]).
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
        kernel.set_environment(options.environment.as_ref());
        kernel.set_pixel_cones(options.pixel_cones);
        // Sub-pixel detail (M13): the path the gates measured (profile stage
        // 20 with pixel cones, hardware rays, prefiltered coverage at eight
        // bins) with a fixed bake budget.
        // (Coverage maps exist for at most 64 placed geometries: a record
        // names its map in 16 bits, 1,024 directions a geometry.)
        // Deforming geometries' instances move as movers do (M12).
        let mut movable = options.movable.clone();
        for (k, i) in scene.instances.iter().enumerate() {
            if options.deforming.contains(&i.geometry) && !movable.contains(&k) {
                movable.push(k);
            }
        }
        let placed = (0..scene.geometries.len())
            .filter(|&g| scene.instances.iter().any(|i| i.geometry == g))
            .count();
        let sub_pixel = options.sub_pixel
            && options.cache
            && kernel.has_hardware_rays()
            && options.movable.is_empty()
            && options.deforming.is_empty()
            && options.moving_lights.is_empty()
            && options.streaming.is_none()
            && placed <= 64;
        let mut cache_options = options.cache_options;
        if sub_pixel {
            kernel.set_prefilter(8.0);
            kernel.set_hardware_rays(true);
            kernel.set_pixel_cones(true);
            kernel.set_profile_stage(20);
            cache_options.frame_target_ms = 0.0;
            cache_options.frame_budget_ms = 0.0;
        }
        let cache = options.cache.then_some(cache_options);
        let inner = match options.streaming {
            Some(stream) if movable.is_empty() && options.moving_lights.is_empty() => {
                InstancedRenderer::with_streaming(
                    &kernel,
                    format,
                    width,
                    height,
                    options.scale,
                    lights,
                    scene,
                    origin,
                    cache,
                    stream,
                )
            }
            _ => InstancedRenderer::with_movers(
                &kernel,
                format,
                width,
                height,
                options.scale,
                lights,
                scene,
                origin,
                cache,
                &movable,
                &options.moving_lights,
            ),
        };
        if sub_pixel {
            kernel.set_profile_stage(0);
        }
        let mut inner = inner;
        if !options.deforming.is_empty() {
            inner.set_deforming(scene, &options.deforming);
        }
        Renderer { kernel, inner }
    }

    /// Moves (and re-colors) light `index`, one of
    /// [`RendererOptions::moving_lights`], from the next frame.
    pub fn set_light(&mut self, index: usize, light: Light) {
        self.inner.set_light(index, light);
    }

    /// Poses geometry `geometry` (one of [`RendererOptions::deforming`]) from
    /// the next frame: its triangles and each corner's shading normal, as
    /// many as the scene's and in its order. Its charts and texture
    /// coordinates stay the scene's (a texel follows its triangle); normal
    /// maps keep the scene's tangents.
    pub fn set_vertices(
        &mut self,
        geometry: usize,
        triangles: Vec<[Vec3; 3]>,
        normals: Vec<[Vec3; 3]>,
    ) {
        self.inner.set_vertices(geometry, triangles, normals);
    }

    /// Moves instance `instance` (one of [`RendererOptions::movable`]) to
    /// `rotation` (orthonormal) and `position` (in the world, f64) from the
    /// next frame. Its scale stays.
    pub fn set_transform(&mut self, instance: usize, rotation: Mat3, position: DVec3) {
        self.inner
            .set_instance_transform(instance, rotation, position);
    }

    /// Renders a frame seen from `camera` into `target` (a texture of the
    /// format and size given). Returns without waiting for the GPU.
    pub fn render(&mut self, camera: Camera, target: &wgpu::TextureView) {
        self.inner.frame(camera, target);
    }

    /// [`Renderer::render`] without drawing: the frame's colors stay in the
    /// renderer, for [`Renderer::encode_display`]. For engines that order
    /// their own passes (the Bevy plugin): this submits the frame's compute
    /// work now, and the engine draws the result later in its own encoder.
    pub fn compute(&mut self, camera: Camera) {
        self.inner.frame_compute(camera);
    }

    /// Encodes the last frame drawn to `target` (a texture view of the
    /// format and size given) into `encoder`. A float target (such as
    /// `Rgba16Float`) receives linear light for the engine's own tone
    /// mapping; other formats get the viewer's tone curve.
    pub fn encode_display(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.inner.encode_display(encoder, target);
    }

    /// Scales the colors drawn to the target by `factor` (exposure) before
    /// the tone curve, from the next frame: 1 by default, 2 one stop
    /// brighter. A float target gets the scaled linear light;
    /// [`Self::read_colors`] stays unscaled.
    pub fn set_exposure(&mut self, factor: f32) {
        self.inner.set_exposure(factor);
    }

    /// Follows the target's new size.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.inner.resize(width, height);
    }

    /// Changes light `index`'s RGB radiance from the next frame. Lighting is
    /// linear in emitted power, so the cache re-weights what it holds and
    /// bakes nothing. With bounce light, a light past the third has its
    /// bounce baked at its radiance: re-coloring it clears bounce light,
    /// which re-bakes over the following frames (M10 gate 21).
    pub fn set_light_radiance(&mut self, index: usize, radiance: Vec3) {
        self.inner.set_light_radiance(index, radiance);
    }

    /// The last frame the cache has reported on (`None` without the cache).
    pub fn stats(&self) -> Option<CacheStats> {
        self.inner.cache_stats()
    }

    /// What camera-level streaming did in the latest frame (`None` without
    /// it; M14 gate 1).
    pub fn stream_stats(&self) -> Option<StreamStats> {
        self.inner.stream_log().last().copied()
    }

    /// Whether the cache baked nothing and was asked for nothing in the last
    /// `frames` frames it reported on: the view's lighting is complete.
    pub fn is_settled(&self, frames: usize) -> bool {
        let log = self.inner.cache_log();
        let latest = self.stats().map_or(0, |s| s.frame as usize).min(log.len());
        // (Nor re-baking for movers, nor deforming geometry's lighting
        // lagging its pose: M12 gate 3.)
        latest >= frames
            && self.inner.deforming_lag() == 0
            && log[latest - frames..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0 && s.rebaked_pages == 0)
    }

    /// The most frames any light's lighting near deforming geometry lags
    /// its pose (`CacheOptions::deforming_budget`; 0 when caught up).
    pub fn deforming_lag(&self) -> u32 {
        self.inner.deforming_lag()
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

    /// Diagnostics (M9 gate 1): re-bakes every page the cache may show and
    /// counts values that changed (zero when no stale light is held).
    #[doc(hidden)]
    pub fn verify_cache(&mut self) -> crate::CacheCheck {
        self.inner.verify_cache()
    }

    /// Diagnostics (M9 gate 2): re-bakes every page the cache may show and
    /// compares each page's change of moving light `light`'s ratios with the
    /// bound the ledger certified; restores the cache afterwards.
    #[doc(hidden)]
    pub fn verify_ledger(&mut self, light: usize) -> crate::LedgerCheck {
        self.inner.verify_ledger(light)
    }

    /// Diagnostics (M9 gate 3): re-bakes every page the cache may show;
    /// compares its light channels bit for bit and its sky's change with its
    /// sky ledger; restores the cache afterwards.
    #[doc(hidden)]
    pub fn verify_sky(&mut self) -> crate::SkyCheck {
        self.inner.verify_sky()
    }

    /// The kernel, for its device and adapter.
    pub fn kernel(&self) -> &MaskConeKernel {
        &self.kernel
    }
}
