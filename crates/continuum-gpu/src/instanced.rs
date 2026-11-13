//! Interactive rendering of an instanced scene: shared geometry placed many
//! times, lit per pixel with mask cones (no caches), kept on the GPU between
//! frames.

use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Mat3, Vec3};

use crate::hierarchy::{MoverTree, NodeGpu, Placement, mesh_bounds, mover_subtree, union_root};
use crate::hw::HwScene;
use crate::live::{DisplayDims, display_bind, display_pipeline};
use crate::stream::{
    LightingPool, LightingStats, LitTargets, StreamOptions, StreamStats, Streamer, Targets,
};
use crate::vcache::{PAGE_TEXELS, VirtualCache};
use crate::{
    COMPUTE_VISIBILITY_BINDINGS, CacheOptions, CacheStats, Camera, CameraUniform,
    HW_EDGES_BINDINGS, HW_M13_CORNERS_BINDINGS, HW_VC_BAKE_INDIRECT_BINDINGS,
    HW_VISIBILITY_BINDINGS, HW_VISIBILITY_BLEND_BINDINGS, HW_VISIBILITY_MOVERS_BINDINGS,
    InstancedScene, Light, LightGpu, M13_COVERAGE_BINDINGS, M13_MAPS_BINDINGS, M13_NEAR_BINDINGS,
    MaskConeKernel, RECT_LIGHT_BINDINGS, RectLight, SceneTextures, VC_ARGS_BAKE_BINDINGS,
    VC_ARGS_BINDINGS, VC_ARGS_EDGES_BINDINGS, VC_BAKE_BINDINGS, VC_BAKE_INDIRECT_BINDINGS,
    VC_BAKE_SKY_BINDINGS, VC_FILL_BINDINGS, VC_LEDGER_BINDINGS, VC_LEDGER_RESET_BINDINGS,
    VC_SELF_BAKE_BINDINGS, VC_SHADE_BINDINGS, VC_SHADE_CONES_BINDINGS, VC_SHADE_EDGES_BINDINGS,
    VC_STALE_BINDINGS, VC_STALE_TEXELS_BINDINGS, VC_UPDATE_BINDINGS, camera_uniform,
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
    /// Display and the copies for readback.
    pub readback: Duration,
    /// Planning the frame's bake on the CPU (from the requests of two frames
    /// before).
    pub planning: Duration,
    /// The first four camera passes, in order, each submitted and waited
    /// for alone.
    pub passes: [Duration; 4],
    /// The bake's passes, in order, each submitted and waited for alone,
    /// with their labels (diagnostics; empty labels past the last).
    pub bake_passes: [(&'static str, Duration); 8],
}

/// How a camera pass is dispatched (see `InstancedRenderer::camera_dispatch`).
#[derive(Clone, Copy, Debug)]
enum CameraDispatch {
    Screen,
    One,
    Listed,
}

/// How a bake pass is dispatched: workgroups, or from the dispatch-sizes
/// buffer at an offset (after the movers' stale pass).
enum Dispatch {
    Groups(u32, u32),
    Indirect(u64),
}

impl Cache {
    /// The bake passes for `counts` (table updates, sun texels, sky face
    /// jobs, pages; then objects moved, and 1 with moving lights), skipping
    /// empty ones, timed on the GPU where the device has timestamps: the
    /// first pass that runs writes the first query, the last the second.
    /// With movers moved, the stale passes run after the table updates;
    /// with moving lights, the ledger does; either sizes the bake and the
    /// fill.
    fn encode_bake(&self, encoder: &mut wgpu::CommandEncoder, counts: [u32; 6]) {
        self.encode_bake_passes(encoder, counts, None);
    }

    /// `encode_bake`'s passes, or with `only`, that one pass alone and
    /// untimed (diagnostics: `frame_profiled` submits each on its own);
    /// the passes' labels.
    fn encode_bake_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        counts: [u32; 6],
        only: Option<usize>,
    ) -> Vec<&'static str> {
        let groups = |n: u32| Dispatch::Groups(n.div_ceil(64), 1);
        let slots = |n: u32| {
            let g = n.div_ceil(64);
            Dispatch::Groups(g.min(65_535), g.div_ceil(65_535))
        };
        let mut list: Vec<(
            &'static str,
            &wgpu::ComputePipeline,
            &wgpu::BindGroup,
            Dispatch,
        )> = Vec::new();
        if counts[0] > 0 {
            list.push(("update", &self.passes[0], &self.binds[0], groups(counts[0])));
        }
        let moved = counts[4] > 0 && self.has_movers;
        let lit = counts[5] > 0 && self.moving_lights;
        let indirect = moved || lit;
        if let Some((passes, binds)) = &self.movers {
            if self.moving_lights && counts[3] > 0 {
                list.push(("ledger reset", &passes[4], &binds[4], groups(counts[3])));
            }
            if moved {
                list.push(("stale", &passes[0], &binds[0], slots(self.state.pages())));
                list.push(("stale args", &passes[1], &binds[1], Dispatch::Groups(1, 1)));
                list.push(("stale texels", &passes[2], &binds[2], Dispatch::Indirect(0)));
            }
            if lit {
                list.push(("ledger", &passes[5], &binds[5], slots(self.state.pages())));
            }
            if indirect {
                list.push(("bake args", &passes[3], &binds[3], Dispatch::Groups(1, 1)));
                list.push((
                    "bake",
                    &self.passes[1],
                    &self.binds[1],
                    Dispatch::Indirect(12),
                ));
            }
        }
        if indirect {
        } else if counts[1] > 0 {
            list.push(("bake", &self.passes[1], &self.binds[1], groups(counts[1])));
        }
        // The stale pass may append sky samples (M9 gate 3).
        if indirect && self.state.has_sky() {
            list.push((
                "sky",
                &self.passes[2],
                &self.binds[2],
                Dispatch::Indirect(36),
            ));
        } else if counts[2] > 0 {
            list.push(("sky", &self.passes[2], &self.binds[2], groups(counts[2])));
        }
        let bounce = self.state.indirect_jobs();
        let bounce_pass = self
            .indirect
            .as_ref()
            .and_then(|(p, b)| b.as_ref().map(|b| (p, b)))
            .filter(|_| bounce > 0);
        if indirect {
            list.push((
                "fill",
                &self.passes[3],
                &self.binds[3],
                Dispatch::Indirect(24),
            ));
        } else if counts[3] > 0 {
            list.push(("fill", &self.passes[3], &self.binds[3], groups(counts[3])));
        }
        // Bounce samples read the direct light just baked: after the fill.
        if let Some((pipeline, bind)) = bounce_pass {
            list.push((
                "bounce",
                pipeline,
                bind,
                Dispatch::Groups(bounce.min(65_535), bounce.div_ceil(65_535)),
            ));
        }
        let set = self.state.timer.as_ref().map(|t| &t.set);
        let ran: Vec<usize> = (0..list.len()).collect();
        let args = self.state.buffer(66);
        for (k, (_, pipeline, bind, dispatch)) in list.iter().enumerate() {
            if only.is_some_and(|o| o != k) {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: if only.is_some() {
                    None
                } else {
                    timestamps(set, &ran, k, 0)
                },
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, *bind, &[]);
            match dispatch {
                Dispatch::Groups(x, y) => pass.dispatch_workgroups(*x, *y, 1),
                Dispatch::Indirect(offset) => pass.dispatch_workgroups_indirect(args, *offset),
            }
        }
        list.iter().map(|e| e.0).collect()
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
    /// With movers or moving lights: the stale passes (pages, then
    /// candidates' texels), each followed by a dispatch-size pass, then the
    /// ledger's reset and check passes, and their bind groups.
    movers: Option<([wgpu::ComputePipeline; 6], Vec<wgpu::BindGroup>)>,
    /// Movers in the scene, moving lights tracked.
    has_movers: bool,
    moving_lights: bool,
    /// Bounce light's sample pass and its bind group (M10 gate 1).
    indirect: Option<(wgpu::ComputePipeline, Option<wgpu::BindGroup>)>,
    /// Whether that pass traces through hardware ray queries (M14 gate 10).
    indirect_hw: bool,
}

/// What [`InstancedRenderer::verify_cache`] found: resident pages that
/// shading may read, their values (texels × channels), and how many of
/// those changed when re-baked from scratch, and by how much at most.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheCheck {
    /// Resident pages shading may read.
    pub pages: u32,
    /// Their values (texels × channels).
    pub values: u64,
    /// Values that changed when re-baked.
    pub differing: u64,
    /// The largest change.
    pub largest: f32,
}

/// What [`InstancedRenderer::verify_ledger`] found for a moving light.
#[derive(Clone, Copy, Debug, Default)]
pub struct LedgerCheck {
    /// Pages shading may read.
    pub pages: u32,
    /// Pages whose largest change of a texel's ratio exceeded the certified
    /// bound (plus 1e-5 for rounding): soundness violations.
    pub violations: u32,
    /// Pages with a change above ε.
    pub over_epsilon: u32,
    /// Pages with any change at all.
    pub changed: u32,
    /// The largest change.
    pub largest_change: f32,
    /// The largest certified bound.
    pub largest_bound: f32,
}

/// What [`InstancedRenderer::verify_sky`] found with movers under a sky.
#[derive(Clone, Copy, Debug, Default)]
pub struct SkyCheck {
    /// Pages shading may read.
    pub pages: u32,
    /// Light-channel values that changed when re-baked (should be none).
    pub light_differing: u64,
    /// Pages whose sky changed by more than its ledger's charge (plus 1e-5
    /// for rounding): soundness violations.
    pub violations: u32,
    /// Pages whose sky changed by more than its ε threshold.
    pub over_threshold: u32,
    /// Pages whose sky changed at all.
    pub changed: u32,
    /// The largest change of a sky value.
    pub largest_change: f32,
    /// The largest charge.
    pub largest_charge: f32,
}

/// Objects that may move (M9): where their subtree lives in the object
/// hierarchy, and their placements and world boxes now.
struct Movers {
    tree: MoverTree,
    origin: DVec3,
    /// Per mover (in `tree.objects` order): its placement, its padded
    /// world box, and whether it moved since the last frame.
    placements: Vec<Placement>,
    bounds: Vec<(Vec3, Vec3)>,
    dirty: Vec<bool>,
    /// Instance index → mover index.
    index: std::collections::HashMap<usize, usize>,
    /// Each geometry's object-space box.
    local: Vec<(Vec3, Vec3)>,
    /// A box around each light's cone ends (as the shader's `light_box`,
    /// a little larger).
    light_boxes: Vec<(Vec3, Vec3)>,
}

/// A box around every point a light cone toward `light` reaches near the
/// light (its bounding sphere; a sphere light's cone window, the square
/// around its disk, reaches about 1.42 radii from its center), a little
/// larger than the shader's `light_box`.
fn light_box(light: &Light) -> (Vec3, Vec3) {
    let (c, r) = match light.emitter {
        crate::Emitter::Rect(r) => (
            r.corner + (r.edge_u + r.edge_v) * 0.5,
            0.5 * (r.edge_u.length() + r.edge_v.length()),
        ),
        crate::Emitter::Sphere(s) => (s.center, 1.5 * s.radius),
    };
    let r = r * 1.01 + 2e-5 * c.abs().max_element().max(1.0) + 1e-3;
    (c - Vec3::splat(r), c + Vec3::splat(r))
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
    /// pass and a shading pass (and with the extra samples' list, the
    /// passes that make and shade it).
    passes: Vec<(wgpu::ComputePipeline, Vec<u32>)>,
    /// How each camera pass is dispatched: a workgroup per 8×8 pixels; one
    /// thread (`vc_args_edges`); or from the dispatch arguments it wrote
    /// (`vc_shade_edges`, M10 gate 19).
    camera_dispatch: Vec<CameraDispatch>,
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
    /// The target holds floats: it gets linear light, for an engine's own
    /// tone mapping.
    float_target: bool,
    /// Where camera rays cross each pixel, from its center, in pixels
    /// (references that average many sub-pixel positions).
    pixel_offset: [f32; 2],
    /// Textures are sampled for this share of a pixel's width (1: the
    /// pixel; 1/16 for references at 16×16 samples per pixel).
    footprint: f32,
    /// The display's exposure (see `set_exposure`).
    exposure: f32,
    /// Textures box-filtered over the pixel's footprint (M7 gate 2; the
    /// default), or isotropic trilinear (as before).
    texture_box: bool,
    movers: Option<Movers>,
    /// Lights that may move (`set_light`).
    moving: Vec<usize>,
    /// Camera levels streamed by need (M14 gate 1).
    stream: Option<Streamer>,
    /// Lighting levels streamed (M14 gate 4).
    lighting: Option<LightingPool>,
    /// The visibility buffer (binding 31), once the targets are made.
    vis: Option<wgpu::Buffer>,
}

/// The buffers lighting levels load into (M14 gate 4).
fn lit_targets<'a>(scene: &'a [(u32, wgpu::Buffer)], cache: &'a Cache) -> LitTargets<'a> {
    let find = |b: u32| {
        &scene
            .iter()
            .find(|(x, _)| *x == b)
            .expect("a scene buffer")
            .1
    };
    LitTargets {
        tris: find(2),
        normals: find(24),
        tex: find(58),
        meshes: find(3),
        nodes: find(25),
        items: find(26),
        cov: find(29),
        uv: cache.state.buffer(23),
        chart: cache.state.buffer(35),
        cells: cache.state.buffer(34),
        cell_tris: cache.state.buffer(54),
    }
}

/// The buffers camera levels upload into (M14 gate 1).
fn stream_targets<'a>(scene: &'a [(u32, wgpu::Buffer)], cache: Option<&'a Cache>) -> Targets<'a> {
    let find = |b: u32| {
        &scene
            .iter()
            .find(|(x, _)| *x == b)
            .expect("a scene buffer")
            .1
    };
    Targets {
        tris: find(2),
        normals: find(24),
        tex: find(58),
        uv: cache.map(|c| c.state.buffer(23)),
        chart: cache.map(|c| c.state.buffer(35)),
        xforms: find(30),
        meshes: find(3),
    }
}

fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
    tlas: &[(u32, &wgpu::Tlas)],
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
    for (binding, tlas) in tlas {
        if keep.contains(binding) {
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: tlas.as_binding(),
            });
        }
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
    if let Some(t) = textures
        && keep.contains(&79)
    {
        entries.push(wgpu::BindGroupEntry {
            binding: 79,
            resource: wgpu::BindingResource::TextureView(&t.normals),
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
            &[],
            &[],
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
            &[],
            &[],
            None,
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
            &[],
            &[],
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
            &[],
            &[],
            None,
        )
    }

    /// A renderer whose instances `movable` may move
    /// ([`Self::set_instance_transform`]), with or without a lighting cache
    /// (M9). The cache re-bakes what a move makes stale within the frame
    /// and never shows stale light; it does not support a sky with movers
    /// yet.
    #[allow(clippy::too_many_arguments)]
    pub fn with_movers(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
        cache: Option<CacheOptions>,
        movable: &[usize],
        moving_lights: &[usize],
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
            cache,
            movable,
            moving_lights,
            None,
        )
    }

    /// [`Self::with_movers`] without movers, with camera levels streamed by
    /// need (M14 gate 1): the scene's `geometry_levels` stream within
    /// `stream`'s budget (with hardware camera rays; otherwise ignored).
    #[allow(clippy::too_many_arguments)]
    pub fn with_streaming(
        kernel: &MaskConeKernel,
        target_format: wgpu::TextureFormat,
        target_width: u32,
        target_height: u32,
        scale: u32,
        lights: &[Light],
        scene: &InstancedScene,
        origin: DVec3,
        cache: Option<CacheOptions>,
        stream: StreamOptions,
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
            cache,
            &[],
            &[],
            Some(stream),
        )
    }

    /// Moves (and re-colors) light `index`, one of the moving lights given
    /// at construction, from the next frame: per pixel, the next frame is
    /// exact; with the cache, the ledger re-bakes what may be more than
    /// `CacheOptions::ledger_epsilon` off (M9 gate 2). Spots cannot move.
    pub fn set_light(&mut self, index: usize, light: Light) {
        assert!(
            self.moving.contains(&index),
            "light {index} is not a moving light"
        );
        assert!(
            !matches!(
                light.emitter,
                crate::Emitter::Sphere(crate::SphereLight { spot: Some(_), .. })
            ),
            "moving spots are not supported"
        );
        let gpu = LightGpu::from(&light);
        let buffer = &self
            .scene
            .iter()
            .find(|(b, _)| *b == 52)
            .expect("lights buffer")
            .1;
        self.queue.write_buffer(
            buffer,
            (index * std::mem::size_of::<LightGpu>()) as u64,
            bytemuck::bytes_of(&gpu),
        );
        if let Some(cache) = &mut self.cache {
            cache.state.set_light_pose(index, gpu);
        }
    }

    /// Moves movable instance `instance` (given at construction) to
    /// `rotation` (orthonormal) and `position` from the next frame. Its
    /// scale stays.
    pub fn set_instance_transform(&mut self, instance: usize, rotation: Mat3, position: DVec3) {
        let movers = self.movers.as_mut().expect("a renderer with movers");
        let &k = movers
            .index
            .get(&instance)
            .unwrap_or_else(|| panic!("instance {instance} is not movable"));
        let p = &mut movers.placements[k];
        let translation = (position - movers.origin).as_vec3();
        if p.rotation != rotation || p.translation != translation {
            p.rotation = rotation;
            p.translation = translation;
            movers.dirty[k] = true;
        }
    }

    /// Applies the moves since the last frame: the movers' transforms, their
    /// subtree and the root, their hardware structure (in `encoder`), and
    /// the cache's list of moved objects for its stale pass.
    fn apply_moves(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(movers) = &mut self.movers else {
            return;
        };
        if !movers.dirty.iter().any(|&d| d) {
            return;
        }
        let buffer = |b: u32| {
            &self
                .scene
                .iter()
                .find(|(k, _)| *k == b)
                .expect("scene buffer")
                .1
        };
        let mut moved = Vec::new();
        for k in 0..movers.placements.len() {
            if !std::mem::take(&mut movers.dirty[k]) {
                continue;
            }
            let p = movers.placements[k];
            let object = movers.tree.objects[k];
            let old = movers.bounds[k];
            let new = p.world_bounds(movers.local[p.geometry]);
            movers.bounds[k] = new;
            moved.push((object, [old.0, old.1], [new.0, new.1]));
            self.queue.write_buffer(
                buffer(30),
                object as u64 * 64,
                bytemuck::cast_slice(&p.rows()),
            );
        }
        let (root, region, items) = mover_subtree(&movers.tree, &movers.bounds);
        let node = std::mem::size_of::<NodeGpu>() as u64;
        let nodes = buffer(9);
        let top = union_root(movers.tree.fixed_root, root);
        self.queue.write_buffer(nodes, 0, bytemuck::bytes_of(&top));
        self.queue
            .write_buffer(nodes, 2 * node, bytemuck::bytes_of(&root));
        if !region.is_empty() {
            self.queue.write_buffer(
                nodes,
                movers.tree.first_node as u64 * node,
                bytemuck::cast_slice(&region),
            );
        }
        self.queue.write_buffer(
            buffer(10),
            movers.tree.first_item as u64 * 4,
            bytemuck::cast_slice(&items),
        );
        if let Some(hw) = &mut self.hw {
            hw.update_movers(encoder, &movers.placements);
        }
        if let Some(cache) = &mut self.cache {
            // Every texel's apex lies within the objects' boxes; the
            // movers' old boxes stay in for this frame's test.
            let (mut lo, mut hi) = top.bounds();
            for (_, old, _) in &moved {
                (lo, hi) = (lo.min(old[0]), hi.max(old[1]));
            }
            let pad = 1e-3 * lo.abs().max(hi.abs()).max_element().max(1.0);
            let scene = (lo - Vec3::splat(pad), hi + Vec3::splat(pad));
            cache.state.set_moved(&moved, &movers.light_boxes, scene);
        }
    }

    /// How the display pass writes colors (`display.wgsl`): 0 with the tone
    /// curve to an sRGB target, 1 with the curve and gamma, 2 linear (float
    /// targets).
    fn display_mode(&self) -> u32 {
        if self.float_target {
            2
        } else {
            !self.srgb as u32
        }
    }

    /// Camera rays cross each pixel `(dx, dy)` pixels from its center (x
    /// right, y down) from the next frame: for supersampled references,
    /// which average frames over a grid of offsets. Pixel cones ignore it.
    pub fn set_pixel_offset(&mut self, dx: f32, dy: f32) {
        self.pixel_offset = [dx, dy];
    }

    /// Scales the colors drawn to the target by `factor` before the tone
    /// curve (1 by default; a float target gets them scaled, linear) from
    /// the next frame. `read_colors` stays unscaled.
    pub fn set_exposure(&mut self, factor: f32) {
        self.exposure = factor.max(0.0);
    }

    /// Textures are sampled at the footprint of `share` of a pixel's width
    /// from the next frame (1 by default): for references that average
    /// `n × n` sub-pixel samples, `1 / n`, as at `n` times the resolution.
    pub fn set_footprint(&mut self, share: f32) {
        self.footprint = share;
    }

    /// Experiments: textures box-filtered over each pixel's footprint (the
    /// default, M7 gate 2) or isotropic trilinear, from the next frame.
    #[doc(hidden)]
    pub fn set_texture_box(&mut self, on: bool) {
        self.texture_box = on;
    }

    /// Changes light `index`'s RGB radiance from the next frame. Lighting is
    /// linear in emitted power: per-pixel frames shade with it, and the cache
    /// re-weights its stored channels, baking nothing (except bounce light
    /// for a light past the third, which re-bakes: M10 gate 21).
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
        // Bounce light holds lights past the third at their radiance (M10
        // gate 21): cleared, pages ask for their samples again.
        if index >= 3
            && let Some(cache) = &self.cache
            && cache.state.indirect()
        {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(cache.state.buffer(73), 0, None);
            self.queue.submit([encoder.finish()]);
        }
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
        movable: &[usize],
        moving_lights: &[usize],
        stream: Option<StreamOptions>,
    ) -> Self {
        // Alpha blending needs the cache's extra-samples pass: hardware camera
        // rays, no movers (M10 gate 16); otherwise its fallback.
        let blending = cache.is_some()
            && kernel.uses_hardware_rays()
            && !kernel.pixel_cones.get()
            && movable.is_empty();
        let fallback;
        let scene: &InstancedScene =
            if !blending && scene.geometry_blend.iter().any(|b| b.is_some()) {
                fallback = scene.blend_as_mask();
                &fallback
            } else {
                scene
            };
        let cut = scene.with_cutouts();
        let scene: &InstancedScene = &cut;
        let device = kernel.device.clone();
        let scale = scale.max(1);
        let (width, height) = (target_width.div_ceil(scale), target_height.div_ceil(scale));
        // Camera levels stream with hardware camera rays only (M14 gate 1).
        let stream = stream.filter(|_| kernel.uses_hardware_rays() && !kernel.pixel_cones.get());
        assert!(
            stream.is_none() || movable.is_empty(),
            "streamed camera levels with movers are not supported yet"
        );
        let pool = stream.map_or(0, |o| crate::stream::pool_triangles(scene, &o));
        let (layout, lit_packages) = match stream
            .and_then(|o| crate::stream::lighting_layout(scene, &o, kernel.mikktspace.get()))
        {
            Some((l, p)) => (Some(l), Some(p)),
            None => (None, None),
        };
        if layout.is_some() {
            assert!(
                cache.is_some() && moving_lights.is_empty(),
                "streamed lighting levels need the cache, without moving lights (yet)"
            );
        }
        let (scene_buffers, tree, lit_nodes) =
            kernel.instanced_buffers(lights, scene, origin, movable, pool, layout.as_ref());
        let movers = tree.map(|tree| {
            let local: Vec<(Vec3, Vec3)> = scene
                .geometries
                .iter()
                .map(|g| mesh_bounds(&g.triangles))
                .collect();
            let placements: Vec<Placement> = tree
                .objects
                .iter()
                .map(|&m| {
                    let i = &scene.instances[m as usize];
                    Placement {
                        geometry: i.geometry,
                        rotation: i.rotation,
                        scale: i.scale,
                        translation: (i.position - origin).as_vec3(),
                        clear: scene.blend_clear(i.geometry),
                    }
                })
                .collect();
            Movers {
                bounds: placements
                    .iter()
                    .map(|p| p.world_bounds(local[p.geometry]))
                    .collect(),
                dirty: vec![false; placements.len()],
                index: tree
                    .objects
                    .iter()
                    .enumerate()
                    .map(|(k, &m)| (m as usize, k))
                    .collect(),
                placements,
                local,
                origin,
                light_boxes: lights.iter().map(light_box).collect(),
                tree,
            }
        });
        let camera_buf = uniform_buffer(
            &device,
            "camera",
            &[0; std::mem::size_of::<CameraUniform>()],
        );
        let dims_buf = uniform_buffer(&device, "dims", &[0; std::mem::size_of::<DisplayDims>()]);
        let display = display_pipeline(&device, target_format);
        let pixel_cones = kernel.pixel_cones.get();
        // Pixel cones need no camera rays, so no acceleration structure.
        // M13 gate 4 (profile stage 16) traces corner rays in hardware.
        let hw = (kernel.uses_hardware_rays()
            && (!pixel_cones || kernel.profile_stage.get() == 16))
            .then(|| {
                // Top-level slots for objects showing finer levels than
                // their proxies (M14 gate 6).
                let dynamic = stream.map_or(0, |o| crate::stream::detailed_capacity(scene, &o));
                kernel.hw_scene(scene, origin, movable, dynamic)
            });
        // Bounce samples trace through the camera's structures where there
        // are some (M14 gate 10).
        let bounce_hw = hw.is_some() && !pixel_cones;
        let visibility = hw.as_ref().map(|hw| {
            if hw.movers.is_some() {
                (
                    kernel.entry("hw_visibility_movers"),
                    HW_VISIBILITY_MOVERS_BINDINGS.to_vec(),
                )
            } else {
                (
                    kernel.entry("hw_visibility"),
                    HW_VISIBILITY_BINDINGS.to_vec(),
                )
            }
        });
        // Silhouette rays through ray queries (M7 gate 4): with the cache's
        // edge rays and hardware camera rays (not with movers, whose own
        // structure the pass does not trace).
        let blend = scene.geometry_blend.iter().any(|b| b.is_some());
        let hw_edges = cache.as_ref().is_some_and(|o| o.edge_rays || blend)
            && hw.as_ref().is_some_and(|hw| hw.movers.is_none());
        let shade_bindings = |list: &[u32]| {
            let mut keep = list.to_vec();
            keep.push(31);
            keep
        };
        let passes = match (cache.is_some(), visibility) {
            // M13: the scalar coverage walk in its own entry point (profile
            // stage 14), so it does not run at the pixel-cone entry's
            // occupancy.
            (false, _) if pixel_cones && kernel.profile_stage.get() == 16 => vec![
                (
                    kernel.entry("hw_m13_corners"),
                    HW_M13_CORNERS_BINDINGS.to_vec(),
                ),
                (kernel.entry("hw_m13_near"), M13_NEAR_BINDINGS.to_vec()),
            ],
            (false, _) if pixel_cones && kernel.profile_stage.get() == 15 => {
                vec![(
                    kernel.entry("m13_coverage_maps"),
                    M13_MAPS_BINDINGS.to_vec(),
                )]
            }
            (false, _) if pixel_cones && kernel.profile_stage.get() == 14 => {
                vec![(kernel.entry("m13_coverage"), M13_COVERAGE_BINDINGS.to_vec())]
            }
            (false, _) if pixel_cones => {
                vec![(kernel.entry("render_cones"), RECT_LIGHT_BINDINGS.to_vec())]
            }
            (true, _) if pixel_cones => vec![(
                kernel.entry("vc_shade_cones"),
                VC_SHADE_CONES_BINDINGS.to_vec(),
            )],
            (false, Some(v)) => vec![
                v,
                (kernel.entry("shade"), shade_bindings(RECT_LIGHT_BINDINGS)),
            ],
            (false, None) => vec![(kernel.entry("render"), RECT_LIGHT_BINDINGS.to_vec())],
            (true, Some(v)) if hw_edges => {
                // With blending, the visibility pass lists the layers behind
                // blended hits (M10 gate 19); `hw_edges` runs for
                // silhouette rays alone.
                let mut list = vec![if blend {
                    (
                        kernel.entry("hw_visibility_blend"),
                        HW_VISIBILITY_BLEND_BINDINGS.to_vec(),
                    )
                } else {
                    v
                }];
                if cache.as_ref().is_some_and(|o| o.edge_rays) {
                    list.push((kernel.entry("hw_edges"), HW_EDGES_BINDINGS.to_vec()));
                }
                list.push((kernel.entry("vc_shade"), VC_SHADE_BINDINGS.to_vec()));
                list.push((
                    kernel.entry("vc_args_edges"),
                    VC_ARGS_EDGES_BINDINGS.to_vec(),
                ));
                list.push((
                    kernel.entry("vc_shade_edges"),
                    VC_SHADE_EDGES_BINDINGS.to_vec(),
                ));
                list
            }
            (true, v) => vec![
                v.unwrap_or_else(|| {
                    (
                        kernel.entry("visibility"),
                        COMPUTE_VISIBILITY_BINDINGS.to_vec(),
                    )
                }),
                (kernel.entry("vc_shade"), VC_SHADE_BINDINGS.to_vec()),
            ],
        };
        let camera_dispatch: Vec<CameraDispatch> = passes
            .iter()
            .map(|(_, bindings)| match bindings.as_slice() {
                b if b == VC_ARGS_EDGES_BINDINGS => CameraDispatch::One,
                b if b == VC_SHADE_EDGES_BINDINGS => CameraDispatch::Listed,
                _ => CameraDispatch::Screen,
            })
            .collect();
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
                movable.len() as u32,
                moving_lights,
                &lights.iter().map(LightGpu::from).collect::<Vec<_>>(),
                pool,
                layout.as_ref(),
            ),
            movers: (!movable.is_empty() || !moving_lights.is_empty()).then(|| {
                (
                    [
                        kernel.entry("vc_stale"),
                        kernel.entry("vc_args"),
                        kernel.entry("vc_stale_texels"),
                        kernel.entry("vc_args_bake"),
                        kernel.entry("vc_ledger_reset"),
                        kernel.entry("vc_ledger"),
                    ],
                    Vec::new(),
                )
            }),
            has_movers: !movable.is_empty(),
            moving_lights: !moving_lights.is_empty(),
            // Bounce rays through hardware ray queries where camera rays
            // use them (M14 gate 10).
            indirect: options.indirect.then(|| {
                let name = if bounce_hw {
                    "hw_vc_bake_indirect"
                } else {
                    "vc_bake_indirect"
                };
                (kernel.entry(name), None)
            }),
            indirect_hw: bounce_hw,
            passes: [
                kernel.entry("vc_update"),
                kernel.entry("vc_bake"),
                kernel.entry("vc_bake_sky"),
                kernel.entry("vc_fill"),
            ],
            binds: Vec::new(),
        });
        let mut cache = cache;
        if hw_edges && let Some(c) = &mut cache {
            c.state.set_hardware_edges(blend);
        }
        let colors = Self::colors_buffer(&device, width, height);
        let mut renderer = InstancedRenderer {
            device: device.clone(),
            queue: kernel.queue.clone(),
            passes,
            camera_dispatch,
            hw,
            display_bind: display_bind(&device, &display, &colors, &dims_buf),
            mark: kernel.entry("mark"),
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
            textures: {
                // The environment's image after the scene's (M5 gate 12).
                let (all, _) = kernel.textures_with_environment(&scene.textures);
                kernel.scene_textures(&all, &scene.linear_textures())
            },
            srgb: target_format.is_srgb(),
            float_target: matches!(
                target_format,
                wgpu::TextureFormat::Rgba16Float
                    | wgpu::TextureFormat::Rgba32Float
                    | wgpu::TextureFormat::Rg11b10Ufloat
            ),
            pixel_offset: [0.0; 2],
            footprint: 1.0,
            exposure: 1.0,
            texture_box: true,
            movers,
            moving: moving_lights.to_vec(),
            stream: None,
            lighting: None,
            vis: None,
        };
        if let (Some(lay), Some(nodes), Some(packages), Some(options)) =
            (layout.clone(), lit_nodes, lit_packages, stream)
        {
            let cells = renderer
                .cache
                .as_mut()
                .and_then(|c| c.state.take_lit_cells())
                .expect("the cache's lighting tables");
            renderer.lighting = Some(LightingPool::new(
                lay,
                nodes,
                cells,
                packages,
                scene,
                &options,
                kernel.mikktspace.get(),
                &device,
            ));
        }
        if let (Some(options), true) = (stream, pool > 0) {
            let base = layout.as_ref().map_or_else(
                || {
                    scene
                        .geometries
                        .iter()
                        .map(|g| g.triangles.len())
                        .sum::<usize>() as u32
                },
                |l| l.lit_tris(),
            );
            let hw = renderer.hw.as_mut().expect("hardware camera rays");
            let targets = stream_targets(&renderer.scene, renderer.cache.as_ref());
            // The page table follows the objects' ranges (at least one) and
            // the clusters' member lists (M14 gate 6).
            let clustered = scene
                .instances
                .iter()
                .filter(|i| crate::hw::proxied(scene, i.geometry))
                .count() as u32;
            let table_base = 4 * scene.instances.len().max(1) as u32 + 4 * clustered.div_ceil(4);
            renderer.stream = Some(Streamer::new(
                &renderer.device,
                &renderer.queue,
                scene,
                options,
                base,
                table_base,
                hw,
                &targets,
            ));
        }
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
        let pipeline = &kernel.entry("vc_self_bake");
        let group = bind_group(
            &self.device,
            pipeline,
            &all,
            &[],
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if let Some(stream) = &mut self.stream {
            stream.bind(device, &vis);
        }
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
        let mut tlas: Vec<(u32, &wgpu::Tlas)> = Vec::new();
        if let Some(hw) = &self.hw {
            tlas.push((32, &hw.tlas));
            if let Some((movers, _)) = &hw.movers {
                tlas.push((59, movers));
            }
        }
        self.binds = self
            .passes
            .iter()
            .map(|(pipeline, keep)| {
                bind_group(device, pipeline, &all, &tlas, Some(&self.textures), keep)
            })
            .collect();
        let cache_binds = self.cache.as_ref().map(|cache| {
            let binds: Vec<wgpu::BindGroup> = cache
                .passes
                .iter()
                .zip([
                    VC_UPDATE_BINDINGS,
                    VC_BAKE_BINDINGS,
                    VC_BAKE_SKY_BINDINGS,
                    VC_FILL_BINDINGS,
                ])
                .map(|(pipeline, keep)| bind_group(device, pipeline, &all, &[], None, keep))
                .collect();
            let movers: Option<Vec<wgpu::BindGroup>> = cache.movers.as_ref().map(|(passes, _)| {
                passes
                    .iter()
                    .zip([
                        VC_STALE_BINDINGS,
                        VC_ARGS_BINDINGS,
                        VC_STALE_TEXELS_BINDINGS,
                        VC_ARGS_BAKE_BINDINGS,
                        VC_LEDGER_RESET_BINDINGS,
                        VC_LEDGER_BINDINGS,
                    ])
                    .map(|(pipeline, keep)| bind_group(device, pipeline, &all, &[], None, keep))
                    .collect()
            });
            (binds, movers)
        });
        let bounce_bind = self.cache.as_ref().and_then(|cache| {
            cache.indirect.as_ref().map(|(pipeline, _)| {
                if cache.indirect_hw {
                    bind_group(
                        device,
                        pipeline,
                        &all,
                        &tlas,
                        Some(&self.textures),
                        HW_VC_BAKE_INDIRECT_BINDINGS,
                    )
                } else {
                    bind_group(
                        device,
                        pipeline,
                        &all,
                        &[],
                        Some(&self.textures),
                        VC_BAKE_INDIRECT_BINDINGS,
                    )
                }
            })
        });
        if let (Some(cache), Some(bind)) = (&mut self.cache, bounce_bind)
            && let Some((_, b)) = &mut cache.indirect
        {
            *b = Some(bind);
        }
        if let (Some(cache), Some((binds, movers))) = (&mut self.cache, cache_binds) {
            cache.binds = binds;
            if let (Some((_, b)), Some(movers)) = (&mut cache.movers, movers) {
                *b = movers;
            }
        }
        self.display_bind = display_bind(device, &self.display, &self.colors, &self.dims_buf);
        self.vis = Some(vis);
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
        match (self.camera_dispatch[k], &self.cache) {
            (CameraDispatch::One, _) => pass.dispatch_workgroups(1, 1, 1),
            (CameraDispatch::Listed, Some(cache)) => {
                pass.dispatch_workgroups_indirect(cache.state.buffer(66), 36)
            }
            _ => pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1),
        }
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
        self.frame_to(camera, Some(target));
    }

    /// [`Self::frame`] without drawing to a target: the frame's colors stay
    /// in the renderer for [`Self::encode_display`] (or `read_colors`).
    pub fn frame_compute(&mut self, camera: Camera) {
        self.frame_to(camera, None);
    }

    /// Encodes the last frame's colors drawn to `target` (of the format the
    /// renderer was made for) into `encoder`: for engines that order their
    /// own passes, after [`Self::frame_compute`].
    pub fn encode_display(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        crate::live::encode_display(encoder, &self.display, &self.display_bind, target, None);
    }

    fn frame_to(&mut self, camera: Camera, target: Option<&wgpu::TextureView>) {
        let mut cam = camera_uniform(camera, self.width, self.height, self.radiance);
        cam.right[3] = self.pixel_offset[0];
        cam.up[3] = self.pixel_offset[1];
        cam.eye[3] = self.footprint;
        cam.forward[3] = self.texture_box as u32 as f32;
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [self.display_mode(), self.scale],
            exposure: [self.exposure, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        if let Some(cache) = &mut self.cache {
            cache
                .state
                .begin_frame(&self.device, self.lighting.as_ref().map(|l| l.resident()));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let (Some(pool), Some(cache)) = (&mut self.lighting, &self.cache) {
            let (wanted, pinned) = cache.state.lighting_requests();
            let targets = lit_targets(&self.scene, cache);
            pool.update(&mut encoder, &targets, wanted, pinned);
        }
        self.apply_moves(&mut encoder);
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
        if let (Some(stream), Some(hw)) = (&mut self.stream, &mut self.hw) {
            let targets = stream_targets(&self.scene, self.cache.as_ref());
            let per_meter =
                2.0 * (camera.vertical_fov_degrees.to_radians() * 0.5).tan() / self.height as f32;
            stream.begin_frame(
                &self.device,
                &self.queue,
                &mut encoder,
                hw,
                &targets,
                (self.width, self.height),
                per_meter,
            );
        }
        for k in 0..self.passes.len() {
            self.encode_camera_pass(&mut encoder, k);
        }
        self.encode_camera_end(&mut encoder);
        if let Some(stream) = &self.stream {
            stream.encode_feedback(&mut encoder, (self.width, self.height));
        }
        if let Some(cache) = &mut self.cache {
            cache.state.encode_readback(&mut encoder);
        }
        if let Some(target) = target {
            crate::live::encode_display(
                &mut encoder,
                &self.display,
                &self.display_bind,
                target,
                None,
            );
        }
        if let Some(cache) = &self.cache {
            cache.state.encode_timestamps(&mut encoder);
        }
        let index = self.queue.submit([encoder.finish()]);
        if let Some(stream) = &mut self.stream {
            stream.after_submit(index.clone());
        }
        if let Some(cache) = &mut self.cache {
            cache.state.after_submit(index);
        }
    }

    /// What camera-level streaming did in every frame so far (M14 gate 1;
    /// empty without it).
    pub fn stream_log(&self) -> &[StreamStats] {
        self.stream.as_ref().map_or(&[], |s| s.log())
    }

    /// Evicts every streamed level at the next frame (diagnostics: what
    /// loads cost).
    #[doc(hidden)]
    pub fn stream_evict_all(&mut self) {
        if let Some(s) = &mut self.stream {
            s.evict_all();
        }
    }

    /// What lighting-level streaming did in every frame so far (M14 gate 4;
    /// empty without it), and the cache's pages waiting for lighting levels.
    pub fn lighting_log(&self) -> (&[LightingStats], usize) {
        (
            self.lighting.as_ref().map_or(&[], |l| l.log()),
            self.cache.as_ref().map_or(0, |c| c.state.waiting_pages()),
        )
    }

    /// Bytes of camera levels kept in CPU memory for streaming (M14 gate
    /// 1; 0 without streaming).
    pub fn stream_cpu_bytes(&self) -> u64 {
        self.stream.as_ref().map_or(0, |s| s.cpu_bytes())
    }

    /// Triangles of every streamed geometry's finest camera level (M14
    /// gate 1; 0 without streaming).
    pub fn stream_finest_triangles(&self) -> u64 {
        self.stream.as_ref().map_or(0, |s| s.finest_triangles())
    }

    /// The object each pixel's camera ray hit in the latest frame (`u32::MAX`
    /// for none), waiting for the GPU. For tests and gates.
    pub fn read_visibility(&self) -> Vec<u32> {
        let vis = self.vis.as_ref().expect("targets made");
        let size = vis.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(vis, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let words: Vec<u32> = bytemuck::cast_slice::<u8, u32>(
            &staging
                .slice(..)
                .get_mapped_range()
                .expect("mapped visibility"),
        )
        .to_vec();
        let n = (self.width * self.height) as usize;
        words.chunks(4).take(n).map(|v| v[0]).collect()
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
        let mut cam = camera_uniform(camera, self.width, self.height, self.radiance);
        cam.right[3] = self.pixel_offset[0];
        cam.up[3] = self.pixel_offset[1];
        cam.eye[3] = self.footprint;
        cam.forward[3] = self.texture_box as u32 as f32;
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [self.display_mode(), self.scale],
            exposure: [self.exposure, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        self.wait();
        let mut profile = FrameProfile::default();
        let start = Instant::now();
        if let Some(cache) = &mut self.cache {
            cache.state.begin_frame(&self.device, None);
        }
        profile.planning = start.elapsed();
        let start = Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.apply_moves(&mut encoder);
        let mut counts = None;
        if let Some(cache) = &mut self.cache {
            cache.state.encode_clear(&mut encoder);
            counts = Some(cache.state.upload(
                &self.queue,
                self.width,
                self.height,
                camera.vertical_fov_degrees,
            ));
        }
        self.queue.submit([encoder.finish()]);
        self.wait();
        // Each bake pass on its own, waited for.
        if let (Some(cache), Some(counts)) = (&self.cache, counts) {
            let mut none = self.device.create_command_encoder(&Default::default());
            let labels = cache.encode_bake_passes(&mut none, counts, Some(usize::MAX));
            for (k, label) in labels.into_iter().enumerate() {
                let pass_start = Instant::now();
                let mut encoder = self.device.create_command_encoder(&Default::default());
                cache.encode_bake_passes(&mut encoder, counts, Some(k));
                self.queue.submit([encoder.finish()]);
                self.wait();
                if k < 8 {
                    profile.bake_passes[k] = (label, pass_start.elapsed());
                }
            }
        }
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
            if k < 4 {
                profile.passes[k] = start.elapsed();
            }
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

    /// Diagnostics (M9 gate 1): re-bakes every resident page that shading
    /// may read (not marked stale) from scratch, with the scene as it is,
    /// and compares every texel's every channel with what the cache held,
    /// bit for bit. Waits for the GPU. The pages keep the re-baked values;
    /// slot use is restored, so later frames run as they would have.
    pub fn verify_cache(&mut self) -> CacheCheck {
        let Some(cache) = &mut self.cache else {
            return CacheCheck::default();
        };
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let device = &self.device;
        let before = cache.state.read_buffer(device, 37);
        let marks = cache.state.read_buffer(device, 62);
        let used = cache.state.read_buffer(device, 40);
        let pages = cache.state.readable_pages(bytemuck::cast_slice(&marks));
        let chunk = cache.state.rebake_chunk();
        for part in pages.chunks(chunk) {
            let counts = cache.state.upload_rebake(&self.queue, part);
            let mut encoder = device.create_command_encoder(&Default::default());
            cache.state.encode_clear(&mut encoder);
            cache.encode_bake(&mut encoder, counts);
            self.queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
        }
        self.queue.write_buffer(cache.state.buffer(40), 0, &used);
        let after = cache.state.read_buffer(device, 37);
        let (before, after): (&[f32], &[f32]) =
            (bytemuck::cast_slice(&before), bytemuck::cast_slice(&after));
        let channels = self.lights + 1;
        let per_page = PAGE_TEXELS as usize * channels;
        let mut check = CacheCheck {
            pages: pages.len() as u32,
            ..Default::default()
        };
        for &slot in &pages {
            let first = slot as usize * per_page;
            for k in first..first + per_page {
                check.values += 1;
                if before[k].to_bits() != after[k].to_bits() {
                    check.differing += 1;
                    check.largest = check.largest.max((before[k] - after[k]).abs());
                }
            }
        }
        check
    }

    /// Diagnostics (M9 gate 2): for moving light `light`, re-bakes every
    /// page shading may read from scratch and compares each page's largest
    /// change of a texel's ratio with the bound the ledger certified for it.
    /// Restores the cache afterwards, so later frames run as they would
    /// have. Waits for the GPU.
    pub fn verify_ledger(&mut self, light: usize) -> LedgerCheck {
        let mut check = LedgerCheck::default();
        let Some(j) = self.moving.iter().position(|&l| l == light) else {
            return check;
        };
        let Some(cache) = &mut self.cache else {
            return check;
        };
        let device = self.device.clone();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        // Kept on the GPU, restored at the end.
        let keep: Vec<(u32, wgpu::Buffer)> = [37u32, 40, 61, 62, 69, 72]
            .iter()
            .map(|&b| {
                let source = cache.state.buffer(b);
                let copy = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ledger check copy"),
                    size: source.size(),
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut e = device.create_command_encoder(&Default::default());
                e.copy_buffer_to_buffer(source, 0, &copy, 0, None);
                self.queue.submit([e.finish()]);
                (b, copy)
            })
            .collect();
        let before = cache.state.read_buffer(&device, 37);
        let marks = cache.state.read_buffer(&device, 62);
        let bounds = cache.state.read_buffer(&device, 72);
        let records = cache.state.read_buffer(&device, 69);
        let pages = cache.state.readable_pages(bytemuck::cast_slice(&marks));
        let chunk = cache.state.rebake_chunk();
        for part in pages.chunks(chunk) {
            let counts = cache.state.upload_rebake(&self.queue, part);
            let mut encoder = device.create_command_encoder(&Default::default());
            cache.state.encode_clear(&mut encoder);
            cache.encode_bake(&mut encoder, counts);
            self.queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
        }
        let after = cache.state.read_buffer(&device, 37);
        let mut e = device.create_command_encoder(&Default::default());
        for (b, copy) in &keep {
            e.copy_buffer_to_buffer(copy, 0, cache.state.buffer(*b), 0, None);
        }
        self.queue.submit([e.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let (before, after): (&[f32], &[f32]) =
            (bytemuck::cast_slice(&before), bytemuck::cast_slice(&after));
        let bounds: &[f32] = bytemuck::cast_slice(&bounds);
        let channels = self.lights + 1;
        let moving = cache.state.moving_lights() as usize;
        let epsilon = cache.state.epsilon();
        for &slot in &pages {
            let first = slot as usize * PAGE_TEXELS as usize;
            let mut change = 0.0f32;
            for k in first..first + PAGE_TEXELS as usize {
                let at = k * channels + light;
                change = change.max((before[at] - after[at]).abs());
            }
            let bound = bounds[slot as usize * moving + j];
            check.pages += 1;
            check.largest_change = check.largest_change.max(change);
            check.largest_bound = check.largest_bound.max(bound);
            if change > bound + 1e-5 {
                check.violations += 1;
                if std::env::var_os("CONTINUUM_LEDGER_DEBUG").is_some()
                    && (check.violations <= 3 || change > 0.01)
                {
                    let words: &[u32] = bytemuck::cast_slice(&records);
                    let base = (slot as usize * moving + j) * 48;
                    let r = &words[base..base + 48];
                    eprintln!(
                        "slot {slot}: version {}, guard {:e}, u_min {:e}, alpha {:e}, flags {}, bound {bound}",
                        r[0],
                        f32::from_bits(r[1]),
                        f32::from_bits(r[2]),
                        f32::from_bits(r[3]),
                        r[4]
                    );
                    for k in 0..PAGE_TEXELS as usize {
                        let at = (first + k) * channels + light;
                        if before[at] != after[at] {
                            let half = (r[5 + k / 2] >> (16 * (k % 2))) & 0xffff;
                            eprintln!(
                                "  texel {k}: state {}, code {}, offset {}: {} -> {}",
                                half & 3,
                                (half >> 2) & 15,
                                half >> 6,
                                before[at],
                                after[at]
                            );
                        }
                    }
                }
            }
            if change > epsilon {
                check.over_epsilon += 1;
            }
            if change > 0.0 {
                check.changed += 1;
            }
        }
        check
    }

    /// Diagnostics (M9 gate 3): re-bakes every page shading may read from
    /// scratch; compares its light channels bit for bit, and its sky's
    /// largest change with the charge its sky ledger holds and its
    /// threshold. Restores the cache afterwards. Waits for the GPU.
    pub fn verify_sky(&mut self) -> SkyCheck {
        let mut check = SkyCheck::default();
        let Some(cache) = &mut self.cache else {
            return check;
        };
        let device = self.device.clone();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let keep: Vec<(u32, wgpu::Buffer)> = [37u32, 40, 61, 62, 77]
            .iter()
            .map(|&b| {
                let source = cache.state.buffer(b);
                let copy = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sky check copy"),
                    size: source.size(),
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut e = device.create_command_encoder(&Default::default());
                e.copy_buffer_to_buffer(source, 0, &copy, 0, None);
                self.queue.submit([e.finish()]);
                (b, copy)
            })
            .collect();
        let before = cache.state.read_buffer(&device, 37);
        let marks = cache.state.read_buffer(&device, 62);
        let ledger = cache.state.read_buffer(&device, 77);
        let pages = cache.state.readable_pages(bytemuck::cast_slice(&marks));
        let chunk = cache.state.rebake_chunk();
        for part in pages.chunks(chunk) {
            let counts = cache.state.upload_rebake(&self.queue, part);
            let mut encoder = device.create_command_encoder(&Default::default());
            cache.state.encode_clear(&mut encoder);
            cache.encode_bake(&mut encoder, counts);
            self.queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
        }
        let after = cache.state.read_buffer(&device, 37);
        let mut e = device.create_command_encoder(&Default::default());
        for (b, copy) in &keep {
            e.copy_buffer_to_buffer(copy, 0, cache.state.buffer(*b), 0, None);
        }
        self.queue.submit([e.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let (before, after): (&[f32], &[f32]) =
            (bytemuck::cast_slice(&before), bytemuck::cast_slice(&after));
        let ledger: &[f32] = bytemuck::cast_slice(&ledger);
        let channels = self.lights + 1;
        for &slot in &pages {
            let first = slot as usize * PAGE_TEXELS as usize;
            let mut change = 0.0f32;
            for k in first..first + PAGE_TEXELS as usize {
                for l in 0..self.lights {
                    let at = k * channels + l;
                    if before[at].to_bits() != after[at].to_bits() {
                        check.light_differing += 1;
                    }
                }
                let at = k * channels + self.lights;
                change = change.max((before[at] - after[at]).abs());
            }
            let at = 8 * slot as usize;
            let (charge, threshold) = (ledger[at] + ledger[at + 1].max(0.0), ledger[at + 2]);
            check.pages += 1;
            check.largest_change = check.largest_change.max(change);
            check.largest_charge = check.largest_charge.max(charge);
            check.violations += (change > charge + 1e-5) as u32;
            check.over_threshold += (change > threshold) as u32;
            if change > threshold && std::env::var_os("CONTINUUM_SKY_DEBUG").is_some() {
                let w: &[u32] = bytemuck::cast_slice(ledger);
                eprintln!(
                    "slot {slot}: change {change}, ledger {:?}, mask {:x} {:x} {:x}, stride {}, frame {}",
                    &ledger[at..at + 4],
                    w[at + 4],
                    w[at + 5],
                    w[at + 6],
                    w[at + 7],
                    cache.state.slot_debug(slot)
                );
            }
            check.changed += (change > 0.0) as u32;
        }
        check
    }

    /// Diagnostics: every page resident in the lighting cache with channel
    /// `channel` of its texels (one per light, then the sky's). Waits for the
    /// GPU; empty without a cache.
    pub fn read_cache_pages(&self, channel: u32) -> Vec<crate::CachePage> {
        self.cache
            .as_ref()
            .map_or(Vec::new(), |c| c.state.read_pages(&self.device, channel))
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
