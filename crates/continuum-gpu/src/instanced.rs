//! Interactive rendering of an instanced scene: shared geometry placed many
//! times, lit per pixel with mask cones (no caches), kept on the GPU between
//! frames.

use std::time::{Duration, Instant};

use crate::Tri;

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
    HW_EDGES_BINDINGS, HW_LAYERS_BINDINGS, HW_M13_CORNERS_BINDINGS, HW_VC_BAKE_INDIRECT_BINDINGS,
    HW_VC_BAKE_INDIRECT_MOVERS_BINDINGS, HW_VC_BAKE_SKY_BINDINGS, HW_VC_BAKE_SKY_MOVERS_BINDINGS,
    HW_VC_BAKE_WIDE_BINDINGS, HW_VC_ERROR_SHARES_BINDINGS, HW_VC_SHADE_BINDINGS,
    HW_VC_SHADE_EDGES_BINDINGS, HW_VISIBILITY_BINDINGS, HW_VISIBILITY_BLEND_BINDINGS,
    HW_VISIBILITY_MOVERS_BINDINGS, InstancedScene, Light, LightGpu, M13_BIN_ARGS_BINDINGS,
    M13_BIN_RAYS_BINDINGS, M13_BINS_BINDINGS, M13_CORNERS_TILES_BINDINGS, M13_COVERAGE_BINDINGS,
    M13_MAPS_BINDINGS, M13_MISS_ARGS_BINDINGS, M13_NEAR_BINDINGS, M13_PAGE_ARGS_BINDINGS,
    M13_PAGE_BAKE_BINDINGS, M13_PREP_BINDINGS, M13_REFERENCE_BINDINGS, M13_ROOTS_BINDINGS,
    M13_SHADE_ARGS_BINDINGS, M13_SHADE_BACKGROUND_BINDINGS, M13_SHADE_MISSES_BINDINGS,
    M13_SHADE_RAYS_BINDINGS, M13_TILES_BINDINGS, MaskConeKernel, RECT_LIGHT_BINDINGS, RectLight,
    SceneTextures, VC_ARGS_BAKE_BINDINGS, VC_ARGS_BINDINGS, VC_ARGS_EDGES_BINDINGS,
    VC_BAKE_BINDINGS, VC_BAKE_INDIRECT_BINDINGS, VC_BAKE_SKY_BINDINGS, VC_BOUNCE_BINDINGS,
    VC_ERROR_PROBES_BINDINGS, VC_FILL_BINDINGS, VC_LEDGER_BINDINGS, VC_LEDGER_RESET_BINDINGS,
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
    /// The first twelve camera passes, in order, each submitted and waited
    /// for alone.
    pub passes: [Duration; 16],
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
    /// One workgroup, or one invocation, per probe pixel (M4 gate 13).
    ProbeGroups,
    Probes,
    /// One invocation per 8×8 tile (M13 gate 5's roots pass).
    Tiles,
    /// From the arguments the bin rays' list wrote (M13 gate 5).
    M13Rays,
    /// From the arguments the page requests wrote (M13 gate 6).
    M13Pages,
    /// From the colour passes' arguments at a byte offset (M13 appearance,
    /// third).
    M13Args(u64),
    /// One invocation per page of the pool, per object-hierarchy node and
    /// per object (M13 gate 8's once-a-frame pass).
    Prep,
    /// A workgroup per 64 samples of the edge pixels listed for
    /// anti-aliasing (M15), from the dispatch arguments' words 12–14.
    Aa,
    /// A workgroup per 64 of their samples to shade, words 16–18.
    AaShade,
    /// A workgroup per 64 listed pixels, words 20–22.
    AaScatter,
    /// A workgroup per 16 pixels listed again (stage 1), words 24–26.
    AaAgain,
    /// Fog (M18): a workgroup per 8 × 8 cells of each layer of the sun's
    /// visibility grid; per 8 × 8 froxel columns.
    FogGrid,
    FogColumns,
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
        self.encode_bake_passes(encoder, counts, None, false);
    }

    /// `encode_bake` for a diagnostic re-bake that keeps the moving lights'
    /// ledger records: pages re-bake on their own lattices (M9 gate 6).
    fn encode_rebake(&self, encoder: &mut wgpu::CommandEncoder, counts: [u32; 6]) {
        self.encode_bake_passes(encoder, counts, None, true);
    }

    /// `encode_bake`'s passes, or with `only`, that one pass alone and
    /// untimed (diagnostics: `frame_profiled` submits each on its own);
    /// the passes' labels.
    fn encode_bake_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        counts: [u32; 6],
        only: Option<usize>,
        keep_ledger: bool,
    ) -> Vec<&'static str> {
        let groups = |n: u32| Dispatch::Groups(n.div_ceil(64), 1);
        let slots = |n: u32| {
            let g = n.div_ceil(64);
            Dispatch::Groups(g.min(65_535), g.div_ceil(65_535))
        };
        // The bake lists wide texels and coarse ones afresh (M5 gate 15, M22
        // gate 9).
        if (self.wide.is_some() || self.area.is_some()) && only.is_none_or(|o| o == 0) {
            encoder.clear_buffer(self.state.buffer(117), 0, Some(4));
        }
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
            if self.moving_lights && counts[3] > 0 && !keep_ledger {
                list.push(("ledger reset", &passes[4], &binds[4], groups(counts[3])));
            }
            if moved {
                list.push((
                    "stale",
                    &passes[0],
                    &binds[0],
                    slots(self.state.slot_extent()),
                ));
                list.push(("stale args", &passes[1], &binds[1], Dispatch::Groups(1, 1)));
                list.push(("stale texels", &passes[2], &binds[2], Dispatch::Indirect(0)));
            }
            // Bounce light's ledger (M22): charges after a move, or a drain
            // at rest.
            if self.state.bounce_pass().is_some() && !keep_ledger {
                list.push((
                    "bounce ledger",
                    &passes[6],
                    &binds[6],
                    slots(self.state.slot_extent()),
                ));
            }
            if lit {
                list.push((
                    "ledger",
                    &passes[5],
                    &binds[5],
                    slots(self.state.slot_extent()),
                ));
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
        // Rays to wide lights (M5 gate 15): a workgroup a listed texel, at
        // most the frame's texel jobs.
        if let Some((pipeline, Some(bind))) = &self.wide
            && (indirect || counts[1] > 0)
        {
            let n = if indirect {
                crate::vcache::WIDE_CAP
            } else {
                counts[1].min(crate::vcache::WIDE_CAP)
            };
            list.push((
                "wide",
                pipeline,
                bind,
                Dispatch::Groups(n.min(65_535), n.div_ceil(65_535)),
            ));
        }
        // Coarse texels averaging their area (M22 gate 9): likewise.
        if let Some((pipeline, Some(bind))) = &self.area
            && (indirect || counts[1] > 0)
        {
            let n = if indirect {
                crate::vcache::WIDE_CAP
            } else {
                counts[1].min(crate::vcache::WIDE_CAP)
            };
            list.push((
                "area",
                pipeline,
                bind,
                Dispatch::Groups(n.min(65_535), n.div_ceil(65_535)),
            ));
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
    movers: Option<([wgpu::ComputePipeline; 7], Vec<wgpu::BindGroup>)>,
    /// Movers in the scene, moving lights tracked.
    has_movers: bool,
    moving_lights: bool,
    /// Bounce light's sample pass and its bind group (M10 gate 1).
    indirect: Option<(wgpu::ComputePipeline, Option<wgpu::BindGroup>)>,
    /// Whether that pass traces through hardware ray queries (M14 gate 10).
    indirect_hw: bool,
    /// Sky samples through hardware ray queries (M5 gate 13).
    sky_hw: bool,
    /// Rays to wide lights (M5 gate 15): the pass and its bind group.
    wide: Option<(wgpu::ComputePipeline, Option<wgpu::BindGroup>)>,
    /// Coarse texels averaging their area (M22 gate 9): the pass and its
    /// bind group.
    area: Option<(wgpu::ComputePipeline, Option<wgpu::BindGroup>)>,
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

/// What [`InstancedRenderer::verify_bounce`] found (M22): resident pages'
/// bounce samples against a re-bake in place.
#[derive(Clone, Copy, Debug, Default)]
pub struct BounceCheck {
    /// Pages holding bounce samples that shading may read.
    pub pages: u32,
    /// The largest bounce value held or re-baked (the scale of the others).
    pub largest_value: f32,
    /// Pages whose samples changed by more than 1e-4 of that scale.
    pub changed: u32,
    /// Pages that changed by more than ε of that scale.
    pub over_epsilon: u32,
    /// Of those, pages whose charge had not reached ε (the ledger missed
    /// them).
    pub missed: u32,
    /// The largest change.
    pub largest_change: f32,
    /// The largest charge held.
    pub largest_charge: f32,
    /// Diagnostics: of the pages over ε, those read in the last frame.
    pub visible_over: u32,
    /// Of the pages over ε, those listed for a re-bake.
    pub listed_over: u32,
    /// Of the pages over ε, those whose samples were baked while something
    /// moved.
    pub moving_over: u32,
}

/// Two halves packed in a word (low first).
fn half_pair(w: u32) -> [f32; 2] {
    let f = |h: u32| {
        let (sign, exp, man) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
        let v = if exp == 0 {
            man as f32 * 2f32.powi(-24)
        } else if exp == 31 {
            f32::INFINITY
        } else {
            (1.0 + man as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
        };
        if sign == 1 { -v } else { v }
    };
    [f(w & 0xffff), f(w >> 16)]
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

/// The extra mips a deforming object's cache pages take (M12 gate 1): their
/// texels re-bake every frame it is posed, so a quarter as many.
const DEFORMING_MIPS: f32 = 1.0;

/// A geometry whose vertices change (M12 gate 1): where its triangles and
/// triangle hierarchy live, the rest pose's tree to refit, the words its
/// triangles' and normals' w hold (vertex colors, tangents), and the pose
/// to take next frame.
struct Deformer {
    geometry: usize,
    first: u32,
    root: u32,
    nodes: Vec<NodeGpu>,
    items: Vec<u32>,
    tri_w: Vec<f32>,
    normal_w: Vec<f32>,
    pose: Option<(Vec<Tri>, Vec<[Vec3; 3]>)>,
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
    /// Deforming geometry's instances (M12 gate 2): which movers deform;
    /// per mover and light, the box its lighting last caught up with and
    /// the frames it has lagged since; per mover the light catching up now
    /// (a slice of its shadows a frame) and the light after; the slices a
    /// catch-up takes, and per recent cache frame whether it re-baked a
    /// slice (to fit the slices to the budget).
    deforming: Vec<bool>,
    lag_boxes: Vec<Vec<(Vec3, Vec3)>>,
    lag_frames: Vec<Vec<u32>>,
    /// Per mover its poses so far, and per light the pose its lighting
    /// last caught up with (behind while less).
    poses: Vec<u32>,
    light_poses: Vec<Vec<u32>>,
    catching: Vec<Option<CatchUp>>,
    next_light: Vec<usize>,
    slices: u32,
    /// The texel–light cones a whole catch-up is estimated to re-bake (the
    /// largest seen, decaying slowly).
    catchup_cost: f32,
    updates: [(u32, u32); 8],
}

/// A light's lighting near a deforming object catching up with its pose
/// (M12 gate 2), a slice of its shadows a frame: the slice next, of how
/// many; the pose it catches up with (the first slice's) and the union of
/// the object's boxes over the slices so far (what it was baked against).
#[derive(Clone, Copy)]
struct CatchUp {
    light: usize,
    slice: u32,
    slices: u32,
    pose: u32,
    union: (Vec3, Vec3),
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
    /// The last frame's submission: the movers' top-level structure is
    /// rebuilt in place, so a frame that rebuilds it waits for the frame
    /// before (whose rays may still read it) to finish.
    last_submission: Option<wgpu::SubmissionIndex>,
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
    /// M13 gates 5–6: per frame, the pixels tracing bin rays and each
    /// pixel's covered bins (binding 88), and the rays' dispatch arguments
    /// (89).
    m13_work: Option<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>,
    /// Every camera pass's dispatch in one compute pass (M13's colour pass,
    /// profile stage 20).
    one_pass: bool,
    /// The M13 colour pass's cache of the sun's visibility (binding 104),
    /// and whether it must be emptied before the next frame (a light moved).
    m13_sun: Option<wgpu::Buffer>,
    m13_sun_stale: bool,
    /// The large hits' records lit by the cache (M13 gate 12, binding 105).
    m13_hits: Option<wgpu::Buffer>,
    /// The records' count (M13 gate 13: the records themselves plain words).
    m13_hits_count: Option<wgpu::Buffer>,
    /// M13 gate 7: GPU timestamps around the page arguments' pass (0, 1)
    /// and the page bakes (2, 3) (a query set, its resolve buffer, a
    /// readback buffer, nanoseconds a tick), where the device has them.
    m13_timer: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, f32)>,
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
    /// The display's tone map and auto-exposure (M17), and when the last
    /// metered frame was drawn (for the adaptation's rate).
    tone_map: crate::ToneMap,
    auto_exposure: Option<crate::AutoExposure>,
    /// Bloom and the upscaler (M19).
    bloom: Option<crate::Bloom>,
    upscale: crate::Upscale,
    /// Depth of field (M20), and its passes once set.
    dof: Option<crate::DepthOfField>,
    dof_passes: Option<crate::dof::DofPasses>,
    /// Motion blur (M21), its passes once set, the previous camera, and
    /// each mover's previous placement (None: where it is).
    motion: Option<crate::MotionBlur>,
    motion_passes: Option<crate::motion::MotionPasses>,
    camera_before: Option<Camera>,
    movers_before: Vec<Option<crate::hierarchy::Placement>>,
    bloom_passes: crate::exposure::BloomPasses,
    meter: crate::exposure::Meter,
    metered_at: std::cell::Cell<Option<Instant>>,
    /// The sky's radiance the fog scatters (M18), and whether this frame
    /// traces the sun's visibility grid again.
    fog_sky: Vec3,
    fog_trace: bool,
    /// Lights moved so far (`set_light`).
    light_moves: u32,
    /// Time the caller waited for the GPU since the last frame
    /// ([`Self::wait`]): the cache's frame period counts it (M5 gate 14), as
    /// a frame call counts its own wait.
    waited: std::cell::Cell<Duration>,
    /// Textures box-filtered over the pixel's footprint (M7 gate 2; the
    /// default), or isotropic trilinear (as before).
    texture_box: bool,
    movers: Option<Movers>,
    /// Geometries whose vertices change (M12), and the pose each takes
    /// next frame.
    deformers: Vec<Deformer>,
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
    if let Some((view, sampler)) = textures.and_then(|t| t.coverage.as_ref())
        && keep.contains(&85)
    {
        entries.push(wgpu::BindGroupEntry {
            binding: 85,
            resource: wgpu::BindingResource::TextureView(view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 86,
            resource: wgpu::BindingResource::Sampler(sampler),
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
    if std::env::var_os("CONTINUUM_BIND_DEBUG").is_some() {
        eprintln!("bind group: keep {keep:?}, entries {}", entries.len());
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
        // The M13 colour pass's cached visibility of the lights is stale,
        // and the fog's (M18).
        self.m13_sun_stale = true;
        self.light_moves = self.light_moves.wrapping_add(1);
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

    /// Makes geometries `geometries` of `scene` (the scene this renderer was
    /// built from) deforming (M12): [`Self::set_vertices`] then poses them.
    /// Every instance of each must be movable.
    pub fn set_deforming(&mut self, scene: &InstancedScene, geometries: &[usize]) {
        let movers = self
            .movers
            .as_ref()
            .expect("deforming geometry needs movers");
        for &g in geometries {
            assert!(
                scene
                    .instances
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| i.geometry == g)
                    .all(|(k, _)| movers.index.contains_key(&k)),
                "every instance of a deforming geometry must be movable"
            );
            let [first, count, root, _] = movers.tree.geometry_ranges[g];
            let surface = &scene.geometries[g];
            assert_eq!(count as usize, surface.triangles.len());
            let (nodes, items, _) =
                crate::hierarchy::triangle_hierarchy(&surface.triangles, root, first);
            let tri_w = match scene.geometry_colors.get(g) {
                Some(Some(colors)) if colors.len() == surface.triangles.len() => colors
                    .iter()
                    .flat_map(|c| c.map(crate::color_code))
                    .collect(),
                _ => vec![0.0; 3 * surface.triangles.len()],
            };
            let mapped = scene
                .geometry_normal_maps
                .get(g)
                .copied()
                .flatten()
                .is_some()
                && surface.tex.len() == surface.triangles.len()
                && surface.normals.len() == surface.triangles.len();
            let tangents = mapped
                .then(|| crate::tangents::mikktspace(surface))
                .flatten();
            let normal_w = (0..3 * surface.triangles.len())
                .map(|i| {
                    tangents
                        .as_ref()
                        .map_or(0.0, |ts| crate::tangents::encode(ts[i / 3][i % 3]))
                })
                .collect();
            // Its objects take pages one mip coarser (fewer texels re-baked
            // each pose; the albedo's w, read by the cache's shading).
            for (k, i) in scene.instances.iter().enumerate() {
                if i.geometry == g {
                    let a = i.albedo;
                    let buffer = &self
                        .scene
                        .iter()
                        .find(|(b, _)| *b == 8)
                        .expect("albedo buffer")
                        .1;
                    self.queue.write_buffer(
                        buffer,
                        16 * k as u64,
                        bytemuck::cast_slice(&[a.x, a.y, a.z, DEFORMING_MIPS]),
                    );
                }
            }
            self.deformers.push(Deformer {
                geometry: g,
                first,
                root,
                nodes,
                items,
                tri_w,
                normal_w,
                pose: None,
            });
        }
    }

    /// Poses deforming geometry `geometry` (one of [`Self::set_deforming`])
    /// from the next frame: its triangles and each corner's normal, as many
    /// as the rest pose's and in its order (its charts and texture
    /// coordinates stay the rest pose's).
    pub fn set_vertices(&mut self, geometry: usize, triangles: Vec<Tri>, normals: Vec<[Vec3; 3]>) {
        let d = self
            .deformers
            .iter_mut()
            .find(|d| d.geometry == geometry)
            .unwrap_or_else(|| panic!("geometry {geometry} is not deforming"));
        assert_eq!(
            triangles.len() * 3,
            d.tri_w.len(),
            "a pose keeps the triangle count"
        );
        assert_eq!(normals.len(), triangles.len());
        d.pose = Some((triangles, normals));
    }

    /// Development (M12 gate 2): the slices a deforming light's catch-up
    /// takes now.
    #[doc(hidden)]
    pub fn deforming_slices(&self) -> u32 {
        self.movers.as_ref().map_or(0, |m| m.slices)
    }

    /// Diagnostics (M15): the last frame's anti-aliasing counters: edge
    /// pixels listed, samples per listed pixel, samples shaded (no pixel
    /// nearby saw their surface), samples traced, pixels all of whose
    /// samples saw its own surface, samples that did. Waits for the GPU; zeros
    /// without anti-aliasing.
    #[doc(hidden)]
    pub fn aa_counters(&self) -> [u32; 6] {
        let Some(cache) = &self.cache else {
            return [0; 6];
        };
        if !cache.state.anti_aliasing() {
            return [0; 6];
        }
        let words: Vec<u32> =
            bytemuck::cast_slice(&cache.state.read_buffer(&self.device, 110)).to_vec();
        [words[0], words[1], words[2], words[3], words[4], words[5]]
    }

    /// Diagnostics: the slots whose sky is still the guess
    /// (`VirtualCache::sky_guesses`).
    #[doc(hidden)]
    pub fn sky_guesses(&self) -> Vec<crate::SkyGuess> {
        self.cache
            .as_ref()
            .map_or(Vec::new(), |c| c.state.sky_guesses(&self.device))
    }

    /// Whether skies wait to re-bake (pages on moved objects wait
    /// `SKY_SETTLE` frames after their last move).
    pub(crate) fn skies_pending(&self) -> bool {
        self.cache.as_ref().is_some_and(|c| c.state.skies_pending())
    }

    /// The most frames any light's lighting near deforming geometry lags its
    /// pose (M12 gate 2; 0 with `CacheOptions::deforming_budget` 0 or
    /// nothing deforming).
    pub fn deforming_lag(&self) -> u32 {
        self.movers.as_ref().map_or(0, |m| {
            m.lag_frames
                .iter()
                .zip(&m.deforming)
                .filter(|(_, d)| **d)
                .flat_map(|(f, _)| f.iter().copied())
                .max()
                .unwrap_or(0)
        })
    }

    /// Applies the poses given since the last frame (M12): each deformed
    /// geometry's triangles, normals, refit hierarchy and hardware
    /// structure, its new box, and its instances marked moved.
    fn apply_poses(&mut self) {
        if self.deformers.iter().all(|d| d.pose.is_none()) {
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
        let node_size = std::mem::size_of::<NodeGpu>() as u64;
        for d in &mut self.deformers {
            let Some((tris, normals)) = d.pose.take() else {
                continue;
            };
            let words: Vec<[f32; 4]> = tris
                .iter()
                .flatten()
                .zip(&d.tri_w)
                .map(|(v, &w)| [v.x, v.y, v.z, w])
                .collect();
            self.queue
                .write_buffer(buffer(2), 48 * d.first as u64, bytemuck::cast_slice(&words));
            let words: Vec<[f32; 4]> = normals
                .iter()
                .flatten()
                .zip(&d.normal_w)
                .map(|(n, &w)| [n.x, n.y, n.z, w])
                .collect();
            self.queue.write_buffer(
                buffer(24),
                48 * d.first as u64,
                bytemuck::cast_slice(&words),
            );
            let cov = crate::hierarchy::refit_triangle_hierarchy(
                &tris,
                &mut d.nodes,
                &d.items,
                d.root,
                d.first,
            );
            self.queue.write_buffer(
                buffer(25),
                d.root as u64 * node_size,
                bytemuck::cast_slice(&d.nodes),
            );
            self.queue.write_buffer(
                buffer(29),
                4 * d.root as u64 * crate::hierarchy::COVERAGE_SAMPLES as u64,
                bytemuck::cast_slice(&cov),
            );
            if let Some(hw) = &mut self.hw {
                hw.rebuild_geometry(&self.device, &self.queue, d.geometry, &tris);
            }
            if let Some(movers) = &mut self.movers {
                movers.local[d.geometry] = mesh_bounds(&tris);
                for (k, p) in movers.placements.iter().enumerate() {
                    if p.geometry == d.geometry {
                        movers.dirty[k] = true;
                        movers.deforming[k] = true;
                        movers.poses[k] += 1;
                    }
                }
            }
        }
    }

    /// Applies the moves since the last frame: the movers' transforms, their
    /// subtree and the root, their hardware structure (in `encoder`), and
    /// the cache's list of moved objects for its stale pass.
    fn apply_moves(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(movers) = &mut self.movers else {
            return;
        };
        // Deforming objects whose lighting still lags a pose catch up with
        // it in turn after they stop (M12 gate 2).
        for k in 0..movers.placements.len() {
            if movers.deforming[k]
                && movers
                    .light_poses
                    .get(k)
                    .is_some_and(|f| f.iter().any(|&n| n < movers.poses[k]))
            {
                movers.dirty[k] = true;
            }
        }
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
        let lights = movers.light_boxes.len();
        let all = (1u32 << lights) - 1;
        // Deforming geometry's lighting in turn (M12 gate 2): fit the
        // slices a catch-up takes to the re-bakes' measured cost.
        let (budget, frame) = self.cache.as_ref().map_or((0, 0), |c| {
            (c.state.deforming_budget(), c.state.frame_index())
        });
        let most = (crate::vcache::DEFORMING_MAX_LAG / lights.max(1) as u32).max(1);
        if budget > 0
            && let Some(stats) = self.cache.as_ref().map(|c| c.state.stats())
            && let Some(&(_, slices)) = movers.updates.iter().find(|u| u.0 == stats.frame)
            && slices > 0
        {
            // A slice's re-bake times the slices of its catch-up: the whole
            // catch-up's (slices split the pages evenly), the largest seen
            // decaying slowly; the slices a catch-up needs to fit the budget.
            let whole = stats.rebaked_texels as f32 * slices as f32;
            movers.catchup_cost = whole.max(0.98 * movers.catchup_cost);
            movers.slices = ((movers.catchup_cost / budget as f32).ceil() as u32).clamp(1, most);
        }
        if movers.lag_boxes.len() != movers.placements.len() {
            movers.lag_boxes = movers.bounds.iter().map(|&b| vec![b; lights]).collect();
            movers.lag_frames = vec![vec![0; lights]; movers.placements.len()];
            movers.light_poses = vec![vec![0; lights]; movers.placements.len()];
        }
        let mut asked = 0;
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
            self.queue.write_buffer(
                buffer(30),
                object as u64 * 64,
                bytemuck::cast_slice(&p.rows()),
            );
            if !movers.deforming[k] || budget == 0 {
                movers.lag_boxes[k] = vec![new; lights];
                movers.lag_frames[k] = vec![0; lights];
                movers.light_poses[k] = vec![movers.poses[k]; lights];
                moved.push(crate::vcache::Moved {
                    object,
                    old,
                    new,
                    olds: Vec::new(),
                    mask: all,
                    own: all,
                    slice: None,
                });
                continue;
            }
            // One light at a time catches up, a slice of its shadows a
            // frame; the lights take turns.
            if movers.catching[k].is_none() {
                let start = movers.next_light[k];
                for step in 0..lights {
                    let l = (start + step) % lights;
                    if movers.light_poses[k][l] < movers.poses[k] {
                        movers.catching[k] = Some(CatchUp {
                            light: l,
                            slice: 0,
                            slices: movers.slices,
                            pose: movers.poses[k],
                            union: new,
                        });
                        movers.next_light[k] = (l + 1) % lights;
                        break;
                    }
                }
            }
            for l in 0..lights {
                if movers.light_poses[k][l] < movers.poses[k] {
                    movers.lag_frames[k][l] += 1;
                }
            }
            let Some(mut c) = movers.catching[k] else {
                continue;
            };
            let l = c.light;
            c.union = (c.union.0.min(new.0), c.union.1.max(new.1));
            moved.push(crate::vcache::Moved {
                object,
                old,
                new,
                olds: movers.lag_boxes[k].clone(),
                mask: 1 << l,
                // Its own texels lie in its box, so in its shadows' region:
                // they re-bake slice by slice, texel by texel, not as whole
                // pages at once.
                own: 0,
                slice: Some((c.slice, c.slices)),
            });
            asked = asked.max(c.slices);
            c.slice += 1;
            if c.slice >= c.slices {
                // Its texels were baked against the object anywhere in the
                // union: the next catch-up starts from there.
                movers.lag_boxes[k][l] = c.union;
                movers.light_poses[k][l] = c.pose;
                movers.lag_frames[k][l] = 0;
                movers.catching[k] = None;
            } else {
                movers.catching[k] = Some(c);
            }
        }
        movers.updates[(frame % 8) as usize] = (frame, asked);
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
            // Rebuilt in place: not while the frame before may still trace
            // it (pipelined frames lost an animated fox from camera rays
            // in the viewer).
            if let Some(index) = self.last_submission.take() {
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                });
            }
            hw.update_movers(encoder, &movers.placements);
        }
        if let Some(cache) = &mut self.cache {
            // Every texel's apex lies within the objects' boxes; the
            // movers' old boxes stay in for this frame's test.
            let (mut lo, mut hi) = top.bounds();
            for m in &moved {
                (lo, hi) = (lo.min(m.old.0), hi.max(m.old.1));
                for o in &m.olds {
                    (lo, hi) = (lo.min(o.0), hi.max(o.1));
                }
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
        // Alpha blending and transmission need the cache's extra-samples
        // pass: hardware camera rays, no movers (M10 gate 16, M16);
        // otherwise their fallback.
        let blending = cache.is_some()
            && kernel.uses_hardware_rays()
            && !kernel.pixel_cones.get()
            && movable.is_empty();
        let fallback;
        let scene: &InstancedScene = if !blending && scene.sees_through() {
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
                        clear: scene.instance_clear(i),
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
                deforming: vec![false; tree.objects.len()],
                lag_boxes: Vec::new(),
                lag_frames: Vec::new(),
                poses: vec![0; tree.objects.len()],
                light_poses: Vec::new(),
                catching: vec![None; tree.objects.len()],
                next_light: vec![0; tree.objects.len()],
                slices: 1,
                catchup_cost: 0.0,
                updates: [(u32::MAX, 0); 8],
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
        // The auto-exposure meter (M17) and bloom (M19), bound with the
        // colours below.
        let meter = crate::exposure::Meter::new(&device);
        let bloom_passes = crate::exposure::BloomPasses::new(&device, (width, height));
        let pixel_cones = kernel.pixel_cones.get();
        // Pixel cones need no camera rays, so no acceleration structure.
        // M13 gate 4 (profile stage 16) traces corner rays in hardware.
        let hw = (kernel.uses_hardware_rays()
            && (!pixel_cones || matches!(kernel.profile_stage.get(), 16..=20)))
        .then(|| {
            // Top-level slots for objects showing finer levels than
            // their proxies (M14 gate 6).
            let dynamic = stream.map_or(0, |o| crate::stream::detailed_capacity(scene, &o));
            kernel.hw_scene(scene, origin, movable, dynamic)
        });
        // Bounce samples trace through the camera's structures where there
        // are some (M14 gate 10).
        let bounce_hw = hw.is_some() && !pixel_cones;
        // Sky samples through the same structures (M5 gate 13), where they
        // hold the lighting's geometry: with movers through theirs too; not
        // with streamed levels (camera levels differ from lighting levels)
        // or alpha-blended objects (cones weigh their opacity).
        let sky_rays = bounce_hw
            && cache.is_some_and(|o| o.sky_rays)
            && stream.is_none()
            && !scene.sees_through();
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
        // Layers behind surfaces camera rays see through: alpha blending
        // (M10 gate 16) and transmission (M16).
        let blend = scene.sees_through();
        let hw_edges = cache.as_ref().is_some_and(|o| o.edge_rays || blend)
            && hw.as_ref().is_some_and(|hw| hw.movers.is_none());
        let shade_bindings = |list: &[u32]| {
            let mut keep = list.to_vec();
            keep.push(31);
            keep
        };
        // The cache's shading with its reflection rays through ray queries
        // (not with movers, whose structure they do not trace).
        let hw_shade = hw.as_ref().is_some_and(|hw| hw.movers.is_none())
            && cache.as_ref().is_none_or(|o| o.hardware_shading);
        let shading = if hw_shade {
            (kernel.entry("hw_vc_shade"), HW_VC_SHADE_BINDINGS.to_vec())
        } else {
            (kernel.entry("vc_shade"), VC_SHADE_BINDINGS.to_vec())
        };
        let shading_edges = if hw_shade {
            (
                kernel.entry("hw_vc_shade_edges"),
                HW_VC_SHADE_EDGES_BINDINGS.to_vec(),
            )
        } else {
            (
                kernel.entry("vc_shade_edges"),
                VC_SHADE_EDGES_BINDINGS.to_vec(),
            )
        };
        let is_shade = |b: &[u32]| b == VC_SHADE_BINDINGS || b == HW_VC_SHADE_BINDINGS;
        let is_shade_edges =
            |b: &[u32]| b == VC_SHADE_EDGES_BINDINGS || b == HW_VC_SHADE_EDGES_BINDINGS;
        let mut passes = match (cache.is_some(), visibility) {
            // M13: the scalar coverage walk in its own entry point (profile
            // stage 14), so it does not run at the pixel-cone entry's
            // occupancy.
            (false, _) if pixel_cones && matches!(kernel.profile_stage.get(), 17 | 18 | 20) => {
                vec![
                    (kernel.entry("hw_m13_prep"), M13_PREP_BINDINGS.to_vec()),
                    (
                        kernel.entry("hw_m13_tile_roots"),
                        M13_ROOTS_BINDINGS.to_vec(),
                    ),
                    // After the roots: corners only in tiles that may hold a
                    // large instance (M13 gate 8).
                    (
                        kernel.entry("hw_m13_corners_tiles"),
                        M13_CORNERS_TILES_BINDINGS.to_vec(),
                    ),
                    (kernel.entry("hw_m13_bins"), M13_BINS_BINDINGS.to_vec()),
                    (
                        kernel.entry("hw_m13_bin_args"),
                        M13_BIN_ARGS_BINDINGS.to_vec(),
                    ),
                    (
                        kernel.entry("hw_m13_bin_rays"),
                        M13_BIN_RAYS_BINDINGS.to_vec(),
                    ),
                    (
                        kernel.entry("hw_m13_shade_args"),
                        M13_SHADE_ARGS_BINDINGS.to_vec(),
                    ),
                    (
                        kernel.entry("hw_m13_shade_rays"),
                        M13_SHADE_RAYS_BINDINGS.to_vec(),
                    ),
                    // M13 appearance, third (development, off): backgrounds'
                    // bins from the sun's cache where their corners disagree
                    // (`hw_m13_shade_background`, `hw_m13_miss_args`,
                    // `hw_m13_shade_misses`; G10_BG_SHADOWS in the shader).
                    (kernel.entry("hw_m13_tiles"), M13_TILES_BINDINGS.to_vec()),
                    (
                        kernel.entry("hw_m13_page_args"),
                        M13_PAGE_ARGS_BINDINGS.to_vec(),
                    ),
                    (
                        kernel.entry("hw_m13_page_bake"),
                        M13_PAGE_BAKE_BINDINGS.to_vec(),
                    ),
                ]
            }
            (false, _) if pixel_cones && kernel.profile_stage.get() == 19 => vec![(
                kernel.entry("hw_m13_reference"),
                M13_REFERENCE_BINDINGS.to_vec(),
            )],
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
            // M13 gate 12 (development): the sub-pixel pass lit by the cache:
            // the passes that shade from the variant module, every binding
            // list from naga's analysis.
            (true, _) if pixel_cones && kernel.profile_stage.get() == 20 => {
                // (Every pass that reads the frames' stamp from the variant,
                // which mirrors it.)
                let plain =
                    |name: &'static str| (kernel.entry(name), kernel.entry_bindings(name, false));
                let lit =
                    |name: &'static str| (kernel.entry_vc(name), kernel.entry_bindings(name, true));
                vec![
                    lit("hw_m13_prep"),
                    lit("hw_m13_tile_roots"),
                    lit("hw_m13_corners_tiles"),
                    lit("hw_m13_hits_args_a"),
                    lit("hw_m13_shade_hits_a"),
                    lit("hw_m13_bins"),
                    plain("hw_m13_bin_args"),
                    lit("hw_m13_bin_rays"),
                    plain("hw_m13_shade_args"),
                    lit("hw_m13_shade_rays"),
                    lit("hw_m13_hits_args_b"),
                    lit("hw_m13_shade_hits_b"),
                    lit("hw_m13_scatter_hits_b"),
                    lit("hw_m13_tiles"),
                    lit("hw_m13_page_args"),
                    lit("hw_m13_page_bake"),
                ]
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
                if blend {
                    // The layers behind see-through hits, for the records
                    // listed so far (M16 gate 3).
                    list.push((
                        kernel.entry("vc_args_edges"),
                        VC_ARGS_EDGES_BINDINGS.to_vec(),
                    ));
                    list.push((kernel.entry("hw_layers"), HW_LAYERS_BINDINGS.to_vec()));
                }
                list.push(shading.clone());
                list.push((
                    kernel.entry("vc_args_edges"),
                    VC_ARGS_EDGES_BINDINGS.to_vec(),
                ));
                list.push(shading_edges.clone());
                list
            }
            (true, v) => vec![
                v.unwrap_or_else(|| {
                    (
                        kernel.entry("visibility"),
                        COMPUTE_VISIBILITY_BINDINGS.to_vec(),
                    )
                }),
                shading.clone(),
            ],
        };
        // The error while flying, reported (M4 gate 13): probe pixels after
        // shading, through ray queries (not with movers, whose own
        // structure the sky's rays do not trace).
        if cache.as_ref().is_some_and(|o| o.error_probes > 0)
            && !pixel_cones
            && hw.as_ref().is_some_and(|hw| hw.movers.is_none())
        {
            passes.push((
                kernel.entry("hw_vc_error_shares"),
                HW_VC_ERROR_SHARES_BINDINGS.to_vec(),
            ));
            passes.push((
                kernel.entry("vc_error_probes"),
                VC_ERROR_PROBES_BINDINGS.to_vec(),
            ));
        }
        let m13_lit = cache.is_some() && pixel_cones && kernel.profile_stage.get() == 20;
        let mut camera_dispatch: Vec<CameraDispatch> = if m13_lit {
            // Gate 12's passes in order (prep, roots, corners, the first
            // records' arguments and lighting, bins, bin arguments, bin
            // rays, shading arguments, shading, the other records' arguments,
            // lighting and scatter, pixels, page arguments, bakes).
            let mut d = vec![
                CameraDispatch::Prep,
                CameraDispatch::Tiles,
                CameraDispatch::Screen,
                CameraDispatch::One,
                CameraDispatch::M13Args(16),
                CameraDispatch::Screen,
                CameraDispatch::One,
                CameraDispatch::M13Rays,
                CameraDispatch::One,
                CameraDispatch::M13Rays,
                CameraDispatch::One,
                CameraDispatch::M13Args(32),
                CameraDispatch::M13Args(32),
                CameraDispatch::Screen,
                CameraDispatch::One,
                CameraDispatch::M13Pages,
            ];
            d.resize(passes.len(), CameraDispatch::Screen);
            d
        } else {
            passes
                .iter()
                .map(|(_, bindings)| match bindings.as_slice() {
                    b if b == HW_VC_ERROR_SHARES_BINDINGS => CameraDispatch::ProbeGroups,
                    b if b == VC_ERROR_PROBES_BINDINGS => CameraDispatch::Probes,
                    b if b == VC_ARGS_EDGES_BINDINGS => CameraDispatch::One,
                    b if is_shade_edges(b) => CameraDispatch::Listed,
                    b if b == HW_LAYERS_BINDINGS => CameraDispatch::Listed,
                    b if b == M13_ROOTS_BINDINGS => CameraDispatch::Tiles,
                    b if b == M13_BIN_ARGS_BINDINGS => CameraDispatch::One,
                    b if b == M13_BIN_RAYS_BINDINGS => CameraDispatch::M13Rays,
                    b if b == M13_SHADE_ARGS_BINDINGS => CameraDispatch::One,
                    b if b == M13_SHADE_RAYS_BINDINGS => CameraDispatch::M13Rays,
                    b if b == M13_SHADE_BACKGROUND_BINDINGS => CameraDispatch::M13Args(16),
                    b if b == M13_MISS_ARGS_BINDINGS => CameraDispatch::One,
                    b if b == M13_SHADE_MISSES_BINDINGS => CameraDispatch::M13Args(32),
                    b if b == M13_PAGE_ARGS_BINDINGS => CameraDispatch::One,
                    b if b == M13_PAGE_BAKE_BINDINGS => CameraDispatch::M13Pages,
                    b if b == M13_PREP_BINDINGS => CameraDispatch::Prep,
                    _ => CameraDispatch::Screen,
                })
                .collect()
        };
        // Anti-aliasing on the default path (M15), with hardware camera
        // rays: the edge list before shading, the
        // samples after it (and after any layers behind blended surfaces).
        let aa = cache.as_ref().is_some_and(|o| o.anti_aliasing)
            && hw.is_some()
            && !pixel_cones
            && passes.iter().any(|(_, keep)| is_shade(keep));
        if aa {
            let pass =
                |name: &'static str| (kernel.entry(name), kernel.entry_bindings(name, false));
            let shade = passes
                .iter()
                .position(|(_, keep)| is_shade(keep))
                .expect("a shading pass");
            passes.insert(shade, pass("hw_aa_list"));
            camera_dispatch.insert(shade, CameraDispatch::Screen);
            let after = passes
                .iter()
                .rposition(|(_, keep)| is_shade(keep) || is_shade_edges(keep))
                .expect("a shading pass")
                + 1;
            let movers = hw.as_ref().is_some_and(|hw| hw.movers.is_some());
            for (k, (name, dispatch)) in [
                ("hw_aa_args", CameraDispatch::One),
                (
                    if movers { "hw_aa_movers" } else { "hw_aa" },
                    CameraDispatch::Aa,
                ),
                ("hw_aa_args_again", CameraDispatch::One),
                (
                    if movers {
                        "hw_aa_again_movers"
                    } else {
                        "hw_aa_again"
                    },
                    CameraDispatch::AaAgain,
                ),
                ("hw_aa_args_shade", CameraDispatch::One),
                ("hw_aa_retrace", CameraDispatch::AaShade),
                (
                    if hw_shade {
                        "hw_vc_aa_shade"
                    } else {
                        "vc_aa_shade"
                    },
                    CameraDispatch::AaShade,
                ),
                ("hw_aa_scatter", CameraDispatch::AaScatter),
            ]
            .into_iter()
            .enumerate()
            {
                passes.insert(after + k, pass(name));
                camera_dispatch.insert(after + k, dispatch);
            }
        }
        // Fog (M18), with hardware camera rays and no movers: after shading
        // and anti-aliasing (before the error probes).
        let fog = cache.as_ref().is_some_and(|o| o.fog.is_some())
            && hw.as_ref().is_some_and(|hw| hw.movers.is_none())
            && !pixel_cones
            && passes.iter().any(|(_, keep)| is_shade(keep));
        if fog {
            let at = passes
                .iter()
                .position(|(_, keep)| keep.as_slice() == HW_VC_ERROR_SHARES_BINDINGS)
                .unwrap_or(passes.len());
            for (k, (name, dispatch)) in [
                ("hw_fog_visibility", CameraDispatch::FogGrid),
                ("hw_fog_froxels", CameraDispatch::FogColumns),
                ("hw_fog_apply", CameraDispatch::Screen),
            ]
            .into_iter()
            .enumerate()
            {
                passes.insert(
                    at + k,
                    (kernel.entry(name), kernel.entry_bindings(name, false)),
                );
                camera_dispatch.insert(at + k, dispatch);
            }
        }
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
                        kernel.entry("vc_bounce"),
                    ],
                    Vec::new(),
                )
            }),
            has_movers: !movable.is_empty(),
            moving_lights: !moving_lights.is_empty(),
            // Bounce rays through hardware ray queries where camera rays
            // use them (M14 gate 10).
            indirect: options.indirect.then(|| {
                let name = if bounce_hw && !movable.is_empty() {
                    "hw_vc_bake_indirect_movers"
                } else if bounce_hw {
                    "hw_vc_bake_indirect"
                } else {
                    "vc_bake_indirect"
                };
                (kernel.entry(name), None)
            }),
            indirect_hw: bounce_hw,
            sky_hw: sky_rays,
            // Rays to wide lights where sky samples may use rays (the same
            // structures hold the lighting's geometry).
            wide: (bounce_hw
                && cache.is_some_and(|o| o.wide_rays)
                && movable.is_empty()
                && stream.is_none()
                && !scene.sees_through())
            .then(|| (kernel.entry("hw_vc_bake_wide"), None)),
            // Coarse texels averaging their area where sky samples use rays.
            area: (sky_rays && cache.is_some_and(|o| o.area_texels) && movable.is_empty())
                .then(|| (kernel.entry("hw_vc_bake_area"), None)),
            passes: [
                kernel.entry("vc_update"),
                kernel.entry("vc_bake"),
                kernel.entry(if sky_rays && !movable.is_empty() {
                    "hw_vc_bake_sky_movers"
                } else if sky_rays {
                    "hw_vc_bake_sky"
                } else {
                    "vc_bake_sky"
                }),
                kernel.entry("vc_fill"),
            ],
            binds: Vec::new(),
        });
        let mut cache = cache;
        if let Some(c) = &mut cache {
            c.state.set_sky_mean(kernel.sky());
        }
        if !fog && let Some(c) = &mut cache {
            // Without the passes, no fog in shading either.
            c.state.drop_fog();
        }
        if hw_edges && let Some(c) = &mut cache {
            c.state.set_hardware_edges(blend);
        }
        if aa && let Some(c) = &mut cache {
            c.state.set_anti_aliasing();
        }
        if let Some(c) = &mut cache
            && c.sky_hw
        {
            c.state.set_sky_rays();
        }
        if let Some(c) = &mut cache
            && c.area.is_some()
        {
            c.state.set_area_rays();
        }
        let colors = Self::colors_buffer(&device, width, height);
        let mut renderer = InstancedRenderer {
            device: device.clone(),
            queue: kernel.queue.clone(),
            last_submission: None,
            passes,
            camera_dispatch,
            hw,
            display_bind: display_bind(
                &device,
                &display,
                &colors,
                &dims_buf,
                &meter.state,
                &bloom_passes.out,
            ),
            mark: kernel.entry("mark"),
            display,
            scene: scene_buffers,
            camera_buf,
            dims_buf,
            binds: Vec::new(),
            colors,
            m13_work: None,
            m13_timer: None,
            // (`CONTINUUM_M13_PASSES` in development: a compute pass a
            // dispatch, for per-pass times in a GPU trace.)
            one_pass: pixel_cones
                && kernel.profile_stage.get() == 20
                && std::env::var_os("CONTINUUM_M13_PASSES").is_none(),
            m13_sun: None,
            m13_sun_stale: false,
            m13_hits: None,
            m13_hits_count: None,
            cache,
            width,
            height,
            scale,
            // Shading reads the lights' buffer; the camera's radiance is unused.
            radiance: lights[0].radiance.x,
            lights: lights.len(),
            textures: {
                // The environment's image after the scene's (M5 gate 12).
                let (all, layer) = kernel.textures_with_environment(&scene.textures);
                let mut t =
                    kernel.scene_textures_with_environment(&all, &scene.linear_textures(), layer);
                t.coverage = kernel.take_coverage_texture();
                t
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
            tone_map: crate::ToneMap::Reinhard,
            auto_exposure: None,
            bloom: None,
            upscale: crate::Upscale::Nearest,
            dof: None,
            dof_passes: None,
            motion: None,
            motion_passes: None,
            camera_before: None,
            movers_before: Vec::new(),
            bloom_passes,
            meter,
            metered_at: std::cell::Cell::new(None),
            fog_sky: kernel.sky(),
            fog_trace: true,
            light_moves: 0,
            waited: std::cell::Cell::new(Duration::ZERO),
            texture_box: true,
            movers,
            deformers: Vec::new(),
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

    /// M13 gate 6 (profile stage 18): coverage pages baked a frame at most.
    #[doc(hidden)]
    pub fn set_m13_page_budget(&self, pages: u32) {
        if let Some((_, state)) = self.scene.iter().find(|(b, _)| *b == 92) {
            self.queue.write_buffer(state, 16, &pages.to_le_bytes());
        }
    }

    /// M13 (development): sets word `index` of the coverage pages' state
    /// (see `g6_state` in `hw_visibility.wgsl`; 28: the pixels' pass writes
    /// each pixel's class instead of its colour).
    #[doc(hidden)]
    pub fn set_m13_state_word(&self, index: u32, value: u32) {
        if let Some((_, state)) = self.scene.iter().find(|(b, _)| *b == 92) {
            self.queue
                .write_buffer(state, 4 * index as u64, &value.to_le_bytes());
        }
    }

    /// M13 gate 7: the GPU time of the last frame's page passes (arguments
    /// and bakes, each timed on its own) in milliseconds, after the frame is
    /// waited for; None without timestamps.
    #[doc(hidden)]
    pub fn m13_page_ms(&self) -> Option<f64> {
        let (_, _, readback, period) = self.m13_timer.as_ref()?;
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let ticks: [u64; 4] = {
            let data = slice.get_mapped_range().expect("mapped timestamps");
            let words: &[u64] = bytemuck::cast_slice(&data);
            [words[0], words[1], words[2], words[3]]
        };
        readback.unmap();
        let span = |a: u64, b: u64| b.saturating_sub(a) as f64 * *period as f64 / 1e6;
        Some(span(ticks[0], ticks[1]) + span(ticks[2], ticks[3]))
    }

    #[doc(hidden)]
    /// The M13 colour pass's work counters (development): word 0 the pixels
    /// tracing bin rays, 1 the sun cache's misses, 2 the backgrounds shaded
    /// bin by bin, this frame.
    pub fn m13_work_counters(&self) -> [u32; 8] {
        let Some((work, _, _, _)) = &self.m13_work else {
            return [0; 8];
        };
        self.read_head(work)
    }

    #[doc(hidden)]
    /// The large hits' records lit by the cache in the last frame (M13 gate
    /// 12, development): word 0 all of them, 1 those of the first lighting
    /// pass (the corners').
    pub fn m13_hit_counts(&self) -> [u32; 8] {
        let Some(hits) = &self.m13_hits else {
            return [0; 8];
        };
        self.read_head(hits)
    }

    #[doc(hidden)]
    /// The large hits' records of the last frame (M13 gate 13, development):
    /// five words each after the header (see `g12_hits` in `hw_vc.wgsl`).
    pub fn m13_hit_records(&self) -> Vec<u32> {
        let Some(hits) = &self.m13_hits else {
            return Vec::new();
        };
        let n = (self.read_head(hits)[0] as u64).min(3_000_000);
        let size = (8 + 5 * n) * 4;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit records"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(hits, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let words: Vec<u32> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped records")).to_vec();
        words[8..].to_vec()
    }

    /// The first eight words of `buffer` (development counters).
    fn read_head(&self, buffer: &wgpu::Buffer) -> [u32; 8] {
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("work counters"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, 32);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let words: Vec<u32> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped counters")).to_vec();
        let mut out = [0; 8];
        out.copy_from_slice(&words);
        out
    }

    /// M13 gate 6: the coverage pages' counters after the last frame (see
    /// `g6_state` in `hw_visibility.wgsl`: 2 bakes, 6 requests, 8 pages
    /// used, 10 baked in all, 11 nodes fitting no page, 12 page roots, 14
    /// the largest shift read in 1/10,000 bin, 26 refreshes baked).
    #[doc(hidden)]
    pub fn m13_page_counters(&self) -> [u32; 32] {
        let Some((_, state)) = self.scene.iter().find(|(b, _)| *b == 92) else {
            return [0; 32];
        };
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("page counters"),
            size: 128,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(state, 0, &readback, 0, 128);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let words: Vec<u32> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped counters")).to_vec();
        let mut out = [0; 32];
        out.copy_from_slice(&words);
        out
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
        // M13 gate 5: each 8×8 tile's subtree roots (a count, then up to 63,
        // each with its projected box).
        let roots_size = if self.passes.iter().any(|(_, keep)| keep.contains(&87)) {
            width.div_ceil(8) as u64 * height.div_ceil(8) as u64 * 416 * 4
        } else {
            16
        };
        let roots = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile roots"),
            size: roots_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        all.push((87, &roots));
        // M13 gates 5–6: a count, then the pixels tracing bin rays and every
        // pixel's covered bins; the rays' dispatch arguments apart (a buffer
        // cannot be read for a dispatch and bound for storage in it).
        self.m13_work = self
            .passes
            .iter()
            .any(|(_, keep)| keep.contains(&88))
            .then(|| {
                let buffer = |label, size, usage| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size,
                        usage: wgpu::BufferUsages::STORAGE | usage,
                        mapped_at_creation: false,
                    })
                };
                let pixels = width as u64 * height as u64;
                let colour = self.passes.iter().any(|(_, keep)| keep.contains(&102));
                (
                    // The header, the list, the masks, and (M13 gates 9–10)
                    // three words a pixel that bin rays claim.
                    buffer(
                        "m13 work",
                        (8 + 5 * pixels) * 4,
                        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    ),
                    buffer("m13 rays args", 48, wgpu::BufferUsages::INDIRECT),
                    // Colour (M13 gates 9–10): per pixel 20 words, per corner
                    // 8, each field an array.
                    buffer(
                        "m13 colour",
                        if colour { (28 * pixels + 16) * 4 } else { 16 },
                        wgpu::BufferUsages::empty(),
                    ),
                    // The shaded bins' list (M13 gate 10): 20 words an entry, then
                    // the list of entries the shading pass shades.
                    buffer(
                        "m13 shaded bins",
                        if colour {
                            (22 * 524_288 + 6 * 1_048_576) * 4
                        } else {
                            16
                        },
                        wgpu::BufferUsages::empty(),
                    ),
                )
            });
        // The sun's cache (M13 appearance, third): 2^20 buckets of 16 words,
        // then a word per object (the albedo buffer's entries).
        let sun_cache = self
            .passes
            .iter()
            .any(|(_, keep)| keep.contains(&104))
            .then(|| {
                let objects = self
                    .scene
                    .iter()
                    .find(|(b, _)| *b == 8)
                    .map_or(0, |(_, buf)| buf.size() / 16);
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("m13 sun cache"),
                    size: (16 * 1_048_576 + objects + 16) * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
        if let Some(buf) = &sun_cache {
            all.push((104, buf));
        }
        self.m13_sun = sun_cache.clone();
        // The large hits' records lit by the cache (M13 gate 12): a header
        // of eight words, then five words a record.
        self.m13_hits = self
            .passes
            .iter()
            .any(|(_, keep)| keep.contains(&105))
            .then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("m13 hits"),
                    size: (8 + 5 * 3_000_000) * 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            });
        if let Some(buf) = &self.m13_hits {
            all.push((105, buf));
        }
        self.m13_hits_count = self
            .passes
            .iter()
            .any(|(_, keep)| keep.contains(&106))
            .then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("m13 hits count"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
        if let Some(buf) = &self.m13_hits_count {
            all.push((106, buf));
        }
        if let Some((work, args, plain, shaded)) = &self.m13_work {
            all.push((88, work));
            all.push((89, args));
            all.push((102, plain));
            all.push((103, shaded));
        }
        self.m13_timer = (self
            .passes
            .iter()
            .any(|(_, keep)| keep.as_slice() == M13_PAGE_BAKE_BINDINGS)
            && device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
        .then(|| {
            let buffer = |label, usage| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 32,
                    usage,
                    mapped_at_creation: false,
                })
            };
            (
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("page timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 4,
                }),
                buffer(
                    "page timestamps resolve",
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                ),
                buffer(
                    "page timestamps readback",
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                ),
                self.queue.get_timestamp_period(),
            )
        });
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
            tlas.push((101, &hw.background));
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
                    if cache.sky_hw && cache.has_movers {
                        HW_VC_BAKE_SKY_MOVERS_BINDINGS
                    } else if cache.sky_hw {
                        HW_VC_BAKE_SKY_BINDINGS
                    } else {
                        VC_BAKE_SKY_BINDINGS
                    },
                    VC_FILL_BINDINGS,
                ])
                .enumerate()
                .map(|(k, (pipeline, keep))| {
                    let tl: &[_] = if k == 2 && cache.sky_hw { &tlas } else { &[] };
                    bind_group(device, pipeline, &all, tl, None, keep)
                })
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
                        VC_BOUNCE_BINDINGS,
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
                        if cache.has_movers {
                            HW_VC_BAKE_INDIRECT_MOVERS_BINDINGS
                        } else {
                            HW_VC_BAKE_INDIRECT_BINDINGS
                        },
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
        let wide_bind = self.cache.as_ref().and_then(|cache| {
            cache.wide.as_ref().map(|(pipeline, _)| {
                bind_group(
                    device,
                    pipeline,
                    &all,
                    &tlas,
                    None,
                    HW_VC_BAKE_WIDE_BINDINGS,
                )
            })
        });
        let area_bind = self.cache.as_ref().and_then(|cache| {
            cache.area.as_ref().map(|(pipeline, _)| {
                bind_group(
                    device,
                    pipeline,
                    &all,
                    &tlas,
                    None,
                    HW_VC_BAKE_WIDE_BINDINGS,
                )
            })
        });
        if let (Some(cache), Some(bind)) = (&mut self.cache, bounce_bind)
            && let Some((_, b)) = &mut cache.indirect
        {
            *b = Some(bind);
        }
        if let (Some(cache), Some(bind)) = (&mut self.cache, wide_bind)
            && let Some((_, b)) = &mut cache.wide
        {
            *b = Some(bind);
        }
        if let (Some(cache), Some(bind)) = (&mut self.cache, area_bind)
            && let Some((_, b)) = &mut cache.area
        {
            *b = Some(bind);
        }
        if let (Some(cache), Some((binds, movers))) = (&mut self.cache, cache_binds) {
            cache.binds = binds;
            if let (Some((_, b)), Some(movers)) = (&mut cache.movers, movers) {
                *b = movers;
            }
        }
        self.bloom_passes = crate::exposure::BloomPasses::new(device, (width, height));
        self.bloom_passes.bind(device, &self.colors);
        self.display_bind = display_bind(
            device,
            &self.display,
            &self.colors,
            &self.dims_buf,
            &self.meter.state,
            &self.bloom_passes.out,
        );
        // Each pixel's hit, where the cache's camera passes write it.
        self.meter
            .bind(device, &self.colors, self.cache.is_some().then_some(&vis));
        self.vis = Some(vis);
        self.dof_passes = None;
        self.make_dof();
        self.motion_passes = None;
        self.make_motion();
    }

    /// Makes and binds motion blur's passes if it is on and they do not fit
    /// the image (M21).
    fn make_motion(&mut self) {
        let (Some(_), Some(vis)) = (self.motion, &self.vis) else {
            return;
        };
        let size = (self.width, self.height);
        if self.motion_passes.as_ref().is_some_and(|p| p.fits(size)) {
            return;
        }
        let find = |b: u32| self.scene.iter().find(|(x, _)| *x == b).map(|(_, buf)| buf);
        let (Some(meshes), Some(xforms)) = (find(3), find(30)) else {
            return;
        };
        let mut mover_of = vec![0u32; (xforms.size() / 64) as usize];
        let mut movers = 0;
        if let Some(m) = &self.movers {
            movers = m.tree.objects.len();
            for (k, &x) in m.tree.objects.iter().enumerate() {
                if let Some(slot) = mover_of.get_mut(x as usize) {
                    *slot = k as u32 + 1;
                }
            }
        }
        let mut passes = crate::motion::MotionPasses::new(&self.device, size, &mover_of, movers);
        passes.bind(
            &self.device,
            &self.camera_buf,
            &self.colors,
            vis,
            meshes,
            xforms,
        );
        self.motion_passes = Some(passes);
    }

    /// Records motion blur's passes (M21), if on, after the camera's (and
    /// depth of field's): the previous camera and movers' placements first.
    fn encode_motion(&self, encoder: &mut wgpu::CommandEncoder, camera: &Camera) {
        let (Some(blur), Some(passes)) = (&self.motion, &self.motion_passes) else {
            return;
        };
        let before = camera_uniform(
            self.camera_before.unwrap_or(*camera),
            self.width,
            self.height,
            self.radiance,
        );
        self.queue
            .write_buffer(&passes.before, 0, bytemuck::bytes_of(&before));
        let mut moved = false;
        if let Some(m) = &self.movers
            && !m.placements.is_empty()
        {
            let rows: Vec<[[f32; 4]; 4]> = m
                .placements
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    let b = self.movers_before.get(k).copied().flatten().unwrap_or(*p);
                    moved |= b.rotation != p.rotation || b.translation != p.translation;
                    b.rows()
                })
                .collect();
            self.queue
                .write_buffer(&passes.moved, 0, bytemuck::cast_slice(&rows));
        }
        passes.encode(&self.queue, encoder, blur, moved);
    }

    /// Makes and binds depth of field's passes if it is on and they do not
    /// fit the image (M20).
    fn make_dof(&mut self) {
        let (Some(lens), Some(vis)) = (self.dof, &self.vis) else {
            return;
        };
        let size = (self.width, self.height);
        let max_px = lens.max_blur * self.height as f32;
        if self
            .dof_passes
            .as_ref()
            .is_some_and(|p| p.fits(size, max_px))
        {
            return;
        }
        let mut passes = crate::dof::DofPasses::new(&self.device, size, max_px);
        let lights = self.scene.iter().find(|(b, _)| *b == 52).map(|(_, b)| b);
        passes.bind(&self.device, &self.camera_buf, &self.colors, vis, lights);
        self.dof_passes = Some(passes);
    }

    /// Records depth of field's passes (M20), if on, after the camera's.
    fn encode_dof(&self, encoder: &mut wgpu::CommandEncoder, camera: &Camera) {
        if let (Some(lens), Some(passes)) = (&self.dof, &self.dof_passes) {
            passes.encode(&self.queue, encoder, camera.vertical_fov_degrees, lens);
        }
    }

    /// Camera pass `k` (of `passes`), timed on the GPU where the cache has
    /// timestamps: the first pass writes query 2 as it begins.
    fn encode_camera_pass(&self, encoder: &mut wgpu::CommandEncoder, k: usize) {
        let set = self
            .cache
            .as_ref()
            .and_then(|c| c.state.timer.as_ref())
            .map(|t| &t.set);
        let (_, keep) = &self.passes[k];
        // M13 gate 7: each page pass's span (queries 0–1 the arguments, 2–3
        // the bakes).
        let page_writes = self.m13_timer.as_ref().and_then(|(query_set, ..)| {
            let first = if keep.as_slice() == M13_PAGE_ARGS_BINDINGS {
                0
            } else if keep.as_slice() == M13_PAGE_BAKE_BINDINGS {
                2
            } else {
                return None;
            };
            Some(wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(first),
                end_of_pass_write_index: Some(first + 1),
            })
        });
        // Only the first writes (query 2); the display pass ends the span.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: page_writes.or(set.filter(|_| k == 0).map(|query_set| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: None,
                }
            })),
        });
        self.dispatch_camera(&mut pass, k);
    }

    /// Pass `k`'s pipeline, bindings and dispatch into an open compute pass.
    fn dispatch_camera(&self, pass: &mut wgpu::ComputePass, k: usize) {
        let (pipeline, _) = &self.passes[k];
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.binds[k], &[]);
        match (self.camera_dispatch[k], &self.cache) {
            (CameraDispatch::One, _) => pass.dispatch_workgroups(1, 1, 1),
            // A workgroup per probe for the shares, an invocation per probe
            // for the shading.
            (CameraDispatch::ProbeGroups, Some(cache)) => {
                pass.dispatch_workgroups(cache.state.probes(), 1, 1)
            }
            (CameraDispatch::Probes, Some(cache)) => {
                pass.dispatch_workgroups(cache.state.probes().div_ceil(64), 1, 1)
            }
            (CameraDispatch::Listed, Some(cache)) => {
                pass.dispatch_workgroups_indirect(cache.state.buffer(66), 36)
            }
            (CameraDispatch::Aa, Some(cache)) => {
                pass.dispatch_workgroups_indirect(cache.state.buffer(66), 48)
            }
            (CameraDispatch::AaShade, Some(cache)) => {
                pass.dispatch_workgroups_indirect(cache.state.buffer(66), 64)
            }
            (CameraDispatch::AaScatter, Some(cache)) => {
                pass.dispatch_workgroups_indirect(cache.state.buffer(66), 80)
            }
            (CameraDispatch::AaAgain, Some(cache)) => {
                pass.dispatch_workgroups_indirect(cache.state.buffer(66), 96)
            }
            (CameraDispatch::FogGrid, _) => {
                // Kept unless its cells or the light moved.
                if self.fog_trace {
                    let [gx, gy, gz] = crate::fog::GRID;
                    pass.dispatch_workgroups(gx.div_ceil(8), gy.div_ceil(8), gz)
                }
            }
            (CameraDispatch::FogColumns, _) => {
                let t = crate::fog::TILE;
                pass.dispatch_workgroups(
                    self.width.div_ceil(t).div_ceil(8),
                    self.height.div_ceil(t).div_ceil(8),
                    1,
                )
            }
            (CameraDispatch::Tiles, _) => {
                pass.dispatch_workgroups(self.width.div_ceil(64), self.height.div_ceil(64), 1)
            }
            (CameraDispatch::M13Rays, _) => {
                if let Some((_, args, _, _)) = &self.m13_work {
                    pass.dispatch_workgroups_indirect(args, 0);
                }
            }
            (CameraDispatch::M13Args(offset), _) => {
                if let Some((_, args, _, _)) = &self.m13_work {
                    pass.dispatch_workgroups_indirect(args, offset);
                }
            }
            (CameraDispatch::Prep, _) => {
                let words = |binding: u32, per: u64| {
                    self.scene
                        .iter()
                        .find(|(b, _)| *b == binding)
                        .map_or(0, |(_, buf)| (buf.size() / per) as u32)
                };
                // Pages, nodes (16 bytes a box), objects (32 bytes a sphere).
                let count = crate::M13_PAGE_POOL as u32 + words(98, 16) + words(90, 32);
                let groups = count.div_ceil(64).max(1);
                let x = groups.min(65_535);
                pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
            }
            (CameraDispatch::M13Pages, _) => {
                if let Some((_, args)) = self.scene.iter().find(|(b, _)| *b == 93) {
                    pass.dispatch_workgroups_indirect(args, 0);
                }
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

    /// Diagnostics: GPU memory by buffer (binding, bytes), the scene's then
    /// the cache's, largest first.
    #[doc(hidden)]
    pub fn memory_report(&self) -> (crate::BufferSizes, crate::BufferSizes) {
        let mut scene: Vec<(u32, u64)> = self.scene.iter().map(|(k, b)| (*k, b.size())).collect();
        scene.sort_by_key(|&(_, size)| std::cmp::Reverse(size));
        let cache = self
            .cache
            .as_ref()
            .map_or(Vec::new(), |c| c.state.buffer_sizes());
        (scene, cache)
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
        self.encode_meter(encoder);
        crate::live::encode_display(encoder, &self.display, &self.display_bind, target, None);
    }

    /// Records the auto-exposure meter (M17) before a display pass, if on:
    /// its adaptation's rate from the time since the last metered frame.
    fn encode_meter(&self, encoder: &mut wgpu::CommandEncoder) {
        // Bloom first (M19), from the rendered colours.
        if let Some(b) = &self.bloom {
            self.bloom_passes
                .encode(&self.queue, encoder, (self.width, self.height), b);
        }
        let Some(auto) = self.auto_exposure else {
            return;
        };
        let now = Instant::now();
        let rate = match self.metered_at.get() {
            Some(t) if auto.seconds > 0.0 => {
                let dt = now.duration_since(t).as_secs_f32().min(0.25);
                1.0 - (-dt / auto.seconds).exp()
            }
            _ => 1.0,
        };
        self.metered_at.set(Some(now));
        self.meter
            .encode(&self.queue, encoder, (self.width, self.height), &auto, rate);
    }

    /// Bloom for the display (M19; off here by default), from the next
    /// frame.
    pub fn set_bloom(&mut self, bloom: Option<crate::Bloom>) {
        self.bloom = bloom;
    }

    /// How the display scales the image up to its target with a render
    /// scale above 1 (M19; `Upscale::Nearest` here by default).
    pub fn set_upscale(&mut self, upscale: crate::Upscale) {
        self.upscale = upscale;
    }

    /// Diagnostics (M20): the last frame's depth-of-field highlights, and
    /// the tiles with more than their lists hold.
    pub fn depth_of_field_highlights(&self) -> Option<(u32, u32)> {
        self.dof_passes
            .as_ref()
            .map(|p| p.highlights(&self.device, &self.queue))
    }

    /// Motion blur (M21; off by default), from the next frame: a shutter of
    /// `shutter` times the motion since [`Self::set_previous_camera`] and
    /// [`Self::set_instance_previous_transform`] (none set: no motion),
    /// gathered into the colours (`read_colors` returns them blurred). With
    /// hardware ray queries.
    pub fn set_motion_blur(&mut self, blur: Option<crate::MotionBlur>) {
        self.motion = blur.filter(|_| self.hw.is_some());
        self.make_motion();
    }

    /// The camera the motion blur's motion starts from (M21): usually the
    /// previous frame's. It stays until set again.
    pub fn set_previous_camera(&mut self, camera: Option<Camera>) {
        self.camera_before = camera;
    }

    /// Where movable instance `instance` was (M21), for motion blur: usually
    /// its previous frame's placement. It stays until set again; unset, the
    /// instance is taken as still.
    pub fn set_instance_previous_transform(
        &mut self,
        instance: usize,
        rotation: Mat3,
        position: DVec3,
    ) {
        let movers = self.movers.as_ref().expect("a renderer with movers");
        let &k = movers
            .index
            .get(&instance)
            .unwrap_or_else(|| panic!("instance {instance} is not movable"));
        let mut before = movers.placements[k];
        before.rotation = rotation;
        before.translation = (position - movers.origin).as_vec3();
        let n = movers.placements.len();
        self.movers_before.resize(n, None);
        self.movers_before[k] = Some(before);
    }

    /// Depth of field (M20; off by default), from the next frame: a thin
    /// lens's blur gathered from the frame's colours and depths, written
    /// into the colours (`read_colors` returns them blurred). With hardware
    /// ray queries.
    pub fn set_depth_of_field(&mut self, lens: Option<crate::DepthOfField>) {
        self.dof = lens.filter(|_| self.hw.is_some());
        self.make_dof();
    }

    /// The display's tone map (M17; `ToneMap::Reinhard` here by default,
    /// `Renderer` defaults to `ToneMap::PbrNeutral`), from the next frame.
    pub fn set_tone_map(&mut self, tone_map: crate::ToneMap) {
        self.tone_map = tone_map;
    }

    /// Auto-exposure for the display (M17; off here by default), from the
    /// next frame. It multiplies `set_exposure`'s factor.
    pub fn set_auto_exposure(&mut self, auto: Option<crate::AutoExposure>) {
        if auto.is_some() && self.auto_exposure.is_none() {
            self.metered_at.set(None);
        }
        self.auto_exposure = auto;
    }

    /// Diagnostics (M17): the auto-exposure meter's exposure and the frames
    /// it has metered. Waits for the GPU.
    #[doc(hidden)]
    pub fn metered_exposure(&self) -> (f32, f32) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meter readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.meter.state, 0, &staging, 0, 16);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let words: Vec<f32> = bytemuck::cast_slice::<u8, f32>(
            &staging.slice(..).get_mapped_range().expect("mapped meter"),
        )
        .to_vec();
        (words[0], words[1])
    }

    fn frame_to(&mut self, camera: Camera, target: Option<&wgpu::TextureView>) {
        let started = Instant::now();
        let mut cam = camera_uniform(camera, self.width, self.height, self.radiance);
        cam.right[3] = self.pixel_offset[0];
        cam.up[3] = self.pixel_offset[1];
        cam.eye[3] = self.footprint;
        cam.forward[3] = self.texture_box as u32 as f32;
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        if let Some(c) = &mut self.cache {
            self.fog_trace = c.state.write_fog(
                &self.queue,
                camera.eye,
                self.fog_sky,
                (self.width, self.height),
                self.light_moves,
            );
        }
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [self.display_mode(), self.scale],
            exposure: [
                self.exposure,
                self.tone_map.code(),
                self.auto_exposure.is_some() as u32 as f32,
                self.upscale.code() as f32,
            ],
            bloom: match self.bloom {
                Some(b) => {
                    let (qw, qh) = self.bloom_passes.quarter();
                    [b.strength.clamp(0.0, 1.0), qw as f32, qh as f32, 1.0]
                }
                None => [0.0; 4],
            },
        };
        self.queue
            .write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&dims));
        if let Some(cache) = &mut self.cache {
            cache.state.add_wait(self.waited.take());
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
        self.apply_poses();
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
        // M13's colour pass (profile stage 20): the claims (three words a
        // pixel after the header, the list and the masks) start at zero.
        if let Some((work, _, plain, _)) = &self.m13_work
            && plain.size() > 16
        {
            let pixels = self.width as u64 * self.height as u64;
            encoder.clear_buffer(work, (8 + 2 * pixels) * 4, Some(3 * pixels * 4));
        }
        // The sun's cache holds only while the lights and the objects are
        // still: emptied after a light moved, and every frame with movers.
        let movers = self.hw.as_ref().is_some_and(|hw| hw.movers.is_some());
        if let Some(sun) = &self.m13_sun
            && (self.m13_sun_stale || movers)
        {
            encoder.clear_buffer(sun, 0, None);
        }
        self.m13_sun_stale = false;
        if let Some(hits) = &self.m13_hits {
            encoder.clear_buffer(hits, 0, Some(32));
        }
        if let Some(count) = &self.m13_hits_count {
            encoder.clear_buffer(count, 0, None);
        }
        if self.one_pass {
            // Every pass's dispatch in one compute pass (wgpu orders the
            // dispatches' storage); with the cache's timer, the pass's start
            // is the camera passes' (query 2), as the first pass's is.
            let set = self
                .cache
                .as_ref()
                .and_then(|c| c.state.timer.as_ref())
                .map(|t| &t.set);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: set.map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: None,
                }),
            });
            for k in 0..self.passes.len() {
                self.dispatch_camera(&mut pass, k);
            }
        } else {
            for k in 0..self.passes.len() {
                self.encode_camera_pass(&mut encoder, k);
            }
        }
        if let Some((set, resolve, readback, _)) = &self.m13_timer {
            encoder.resolve_query_set(set, 0..4, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 32);
        }
        self.encode_dof(&mut encoder, &camera);
        self.encode_motion(&mut encoder, &camera);
        self.encode_camera_end(&mut encoder);
        if let Some(stream) = &self.stream {
            stream.encode_feedback(&mut encoder, (self.width, self.height));
        }
        if let Some(cache) = &mut self.cache {
            cache.state.encode_readback(&mut encoder);
        }
        if let Some(target) = target {
            self.encode_meter(&mut encoder);
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
        self.last_submission = Some(index.clone());
        if let Some(stream) = &mut self.stream {
            stream.after_submit(index.clone());
        }
        if let Some(cache) = &mut self.cache {
            cache.state.after_submit(index);
            cache.state.end_frame(started.elapsed());
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

    /// Diagnostics (M15): each pixel's camera hit in the latest frame:
    /// object (`u32::MAX` for none), triangle, distance and light seen + 1.
    /// Waits for the GPU.
    #[doc(hidden)]
    pub fn read_hits(&self) -> Vec<(u32, u32, f32, u32)> {
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
        words
            .chunks(4)
            .take(n)
            .map(|v| (v[0], v[1], f32::from_bits(v[2]), v[3]))
            .collect()
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
        if let Some(c) = &mut self.cache {
            self.fog_trace = c.state.write_fog(
                &self.queue,
                camera.eye,
                self.fog_sky,
                (self.width, self.height),
                self.light_moves,
            );
        }
        let dims = DisplayDims {
            size: [self.width, self.height],
            flags: [self.display_mode(), self.scale],
            exposure: [
                self.exposure,
                self.tone_map.code(),
                self.auto_exposure.is_some() as u32 as f32,
                self.upscale.code() as f32,
            ],
            bloom: match self.bloom {
                Some(b) => {
                    let (qw, qh) = self.bloom_passes.quarter();
                    [b.strength.clamp(0.0, 1.0), qw as f32, qh as f32, 1.0]
                }
                None => [0.0; 4],
            },
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
        self.apply_poses();
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
            let labels = cache.encode_bake_passes(&mut none, counts, Some(usize::MAX), false);
            for (k, label) in labels.into_iter().enumerate() {
                let pass_start = Instant::now();
                let mut encoder = self.device.create_command_encoder(&Default::default());
                cache.encode_bake_passes(&mut encoder, counts, Some(k), false);
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
            if k < 12 {
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
        self.encode_dof(&mut encoder, &camera);
        self.encode_motion(&mut encoder, &camera);
        self.encode_meter(&mut encoder);
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

    /// Diagnostics (M22): re-bakes every resident page's bounce samples
    /// alone, in place, from the direct light the cache holds now, compares
    /// them with what the cache held, and restores what it held (so a check
    /// does not change the flight it measures). `epsilon`: bounce light's
    /// threshold, as `CacheOptions::bounce_epsilon`.
    pub fn verify_bounce(&mut self, epsilon: f32) -> BounceCheck {
        self.check_bounce(epsilon, true)
    }

    /// Diagnostics (M5 gate 15): texels the last bake listed for rays to
    /// wide lights, and whether the pass that traces them runs.
    pub fn wide_listed(&mut self) -> (u32, bool) {
        let Some(cache) = &self.cache else {
            return (0, false);
        };
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let raw = cache.state.read_buffer(&self.device, 117);
        let w: &[u32] = bytemuck::cast_slice(&raw);
        (w[0], cache.wide.as_ref().is_some_and(|(_, b)| b.is_some()))
    }

    /// Diagnostics (M22): [`Self::verify_bounce`] without restoring: every
    /// resident page's bounce samples stay re-baked from the direct light
    /// the cache holds now (a reference for the image the ledger shows).
    pub fn rebake_bounce(&mut self, epsilon: f32) -> BounceCheck {
        self.check_bounce(epsilon, false)
    }

    fn check_bounce(&mut self, epsilon: f32, restore: bool) -> BounceCheck {
        let Some(cache) = &mut self.cache else {
            return BounceCheck::default();
        };
        if cache.indirect.is_none() {
            return BounceCheck::default();
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let device = &self.device;
        let before = cache.state.read_buffer(device, 73);
        let info = cache.state.read_buffer(device, 61);
        let marks = cache.state.read_buffer(device, 62);
        let used = cache.state.read_buffer(device, 40);
        let readable = cache.state.readable_pages(bytemuck::cast_slice(&marks));
        let pages = cache.state.bounce_pages(&readable);
        let chunk = (cache.state.rebake_chunk() / 9).max(1);
        for part in pages.chunks(chunk) {
            let counts = cache.state.upload_rebake_bounce(&self.queue, part);
            let mut encoder = device.create_command_encoder(&Default::default());
            cache.encode_bake(&mut encoder, counts);
            self.queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
        }
        let after = cache.state.read_buffer(device, 73);
        if restore {
            self.queue.write_buffer(cache.state.buffer(73), 0, &before);
            self.queue.write_buffer(cache.state.buffer(61), 0, &info);
        }
        self.queue.write_buffer(cache.state.buffer(40), 0, &used);
        let (b, a): (&[u32], &[u32]) =
            (bytemuck::cast_slice(&before), bytemuck::cast_slice(&after));
        let info: &[u32] = bytemuck::cast_slice(&info);
        // Per slot 9 samples × 5 channels, each (r, g) and (b, bounces held)
        // as halves.
        let per = 9 * 5 * 2;
        let rgb = |w: &[u32], k: usize| {
            let lo = half_pair(w[k]);
            [lo[0], lo[1], half_pair(w[k + 1])[0]]
        };
        let mut largest_value = 0.0f32;
        let mut changes = Vec::with_capacity(pages.len());
        for &slot in &pages {
            let first = slot as usize * per;
            let (mut change, mut value) = (0.0f32, 0.0f32);
            for k in (first..first + per).step_by(2) {
                let (x, y) = (rgb(b, k), rgb(a, k));
                for c in 0..3 {
                    change = change.max((x[c] - y[c]).abs());
                    value = value.max(x[c].abs().max(y[c].abs()));
                }
            }
            largest_value = largest_value.max(value);
            let charge = f32::from_bits(info[12 * slot as usize + 7]);
            changes.push((slot, change, charge));
        }
        let used: &[u32] = bytemuck::cast_slice(&used);
        let frame = cache.state.frame_number();
        let trace = std::env::var_os("CONTINUUM_TRACE_BOUNCE").is_some();
        let mut check = BounceCheck {
            pages: pages.len() as u32,
            largest_value,
            ..Default::default()
        };
        for (slot, change, charge) in changes {
            let stale = change > epsilon * largest_value;
            if stale {
                let w = info[12 * slot as usize + 3];
                let visible = used[slot as usize].saturating_add(1) >= frame;
                check.visible_over += visible as u32;
                check.listed_over += (w & 2 != 0) as u32;
                check.moving_over += (w & 4 != 0) as u32;
                if trace {
                    eprintln!(
                        "bounce over eps: key {:?} slot {slot} charge {charge:.3} visible {visible} listed {} at {} moving {} change {:.2}",
                        cache.state.slot_key_of(slot),
                        w & 2 != 0,
                        w >> 8,
                        w & 4 != 0,
                        change / (epsilon * largest_value)
                    );
                }
            }
            check.changed += (change > 1e-4 * largest_value) as u32;
            check.over_epsilon += stale as u32;
            check.missed += (stale && charge <= epsilon) as u32;
            check.largest_change = check.largest_change.max(change);
            check.largest_charge = check.largest_charge.max(charge);
        }
        check
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
    /// A moving rect light re-bakes on each page's own lattice (M9 gate 6:
    /// the ledger's records are kept for the re-bake). Restores the cache
    /// afterwards, so later frames run as they would have. Waits for the
    /// GPU.
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
            cache.encode_rebake(&mut encoder, counts);
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
            if std::env::var_os("CONTINUUM_LEDGER_PAGES").is_some() {
                let words: &[u32] = bytemuck::cast_slice(&records);
                let base = (slot as usize * moving + j) * 48;
                let r = &words[base..base + 48];
                if r[4] & 1 != 0 {
                    eprintln!(
                        "page {slot}: version {}, slack {:e}, kappa {:e}, since {}, flags {}, bound {bound:e}, change {change:e}",
                        r[0],
                        f32::from_bits(r[1]),
                        f32::from_bits(r[2]),
                        r[3],
                        r[4]
                    );
                }
            }
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
                        "slot {slot}: version {}, slack {:e}, kappa {:e}, since {}, flags {}, bound {bound}",
                        r[0],
                        f32::from_bits(r[1]),
                        f32::from_bits(r[2]),
                        r[3],
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
                        if std::env::var_os("CONTINUUM_LIGHT_DEBUG").is_some() {
                            eprintln!(
                                "slot {slot} texel {} light {l}: {} vs {}, {}",
                                k - first,
                                before[at],
                                after[at],
                                cache.state.slot_debug(slot)
                            );
                        }
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

    /// Blocks until submitted GPU work is done. The time it blocks counts
    /// toward the frame period the lighting cache's bake budget adapts to
    /// (as a frame call's own wait for the GPU does).
    pub fn wait(&self) {
        let start = Instant::now();
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        self.waited.set(self.waited.get() + start.elapsed());
    }
}
