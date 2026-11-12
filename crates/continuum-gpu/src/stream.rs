//! Unique geometry streamed by need (M14 gate 1): finer *camera levels* of
//! each geometry, kept on the GPU within a triangle budget where the view
//! needs their detail.
//!
//! A geometry may carry finer levels ([`Levels`]). The geometry itself is
//! the *lighting level*: always resident, what mask cones, the cache's bakes
//! and bounce light see, and what camera rays see where no finer level is
//! resident. Camera rays (hardware ray queries) see each geometry at its
//! finest resident level. After the visibility pass a feedback pass
//! (`stream.wgsl`) turns each pixel's hit distance into the level its object
//! needs (the coarsest whose error stays under
//! [`StreamOptions::pixel_error`] pixels) and keeps the highest per
//! geometry. Two frames later the CPU loads the most needed levels into a
//! fixed pool within [`StreamOptions::budget_triangles`], evicting the least
//! recently seen, and points the geometry's top-level instances at them.
//!
//! The pool lies after the lighting level's triangles in every per-triangle
//! buffer the camera's shading reads (positions, normals, texture and chart
//! coordinates, charts), so a hit on a camera level names a triangle the
//! shading reads like any other. An object's resident level's first
//! triangle is kept in its transform's unused word.

use std::collections::BTreeSet;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use continuum::mesh::Surface;
use wgpu::util::DeviceExt;

use crate::InstancedScene;
use crate::hw::HwScene;

/// A finer version of a geometry, for camera rays (M14 gate 1).
#[derive(Clone, Debug, Default)]
pub struct Level {
    /// Its triangles, with shading normals, chart coordinates and charts in
    /// the geometry's own charts (the same chart indices and `[0, 1]²`
    /// coordinates, so the lighting cache's pages serve every level), and
    /// texture coordinates where the geometry has a texture. Its `charts`
    /// and `chart_grid` are not used.
    pub surface: Surface,
    /// The largest distance from this level's surface to the finest
    /// level's, in the geometry's own units.
    pub error: f32,
    /// For a level the scene's [`LevelSource`] gives (`surface` empty): at
    /// least as many triangles as it will have (sizes its page table).
    /// Ignored when `surface` holds the triangles.
    pub triangles: u32,
}

/// Gives camera levels on demand (M14 gate 3), on worker threads, for
/// levels whose `Level::surface` is empty: the stand-in for reading them
/// from a disk, or for generating them.
pub trait LevelSource: Send + Sync + std::fmt::Debug {
    /// Level `level` (an index into `Levels::finer`) of geometry
    /// `geometry`, as [`Level::surface`] would hold it.
    fn level(&self, geometry: usize, level: usize) -> Surface;

    /// Geometry `geometry`'s lighting level (M14 gate 5), for a geometry
    /// whose lighting level streams ([`StreamOptions::lighting_triangles`])
    /// and whose surface in the scene holds no triangles (only its charts
    /// and chart grid): produced once when the renderer is made (for the
    /// record that stays resident) and again on worker threads whenever a
    /// bake needs it and it is not in CPU memory. `None` (the default): the
    /// scene holds every lighting level.
    fn lighting(&self, geometry: usize) -> Option<Surface> {
        let _ = geometry;
        None
    }
}

/// A geometry's camera levels (M14 gate 1): finer versions that camera rays
/// see where the view needs them, streamed within a budget
/// ([`StreamOptions`]). Lighting keeps the geometry itself.
///
/// Not supported yet on camera levels: vertex colors, normal maps'
/// tangents, alpha cutouts and blending, and movers.
#[derive(Clone, Debug, Default)]
pub struct Levels {
    /// The largest distance from the geometry itself (the lighting level)
    /// to the finest level, in its own units.
    pub base_error: f32,
    /// Finer levels, coarse to fine; the last is the finest.
    pub finer: Vec<Level>,
    /// A coarse level camera rays see until a finer one is resident (M14
    /// gate 4): resident from the start, outside the budget, its surface
    /// given (not by a source). With one, camera rays never see the
    /// geometry itself (it needs no hardware structure), and its error
    /// stands for `base_error`.
    pub proxy: Option<Level>,
}

/// How camera levels are streamed (M14 gate 1).
#[derive(Clone, Copy, Debug)]
pub struct StreamOptions {
    /// Triangles of camera levels kept resident at most (the pool's size,
    /// rounded up to 256).
    pub budget_triangles: u32,
    /// The largest error a level may show, in pixels: each object gets the
    /// coarsest level whose error is at most this many pixels at its
    /// distance.
    pub pixel_error: f32,
    /// The time levels' structures may take to build per frame, in ms (at
    /// least one level a frame): the streaming time budget (M14 gate 3),
    /// at `build_ns_per_triangle`. Loads past it wait for later frames.
    pub frame_ms: f32,
    /// What building a level's structure and uploading it costs per
    /// triangle, in ns (about 30 on Apple M4, M14 gate 2).
    pub build_ns_per_triangle: f32,
    /// Bytes of levels from a [`LevelSource`] kept in CPU memory at most
    /// (the least recently used dropped first).
    pub cpu_cache_bytes: u64,
    /// Threads producing levels from a [`LevelSource`].
    pub workers: usize,
    /// Triangles of lighting levels (the geometries themselves) kept on the
    /// GPU at most (M14 gate 4; 0: all resident). Needs every geometry with
    /// camera levels to have a proxy, those geometries last in the scene,
    /// and no sky, bounce light or movers.
    pub lighting_triangles: u32,
    /// Bytes of lighting levels produced on demand (M14 gate 5; see
    /// [`LevelSource::lighting`]) kept in CPU memory at most, the least
    /// recently wanted dropped first.
    pub lighting_cache_bytes: u64,
    /// Every geometry's finest level resident from the start and nothing
    /// streamed: the fully resident reference (the budget is ignored).
    pub resident: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            budget_triangles: 4 << 20,
            pixel_error: 0.5,
            frame_ms: 2.0,
            build_ns_per_triangle: 30.0,
            cpu_cache_bytes: 2 << 30,
            workers: 4,
            lighting_triangles: 0,
            lighting_cache_bytes: 256 << 20,
            resident: false,
        }
    }
}

/// What streaming did in a frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    /// Triangles of the camera levels resident after the frame's loads.
    pub resident_triangles: u64,
    /// The pool's capacity in triangles.
    pub capacity_triangles: u64,
    /// Levels loaded and evicted in the frame.
    pub loads: u32,
    /// Levels evicted in the frame.
    pub evictions: u32,
    /// Of the loads: upgrades of a geometry with a coarser level resident,
    /// and reloads of a geometry evicted in the last 120 frames.
    pub upgrades: u32,
    /// See `upgrades`.
    pub reloads: u32,
    /// Of the evictions: geometries seen in the last 30 frames (but not the
    /// latest feedback), and over-detailed ones seen in the latest.
    pub evicted_recent: u32,
    /// See `evicted_recent`.
    pub evicted_over: u32,
    /// Triangles loaded again: levels loaded within the last 600 frames.
    pub repeat_triangles: u64,
    /// Wants left for later frames by the time budget.
    pub deferred: u32,
    /// Levels a [`LevelSource`] produced (arriving this frame), the frames
    /// they took from request to arrival in all, and wants waiting for
    /// one.
    pub produced: u32,
    /// See `produced`.
    pub production_frames: u32,
    /// See `produced`.
    pub waiting_production: u32,
    /// Bytes of produced levels kept in CPU memory.
    pub cpu_cache_bytes: u64,
    /// Streamed geometries seen in the latest feedback.
    pub seen: u32,
    /// Triangles loaded in the frame.
    pub loaded_triangles: u64,
    /// Geometries seen in the latest feedback wanting a finer level than
    /// the one resident, after the frame's loads.
    pub wanting: u32,
    /// Triangles of the levels the latest feedback wanted (the view's
    /// working set; lighting levels count 0).
    pub wanted_triangles: u64,
    /// CPU time spent reading feedback, planning and uploading (ms).
    pub cpu_ms: f32,
}

/// Triangles per page of the pool. A level occupies whole pages anywhere
/// in the pool; its bottom-level structure has one geometry per page, so a
/// hit's geometry index names the page, and the pool never fragments (M14
/// gate 1's buddy allocator caused nearly all of its evictions).
const PAGE: u32 = 4096;
/// Evicted bottom-level structures are dropped after this many frames, once
/// frames in flight no longer trace them.
const GRAVE_FRAMES: u64 = 4;
const NONE: u32 = u32::MAX;
/// Marks a transform's spare word as holding a camera level's page-table
/// entry (so the float's bits are a normal number).
const LEVEL_FLAG: u32 = 0x4000_0000;

fn pages_of(triangles: u32) -> u32 {
    triangles.div_ceil(PAGE).max(1)
}

/// The pool's free pages, lowest first, so a level's pages tend to be
/// consecutive (fewer, larger uploads).
struct Pages {
    free: BTreeSet<u32>,
}

impl Pages {
    fn new(count: u32) -> Self {
        Pages {
            free: (0..count).collect(),
        }
    }

    fn take(&mut self, n: u32) -> Option<Vec<u32>> {
        if self.free.len() < n as usize {
            return None;
        }
        Some((0..n).filter_map(|_| self.free.pop_first()).collect())
    }

    fn release(&mut self, pages: &[u32]) {
        self.free.extend(pages.iter().copied());
    }
}

/// Why a level was evicted (for `StreamStats`).
enum EvictClass {
    /// Not seen for 30 frames: dropped.
    Old,
    /// A coarser level under a finer one shown: dropped.
    Hidden,
    /// Seen within 30 frames, but not in the latest feedback: dropped.
    Recent,
    /// Seen, more detailed than wanted: downgraded (or dropped if it wants
    /// its lighting level).
    Over,
}

/// A level as the streamer keeps it: its triangles (or their bound, before
/// a source produces it) and its packed data, if in CPU memory.
struct Packed {
    triangles: u32,
    data: Option<Arc<PackedData>>,
    /// Given by the scene's source (its data may be dropped and produced
    /// again).
    sourced: bool,
    /// The frame its production was requested (0: not in production).
    requested: u64,
    /// The latest frame it was produced or loaded.
    used: u64,
}

/// A level's per-triangle data as the camera's buffers hold it.
struct PackedData {
    triangles: u32,
    tris: Vec<[f32; 4]>,
    normals: Vec<[f32; 4]>,
    uv: Vec<[f32; 2]>,
    chart: Vec<u32>,
    tex: Vec<[f32; 2]>,
}

impl PackedData {
    fn bytes(&self) -> u64 {
        (16 * (self.tris.len() + self.normals.len())
            + 8 * (self.uv.len() + self.tex.len())
            + 4 * self.chart.len()) as u64
    }
}

fn level_triangles(l: &Level) -> u32 {
    if l.surface.triangles.is_empty() {
        l.triangles
    } else {
        l.surface.triangles.len() as u32
    }
}

fn pack(s: &Surface, first_chart: u32, textured: bool) -> PackedData {
    let n = s.triangles.len();
    assert!(
        s.normals.len() == n && s.uv.len() == n && s.chart.len() == n,
        "a camera level needs normals, chart coordinates and charts per triangle"
    );
    PackedData {
        triangles: n as u32,
        tris: s
            .triangles
            .iter()
            .flat_map(|t| t.map(|v| [v.x, v.y, v.z, 0.0]))
            .collect(),
        normals: s
            .normals
            .iter()
            .flat_map(|c| c.map(|v| [v.x, v.y, v.z, 0.0]))
            .collect(),
        uv: s.uv.iter().flat_map(|c| c.map(|v| v.to_array())).collect(),
        chart: s.chart.iter().map(|&c| first_chart + c).collect(),
        tex: if !textured {
            Vec::new()
        } else if s.tex.len() == n {
            s.tex.iter().flat_map(|c| c.map(|v| v.to_array())).collect()
        } else {
            vec![[0.0; 2]; 3 * n]
        },
    }
}

struct Resident {
    /// The pool pages holding it, in order.
    pages: Vec<u32>,
    blas: wgpu::Blas,
}

struct Streamed {
    geometry: usize,
    objects: Vec<u32>,
    levels: Vec<Packed>,
    /// Each level's resident copy, if any. Camera rays see the finest
    /// resident; coarser ones stay while memory allows (as a texture keeps
    /// its coarser mips), so stepping down costs nothing.
    held: Vec<Option<Resident>>,
    /// The level wanted in the latest feedback that saw it (0: the lighting
    /// level), and that feedback's frame.
    wanted: u32,
    seen: u64,
    /// The frame it was last evicted in (0: never).
    evicted: u64,
    /// Each level's first page-table entry, after the table's base (as
    /// many entries as the level has pages).
    tables: Vec<u32>,
    /// The frame each level was last loaded in (0: never).
    loaded: Vec<u64>,
    /// Its first chart, and whether its levels carry texture coordinates
    /// (for packing produced levels).
    first_chart: u32,
    textured: bool,
    /// Its proxy, if any: its data, then (once loaded) its structure; its
    /// page-table entry is `proxy_table`.
    proxy: Option<(Arc<PackedData>, Option<wgpu::Blas>)>,
    proxy_table: u32,
}

impl Streamed {
    /// The level camera rays see: the finest resident (from 1; 0: the
    /// lighting level).
    fn level(&self) -> u32 {
        self.held
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |l| l as u32 + 1)
    }

    fn triangles(&self, level: u32) -> u32 {
        if level == 0 {
            0
        } else {
            self.levels[level as usize - 1].triangles
        }
    }

    /// Triangles of every resident level.
    fn resident_triangles(&self) -> u64 {
        self.held
            .iter()
            .zip(&self.levels)
            .filter(|(h, _)| h.is_some())
            .map(|(_, p)| p.triangles as u64)
            .sum()
    }
}

/// The camera's per-triangle buffers and the transforms, for uploads.
pub(crate) struct Targets<'a> {
    pub tris: &'a wgpu::Buffer,
    pub normals: &'a wgpu::Buffer,
    pub tex: &'a wgpu::Buffer,
    /// The cache's chart coordinates and charts (none without the cache).
    pub uv: Option<&'a wgpu::Buffer>,
    pub chart: Option<&'a wgpu::Buffer>,
    pub xforms: &'a wgpu::Buffer,
    /// The objects' ranges, which the page table follows.
    pub meshes: &'a wgpu::Buffer,
}

struct Readback {
    buffer: wgpu::Buffer,
    pending: Option<(u64, wgpu::SubmissionIndex)>,
}

pub(crate) struct Streamer {
    options: StreamOptions,
    /// The pool's first triangle in the camera's buffers (proxies), and its
    /// pages' first.
    base: u32,
    pages_base: u32,
    capacity: u32,
    alloc: Pages,
    /// The page table's first entry (a word index into the objects'
    /// ranges' buffer).
    table_base: u32,
    textured: bool,
    geoms: Vec<Streamed>,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    object_stream: wgpu::Buffer,
    errors: wgpu::Buffer,
    wanted: wgpu::Buffer,
    bind: Option<wgpu::BindGroup>,
    readbacks: [Readback; 2],
    levels: u32,
    objects: u32,
    /// The frame being prepared (from 1), and the latest whose feedback was
    /// read.
    frame: u64,
    latest: u64,
    graveyard: Vec<(u64, wgpu::Blas)>,
    log: Vec<StreamStats>,
    /// Evict every level at the next frame (diagnostics).
    evict_all: bool,
    /// Reused mapped staging for uploads (fresh staging per write cost
    /// about half the uploads' CPU time in development).
    belt: wgpu::util::StagingBelt,
    /// Worker threads producing levels from the scene's source, if any.
    producer: Option<Producer>,
    /// Bytes of produced levels' data kept in CPU memory.
    cache_bytes: u64,
}

/// A level to produce: its streamed geometry, the scene's geometry, the
/// level (from 1), the geometry's first chart, and whether it is textured.
struct Job {
    s: usize,
    geometry: usize,
    level: u32,
    first_chart: u32,
    textured: bool,
}

/// Worker threads calling the scene's [`LevelSource`]; they end when the
/// streamer (and so the job queue) is dropped.
struct Producer {
    jobs: mpsc::Sender<Job>,
    done: mpsc::Receiver<(usize, u32, Arc<PackedData>)>,
}

impl Producer {
    fn new(source: Arc<dyn LevelSource>, workers: usize) -> Self {
        let (jobs, queue) = mpsc::channel::<Job>();
        let (finished, done) = mpsc::channel();
        let queue = Arc::new(Mutex::new(queue));
        for _ in 0..workers.max(1) {
            let (queue, finished, source) = (queue.clone(), finished.clone(), source.clone());
            std::thread::spawn(move || {
                loop {
                    let Ok(job) = queue.lock().expect("job queue").recv() else {
                        return;
                    };
                    let surface = source.level(job.geometry, job.level as usize - 1);
                    let data = Arc::new(pack(&surface, job.first_chart, job.textured));
                    if finished.send((job.s, job.level, data)).is_err() {
                        return;
                    }
                }
            });
        }
        Producer { jobs, done }
    }
}

fn streamed(scene: &InstancedScene) -> impl Iterator<Item = &Levels> {
    scene
        .geometry_levels
        .iter()
        .take(scene.geometries.len())
        .flatten()
        .filter(|l| !l.finer.is_empty())
}

fn finest_pages(l: &Levels) -> u32 {
    pages_of(level_triangles(l.finer.last().expect("a finer level")))
}

fn all_pages(l: &Levels) -> u32 {
    l.finer.iter().map(|f| pages_of(level_triangles(f))).sum()
}

/// The pool's size in triangles for `scene` under `options` (0: nothing to
/// stream).
pub(crate) fn pool_triangles(scene: &InstancedScene, options: &StreamOptions) -> u32 {
    if streamed(scene).count() == 0 {
        0
    } else if options.resident {
        proxy_triangles(scene) + streamed(scene).map(finest_pages).sum::<u32>() * PAGE
    } else {
        proxy_triangles(scene) + options.budget_triangles.div_ceil(PAGE).max(1) * PAGE
    }
}

/// Triangles of every streamed geometry's proxy (packed at the pool's
/// start, ahead of its pages).
fn proxy_triangles(scene: &InstancedScene) -> u32 {
    streamed(scene)
        .filter_map(|l| l.proxy.as_ref())
        .map(|p| p.surface.triangles.len() as u32)
        .sum()
}

/// Entries of the page table (one per page of every streamed geometry's
/// every level).
pub(crate) fn table_entries(scene: &InstancedScene) -> u32 {
    streamed(scene)
        .map(|l| all_pages(l) + l.proxy.is_some() as u32)
        .sum()
}

/// Whether any camera level carries texture coordinates for a textured
/// geometry (the pool then extends the texture coordinates' buffer).
pub(crate) fn textured(scene: &InstancedScene) -> bool {
    scene
        .geometry_levels
        .iter()
        .enumerate()
        .any(|(g, l)| l.is_some() && scene.geometry_textures.get(g).copied().flatten().is_some())
}

fn map_wait(device: &wgpu::Device, buffer: &wgpu::Buffer, index: wgpu::SubmissionIndex) -> Vec<u8> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("GPU poll failed");
    let data = buffer
        .slice(..)
        .get_mapped_range()
        .expect("mapped buffer")
        .to_vec();
    buffer.unmap();
    data
}

impl Streamer {
    /// A streamer for `scene` (as uploaded: after cutouts), whose pool
    /// starts at triangle `base` of the camera's buffers. With
    /// `options.resident`, loads every finest level now.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &InstancedScene,
        options: StreamOptions,
        base: u32,
        table_base: u32,
        hw: &mut HwScene,
        targets: &Targets,
    ) -> Self {
        let textured = textured(scene);
        let mut first_chart = Vec::with_capacity(scene.geometries.len());
        let mut charts = 0u32;
        for g in &scene.geometries {
            first_chart.push(charts);
            charts += g.charts.len() as u32;
        }
        let mut objects_of: Vec<Vec<u32>> = vec![Vec::new(); scene.geometries.len()];
        for (k, i) in scene.instances.iter().enumerate() {
            objects_of[i.geometry].push(k as u32);
        }
        let mut geoms = Vec::new();
        let mut index = vec![NONE; scene.geometries.len()];
        let mut errors_of: Vec<Vec<f32>> = Vec::new();
        let mut table = 0u32;
        let mut proxies = 0u32;
        for (g, levels) in scene.geometry_levels.iter().enumerate() {
            let Some(levels) = levels else { continue };
            if levels.finer.is_empty() || g >= scene.geometries.len() {
                continue;
            }
            let has_tex = scene.geometry_textures.get(g).copied().flatten().is_some();
            index[g] = geoms.len() as u32;
            let proxy_table = table;
            if levels.proxy.is_some() {
                table += 1;
            }
            errors_of.push(
                std::iter::once(levels.proxy.as_ref().map_or(levels.base_error, |p| p.error))
                    .chain(levels.finer.iter().map(|l| l.error))
                    .collect(),
            );
            geoms.push(Streamed {
                geometry: g,
                objects: std::mem::take(&mut objects_of[g]),
                levels: levels
                    .finer
                    .iter()
                    .map(|l| {
                        let sourced =
                            l.surface.triangles.is_empty() && scene.level_source.is_some();
                        Packed {
                            triangles: level_triangles(l),
                            data: (!sourced).then(|| {
                                Arc::new(pack(&l.surface, first_chart[g], textured && has_tex))
                            }),
                            sourced,
                            requested: 0,
                            used: 0,
                        }
                    })
                    .collect(),
                first_chart: first_chart[g],
                textured: textured && has_tex,
                proxy: levels.proxy.as_ref().map(|p| {
                    proxies += p.surface.triangles.len() as u32;
                    (
                        Arc::new(pack(&p.surface, first_chart[g], textured && has_tex)),
                        None,
                    )
                }),
                proxy_table,
                held: levels.finer.iter().map(|_| None).collect(),
                wanted: 0,
                seen: 0,
                evicted: 0,
                tables: levels
                    .finer
                    .iter()
                    .scan(table, |t, l| {
                        let first = *t;
                        *t += pages_of(level_triangles(l));
                        Some(first)
                    })
                    .collect(),
                loaded: vec![0; levels.finer.len()],
            });
            table += all_pages(levels);
        }
        let levels = errors_of.iter().map(Vec::len).max().unwrap_or(1) as u32;
        let mut errors: Vec<f32> = Vec::with_capacity(geoms.len() * levels as usize);
        for e in &errors_of {
            errors.extend(e);
            errors.extend(std::iter::repeat_n(0.0, levels as usize - e.len()));
        }
        if errors.is_empty() {
            errors.push(0.0);
        }
        let object_stream: Vec<[u32; 2]> = scene
            .instances
            .iter()
            .map(|i| [index[i.geometry], i.scale.to_bits()])
            .chain(std::iter::once([NONE, 0]))
            .collect();
        let storage = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let wanted_bytes = (4 * geoms.len() as u64).max(16);
        let wanted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream wanted"),
            size: wanted_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = || Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stream readback"),
                size: wanted_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            pending: None,
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stream feedback"),
            source: wgpu::ShaderSource::Wgsl(include_str!("stream.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream feedback"),
            layout: None,
            module: &module,
            entry_point: Some("stream_feedback"),
            compilation_options: Default::default(),
            cache: None,
        });
        let capacity = pool_triangles(scene, &options);
        let mut streamer = Streamer {
            options,
            base,
            pages_base: base + proxies,
            capacity: capacity - proxies,
            alloc: Pages::new((capacity - proxies) / PAGE),
            table_base,
            textured,
            geoms,
            pipeline,
            params: crate::uniform_buffer(device, "stream params", &[0; 32]),
            object_stream: storage("object streams", bytemuck::cast_slice(&object_stream)),
            errors: storage("level errors", bytemuck::cast_slice(&errors)),
            wanted,
            bind: None,
            readbacks: [readback(), readback()],
            levels,
            objects: scene.instances.len() as u32,
            frame: 1,
            latest: 0,
            graveyard: Vec::new(),
            log: Vec::new(),
            evict_all: false,
            belt: wgpu::util::StagingBelt::new(device.clone(), 16 << 20),
            producer: None,
            cache_bytes: 0,
        };
        if let Some(source) = &scene.level_source
            && !options.resident
            && streamer
                .geoms
                .iter()
                .any(|g| g.levels.iter().any(|l| l.sourced))
        {
            streamer.producer = Some(Producer::new(source.clone(), options.workers));
        }
        streamer.load_proxies(device, queue, hw, targets);
        if options.resident {
            // In batches, so build inputs and scratch stay bounded.
            let all: Vec<usize> = (0..streamer.geoms.len()).collect();
            for batch in all.chunks(32) {
                let mut encoder = device.create_command_encoder(&Default::default());
                for &s in batch {
                    let g = &mut streamer.geoms[s];
                    let last = g.levels.len() - 1;
                    if g.levels[last].data.is_none() {
                        let source = scene.level_source.as_ref().expect("a level source");
                        let surface = source.level(g.geometry, last);
                        g.levels[last].triangles = surface.triangles.len() as u32;
                        g.levels[last].data =
                            Some(Arc::new(pack(&surface, g.first_chart, g.textured)));
                    }
                }
                let loads: Vec<(usize, u32, Vec<u32>)> = batch
                    .iter()
                    .map(|&s| {
                        let level = streamer.geoms[s].levels.len() as u32;
                        let n = streamer.geoms[s].levels[level as usize - 1].triangles;
                        let pages = streamer
                            .alloc
                            .take(pages_of(n))
                            .expect("the pool holds every finest level");
                        (s, level, pages)
                    })
                    .collect();
                streamer.apply(device, queue, &mut encoder, hw, targets, &loads, true);
                queue.submit([encoder.finish()]);
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU poll failed");
            }
            // Nothing streams later: keep only the counts.
            for g in &mut streamer.geoms {
                for p in &mut g.levels {
                    p.data = None;
                }
            }
        }
        streamer
    }

    /// Binds the visibility buffer (after the targets are made).
    pub(crate) fn bind(&mut self, device: &wgpu::Device, vis: &wgpu::Buffer) {
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream feedback"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vis.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.object_stream.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.errors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wanted.as_entire_binding(),
                },
            ],
        }));
    }

    /// Before the frame's camera passes: reads the feedback of two frames
    /// ago, loads and evicts levels (uploads queued, structures built in
    /// `encoder`), and sets the frame's feedback parameters: `per_meter` is
    /// one rendered pixel's height per meter of distance.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hw: &mut HwScene,
        targets: &Targets,
        (width, height): (u32, u32),
        per_meter: f32,
    ) {
        let start = Instant::now();
        let frame = self.frame;
        self.graveyard.retain(|(f, _)| f + GRAVE_FRAMES > frame);
        let mut stats = StreamStats {
            capacity_triangles: self.capacity as u64,
            ..Default::default()
        };
        if std::mem::take(&mut self.evict_all) {
            for s in 0..self.geoms.len() {
                for l in 0..self.geoms[s].held.len() {
                    if let Some(r) = self.geoms[s].held[l].take() {
                        self.alloc.release(&r.pages);
                        self.graveyard.push((frame, r.blas));
                    }
                }
                self.display(s, queue, hw, targets);
            }
            encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&hw.tlas));
        }
        if !self.options.resident {
            let slot = (frame % 2) as usize;
            if let Some((seen, index)) = self.readbacks[slot].pending.take() {
                let raw = map_wait(device, &self.readbacks[slot].buffer, index);
                let words: &[u32] = bytemuck::cast_slice(&raw);
                for (g, &w) in self.geoms.iter_mut().zip(words) {
                    if w > 0 {
                        g.wanted = (w - 1).min(g.levels.len() as u32);
                        g.seen = seen;
                    }
                }
                self.latest = seen;
            }
            self.plan(device, queue, encoder, hw, targets, &mut stats);
            let params: [u32; 8] = [
                width,
                height,
                self.objects,
                self.levels,
                (self.options.pixel_error * per_meter).to_bits(),
                0,
                0,
                0,
            ];
            queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
            encoder.clear_buffer(&self.wanted, 0, None);
        }
        stats.resident_triangles = self.geoms.iter().map(Streamed::resident_triangles).sum();
        stats.seen = self
            .geoms
            .iter()
            .filter(|g| self.latest > 0 && g.seen == self.latest)
            .count() as u32;
        stats.wanting = self
            .geoms
            .iter()
            .filter(|g| self.latest > 0 && g.seen == self.latest && g.wanted > g.level())
            .count() as u32;
        stats.wanted_triangles = self
            .geoms
            .iter()
            .filter(|g| self.latest > 0 && g.seen == self.latest && g.wanted > 0)
            .map(|g| g.levels[g.wanted as usize - 1].triangles as u64)
            .sum();
        stats.cpu_ms = start.elapsed().as_secs_f32() * 1e3;
        self.log.push(stats);
    }

    /// Loads the most needed levels the budget allows, evicting the least
    /// recently seen (then the most over-detailed) to make room.
    fn plan(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hw: &mut HwScene,
        targets: &Targets,
        stats: &mut StreamStats,
    ) {
        self.receive_produced(stats);
        let latest = self.latest;
        if latest == 0 {
            return;
        }
        let mut wants: Vec<usize> = (0..self.geoms.len())
            .filter(|&s| {
                let g = &self.geoms[s];
                g.seen == latest && g.wanted > g.level()
            })
            .collect();
        wants.sort_by_key(|&s| {
            let g = &self.geoms[s];
            (
                std::cmp::Reverse(g.wanted - g.level()),
                g.levels[g.wanted as usize - 1].triangles,
                s,
            )
        });
        let mut loads: Vec<(usize, u32, Vec<u32>)> = Vec::new();
        let mut uploaded = 0u64;
        let mut evicted = false;
        // The streaming time budget, in triangles.
        let allowance =
            (self.options.frame_ms as f64 * 1e6 / self.options.build_ns_per_triangle as f64) as u64;
        for s in wants {
            let level = self.geoms[s].wanted;
            if !self.have(s, level) {
                stats.waiting_production += 1;
                continue;
            }
            let n = self.geoms[s].levels[level as usize - 1].triangles;
            if uploaded > 0 && uploaded + n as u64 > allowance {
                stats.deferred += 1;
                continue;
            }
            let pages = loop {
                if let Some(p) = self.alloc.take(pages_of(n)) {
                    break Some(p);
                }
                let Some(class) = self.evict_one(s, &mut loads, queue, hw, targets) else {
                    break None;
                };
                match class {
                    EvictClass::Recent => stats.evicted_recent += 1,
                    EvictClass::Over => stats.evicted_over += 1,
                    EvictClass::Old | EvictClass::Hidden => {}
                }
                evicted = true;
                stats.evictions += 1;
            };
            let Some(pages) = pages else { continue };
            let g = &self.geoms[s];
            if g.level() > 0 {
                stats.upgrades += 1;
            } else if g.evicted > 0 && g.evicted + 120 > self.frame {
                stats.reloads += 1;
            }
            loads.push((s, level, pages));
            uploaded += n as u64;
        }
        for (s, level, _) in &loads {
            let g = &mut self.geoms[*s];
            let last = &mut g.loaded[*level as usize - 1];
            if *last > 0 && *last + 600 > self.frame {
                stats.repeat_triangles += g.levels[*level as usize - 1].triangles as u64;
            }
            *last = self.frame;
        }
        for (s, level, _) in &loads {
            self.geoms[*s].levels[*level as usize - 1].used = self.frame;
        }
        stats.loads = loads.len() as u32;
        stats.loaded_triangles = loads
            .iter()
            .map(|(s, level, _)| self.geoms[*s].levels[*level as usize - 1].triangles as u64)
            .sum();
        if !loads.is_empty() || evicted {
            self.apply(device, queue, encoder, hw, targets, &loads, true);
        }
        self.trim_cache();
        stats.cpu_cache_bytes = self.cache_bytes;
    }

    /// Whether level `level` of streamed geometry `s` is in CPU memory;
    /// if not, asks for it (once) from the source's workers.
    fn have(&mut self, s: usize, level: u32) -> bool {
        let frame = self.frame;
        let g = &mut self.geoms[s];
        let (geometry, first_chart, textured) = (g.geometry, g.first_chart, g.textured);
        let l = &mut g.levels[level as usize - 1];
        if l.data.is_some() {
            return true;
        }
        if l.requested == 0
            && let Some(p) = &self.producer
        {
            l.requested = frame;
            let _ = p.jobs.send(Job {
                s,
                geometry,
                level,
                first_chart,
                textured,
            });
        }
        false
    }

    /// Takes in what the workers produced.
    fn receive_produced(&mut self, stats: &mut StreamStats) {
        let Some(p) = &self.producer else { return };
        let done: Vec<(usize, u32, Arc<PackedData>)> = p.done.try_iter().collect();
        for (s, level, data) in done {
            let l = &mut self.geoms[s].levels[level as usize - 1];
            stats.produced += 1;
            stats.production_frames += (self.frame - l.requested) as u32;
            l.requested = 0;
            l.used = self.frame;
            l.triangles = data.triangles;
            self.cache_bytes += data.bytes();
            l.data = Some(data);
        }
    }

    /// Drops produced levels' data, least recently used first, to the CPU
    /// cache's cap.
    fn trim_cache(&mut self) {
        while self.cache_bytes > self.options.cpu_cache_bytes {
            let Some((s, l)) = self
                .geoms
                .iter()
                .enumerate()
                .flat_map(|(s, g)| {
                    g.levels
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| l.sourced && l.data.is_some())
                        .map(move |(l, p)| (p.used, s, l))
                })
                .min()
                .map(|(_, s, l)| (s, l))
            else {
                break;
            };
            let data = self.geoms[s].levels[l].data.take().expect("data");
            self.cache_bytes -= data.bytes();
        }
    }

    /// Makes room for geometry `s` by dropping one resident level, in this
    /// order: the level shown of a seen geometry more detailed than it
    /// wants, when a resident coarser level still meets its want (a free
    /// step down; the largest first); any level of the geometry least
    /// recently seen, if not seen for 30 frames (its finest first); a
    /// coarser level hidden under a finer one (those below what it wants
    /// first, the least recently seen first); the level shown of a seen
    /// geometry more detailed than it wants without such a coarser level,
    /// replaced at once by the level it wants (a load appended to `loads`,
    /// so it is never shown coarser than it needs); the level shown of the
    /// geometry least recently seen, even within 30 frames. Never a level of
    /// `s` or of one loading this frame, nor one shown at no more detail than
    /// wanted. What kind was dropped, if any.
    fn evict_one(
        &mut self,
        s: usize,
        loads: &mut Vec<(usize, u32, Vec<u32>)>,
        queue: &wgpu::Queue,
        hw: &mut HwScene,
        targets: &Targets,
    ) -> Option<EvictClass> {
        let latest = self.latest;
        let mut best: Option<((u32, i64, i64), usize, u32)> = None;
        for (i, g) in self.geoms.iter().enumerate() {
            if i == s || loads.iter().any(|l| l.0 == i) {
                continue;
            }
            let shown = g.level();
            for (l, h) in g.held.iter().enumerate() {
                if h.is_none() {
                    continue;
                }
                let level = l as u32 + 1;
                let below = g.held[..l]
                    .iter()
                    .rposition(Option::is_some)
                    .map_or(0, |b| b as u32 + 1);
                let rank = if level == shown
                    && level > g.wanted
                    && below >= g.wanted
                    && g.seen == latest
                {
                    (0, -(g.triangles(level) as i64), 0)
                } else if g.seen + 30 <= latest {
                    (1, g.seen as i64, -(level as i64))
                } else if level < shown {
                    ((level >= g.wanted) as u32 + 2, g.seen as i64, level as i64)
                } else if g.seen == latest && level > g.wanted {
                    (4, -(g.triangles(level) as i64), 0)
                } else if g.seen < latest {
                    (5, g.seen as i64, 0)
                } else {
                    continue;
                };
                if best.as_ref().is_none_or(|b| rank < b.0) {
                    best = Some((rank, i, level));
                }
            }
        }
        let ((rank, _, _), v, level) = best?;
        let class = match rank {
            1 => EvictClass::Old,
            2 | 3 => EvictClass::Hidden,
            0 | 4 => EvictClass::Over,
            _ => EvictClass::Recent,
        };
        let r = self.geoms[v].held[level as usize - 1]
            .take()
            .expect("a resident level");
        self.alloc.release(&r.pages);
        self.graveyard.push((self.frame, r.blas));
        if rank != 2 && rank != 3 {
            self.geoms[v].evicted = self.frame;
        }
        let wanted = self.geoms[v].wanted;
        if rank == 4 && wanted > 0 && self.have(v, wanted) {
            // Step down to the level it wants, loaded into the pages just
            // freed; its objects switch in this frame's build. (Without its
            // data in CPU memory it drops to a coarser level until then.)
            let n = pages_of(self.geoms[v].triangles(wanted));
            let pages = self.alloc.take(n).expect("a step down fits its own pages");
            loads.push((v, wanted, pages));
        } else {
            self.display(v, queue, hw, targets);
        }
        Some(class)
    }

    /// Uploads every proxy (at the pool's start, packed one after another)
    /// and builds their structures, waiting for them (once, at creation).
    fn load_proxies(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hw: &mut HwScene,
        targets: &Targets,
    ) {
        let mut first = self.base;
        let mut built: Vec<(usize, wgpu::BlasTriangleGeometrySizeDescriptor, u32)> = Vec::new();
        for s in 0..self.geoms.len() {
            let Some((p, _)) = &self.geoms[s].proxy else {
                continue;
            };
            let t = first as u64;
            queue.write_buffer(targets.tris, 48 * t, bytemuck::cast_slice(&p.tris));
            queue.write_buffer(targets.normals, 48 * t, bytemuck::cast_slice(&p.normals));
            if let (Some(uv), Some(chart)) = (targets.uv, targets.chart) {
                queue.write_buffer(uv, 24 * t, bytemuck::cast_slice(&p.uv));
                queue.write_buffer(chart, 4 * t, bytemuck::cast_slice(&p.chart));
            }
            if self.textured && !p.tex.is_empty() {
                queue.write_buffer(targets.tex, 24 * t, bytemuck::cast_slice(&p.tex));
            }
            let table = self.table_base + self.geoms[s].proxy_table;
            queue.write_buffer(targets.meshes, 4 * table as u64, bytemuck::bytes_of(&first));
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: 3 * p.triangles,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
            built.push((s, size, 3 * first));
            first += p.triangles;
        }
        for batch in built.chunks(256) {
            let blases: Vec<wgpu::Blas> = batch
                .iter()
                .map(|(_, size, _)| {
                    device.create_blas(
                        &wgpu::CreateBlasDescriptor {
                            label: Some("proxy"),
                            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                        },
                        wgpu::BlasGeometrySizeDescriptors::Triangles {
                            descriptors: vec![size.clone()],
                        },
                    )
                })
                .collect();
            for ((s, _, _), blas) in batch.iter().zip(&blases) {
                if let Some((_, b)) = &mut self.geoms[*s].proxy {
                    *b = Some(blas.clone());
                }
                self.display(*s, queue, hw, targets);
            }
            let entries: Vec<wgpu::BlasBuildEntry> = batch
                .iter()
                .zip(&blases)
                .map(|((_, size, first_vertex), blas)| wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size,
                            vertex_buffer: targets.tris,
                            first_vertex: *first_vertex,
                            vertex_stride: 16,
                            index_buffer: None,
                            first_index: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                })
                .collect();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(entries.iter(), std::iter::once(&hw.tlas));
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll failed");
        }
    }

    /// Points geometry `s`'s objects at its finest resident level (or its
    /// proxy, or its lighting level).
    fn display(&self, s: usize, queue: &wgpu::Queue, hw: &mut HwScene, targets: &Targets) {
        let g = &self.geoms[s];
        let level = g.level();
        let (blas, word) = if level == 0 {
            match &g.proxy {
                Some((_, Some(b))) => (Some(b), LEVEL_FLAG | (self.table_base + g.proxy_table)),
                _ => (None, 0),
            }
        } else {
            let r = g.held[level as usize - 1]
                .as_ref()
                .expect("a resident level");
            (
                Some(&r.blas),
                LEVEL_FLAG | (self.table_base + g.tables[level as usize - 1]),
            )
        };
        for &k in &g.objects {
            hw.point_at(k as usize, g.geometry, blas);
            write_first(queue, targets.xforms, k, word);
        }
    }

    /// Uploads `loads` (streamed geometry, level from 1, its pool pages),
    /// points their objects at them, and rebuilds the top-level structure in
    /// `encoder` (with `rebuild`, even without loads). A replaced level's
    /// pages were freed by the caller.
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hw: &mut HwScene,
        targets: &Targets,
        loads: &[(usize, u32, Vec<u32>)],
        rebuild: bool,
    ) {
        // Per load: the geometry, its level, and each page's size and first
        // vertex in the triangles' buffer, which the structure builds from.
        type Built = (
            usize,
            u32,
            Vec<wgpu::BlasTriangleGeometrySizeDescriptor>,
            Vec<u32>,
        );
        let mut built: Vec<Built> = Vec::with_capacity(loads.len());
        for (s, level, pages) in loads {
            let (s, level) = (*s, *level);
            let p = self.geoms[s].levels[level as usize - 1]
                .data
                .clone()
                .expect("a level in CPU memory");
            let n = p.triangles as usize;
            // Consecutive pages upload together.
            let mut j = 0;
            while j < pages.len() {
                let mut k = j + 1;
                while k < pages.len() && pages[k] == pages[k - 1] + 1 {
                    k += 1;
                }
                let (t0, t1) = (j * PAGE as usize, (k * PAGE as usize).min(n));
                let t = (self.pages_base + pages[j] * PAGE) as u64;
                let belt = &mut self.belt;
                let mut put = |target: &wgpu::Buffer, offset: u64, data: &[u8]| {
                    if let Some(size) = wgpu::BufferSize::new(data.len() as u64) {
                        belt.write_buffer(encoder, target, offset, size)
                            .copy_from_slice(data);
                    }
                };
                put(
                    targets.tris,
                    48 * t,
                    bytemuck::cast_slice(&p.tris[3 * t0..3 * t1]),
                );
                put(
                    targets.normals,
                    48 * t,
                    bytemuck::cast_slice(&p.normals[3 * t0..3 * t1]),
                );
                if let (Some(uv), Some(chart)) = (targets.uv, targets.chart) {
                    put(uv, 24 * t, bytemuck::cast_slice(&p.uv[3 * t0..3 * t1]));
                    put(chart, 4 * t, bytemuck::cast_slice(&p.chart[t0..t1]));
                }
                if self.textured && !p.tex.is_empty() {
                    put(
                        targets.tex,
                        24 * t,
                        bytemuck::cast_slice(&p.tex[3 * t0..3 * t1]),
                    );
                }
                j = k;
            }
            let table = self.table_base + self.geoms[s].tables[level as usize - 1];
            let entries: Vec<u32> = pages
                .iter()
                .map(|&pg| self.pages_base + pg * PAGE)
                .collect();
            queue.write_buffer(
                targets.meshes,
                4 * table as u64,
                bytemuck::cast_slice(&entries),
            );
            let sizes: Vec<wgpu::BlasTriangleGeometrySizeDescriptor> = (0..pages.len())
                .map(|j| wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: 3 * (PAGE as usize).min(n - j * PAGE as usize) as u32,
                    index_format: None,
                    index_count: None,
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                })
                .collect();
            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("camera level"),
                    // Streamed levels build for speed of building (a third
                    // of a load's cost in development); levels loaded once
                    // for good, for speed of tracing.
                    flags: if self.options.resident {
                        wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
                    } else {
                        wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD
                    },
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: sizes.clone(),
                },
            );
            let firsts = pages
                .iter()
                .map(|&pg| 3 * (self.pages_base + pg * PAGE))
                .collect();
            // Resident and shown before the top-level build below reads the
            // objects' instances.
            let old = self.geoms[s].held[level as usize - 1].replace(Resident {
                pages: pages.clone(),
                blas,
            });
            if let Some(old) = old {
                self.alloc.release(&old.pages);
                self.graveyard.push((self.frame, old.blas));
            }
            self.display(s, queue, hw, targets);
            built.push((s, level, sizes, firsts));
        }
        self.belt.finish_and_recall_on_submit(encoder);
        if built.is_empty() && !rebuild {
            return;
        }
        let entries: Vec<wgpu::BlasBuildEntry> = built
            .iter()
            .map(|(s, level, sizes, firsts)| wgpu::BlasBuildEntry {
                blas: &self.geoms[*s].held[*level as usize - 1]
                    .as_ref()
                    .expect("a level just loaded")
                    .blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(
                    sizes
                        .iter()
                        .zip(firsts)
                        .map(|(size, &first_vertex)| wgpu::BlasTriangleGeometry {
                            size,
                            vertex_buffer: targets.tris,
                            first_vertex,
                            vertex_stride: 16,
                            index_buffer: None,
                            first_index: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        })
                        .collect(),
                ),
            })
            .collect();
        encoder.build_acceleration_structures(entries.iter(), std::iter::once(&hw.tlas));
        drop(entries);
    }

    /// After the camera's visibility pass: the feedback pass, and its copy
    /// for reading back.
    pub(crate) fn encode_feedback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        (width, height): (u32, u32),
    ) {
        if self.options.resident {
            return;
        }
        let Some(bind) = &self.bind else { return };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream feedback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(width.div_ceil(32), height.div_ceil(32), 1);
        }
        let slot = (self.frame % 2) as usize;
        encoder.copy_buffer_to_buffer(&self.wanted, 0, &self.readbacks[slot].buffer, 0, None);
    }

    /// After the frame is submitted.
    pub(crate) fn after_submit(&mut self, index: wgpu::SubmissionIndex) {
        if !self.options.resident && self.bind.is_some() {
            let r = &mut self.readbacks[(self.frame % 2) as usize];
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
            r.pending = Some((self.frame, index));
        }
        self.frame += 1;
    }

    /// Evicts every level at the next frame (diagnostics).
    pub(crate) fn evict_all(&mut self) {
        self.evict_all = true;
    }

    /// What streaming did in every frame so far.
    pub(crate) fn log(&self) -> &[StreamStats] {
        &self.log
    }

    /// Bytes of camera levels kept in CPU memory for loading.
    pub(crate) fn cpu_bytes(&self) -> u64 {
        self.geoms
            .iter()
            .flat_map(|g| &g.levels)
            .filter_map(|p| p.data.as_ref().map(|d| d.bytes()))
            .sum()
    }

    /// Triangles of every geometry's finest camera level.
    pub(crate) fn finest_triangles(&self) -> u64 {
        self.geoms
            .iter()
            .map(|g| g.levels.last().map_or(0, |l| l.triangles as u64))
            .sum()
    }
}

/// Writes object `k`'s resident camera level's page-table entry (flagged;
/// 0: the lighting level) into its transform's spare word.
fn write_first(queue: &wgpu::Queue, xforms: &wgpu::Buffer, k: u32, word: u32) {
    queue.write_buffer(xforms, (64 * k as u64) + 60, bytemuck::bytes_of(&word));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_come_lowest_first_and_return() {
        let mut p = Pages::new(8);
        let a = p.take(3).unwrap();
        assert_eq!(a, vec![0, 1, 2]);
        let b = p.take(4).unwrap();
        assert_eq!(b, vec![3, 4, 5, 6]);
        assert!(p.take(2).is_none());
        p.release(&a);
        assert_eq!(p.take(4).unwrap(), vec![0, 1, 2, 7]);
    }
}

/// Lighting levels streamed (M14 gate 4): the geometries from `first` on
/// (the scene's last) stream; the others stay resident, as a prefix of
/// every per-triangle buffer, then `slots` slots of `slot_tris` triangles
/// (and `slot_nodes` hierarchy nodes, `slot_cells` cell-list entries).
/// Per streamed geometry, its record and its place in the chart tables.
#[derive(Clone, Debug)]
pub(crate) struct LightingLayout {
    pub first: usize,
    pub static_tris: u32,
    pub slot_tris: u32,
    pub slots: u32,
    pub slot_nodes: u32,
    pub slot_cells: u32,
    pub records: Arc<Vec<LitRecord>>,
    pub places: Arc<Vec<ChartPlace>>,
}

impl LightingLayout {
    /// Triangles before the camera's pool: the resident prefix and the
    /// slots.
    pub(crate) fn lit_tris(&self) -> u32 {
        self.static_tris + self.slots * self.slot_tris
    }
}

/// What stays resident of a streamed lighting level (M14 gate 5): its
/// hierarchy's root (the *stub*) with the root's coverage, its triangles'
/// bounds, and its package's sizes (which size the slots).
#[derive(Clone, Debug)]
pub(crate) struct LitRecord {
    pub root: crate::hierarchy::NodeGpu,
    pub cov: [f32; crate::hierarchy::COVERAGE_SAMPLES],
    pub bounds: (continuum::glam::Vec3, continuum::glam::Vec3),
    pub tris: u32,
    pub nodes: u32,
    pub cell_entries: u32,
}

/// A geometry's place in the lighting cache's chart tables (as
/// `vcache::chart_tables` numbers them): its first chart, and per chart its
/// first cell and grid.
#[derive(Clone, Debug, Default)]
pub(crate) struct ChartPlace {
    pub first_chart: u32,
    pub charts: Vec<(u32, [u32; 2])>,
}

/// Every geometry's place in the chart tables.
pub(crate) fn chart_places(scene: &InstancedScene) -> Vec<ChartPlace> {
    let (mut chart, mut cell) = (0u32, 0u32);
    scene
        .geometries
        .iter()
        .map(|g| {
            let place = ChartPlace {
                first_chart: chart,
                charts: g
                    .chart_grid
                    .iter()
                    .map(|&grid| {
                        let first = cell;
                        cell += grid[0] * grid[1];
                        (first, grid)
                    })
                    .collect(),
            };
            chart += g.charts.len() as u32;
            place
        })
        .collect()
}

/// Builds a streamed geometry's lighting package from its surface, alone
/// (M14 gate 5): what the renderer's buffers would hold for it, with
/// triangles, nodes and cell-list entries counted from 0. `colors`: its
/// vertex colors, if any; `tangents`: whether its normals carry MikkTSpace
/// tangents (a normal-mapped geometry, with the kernel's MikkTSpace on).
pub(crate) fn lit_package(
    surface: &Surface,
    place: &ChartPlace,
    colors: Option<&[[continuum::glam::Vec3; 3]]>,
    tangents: bool,
) -> (LitPackage, LitRecord) {
    use crate::hierarchy::COVERAGE_SAMPLES;
    let n = surface.triangles.len();
    let colors = colors.filter(|c| c.len() == n);
    let mut tris = Vec::with_capacity(3 * n);
    for (t, corners) in surface.triangles.iter().enumerate() {
        for (k, v) in corners.iter().enumerate() {
            let w = colors.map_or(0.0, |c| crate::color_code(c[t][k]));
            tris.push([v.x, v.y, v.z, w]);
        }
    }
    let mapped = tangents && surface.tex.len() == n && surface.normals.len() == n;
    let codes = mapped
        .then(|| crate::tangents::mikktspace(surface))
        .flatten();
    let mut normals = Vec::with_capacity(3 * n);
    for (t, corners) in surface.normals.iter().enumerate() {
        for (k, v) in corners.iter().enumerate() {
            let w = codes
                .as_ref()
                .map_or(0.0, |ts| crate::tangents::encode(ts[t][k]));
            normals.push([v.x, v.y, v.z, w]);
        }
    }
    let tex: Vec<[f32; 2]> = if surface.tex.len() == n {
        surface.tex.iter().flatten().map(|v| v.to_array()).collect()
    } else {
        vec![[0.0; 2]; 3 * n]
    };
    let (nodes, items, cov) = crate::hierarchy::triangle_hierarchy(&surface.triangles, 0, 0);
    // Chart coordinates, charts and cell lists, as `vcache::chart_tables`
    // builds them: each triangle in every cell its chart coordinates'
    // bounds meet (shrunk by a thousandth of a cell).
    let cell_count: u32 = place.charts.iter().map(|(_, g)| g[0] * g[1]).sum();
    let first_cell = place.charts.first().map_or(0, |c| c.0);
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); cell_count as usize];
    let mut uv = Vec::with_capacity(3 * n);
    let mut chart = Vec::with_capacity(n);
    for (k, (corners, &c)) in surface.uv.iter().zip(&surface.chart).enumerate() {
        let (cell0, grid) = place.charts[c as usize];
        let size = continuum::glam::Vec2::new(grid[0] as f32, grid[1] as f32);
        let lo = (corners[0].min(corners[1]).min(corners[2]) * size
            + continuum::glam::Vec2::splat(1e-3))
        .floor();
        let hi = (corners[0].max(corners[1]).max(corners[2]) * size
            - continuum::glam::Vec2::splat(1e-3))
        .floor();
        let clamp = |v: f32, n: u32| (v.max(0.0) as u32).min(n - 1);
        let (x0, x1) = (clamp(lo.x, grid[0]), clamp(hi.x.max(lo.x), grid[0]));
        let (y0, y1) = (clamp(lo.y, grid[1]), clamp(hi.y.max(lo.y), grid[1]));
        for y in y0..=y1 {
            for x in x0..=x1 {
                lists[(cell0 - first_cell + y * grid[0] + x) as usize].push(k as u32);
            }
        }
        chart.push(place.first_chart + c);
        uv.extend(corners.iter().map(|v| v.to_array()));
    }
    let mut cells = Vec::with_capacity(lists.len());
    let mut cell_tris = Vec::new();
    for (c, list) in lists.into_iter().enumerate() {
        cells.push([
            first_cell + c as u32,
            cell_tris.len() as u32,
            list.len() as u32,
        ]);
        cell_tris.extend(list);
    }
    let record = LitRecord {
        root: nodes[0],
        cov: cov[..COVERAGE_SAMPLES]
            .try_into()
            .expect("a root's coverage"),
        bounds: crate::hierarchy::mesh_bounds(&surface.triangles),
        tris: n as u32,
        nodes: nodes.len() as u32,
        cell_entries: cell_tris.len() as u32,
    };
    (
        LitPackage {
            tris,
            normals,
            tex,
            nodes,
            items,
            cov,
            uv,
            chart,
            cells,
            cell_tris,
        },
        record,
    )
}

/// Bytes a geometry's lighting level takes packaged for streaming (M14
/// gate 5): its package as gate 4 kept every one in CPU memory, without
/// the surface it was built from.
#[doc(hidden)]
pub fn lighting_package_bytes(surface: &Surface) -> u64 {
    let place = ChartPlace {
        first_chart: 0,
        charts: surface
            .chart_grid
            .iter()
            .scan(0u32, |cell, &grid| {
                let first = *cell;
                *cell += grid[0] * grid[1];
                Some((first, grid))
            })
            .collect(),
    };
    lit_package(surface, &place, None, false).0.bytes()
}

/// Where lighting levels stream, if they do (see
/// [`StreamOptions::lighting_triangles`]), and the packages of those the
/// scene holds (built now and kept; the others' are produced on demand).
/// Writes every streamed geometry's record: a geometry whose surface holds
/// no triangles takes its lighting level from the scene's source, produced
/// once now (on every core) for its record, as an offline build would.
pub(crate) fn lighting_layout(
    scene: &InstancedScene,
    options: &StreamOptions,
    mikktspace: bool,
) -> Option<(LightingLayout, Vec<Option<Arc<LitPackage>>>)> {
    if options.lighting_triangles == 0 || options.resident {
        return None;
    }
    let proxied = |g: usize| {
        scene
            .geometry_levels
            .get(g)
            .and_then(|l| l.as_ref())
            .is_some_and(|l| l.proxy.is_some())
    };
    let first = (0..scene.geometries.len()).find(|&g| proxied(g))?;
    let count = scene.geometries.len();
    assert!(
        (first..count).all(proxied),
        "streamed lighting levels need every geometry from the first with a proxy on to have one"
    );
    assert!(
        (first..count).all(|g| scene.blend_clear(g) <= 0.0),
        "streamed lighting levels with blending are not supported yet"
    );
    let static_tris = scene.geometries[..first]
        .iter()
        .map(|g| g.triangles.len() as u32)
        .sum();
    let places = chart_places(scene);
    let tangents = |g: usize| {
        mikktspace
            && scene
                .geometry_normal_maps
                .get(g)
                .copied()
                .flatten()
                .is_some()
    };
    let build = |g: usize| -> (LitRecord, Option<Arc<LitPackage>>) {
        let held = &scene.geometries[g];
        if held.triangles.is_empty() {
            assert!(
                scene
                    .geometry_colors
                    .get(g)
                    .and_then(|c| c.as_ref())
                    .is_none(),
                "lighting levels produced on demand carry no vertex colors"
            );
            let surface = scene
                .level_source
                .as_ref()
                .and_then(|s| s.lighting(g))
                .expect(
                    "a geometry without triangles takes its lighting level from the scene's source",
                );
            let (_, record) = lit_package(&surface, &places[g], None, tangents(g));
            (record, None)
        } else {
            let colors = scene.geometry_colors.get(g).and_then(|c| c.as_deref());
            let (package, record) = lit_package(held, &places[g], colors, tangents(g));
            (record, Some(Arc::new(package)))
        }
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let all: Vec<usize> = (first..count).collect();
    let chunk = all.len().div_ceil(threads).max(1);
    let built: Vec<(LitRecord, Option<Arc<LitPackage>>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .chunks(chunk)
            .map(|part| scope.spawn(|| part.iter().map(|&g| build(g)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a lighting record thread"))
            .collect()
    });
    let (records, packages): (Vec<LitRecord>, Vec<Option<Arc<LitPackage>>>) =
        built.into_iter().unzip();
    let slot_tris = records.iter().map(|r| r.tris).max().unwrap_or(1).max(1);
    let layout = LightingLayout {
        first,
        static_tris,
        slot_tris,
        slots: (options.lighting_triangles / slot_tris).max(1),
        slot_nodes: records.iter().map(|r| r.nodes).max().unwrap_or(1).max(1),
        slot_cells: records
            .iter()
            .map(|r| r.cell_entries)
            .max()
            .unwrap_or(1)
            .max(1),
        records: Arc::new(records),
        places: Arc::new(places[first..].to_vec()),
    };
    Some((layout, packages))
}

/// A streamed geometry's lighting level as it is uploaded into a slot:
/// per triangle, its corners, normals, texture and chart coordinates and
/// chart; its hierarchy's nodes (children and items relative to the
/// slot), items (triangles relative to the slot) and node coverage; and
/// its charts' cells (global cell, first entry and count relative to the
/// slot's cell list) with that list (triangles relative to the slot).
#[derive(Default)]
pub(crate) struct LitPackage {
    pub tris: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub tex: Vec<[f32; 2]>,
    pub nodes: Vec<crate::hierarchy::NodeGpu>,
    pub items: Vec<u32>,
    pub cov: Vec<f32>,
    pub uv: Vec<[f32; 2]>,
    pub chart: Vec<u32>,
    pub cells: Vec<[u32; 3]>,
    pub cell_tris: Vec<u32>,
}

impl LitPackage {
    /// Bytes of its data.
    pub(crate) fn bytes(&self) -> u64 {
        (16 * (self.tris.len() + self.normals.len())
            + 8 * (self.tex.len() + self.uv.len())
            + 48 * self.nodes.len()
            + 4 * (self.items.len() + self.cov.len() + self.chart.len() + self.cell_tris.len())
            + 12 * self.cells.len()) as u64
    }
}

/// The node part of the layout: resident nodes, the stubs' first (one per
/// streamed geometry: its root), and nodes per slot.
pub(crate) struct LitNodes {
    pub static_nodes: u32,
    pub stubs: u32,
    pub slot_nodes: u32,
}

/// The cell part: resident cell-list entries, and entries per slot.
pub(crate) struct LitCells {
    pub static_cells: u32,
    pub slot_cells: u32,
}

/// The buffers lighting levels load into.
pub(crate) struct LitTargets<'a> {
    pub tris: &'a wgpu::Buffer,
    pub normals: &'a wgpu::Buffer,
    pub tex: &'a wgpu::Buffer,
    pub meshes: &'a wgpu::Buffer,
    pub nodes: &'a wgpu::Buffer,
    pub items: &'a wgpu::Buffer,
    pub cov: &'a wgpu::Buffer,
    pub uv: &'a wgpu::Buffer,
    pub chart: &'a wgpu::Buffer,
    pub cells: &'a wgpu::Buffer,
    pub cell_tris: &'a wgpu::Buffer,
}

/// What lighting-level streaming did in a frame (M14 gate 4).
#[derive(Clone, Copy, Debug, Default)]
pub struct LightingStats {
    /// Lighting levels loaded and evicted in the frame.
    pub loads: u32,
    /// See `loads`.
    pub evictions: u32,
    /// Lighting levels wanted but not loaded (no free slot, all pinned, or
    /// over the frame's count).
    pub refused: u32,
    /// Triangles of the resident lighting levels, and the slots' capacity.
    pub resident_triangles: u64,
    /// See `resident_triangles`.
    pub capacity_triangles: u64,
    /// Geometries pinned (pages wait for them, or are planned on them).
    pub pinned: u32,
    /// Lighting levels produced on demand (M14 gate 5) arriving this frame,
    /// the frames they took from request to arrival in all, and wanted
    /// levels waiting for one.
    pub produced: u32,
    /// See `produced`.
    pub production_frames: u32,
    /// See `produced`.
    pub waiting_production: u32,
    /// Bytes of produced lighting packages kept in CPU memory, after the
    /// frame's trim.
    pub cpu_cache_bytes: u64,
}

/// Lighting levels loaded at most per frame.
const LIT_LOADS_PER_FRAME: usize = 64;
/// Frames a lighting level stays at least once loaded: its pages are
/// planned the next frame and baked the one after.
const MIN_RESIDENT: u64 = 4;

/// Worker threads producing lighting packages from the scene's source (M14
/// gate 5); they end when the pool (and so the job queue) is dropped.
struct LitProducer {
    jobs: mpsc::Sender<usize>,
    done: mpsc::Receiver<(usize, Arc<LitPackage>)>,
}

impl LitProducer {
    fn new(
        source: Arc<dyn LevelSource>,
        layout: &LightingLayout,
        tangents: Arc<Vec<bool>>,
        workers: usize,
    ) -> Self {
        let (jobs, queue) = mpsc::channel::<usize>();
        let (finished, done) = mpsc::channel();
        let queue = Arc::new(Mutex::new(queue));
        for _ in 0..workers.max(1) {
            let (queue, finished, source) = (queue.clone(), finished.clone(), source.clone());
            let (places, tangents, first) = (layout.places.clone(), tangents.clone(), layout.first);
            std::thread::spawn(move || {
                loop {
                    let Ok(g) = queue.lock().expect("job queue").recv() else {
                        return;
                    };
                    let surface = source
                        .lighting(first + g)
                        .expect("the source's lighting level");
                    let (package, _) = lit_package(&surface, &places[g], None, tangents[g]);
                    if finished.send((g, Arc::new(package))).is_err() {
                        return;
                    }
                }
            });
        }
        LitProducer { jobs, done }
    }
}

/// Lighting levels streamed into slots (M14 gate 4), their packages kept in
/// CPU memory or produced on demand (M14 gate 5).
pub(crate) struct LightingPool {
    layout: LightingLayout,
    nodes: LitNodes,
    cells: LitCells,
    /// Per streamed geometry: its package if in CPU memory, whether it is
    /// produced on demand (and so may be dropped), and the frame it was
    /// requested in (0: not).
    packages: Vec<Option<Arc<LitPackage>>>,
    sourced: Vec<bool>,
    requested: Vec<u64>,
    producer: Option<LitProducer>,
    /// Bytes of produced packages held, and their cap.
    cache_bytes: u64,
    cache_cap: u64,
    /// Per streamed geometry: its objects, its slot, the latest frame it
    /// was wanted, touched or pinned.
    objects: Vec<Vec<u32>>,
    slot: Vec<Option<u32>>,
    used: Vec<u64>,
    /// The frame each was loaded in (kept at least `MIN_RESIDENT` frames,
    /// so its pages bake before it can go).
    loaded_at: Vec<u64>,
    /// Per slot, the geometry in it.
    owner: Vec<Option<usize>>,
    free: Vec<u32>,
    /// Per object of the scene: whether its lighting level is resident.
    resident: Vec<bool>,
    /// Per object: its streamed geometry (NONE: resident always).
    of_object: Vec<u32>,
    frame: u64,
    log: Vec<LightingStats>,
}

impl LightingPool {
    pub(crate) fn new(
        layout: LightingLayout,
        nodes: LitNodes,
        cells: LitCells,
        packages: Vec<Option<Arc<LitPackage>>>,
        scene: &InstancedScene,
        options: &StreamOptions,
        mikktspace: bool,
    ) -> Self {
        let count = packages.len();
        let mut objects = vec![Vec::new(); count];
        let mut of_object = vec![NONE; scene.instances.len()];
        let mut resident = vec![true; scene.instances.len()];
        for (k, i) in scene.instances.iter().enumerate() {
            if i.geometry >= layout.first {
                objects[i.geometry - layout.first].push(k as u32);
                of_object[k] = (i.geometry - layout.first) as u32;
                resident[k] = false;
            }
        }
        let sourced: Vec<bool> = packages.iter().map(Option::is_none).collect();
        let producer = sourced.iter().any(|&s| s).then(|| {
            let tangents = (0..count)
                .map(|g| {
                    mikktspace
                        && scene
                            .geometry_normal_maps
                            .get(layout.first + g)
                            .copied()
                            .flatten()
                            .is_some()
                })
                .collect();
            LitProducer::new(
                scene
                    .level_source
                    .clone()
                    .expect("lighting levels produced by the scene's source"),
                &layout,
                Arc::new(tangents),
                options.workers,
            )
        });
        LightingPool {
            free: (0..layout.slots).rev().collect(),
            owner: vec![None; layout.slots as usize],
            layout,
            nodes,
            cells,
            packages,
            sourced,
            requested: vec![0; count],
            producer,
            cache_bytes: 0,
            cache_cap: options.lighting_cache_bytes,
            objects,
            slot: vec![None; count],
            used: vec![0; count],
            loaded_at: vec![0; count],
            resident,
            of_object,
            frame: 1,
            log: Vec::new(),
        }
    }

    /// Per object, whether its lighting level is resident.
    pub(crate) fn resident(&self) -> &[bool] {
        &self.resident
    }

    pub(crate) fn log(&self) -> &[LightingStats] {
        &self.log
    }

    /// Loads the lighting levels of the objects `wanted` (missing from
    /// bakes, or under pages asked for), keeping those of `pinned` objects
    /// (pages wait for them or are baked on them this frame), evicting the
    /// least recently used others to make room. A wanted level produced on
    /// demand and not in CPU memory is asked of the workers and loads once
    /// it arrives.
    pub(crate) fn update(
        &mut self,
        queue: &wgpu::Queue,
        targets: &LitTargets,
        wanted: &[u32],
        pinned: &[u32],
    ) {
        let frame = self.frame;
        let mut stats = LightingStats {
            capacity_triangles: (self.layout.slots * self.layout.slot_tris) as u64,
            ..Default::default()
        };
        if let Some(p) = &self.producer {
            for (g, package) in p.done.try_iter() {
                stats.produced += 1;
                stats.production_frames += (frame - self.requested[g]) as u32;
                self.requested[g] = 0;
                self.cache_bytes += package.bytes();
                self.packages[g] = Some(package);
                self.used[g] = self.used[g].max(frame);
            }
        }
        // Protected: pinned levels, then wanted ones in the order given (the
        // pages waiting longest first) while the slots hold them. Only
        // protected levels load, and only unprotected ones are evicted, so
        // the longest-waiting page's levels always come together.
        let n = self.packages.len();
        let capacity = self.layout.slots as usize;
        let mut protect = vec![false; n];
        let mut protected = Vec::new();
        for &o in pinned {
            if let Some(&g) = self.of_object.get(o as usize)
                && g != NONE
                && !protect[g as usize]
            {
                protect[g as usize] = true;
                protected.push(g as usize);
            }
        }
        stats.pinned = protected.len() as u32;
        let mut wants: Vec<usize> = Vec::new();
        for &o in wanted {
            let Some(&g) = self.of_object.get(o as usize) else {
                continue;
            };
            if g == NONE || protect[g as usize] {
                continue;
            }
            if protected.len() >= capacity {
                stats.refused += 1;
                continue;
            }
            protect[g as usize] = true;
            protected.push(g as usize);
            wants.push(g as usize);
        }
        for &g in &protected {
            self.used[g] = frame;
        }
        let mut loaded = 0;
        for g in wants {
            if self.slot[g].is_some() {
                continue;
            }
            if self.packages[g].is_none() {
                if self.requested[g] == 0
                    && let Some(p) = &self.producer
                {
                    self.requested[g] = frame;
                    let _ = p.jobs.send(g);
                }
                stats.waiting_production += 1;
                continue;
            }
            if loaded >= LIT_LOADS_PER_FRAME {
                stats.refused += 1;
                continue;
            }
            let slot = match self.free.pop() {
                Some(k) => Some(k),
                None => {
                    // The least recently used unprotected one, loaded long
                    // enough ago for its pages to have baked.
                    let victim = self
                        .owner
                        .iter()
                        .flatten()
                        .copied()
                        .filter(|&h| !protect[h] && self.loaded_at[h] + MIN_RESIDENT <= frame)
                        .min_by_key(|&h| (self.used[h], h));
                    victim.map(|h| {
                        stats.evictions += 1;
                        self.evict(queue, targets, h)
                    })
                }
            };
            let Some(k) = slot else {
                stats.refused += 1;
                continue;
            };
            self.load(queue, targets, g, k);
            self.loaded_at[g] = frame;
            loaded += 1;
            stats.loads += 1;
        }
        self.trim(&protect);
        stats.cpu_cache_bytes = self.cache_bytes;
        stats.resident_triangles = self
            .owner
            .iter()
            .flatten()
            .map(|&g| self.layout.records[g].tris as u64)
            .sum();
        self.log.push(stats);
        self.frame += 1;
    }

    /// Drops produced packages to the CPU cache's cap: unprotected ones
    /// first, each the least recently used first.
    fn trim(&mut self, protect: &[bool]) {
        if self.cache_bytes <= self.cache_cap {
            return;
        }
        let mut held: Vec<(bool, u64, usize)> = (0..self.packages.len())
            .filter(|&g| self.sourced[g] && self.packages[g].is_some())
            .map(|g| (protect[g], self.used[g], g))
            .collect();
        held.sort_unstable();
        for (_, _, g) in held {
            if self.cache_bytes <= self.cache_cap {
                break;
            }
            let package = self.packages[g].take().expect("a held package");
            self.cache_bytes -= package.bytes();
        }
    }

    /// Frees geometry `g`'s slot: its objects point at their stub.
    fn evict(&mut self, queue: &wgpu::Queue, targets: &LitTargets, g: usize) -> u32 {
        let k = self.slot[g].take().expect("a resident lighting level");
        self.owner[k as usize] = None;
        let stub = self.nodes.stubs + g as u32;
        for &o in &self.objects[g] {
            self.resident[o as usize] = false;
            queue.write_buffer(
                targets.meshes,
                16 * o as u64,
                bytemuck::cast_slice(&[NONE, 0, stub]),
            );
        }
        k
    }

    /// Uploads geometry `g`'s lighting level into slot `k` and points its
    /// objects at it.
    fn load(&mut self, queue: &wgpu::Queue, targets: &LitTargets, g: usize, k: u32) {
        let p = self.packages[g].clone().expect("a package in CPU memory");
        let tri = self.layout.static_tris + k * self.layout.slot_tris;
        let node = self.nodes.static_nodes + self.packages.len() as u32 + k * self.nodes.slot_nodes;
        let cell = self.cells.static_cells + k * self.cells.slot_cells;
        let t = tri as u64;
        queue.write_buffer(targets.tris, 48 * t, bytemuck::cast_slice(&p.tris));
        queue.write_buffer(targets.normals, 48 * t, bytemuck::cast_slice(&p.normals));
        if !p.tex.is_empty() {
            queue.write_buffer(targets.tex, 24 * t, bytemuck::cast_slice(&p.tex));
        }
        queue.write_buffer(targets.uv, 24 * t, bytemuck::cast_slice(&p.uv));
        queue.write_buffer(targets.chart, 4 * t, bytemuck::cast_slice(&p.chart));
        let items: Vec<u32> = p.items.iter().map(|&i| i + tri).collect();
        queue.write_buffer(targets.items, 4 * t, bytemuck::cast_slice(&items));
        let nodes: Vec<crate::hierarchy::NodeGpu> = p
            .nodes
            .iter()
            .map(|n| n.rebased(node as i64, tri as i64))
            .collect();
        queue.write_buffer(
            targets.nodes,
            48 * node as u64,
            bytemuck::cast_slice(&nodes),
        );
        queue.write_buffer(
            targets.cov,
            4 * (node as u64 * crate::hierarchy::COVERAGE_SAMPLES as u64),
            bytemuck::cast_slice(&p.cov),
        );
        let list: Vec<u32> = p.cell_tris.iter().map(|&i| i + tri).collect();
        if !list.is_empty() {
            queue.write_buffer(
                targets.cell_tris,
                4 * cell as u64,
                bytemuck::cast_slice(&list),
            );
        }
        // The geometry's cells are consecutive: one write.
        if let (Some(first), Some(last)) = (p.cells.first(), p.cells.last()) {
            let mut run = vec![[0u32; 2]; (last[0] - first[0] + 1) as usize];
            for &[c, f, n] in &p.cells {
                run[(c - first[0]) as usize] = [f + cell, n];
            }
            queue.write_buffer(
                targets.cells,
                8 * first[0] as u64,
                bytemuck::cast_slice(&run),
            );
        }
        let count = (p.tris.len() / 3) as u32;
        for &o in &self.objects[g] {
            self.resident[o as usize] = true;
            queue.write_buffer(
                targets.meshes,
                16 * o as u64,
                bytemuck::cast_slice(&[tri, count, node]),
            );
        }
        self.slot[g] = Some(k);
        self.owner[k as usize] = Some(g);
    }
}
