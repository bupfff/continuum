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
    /// Triangles loaded per frame at most (at least one level a frame).
    pub upload_triangles: u32,
    /// Every geometry's finest level resident from the start and nothing
    /// streamed: the fully resident reference (the budget is ignored).
    pub resident: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            budget_triangles: 4 << 20,
            pixel_error: 0.5,
            upload_triangles: 1 << 17,
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

/// A level's per-triangle data as the camera's buffers hold it.
struct Packed {
    triangles: u32,
    tris: Vec<[f32; 4]>,
    normals: Vec<[f32; 4]>,
    uv: Vec<[f32; 2]>,
    chart: Vec<u32>,
    tex: Vec<[f32; 2]>,
}

fn pack(s: &Surface, first_chart: u32, textured: bool) -> Packed {
    let n = s.triangles.len();
    assert!(
        s.normals.len() == n && s.uv.len() == n && s.chart.len() == n,
        "a camera level needs normals, chart coordinates and charts per triangle"
    );
    Packed {
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
    /// The pool's first triangle in the camera's buffers.
    base: u32,
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
    pages_of(
        l.finer
            .last()
            .expect("a finer level")
            .surface
            .triangles
            .len() as u32,
    )
}

fn all_pages(l: &Levels) -> u32 {
    l.finer
        .iter()
        .map(|f| pages_of(f.surface.triangles.len() as u32))
        .sum()
}

/// The pool's size in triangles for `scene` under `options` (0: nothing to
/// stream).
pub(crate) fn pool_triangles(scene: &InstancedScene, options: &StreamOptions) -> u32 {
    if streamed(scene).count() == 0 {
        0
    } else if options.resident {
        streamed(scene).map(finest_pages).sum::<u32>() * PAGE
    } else {
        options.budget_triangles.div_ceil(PAGE).max(1) * PAGE
    }
}

/// Entries of the page table (one per page of every streamed geometry's
/// every level).
pub(crate) fn table_entries(scene: &InstancedScene) -> u32 {
    streamed(scene).map(all_pages).sum()
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
        for (g, levels) in scene.geometry_levels.iter().enumerate() {
            let Some(levels) = levels else { continue };
            if levels.finer.is_empty() || g >= scene.geometries.len() {
                continue;
            }
            let has_tex = scene.geometry_textures.get(g).copied().flatten().is_some();
            index[g] = geoms.len() as u32;
            errors_of.push(
                std::iter::once(levels.base_error)
                    .chain(levels.finer.iter().map(|l| l.error))
                    .collect(),
            );
            geoms.push(Streamed {
                geometry: g,
                objects: std::mem::take(&mut objects_of[g]),
                levels: levels
                    .finer
                    .iter()
                    .map(|l| pack(&l.surface, first_chart[g], textured && has_tex))
                    .collect(),
                held: levels.finer.iter().map(|_| None).collect(),
                wanted: 0,
                seen: 0,
                evicted: 0,
                tables: levels
                    .finer
                    .iter()
                    .scan(table, |t, l| {
                        let first = *t;
                        *t += pages_of(l.surface.triangles.len() as u32);
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
            capacity,
            alloc: Pages::new(capacity / PAGE),
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
        };
        if options.resident {
            // In batches, so build inputs and scratch stay bounded.
            let all: Vec<usize> = (0..streamer.geoms.len()).collect();
            for batch in all.chunks(32) {
                let mut encoder = device.create_command_encoder(&Default::default());
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
                    *p = Packed {
                        triangles: p.triangles,
                        tris: Vec::new(),
                        normals: Vec::new(),
                        uv: Vec::new(),
                        chart: Vec::new(),
                        tex: Vec::new(),
                    };
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
        for s in wants {
            let level = self.geoms[s].wanted;
            let n = self.geoms[s].levels[level as usize - 1].triangles;
            if uploaded > 0 && uploaded + n as u64 > self.options.upload_triangles as u64 {
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
        stats.loads = loads.len() as u32;
        stats.loaded_triangles = loads
            .iter()
            .map(|(s, level, _)| self.geoms[*s].levels[*level as usize - 1].triangles as u64)
            .sum();
        if !loads.is_empty() || evicted {
            self.apply(device, queue, encoder, hw, targets, &loads, true);
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
        if rank == 4 && wanted > 0 {
            // Step down to the level it wants, loaded into the pages just
            // freed; its objects switch in this frame's build.
            let n = pages_of(self.geoms[v].triangles(wanted));
            let pages = self.alloc.take(n).expect("a step down fits its own pages");
            loads.push((v, wanted, pages));
        } else {
            self.display(v, queue, hw, targets);
        }
        Some(class)
    }

    /// Points geometry `s`'s objects at its finest resident level (or its
    /// lighting level).
    fn display(&self, s: usize, queue: &wgpu::Queue, hw: &mut HwScene, targets: &Targets) {
        let g = &self.geoms[s];
        let level = g.level();
        let (blas, word) = if level == 0 {
            (None, 0)
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
            let p = &self.geoms[s].levels[level as usize - 1];
            let n = p.triangles as usize;
            // Consecutive pages upload together.
            let mut j = 0;
            while j < pages.len() {
                let mut k = j + 1;
                while k < pages.len() && pages[k] == pages[k - 1] + 1 {
                    k += 1;
                }
                let (t0, t1) = (j * PAGE as usize, (k * PAGE as usize).min(n));
                let t = (self.base + pages[j] * PAGE) as u64;
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
            let entries: Vec<u32> = pages.iter().map(|&pg| self.base + pg * PAGE).collect();
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
                .map(|&pg| 3 * (self.base + pg * PAGE))
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
            .map(|p| {
                (16 * (p.tris.len() + p.normals.len())
                    + 8 * (p.uv.len() + p.tex.len())
                    + 4 * p.chart.len()) as u64
            })
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
