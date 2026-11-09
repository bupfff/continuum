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

/// Triangles per allocation unit of the pool.
const UNIT: u32 = 256;
/// Evicted bottom-level structures are dropped after this many frames, once
/// frames in flight no longer trace them.
const GRAVE_FRAMES: u64 = 4;
const NONE: u32 = u32::MAX;
/// Marks a transform's spare word as holding a camera level's first triangle
/// (so the float's bits are a normal number).
const LEVEL_FLAG: u32 = 0x4000_0000;

/// A buddy allocator over the pool's units.
struct Buddy {
    /// Free blocks per order (`2^order` units), by address in units.
    free: Vec<BTreeSet<u32>>,
}

impl Buddy {
    fn new(units: u32) -> Self {
        let orders = (32 - units.max(1).leading_zeros()) as usize;
        let mut free = vec![BTreeSet::new(); orders + 1];
        let mut addr = 0u32;
        while addr < units {
            let mut k = if addr == 0 {
                orders as u32
            } else {
                addr.trailing_zeros().min(orders as u32)
            };
            while (1u64 << k) > (units - addr) as u64 {
                k -= 1;
            }
            free[k as usize].insert(addr);
            addr += 1 << k;
        }
        Buddy { free }
    }

    fn alloc(&mut self, order: usize) -> Option<u32> {
        let k = (order..self.free.len()).find(|&k| !self.free[k].is_empty())?;
        let addr = self.free[k].pop_first()?;
        for j in (order..k).rev() {
            self.free[j].insert(addr + (1 << j));
        }
        Some(addr)
    }

    fn release(&mut self, mut addr: u32, mut order: usize) {
        while order + 1 < self.free.len() {
            let buddy = addr ^ (1 << order);
            if !self.free[order].remove(&buddy) {
                break;
            }
            addr = addr.min(buddy);
            order += 1;
        }
        self.free[order].insert(addr);
    }
}

fn order_of(triangles: u32) -> usize {
    triangles
        .div_ceil(UNIT)
        .max(1)
        .next_power_of_two()
        .trailing_zeros() as usize
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
    /// 1 + the index in `levels`.
    level: u32,
    addr: u32,
    order: usize,
    blas: wgpu::Blas,
}

struct Streamed {
    geometry: usize,
    objects: Vec<u32>,
    levels: Vec<Packed>,
    resident: Option<Resident>,
    /// The level wanted in the latest feedback that saw it (0: the lighting
    /// level), and that feedback's frame.
    wanted: u32,
    seen: u64,
}

impl Streamed {
    fn level(&self) -> u32 {
        self.resident.as_ref().map_or(0, |r| r.level)
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
    alloc: Buddy,
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
}

/// The pool's size in triangles for `scene` under `options` (0: nothing to
/// stream).
pub(crate) fn pool_triangles(scene: &InstancedScene, options: &StreamOptions) -> u32 {
    let streamed = scene
        .geometry_levels
        .iter()
        .flatten()
        .filter(|l| !l.finer.is_empty());
    if options.resident {
        streamed
            .map(|l| {
                let finest = l
                    .finer
                    .last()
                    .expect("a finer level")
                    .surface
                    .triangles
                    .len();
                (1u32 << order_of(finest as u32)) * UNIT
            })
            .sum()
    } else if streamed.count() == 0 {
        0
    } else {
        options.budget_triangles.div_ceil(UNIT).max(1) * UNIT
    }
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
                resident: None,
                wanted: 0,
                seen: 0,
            });
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
            alloc: Buddy::new(capacity / UNIT),
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
        };
        if options.resident {
            // In batches, so build inputs and scratch stay bounded.
            let all: Vec<usize> = (0..streamer.geoms.len()).collect();
            for batch in all.chunks(32) {
                let mut encoder = device.create_command_encoder(&Default::default());
                let loads: Vec<(usize, u32, u32, usize)> = batch
                    .iter()
                    .map(|&s| {
                        let level = streamer.geoms[s].levels.len() as u32;
                        let order =
                            order_of(streamer.geoms[s].levels[level as usize - 1].triangles);
                        let addr = streamer
                            .alloc
                            .alloc(order)
                            .expect("the pool holds every finest level");
                        (s, level, addr, order)
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
        stats.resident_triangles = self
            .geoms
            .iter()
            .filter_map(|g| {
                g.resident
                    .as_ref()
                    .map(|r| g.levels[r.level as usize - 1].triangles as u64)
            })
            .sum();
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
        let mut loads: Vec<(usize, u32, u32, usize)> = Vec::new();
        let mut uploaded = 0u64;
        let mut evicted = false;
        for s in wants {
            let level = self.geoms[s].wanted;
            let n = self.geoms[s].levels[level as usize - 1].triangles;
            if uploaded > 0 && uploaded + n as u64 > self.options.upload_triangles as u64 {
                continue;
            }
            let order = order_of(n);
            let addr = loop {
                if let Some(a) = self.alloc.alloc(order) {
                    break Some(a);
                }
                if !self.evict_one(s, &loads, queue, hw, targets) {
                    break None;
                }
                evicted = true;
                stats.evictions += 1;
            };
            let Some(addr) = addr else { continue };
            loads.push((s, level, addr, order));
            uploaded += n as u64;
        }
        stats.loads = loads.len() as u32;
        stats.loaded_triangles = uploaded;
        if !loads.is_empty() || evicted {
            self.apply(device, queue, encoder, hw, targets, &loads, true);
        }
    }

    /// Evicts one level to make room for geometry `s`: the least recently
    /// seen not seen in the latest feedback, else the most over-detailed.
    /// Never one loading this frame or needed at its level. Whether one was
    /// evicted.
    fn evict_one(
        &mut self,
        s: usize,
        loads: &[(usize, u32, u32, usize)],
        queue: &wgpu::Queue,
        hw: &mut HwScene,
        targets: &Targets,
    ) -> bool {
        let latest = self.latest;
        let victim = self
            .geoms
            .iter()
            .enumerate()
            .filter(|(i, g)| *i != s && g.resident.is_some() && !loads.iter().any(|l| l.0 == *i))
            .filter_map(|(i, g)| {
                if g.seen < latest {
                    Some((0, g.seen as i64, i))
                } else if g.level() > g.wanted {
                    Some((1, g.wanted as i64 - g.level() as i64, i))
                } else {
                    None
                }
            })
            .min();
        let Some((_, _, v)) = victim else {
            return false;
        };
        let r = self.geoms[v].resident.take().expect("a resident level");
        self.alloc.release(r.addr, r.order);
        let geometry = self.geoms[v].geometry;
        for &k in &self.geoms[v].objects {
            hw.point_at(k as usize, geometry, None);
            write_first(queue, targets.xforms, k, 0);
        }
        self.graveyard.push((self.frame, r.blas));
        true
    }

    /// Uploads `loads` (streamed geometry, level from 1, pool address,
    /// order), points their objects at them, and rebuilds the top-level
    /// structure in `encoder` (with `rebuild`, even without loads).
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hw: &mut HwScene,
        targets: &Targets,
        loads: &[(usize, u32, u32, usize)],
        rebuild: bool,
    ) {
        // Per load: its structure, size and first vertex in the triangles'
        // buffer, which the structure builds from.
        let mut built: Vec<(
            usize,
            wgpu::Blas,
            wgpu::BlasTriangleGeometrySizeDescriptor,
            u32,
        )> = Vec::with_capacity(loads.len());
        for &(s, level, addr, _) in loads {
            let p = &self.geoms[s].levels[level as usize - 1];
            let first = self.base + addr * UNIT;
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
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: 3 * p.triangles,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
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
                    descriptors: vec![size.clone()],
                },
            );
            let geometry = self.geoms[s].geometry;
            for &k in &self.geoms[s].objects {
                hw.point_at(k as usize, geometry, Some(&blas));
                write_first(queue, targets.xforms, k, LEVEL_FLAG | first);
            }
            built.push((s, blas, size, 3 * first));
        }
        if built.is_empty() && !rebuild {
            return;
        }
        let entries: Vec<wgpu::BlasBuildEntry> = built
            .iter()
            .map(|(_, blas, size, first_vertex)| wgpu::BlasBuildEntry {
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
        encoder.build_acceleration_structures(entries.iter(), std::iter::once(&hw.tlas));
        drop(entries);
        for ((s, blas, _, _), &(_, level, addr, order)) in built.into_iter().zip(loads) {
            let old = self.geoms[s].resident.replace(Resident {
                level,
                addr,
                order,
                blas,
            });
            if let Some(old) = old {
                self.alloc.release(old.addr, old.order);
                self.graveyard.push((self.frame, old.blas));
            }
        }
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
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
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

/// Writes object `k`'s resident camera level's first triangle (flagged; 0:
/// the lighting level) into its transform's spare word.
fn write_first(queue: &wgpu::Queue, xforms: &wgpu::Buffer, k: u32, word: u32) {
    queue.write_buffer(xforms, (64 * k as u64) + 60, bytemuck::bytes_of(&word));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buddy_blocks_never_overlap_and_merge_back() {
        let mut b = Buddy::new(13);
        let mut held = Vec::new();
        for order in [0, 2, 1, 0, 3, 0] {
            if let Some(a) = b.alloc(order) {
                held.push((a, order));
            }
        }
        let mut used = [false; 13];
        for &(a, o) in &held {
            for u in a..a + (1 << o) {
                assert!(!used[u as usize], "unit {u} given twice");
                used[u as usize] = true;
            }
        }
        for (a, o) in held {
            b.release(a, o);
        }
        // Back to the first decomposition: 8 + 4 + 1.
        assert_eq!(b.free[3].iter().copied().collect::<Vec<_>>(), vec![0]);
        assert_eq!(b.free[2].iter().copied().collect::<Vec<_>>(), vec![8]);
        assert_eq!(b.free[0].iter().copied().collect::<Vec<_>>(), vec![12]);
        assert!(b.alloc(4).is_none());
    }
}
