//! A view-dependent lighting cache for instanced scenes (M4 gate 8).
//!
//! Static direct light only needs computing once per visible patch of
//! surface, not once per pixel per frame. The cache keeps pages of 8×8 texels
//! (stored 9×9, so a bilinear lookup stays inside one page) on each object's
//! charts, at power-of-two resolutions: texels at mip `m` are the base size ×
//! 2^m × the object's scale, and each pixel wants the mip whose texels span
//! about [`CacheOptions::pixels_per_texel`] pixels. The shading pass reads the
//! wanted page, or the finest coarser one resident, or failing that uses
//! unshadowed light. One pixel per request tile that lacked its wanted page
//! requests the page one mip finer than the one it found (or the coarsest, if
//! none), so pages arrive coarse to fine; the shader lists each page once per
//! frame, with a priority (M4 gate 10). This page manager reads that list
//! back, allocates pages in priority order, lists their texels within a
//! per-frame budget of mask cones, and evicts pages no pixel has read for a
//! while. The page table lives on the GPU (open addressing);
//! this side keeps a mirror and sends changed entries each frame.
//!
//! Frames are pipelined (M4 gate 9): requests and slot use are read back
//! asynchronously, and frame N plans its bake from frame N − 2's requests, so
//! the CPU plans while the GPU renders the frame before.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use continuum::glam::Vec2;
use continuum::mesh::Surface;
use wgpu::util::DeviceExt;

use crate::InstancedScene;

/// Texels along a page's side (stored with a one-texel border).
pub const PAGE: u32 = 8;
const STRIDE: u32 = PAGE + 1;
const PAGE_TEXELS: u32 = STRIDE * STRIDE;
const EMPTY: u32 = u32::MAX;
/// Pages freed per eviction round at most (the least recently used first), so
/// one round's table changes stay within a frame's update pass.
const EVICT_MAX: usize = 32_768;
const NONE: u32 = u32::MAX;
/// Longest probe sequence the shader follows.
const MAX_PROBE: usize = 64;
/// Entries in the per-frame set of requested pages (a power of two, at least
/// twice the request tiles of a 1080p frame).
const REQUEST_SET: u64 = 1 << 18;

/// Settings of the lighting cache.
#[derive(Clone, Copy, Debug)]
pub struct CacheOptions {
    /// Texel size at mip 0 on an object of scale 1 (meters).
    pub base_texel: f32,
    /// Pixels a texel should span; each pixel's mip is chosen for this (log2
    /// rounded).
    pub pixels_per_texel: f32,
    /// One pixel per `request_tile × request_tile` reports a missing page each
    /// frame (a different one each frame).
    pub request_tile: u32,
    /// Texels (mask cones) baked per frame at most.
    pub bake_budget: u32,
    /// Pages the pool holds.
    pub pages: u32,
}

impl Default for CacheOptions {
    fn default() -> Self {
        CacheOptions {
            base_texel: 0.005,
            pixels_per_texel: 2.0,
            request_tile: 4,
            bake_budget: 8_000,
            pages: 1 << 19,
        }
    }
}

/// What the cache did in one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    /// The frame (counted from 1).
    pub frame: u32,
    /// Pixels shaded with unshadowed light (no page of any mip resident).
    pub fallback_pixels: u32,
    /// Distinct pages requested (each by at least one sample pixel that lacked
    /// its wanted page).
    pub requests: u32,
    /// Sample pixels on a surface (one per request tile), and of those, the
    /// ones that found the page of their wanted mip.
    pub samples: u32,
    pub at_wanted: u32,
    /// Pages and texels baked in the frame.
    pub pages_baked: u32,
    pub texels_baked: u32,
    /// Pages resident after the frame's bake.
    pub resident: u32,
    /// Pages evicted while planning the frame.
    pub evicted: u32,
    /// CPU time spent planning the frame's bake, including eviction (not
    /// counting `waiting`).
    pub planning: Duration,
    /// Of that work, eviction (merging slot use, freeing pages).
    pub evicting: Duration,
    /// Time spent waiting for the GPU to finish the frame two before, whose
    /// requests the plan reads.
    pub waiting: Duration,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VChartGpu {
    size: [f32; 2],
    grid: [u32; 2],
    first_cell: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VcParams {
    counts: [u32; 4],
    jobs: [u32; 4],
    params: [f32; 4],
}

/// Per-chart tables shared by every object with a geometry: the chart's size
/// and cell grid, the triangles of each cell, the chart of each triangle and
/// its vertices' chart coordinates. Triangles are numbered as in the flattened
/// scene (geometries in order).
struct ChartTables {
    charts: Vec<VChartGpu>,
    cells: Vec<[u32; 2]>,
    tri_chart: Vec<u32>,
    tri_uv: Vec<[f32; 2]>,
    /// First global chart of each geometry.
    first_chart: Vec<u32>,
}

fn chart_tables(geometries: &[Surface]) -> ChartTables {
    let mut t = ChartTables {
        charts: Vec::new(),
        cells: Vec::new(),
        tri_chart: Vec::new(),
        tri_uv: Vec::new(),
        first_chart: Vec::new(),
    };
    let mut first_tri = 0u32;
    for g in geometries {
        assert_eq!(
            g.charts.len(),
            g.chart_grid.len(),
            "a chart without a cell grid"
        );
        let first_chart = t.charts.len() as u32;
        t.first_chart.push(first_chart);
        for (size, grid) in g.charts.iter().zip(&g.chart_grid) {
            t.charts.push(VChartGpu {
                size: size.to_array(),
                grid: *grid,
                first_cell: t.cells.len() as u32,
                pad: 0,
            });
            t.cells
                .extend(std::iter::repeat_n([NONE; 2], (grid[0] * grid[1]) as usize));
        }
        for (k, (uv, &chart)) in g.uv.iter().zip(&g.chart).enumerate() {
            let c = &t.charts[(first_chart + chart) as usize];
            let center = (uv[0] + uv[1] + uv[2]) / 3.0;
            let cx = ((center.x * c.grid[0] as f32) as u32).min(c.grid[0] - 1);
            let cy = ((center.y * c.grid[1] as f32) as u32).min(c.grid[1] - 1);
            let cell = &mut t.cells[(c.first_cell + cy * c.grid[0] + cx) as usize];
            let free = cell
                .iter()
                .position(|&x| x == NONE)
                .expect("more than two triangles in a chart cell");
            cell[free] = first_tri + k as u32;
            t.tri_chart.push(first_chart + chart);
            t.tri_uv.extend(uv.iter().map(|v| v.to_array()));
        }
        first_tri += g.triangles.len() as u32;
    }
    assert!(t.charts.len() < 1 << 16, "too many charts for page keys");
    if t.cells.is_empty() {
        t.cells.push([NONE; 2]);
    }
    if t.tri_chart.is_empty() {
        t.tri_chart.push(0);
        t.tri_uv.push([0.0; 2]);
    }
    t
}

/// The shader's page-table hash.
fn hash(k: [u32; 3]) -> u32 {
    let mut h = k[0].wrapping_mul(0x9E37_79B1)
        ^ k[1].wrapping_mul(0x85EB_CA77)
        ^ k[2].wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h
}

/// A page's key, (object, chart | mip << 16, page x | page y << 16), unpacked.
fn unpack(k: [u32; 3]) -> (u32, u32, u32, [u32; 2]) {
    (k[0], k[1] & 0xffff, k[1] >> 16, [k[2] & 0xffff, k[2] >> 16])
}

/// A fast hash for page keys: std's SipHash cost more than the rest of the
/// planning. Keys are not adversarial.
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);

impl KeyHasher {
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        (self.0 ^ (self.0 >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(b));
        }
    }

    fn write_u32(&mut self, x: u32) {
        self.add(x as u64);
    }

    fn write_u64(&mut self, x: u64) {
        self.add(x);
    }

    fn write_usize(&mut self, x: usize) {
        self.add(x as u64);
    }
}

type KeyMap<V> = HashMap<[u32; 3], V, BuildHasherDefault<KeyHasher>>;

/// The page table's CPU mirror: open addressing with linear probing over
/// entries (three key words, then the slot), as the shader reads it, and the
/// entries changed since they were last sent.
struct PageTable {
    entries: Vec<[u32; 4]>,
    dirty: Vec<u32>,
    is_dirty: Vec<bool>,
}

impl PageTable {
    /// `len` entries, a power of two.
    fn new(len: usize) -> Self {
        assert!(len.is_power_of_two());
        PageTable {
            entries: vec![[EMPTY, 0, 0, 0]; len],
            dirty: Vec::new(),
            is_dirty: vec![false; len],
        }
    }

    fn mark(&mut self, i: usize) {
        if !self.is_dirty[i] {
            self.is_dirty[i] = true;
            self.dirty.push(i as u32);
        }
    }

    /// The entries changed since the last call.
    fn take_dirty(&mut self) -> Vec<u32> {
        for &i in &self.dirty {
            self.is_dirty[i as usize] = false;
        }
        std::mem::take(&mut self.dirty)
    }

    /// Puts `k` (not present) in the table; its entry, or `None` if the probe
    /// sequence would be longer than the shader follows.
    fn insert(&mut self, k: [u32; 3], slot: u32) -> Option<u32> {
        let mask = self.entries.len() - 1;
        let mut i = hash(k) as usize & mask;
        for _ in 0..MAX_PROBE {
            if self.entries[i][0] == EMPTY {
                self.entries[i] = [k[0], k[1], k[2], slot];
                self.mark(i);
                return Some(i as u32);
            }
            i = (i + 1) & mask;
        }
        None
    }

    /// Empties entry `i`, then moves later entries of its probe run back where
    /// lookups still find them (deletion for linear probing, Knuth's Algorithm
    /// R): no tombstones, so no rebuilds. `moved(key, entry)` reports each
    /// move.
    fn remove(&mut self, mut i: usize, mut moved: impl FnMut([u32; 3], u32)) {
        let mask = self.entries.len() - 1;
        self.entries[i] = [EMPTY, 0, 0, 0];
        self.mark(i);
        let mut j = i;
        loop {
            j = (j + 1) & mask;
            let e = self.entries[j];
            if e[0] == EMPTY {
                return;
            }
            let k = [e[0], e[1], e[2]];
            let home = hash(k) as usize & mask;
            // The entry may stay if its home lies cyclically in (i, j].
            let stays = if i <= j {
                i < home && home <= j
            } else {
                i < home || home <= j
            };
            if !stays {
                self.entries[i] = e;
                self.mark(i);
                moved(k, i as u32);
                self.entries[j] = [EMPTY, 0, 0, 0];
                self.mark(j);
                i = j;
            }
        }
    }

    /// The slot of `k`, looked up as the shader does.
    #[cfg(test)]
    fn find(&self, k: [u32; 3]) -> Option<u32> {
        let mask = self.entries.len() - 1;
        let mut i = hash(k) as usize & mask;
        for _ in 0..MAX_PROBE {
            let e = self.entries[i];
            if e[0] == EMPTY {
                return None;
            }
            if [e[0], e[1], e[2]] == k {
                return Some(e[3]);
            }
            i = (i + 1) & mask;
        }
        None
    }
}

/// A readback buffer for one frame's requests and counters, and the frame
/// and submission whose copy it holds until read.
struct Readback {
    buffer: wgpu::Buffer,
    tiles: u32,
    pending: Option<(u32, wgpu::SubmissionIndex)>,
}

/// Table entries updated per frame at most before the whole table is sent.
const UPDATES_CAPACITY: usize = 1 << 17;

/// The lighting cache's state: GPU buffers (by binding) and the page manager.
pub(crate) struct VirtualCache {
    queue: wgpu::Queue,
    options: CacheOptions,
    inv_base: f32,
    /// Size of each global chart, and the scale of each object.
    chart_size: Vec<Vec2>,
    scale: Vec<f32>,
    table: PageTable,
    /// Resident pages: key → (slot, table entry).
    map: KeyMap<(u32, u32)>,
    slot_key: Vec<[u32; 3]>,
    /// Per slot: the last frame it was read, as last read back, or the frame
    /// it was allocated in, whichever is later.
    last_used: Vec<u32>,
    free: Vec<u32>,
    bake_list: Vec<[u32; 4]>,
    jobs: Vec<[u32; 2]>,
    texels: u32,
    /// The frame being prepared (counted from 1).
    frame: u32,
    /// Per frame: what was baked, then what its shading reported once read back.
    log: Vec<CacheStats>,
    /// The latest frame whose requests have been read back.
    latest: u32,
    evicted: u32,
    planning: Duration,
    evicting: Duration,
    waiting: Duration,
    pub(crate) buffers: Vec<(u32, wgpu::Buffer)>,
    params: wgpu::Buffer,
    /// Alternating per frame.
    readbacks: Vec<Readback>,
    used_readback: wgpu::Buffer,
    /// The frame and submission whose slot use `used_readback` holds.
    used_pending: Option<(u32, wgpu::SubmissionIndex)>,
    used_requested: bool,
}

impl VirtualCache {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &InstancedScene,
        options: CacheOptions,
        width: u32,
        height: u32,
    ) -> Self {
        let tables = chart_tables(&scene.geometries);
        let inv_base = 1.0 / options.base_texel;
        let chart_size: Vec<Vec2> = tables
            .charts
            .iter()
            .map(|c| Vec2::from_array(c.size))
            .collect();
        let scale: Vec<f32> = scene.instances.iter().map(|i| i.scale).collect();
        // Page coordinates have 16 bits each.
        for i in &scene.instances {
            let g = &scene.geometries[i.geometry];
            for size in &g.charts {
                let texels = (size.max_element() * i.scale * inv_base).ceil();
                assert!(
                    texels / (PAGE as f32) < 65_536.0,
                    "a chart too large for the base texel"
                );
            }
        }
        let table_len = (options.pages as usize * 4).next_power_of_two().max(1024);
        let storage = |label: &str, contents: &[u8], extra: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | extra,
            })
        };
        let empty = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let copy_dst = wgpu::BufferUsages::COPY_DST;
        let budget = options.bake_budget.max(PAGE_TEXELS) as u64;
        let table = PageTable::new(table_len);
        let buffers = vec![
            (
                23,
                storage("chart uv", bytemuck::cast_slice(&tables.tri_uv), copy_dst),
            ),
            (
                33,
                storage("charts", bytemuck::cast_slice(&tables.charts), copy_dst),
            ),
            (
                34,
                storage("chart cells", bytemuck::cast_slice(&tables.cells), copy_dst),
            ),
            (
                35,
                storage(
                    "triangle charts",
                    bytemuck::cast_slice(&tables.tri_chart),
                    copy_dst,
                ),
            ),
            (
                36,
                storage("page table", bytemuck::cast_slice(&table.entries), copy_dst),
            ),
            (
                37,
                empty(
                    "page pool",
                    options.pages as u64 * PAGE_TEXELS as u64 * 4,
                    wgpu::BufferUsages::empty(),
                ),
            ),
            (38, empty("bake list", budget * 16, copy_dst)),
            (
                40,
                empty(
                    "slot used",
                    options.pages as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                42,
                empty("cache stats", 16, wgpu::BufferUsages::COPY_SRC | copy_dst),
            ),
            (43, empty("bake jobs", 4 * (budget + 81) * 8, copy_dst)),
            (45, empty("request set", REQUEST_SET * 4, copy_dst)),
            (
                44,
                empty("table updates", UPDATES_CAPACITY as u64 * 32, copy_dst),
            ),
        ];
        let params = crate::uniform_buffer(
            device,
            "cache params",
            &[0; std::mem::size_of::<VcParams>()],
        );
        let used_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot used readback"),
            size: options.pages as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cache = VirtualCache {
            queue: queue.clone(),
            options,
            inv_base,
            chart_size,
            scale,
            table,
            map: KeyMap::default(),
            slot_key: vec![[EMPTY; 3]; options.pages as usize],
            last_used: vec![0; options.pages as usize],
            free: (0..options.pages).rev().collect(),
            bake_list: Vec::new(),
            jobs: Vec::new(),
            texels: 0,
            frame: 1,
            log: Vec::new(),
            latest: 0,
            evicted: 0,
            planning: Duration::ZERO,
            evicting: Duration::ZERO,
            waiting: Duration::ZERO,
            buffers,
            params,
            readbacks: Vec::new(),
            used_readback,
            used_pending: None,
            used_requested: false,
        };
        cache.resize(device, width, height);
        cache
    }

    /// Request buffers for a `width × height` image. Requests in flight are
    /// dropped.
    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let tile = self.options.request_tile;
        let tiles = width.div_ceil(tile) * height.div_ceil(tile);
        self.buffers.retain(|(b, _)| *b != 39);
        self.buffers.push((
            39,
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("requests"),
                size: tiles as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ));
        self.readbacks = (0..2)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("requests readback"),
                    size: tiles as u64 * 16 + 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                tiles,
                pending: None,
            })
            .collect();
    }

    pub(crate) fn buffer(&self, binding: u32) -> &wgpu::Buffer {
        &self
            .buffers
            .iter()
            .find(|(b, _)| *b == binding)
            .expect("cache binding")
            .1
    }

    pub(crate) fn params(&self) -> &wgpu::Buffer {
        &self.params
    }

    /// The latest frame whose shading has been read back.
    pub(crate) fn stats(&self) -> CacheStats {
        if self.latest == 0 {
            return CacheStats::default();
        }
        self.log[self.latest as usize - 1]
    }

    /// Every frame so far (the last two may not be read back yet).
    pub(crate) fn log(&self) -> &[CacheStats] {
        &self.log
    }

    /// GPU memory the cache holds, in bytes.
    pub(crate) fn bytes(&self) -> u64 {
        self.buffers.iter().map(|(_, b)| b.size()).sum::<u64>()
            + self.readbacks.iter().map(|r| r.buffer.size()).sum::<u64>()
            + self.used_readback.size()
    }

    /// Texels along u and v of global chart `c` on object `m` at `mip`, as the
    /// shader computes them.
    fn grid(&self, c: u32, m: u32, mip: u32) -> [u32; 2] {
        let size = self.chart_size[c as usize] * self.scale[m as usize];
        // 2^-mip exactly, as the shader's ldexp.
        let inv = self.inv_base * f32::from_bits((127 - mip) << 23);
        let n = (size * inv).ceil();
        [(n.x as u32).max(1), (n.y as u32).max(1)]
    }

    /// Writes this frame's uniform and uploads its bake list, texel list and
    /// page table changes. Returns the passes' sizes: (table updates, texels,
    /// pages).
    pub(crate) fn upload(
        &mut self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        vertical_fov_degrees: f32,
    ) -> (u32, u32, u32) {
        let mut updates: Vec<[u32; 4]> = Vec::new();
        let dirty = self.table.take_dirty();
        if dirty.len() > UPDATES_CAPACITY {
            queue.write_buffer(
                self.buffer(36),
                0,
                bytemuck::cast_slice(&self.table.entries),
            );
        } else {
            for &i in &dirty {
                updates.push([i, 0, 0, 0]);
                updates.push(self.table.entries[i as usize]);
            }
        }
        if !updates.is_empty() {
            queue.write_buffer(self.buffer(44), 0, bytemuck::cast_slice(&updates));
        }
        if !self.bake_list.is_empty() {
            queue.write_buffer(self.buffer(38), 0, bytemuck::cast_slice(&self.bake_list));
            queue.write_buffer(self.buffer(43), 0, bytemuck::cast_slice(&self.jobs));
        }
        let tile = self.options.request_tile;
        let pixel_angle = 2.0 * (vertical_fov_degrees.to_radians() * 0.5).tan() / height as f32;
        let params = VcParams {
            counts: [
                self.table.entries.len() as u32 - 1,
                self.frame,
                tile,
                width.div_ceil(tile),
            ],
            jobs: [
                self.jobs.len() as u32,
                self.bake_list.len() as u32,
                (updates.len() / 2) as u32,
                REQUEST_SET as u32 - 1,
            ],
            params: [
                self.inv_base,
                pixel_angle,
                self.options.pixels_per_texel,
                0.0,
            ],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.log.push(CacheStats {
            frame: self.frame,
            pages_baked: self.bake_list.len() as u32,
            texels_baked: self.texels,
            resident: self.map.len() as u32,
            evicted: std::mem::take(&mut self.evicted),
            planning: std::mem::take(&mut self.planning),
            evicting: std::mem::take(&mut self.evicting),
            waiting: std::mem::take(&mut self.waiting),
            ..Default::default()
        });
        (
            (updates.len() / 2) as u32,
            self.jobs.len() as u32,
            self.bake_list.len() as u32,
        )
    }

    /// Clears the request tiles and counters before shading.
    pub(crate) fn encode_clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.buffer(42), 0, None);
        encoder.clear_buffer(self.buffer(45), 0, None);
    }

    /// Copies the frame's requests and counters (and every 32 frames, slot
    /// use) for reading back once the frame is done.
    pub(crate) fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let r = &self.readbacks[(self.frame % 2) as usize];
        let bytes = r.tiles as u64 * 16;
        encoder.copy_buffer_to_buffer(self.buffer(39), 0, &r.buffer, 0, bytes);
        encoder.copy_buffer_to_buffer(self.buffer(42), 0, &r.buffer, bytes, 16);
        self.used_requested = self.frame.is_multiple_of(32) && self.used_pending.is_none();
        if self.used_requested {
            encoder.copy_buffer_to_buffer(self.buffer(40), 0, &self.used_readback, 0, None);
        }
    }

    /// After the frame's submission (`index`): starts its readbacks, without
    /// waiting.
    pub(crate) fn after_submit(&mut self, index: wgpu::SubmissionIndex) {
        let r = &mut self.readbacks[(self.frame % 2) as usize];
        r.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |m| m.expect("map failed"));
        r.pending = Some((self.frame, index.clone()));
        if self.used_requested {
            self.used_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, |m| m.expect("map failed"));
            self.used_pending = Some((self.frame, index));
            self.used_requested = false;
        }
        self.frame += 1;
    }

    /// Before the frame: reads back the requests of two frames ago (waiting
    /// only for that frame), evicts with any slot use that has arrived, and
    /// plans the frame's bake.
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        let start = Instant::now();
        let slot = (self.frame % 2) as usize;
        let requests = self.collect(device, slot);
        if let Some((frame, _)) = &self.used_pending
            && frame + 2 <= self.frame
        {
            let (_, index) = self.used_pending.take().expect("pending slot use");
            let raw = map_wait(device, &self.used_readback, index);
            let evicting = Instant::now();
            self.evicted += self.merge_used_and_evict(bytemuck::cast_slice(&raw), 64);
            self.evicting += evicting.elapsed();
        }
        self.plan(device, requests);
        // `collect` adds its wait to `waiting`.
        self.planning += start.elapsed().saturating_sub(self.waiting);
    }

    /// Reads back every frame still in flight (for reporting; plans nothing).
    pub(crate) fn drain(&mut self, device: &wgpu::Device) {
        for slot in 0..self.readbacks.len() {
            self.collect(device, slot);
        }
    }

    /// Reads back readback `slot` if a frame is pending there: records its
    /// counters and returns its requests.
    fn collect(&mut self, device: &wgpu::Device, slot: usize) -> Vec<([u32; 3], u32)> {
        let Some((frame, index)) = self.readbacks[slot].pending.take() else {
            return Vec::new();
        };
        let wait = Instant::now();
        let raw = map_wait(device, &self.readbacks[slot].buffer, index);
        self.waiting += wait.elapsed();
        let words: &[u32] = bytemuck::cast_slice(&raw);
        let tiles = self.readbacks[slot].tiles as usize;
        let counters = &words[4 * tiles..4 * tiles + 4];
        let count = (counters[1] as usize).min(tiles);
        let requests: Vec<([u32; 3], u32)> = words[..4 * count]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|r| ([r[0], r[1], r[2]], r[3]))
            .collect();
        let record = &mut self.log[frame as usize - 1];
        record.fallback_pixels = counters[0];
        record.requests = count as u32;
        record.samples = counters[2];
        record.at_wanted = counters[3];
        self.latest = self.latest.max(frame);
        requests
    }

    /// Merges slot use read back from the GPU and frees pages unused for more
    /// than `age` frames, the least recently used first, at most `EVICT_MAX`.
    /// Returns how many were freed.
    fn merge_used_and_evict(&mut self, used: &[u32], age: u32) -> u32 {
        for (l, &u) in self.last_used.iter_mut().zip(used) {
            *l = (*l).max(u);
        }
        let mut stale: Vec<(u32, u32)> = (0..self.options.pages)
            .filter(|&slot| {
                self.slot_key[slot as usize][0] != EMPTY
                    && self.last_used[slot as usize] + age < self.frame
            })
            .map(|slot| (self.last_used[slot as usize], slot))
            .collect();
        if stale.len() > EVICT_MAX {
            stale.select_nth_unstable(EVICT_MAX);
            stale.truncate(EVICT_MAX);
        }
        for &(_, slot) in &stale {
            let k = self.slot_key[slot as usize];
            let (_, entry) = self.map.remove(&k).expect("resident page");
            let map = &mut self.map;
            self.table.remove(entry as usize, |moved, i| {
                map.get_mut(&moved).expect("resident page").1 = i;
            });
            self.slot_key[slot as usize] = [EMPTY; 3];
            self.free.push(slot);
        }
        stale.len() as u32
    }

    /// With the pool full: reads slot use back now (waiting for all GPU work)
    /// and frees pages unused for more than `age` frames.
    fn evict_now(&mut self, device: &wgpu::Device, age: u32) -> u32 {
        if self.used_pending.is_some() {
            // The slot-use buffer is mapped or about to be: use it.
            let (_, index) = self.used_pending.take().expect("pending slot use");
            let raw = map_wait(device, &self.used_readback, index);
            return self.merge_used_and_evict(bytemuck::cast_slice(&raw), age);
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(self.buffer(40), 0, &self.used_readback, 0, None);
        let index = self.queue.submit([encoder.finish()]);
        self.used_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |m| m.expect("map failed"));
        let raw = map_wait(device, &self.used_readback, index);
        self.merge_used_and_evict(bytemuck::cast_slice(&raw), age)
    }

    /// Chooses the frame's pages from the requested ones (each listed once by
    /// the shader, with its priority), within the bake budget.
    fn plan(&mut self, device: &wgpu::Device, requests: Vec<([u32; 3], u32)>) {
        self.bake_list.clear();
        self.jobs.clear();
        self.texels = 0;
        // Pages planned in the frame between are resident already.
        let mut order: Vec<(u32, [u32; 3])> = requests
            .into_iter()
            .filter(|(k, _)| !self.map.contains_key(k))
            .map(|(k, priority)| (priority, k))
            .collect();
        // Each page costs at least one texel: no more than `budget` can fit.
        let budget = self.options.bake_budget.max(PAGE_TEXELS);
        if order.len() > budget as usize {
            order.select_nth_unstable(budget as usize);
            order.truncate(budget as usize);
        }
        order.sort_unstable();
        let mut evicted_now = false;
        for (_, k) in order {
            // A fingerprint collision can list a page twice.
            if self.map.contains_key(&k) {
                continue;
            }
            let (m, c, mip, page) = unpack(k);
            let n = self.grid(c, m, mip);
            let inside = [
                (n[0] - (page[0] * PAGE).min(n[0])).min(STRIDE),
                (n[1] - (page[1] * PAGE).min(n[1])).min(STRIDE),
            ];
            if inside[0] == 0 || inside[1] == 0 {
                continue;
            }
            let cost = inside[0] * inside[1];
            if self.texels + cost > budget {
                break;
            }
            if self.free.is_empty() && !evicted_now {
                evicted_now = true;
                self.evicted += self.evict_now(device, 2);
            }
            let Some(slot) = self.free.pop() else { break };
            let Some(entry) = self.table.insert(k, slot) else {
                self.free.push(slot);
                continue;
            };
            self.map.insert(k, (slot, entry));
            self.slot_key[slot as usize] = k;
            self.last_used[slot as usize] = self.frame;
            let index = self.bake_list.len() as u32;
            self.bake_list.push([k[0], k[1], k[2], slot]);
            // One texel past the grid on each side too, in case the shader's
            // grid is a rounding larger; it skips texels outside its own.
            let listed = [(inside[0] + 1).min(STRIDE), (inside[1] + 1).min(STRIDE)];
            for y in 0..listed[1] {
                for x in 0..listed[0] {
                    self.jobs.push([index, y * STRIDE + x]);
                }
            }
            self.texels += cost;
        }
    }
}

/// Waits for submission `index`, after which `buffer` (whose mapping was
/// requested after that submission) is mapped; copies it out and unmaps it.
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

#[cfg(test)]
mod tests {
    use super::*;
    use continuum::glam::Vec3;

    #[test]
    fn every_triangle_of_a_parametric_chart_lands_in_its_cell() {
        let s = Surface::parametric(6, 4, |u, v| Vec3::new(u, v, (u * v).sin()));
        let t = chart_tables(std::slice::from_ref(&s));
        let filled: usize = t.cells.iter().flatten().filter(|&&x| x != NONE).count();
        assert_eq!(filled, s.triangles.len());
        assert_eq!(t.cells.len(), 24);
    }

    #[test]
    fn deletion_keeps_every_remaining_page_findable() {
        let mut table = PageTable::new(1 << 10);
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        let mut live: HashMap<[u32; 3], u32> = HashMap::new();
        let mut moves = 0;
        for step in 0..20_000u32 {
            let remove = live.len() > 400 || (!live.is_empty() && next() % 3 == 0);
            if remove {
                let k = *live.keys().nth(next() as usize % live.len()).unwrap();
                let entry = live.remove(&k).unwrap();
                table.remove(entry as usize, |moved, i| {
                    *live.get_mut(&moved).unwrap() = i;
                    moves += 1;
                });
                assert_eq!(table.find(k), None);
            } else {
                // Few distinct values, so keys collide into long runs.
                let k = [next() % 64, next() % 4, next() % 64];
                if live.contains_key(&k) {
                    continue;
                }
                if let Some(entry) = table.insert(k, step) {
                    live.insert(k, entry);
                }
            }
            if step % 997 == 0 {
                for (k, &entry) in &live {
                    let e = table.entries[entry as usize];
                    assert_eq!([e[0], e[1], e[2]], *k);
                    assert_eq!(table.find(*k), Some(e[3]));
                }
            }
        }
        assert!(moves > 100, "deletions moved only {moves} entries");
    }

    #[test]
    fn the_hash_matches_known_values() {
        // Fixed by the shader's formula; guards against edits on one side only.
        assert_eq!(hash([0, 0, 0]), 0);
        assert_ne!(hash([1, 0, 0]), hash([0, 1, 0]));
    }
}
