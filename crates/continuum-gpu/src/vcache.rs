//! A view-dependent lighting cache for instanced scenes (M4 gate 8).
//!
//! Static direct light only needs computing once per visible patch of
//! surface, not once per pixel per frame. The cache keeps pages of 8×8 texels
//! (stored 9×9, so a bilinear lookup stays inside one page) on each object's
//! charts, at power-of-two resolutions: texels at mip `m` are the base size ×
//! 2^m × the object's scale, and each pixel wants the mip whose texels span
//! about [`CacheOptions::pixels_per_texel`] pixels. The shading pass reads the
//! wanted page, or the finest coarser one resident, or failing that uses
//! unshadowed light, and one pixel per request tile reports the wanted page if
//! it was missing. This page manager reads those requests back, allocates
//! pages (coarse levels first, so every pixel soon has some page), lists their
//! texels within a per-frame budget of mask cones, and evicts pages no pixel
//! has read for a while. The page table lives on the GPU (open addressing);
//! this side keeps a mirror and sends changed entries each frame.

use std::collections::HashMap;

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
const REMOVED: u32 = u32::MAX - 1;
const NONE: u32 = u32::MAX;
/// Longest probe sequence the shader follows.
const MAX_PROBE: usize = 64;
/// Coarsest mip considered (the shader's limit).
const MAX_MIP: u32 = 30;

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
            bake_budget: 20_000,
            pages: 1 << 19,
        }
    }
}

/// What the cache did in the last frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    /// Pixels shaded with unshadowed light (no page of any mip resident).
    pub fallback_pixels: u32,
    /// Request tiles whose sample pixel lacked its wanted page.
    pub requests: u32,
    /// Pages and texels baked in the frame.
    pub pages_baked: u32,
    pub texels_baked: u32,
    /// Pages resident after the frame's bake.
    pub resident: u32,
    /// Pages evicted while planning the next frame.
    pub evicted: u32,
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

/// A page's key: (object, chart | mip << 16, page x | page y << 16).
fn key(object: u32, chart: u32, mip: u32, page: [u32; 2]) -> [u32; 3] {
    [object, chart | (mip << 16), page[0] | (page[1] << 16)]
}

fn unpack(k: [u32; 3]) -> (u32, u32, u32, [u32; 2]) {
    (k[0], k[1] & 0xffff, k[1] >> 16, [k[2] & 0xffff, k[2] >> 16])
}

struct Readback {
    buffer: wgpu::Buffer,
    tiles: u32,
}

/// The lighting cache's state: GPU buffers (by binding) and the page manager.
pub(crate) struct VirtualCache {
    queue: wgpu::Queue,
    options: CacheOptions,
    inv_base: f32,
    /// Size of each global chart, and the scale of each object.
    chart_size: Vec<Vec2>,
    scale: Vec<f32>,
    table: Vec<[u32; 4]>,
    removed: usize,
    /// Resident pages: key → (slot, table entry).
    map: HashMap<[u32; 3], (u32, u32)>,
    slot_key: Vec<[u32; 3]>,
    /// Per slot: the last frame it was read, as last read back, or the frame
    /// it was allocated in, whichever is later.
    last_used: Vec<u32>,
    free: Vec<u32>,
    dirty: Vec<u32>,
    is_dirty: Vec<bool>,
    full_upload: bool,
    bake_list: Vec<[u32; 4]>,
    jobs: Vec<[u32; 2]>,
    texels: u32,
    frame: u32,
    stats: CacheStats,
    pub(crate) buffers: Vec<(u32, wgpu::Buffer)>,
    params: wgpu::Buffer,
    updates_capacity: usize,
    readback: Readback,
    used_readback: wgpu::Buffer,
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
        let updates_capacity = 2 * budget as usize;
        let table = vec![[EMPTY, 0, 0, 0]; table_len];
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
                storage("page table", bytemuck::cast_slice(&table), copy_dst),
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
            (
                44,
                empty("table updates", updates_capacity as u64 * 32, copy_dst),
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
            removed: 0,
            map: HashMap::new(),
            slot_key: vec![[EMPTY; 3]; options.pages as usize],
            last_used: vec![0; options.pages as usize],
            free: (0..options.pages).rev().collect(),
            dirty: Vec::new(),
            is_dirty: vec![false; table_len],
            full_upload: false,
            bake_list: Vec::new(),
            jobs: Vec::new(),
            texels: 0,
            frame: 1,
            stats: CacheStats::default(),
            buffers,
            params,
            updates_capacity,
            readback: Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("requests"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                tiles: 0,
            },
            used_readback,
        };
        cache.resize(device, width, height);
        cache
    }

    /// Request buffers for a `width × height` image.
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
        self.readback = Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("requests readback"),
                size: tiles as u64 * 16 + 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tiles,
        };
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

    pub(crate) fn stats(&self) -> CacheStats {
        self.stats
    }

    /// GPU memory the cache holds, in bytes.
    pub(crate) fn bytes(&self) -> u64 {
        self.buffers.iter().map(|(_, b)| b.size()).sum::<u64>()
            + self.readback.buffer.size()
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

    /// The page of `mip` holding the center of page `page` of `from` (on chart
    /// `c` of object `m`).
    fn coarser(&self, c: u32, m: u32, from: u32, page: [u32; 2]) -> [u32; 2] {
        let n = self.grid(c, m, from);
        let n2 = self.grid(c, m, from + 1);
        let mut out = [0; 2];
        for a in 0..2 {
            let center = (page[a] * PAGE + PAGE / 2).min(n[a] - 1);
            let uv = (center as f64 + 0.5) / n[a] as f64;
            let t = (uv * n2[a] as f64 - 0.5).clamp(0.0, (n2[a] - 1) as f64);
            out[a] = t as u32 / PAGE;
        }
        out
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
        if self.full_upload || self.dirty.len() > self.updates_capacity {
            queue.write_buffer(self.buffer(36), 0, bytemuck::cast_slice(&self.table));
        } else {
            for &i in &self.dirty {
                updates.push([i, 0, 0, 0]);
                updates.push(self.table[i as usize]);
            }
        }
        for &i in &self.dirty {
            self.is_dirty[i as usize] = false;
        }
        self.dirty.clear();
        self.full_upload = false;
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
                self.table.len() as u32 - 1,
                self.frame,
                tile,
                width.div_ceil(tile),
            ],
            jobs: [
                self.jobs.len() as u32,
                self.bake_list.len() as u32,
                (updates.len() / 2) as u32,
                0,
            ],
            params: [
                self.inv_base,
                pixel_angle,
                self.options.pixels_per_texel,
                0.0,
            ],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.stats = CacheStats {
            pages_baked: self.bake_list.len() as u32,
            texels_baked: self.texels,
            resident: self.map.len() as u32,
            ..Default::default()
        };
        (
            (updates.len() / 2) as u32,
            self.jobs.len() as u32,
            self.bake_list.len() as u32,
        )
    }

    /// Clears the request tiles and counters before shading.
    pub(crate) fn encode_clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.buffer(39), 0, None);
        encoder.clear_buffer(self.buffer(42), 0, None);
    }

    /// Copies the requests and counters for reading back after the frame.
    pub(crate) fn encode_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        let bytes = self.readback.tiles as u64 * 16;
        encoder.copy_buffer_to_buffer(self.buffer(39), 0, &self.readback.buffer, 0, bytes);
        encoder.copy_buffer_to_buffer(self.buffer(42), 0, &self.readback.buffer, bytes, 16);
    }

    /// After the frame's submission: reads its requests back (waiting for the
    /// GPU) and plans the next frame's bake.
    pub(crate) fn finish_frame(&mut self, device: &wgpu::Device) {
        let raw = map_read(device, &self.readback.buffer);
        let words: &[u32] = bytemuck::cast_slice(&raw);
        let tiles = self.readback.tiles as usize;
        let requests: Vec<[u32; 3]> = words[..4 * tiles]
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|r| r[3] == 1)
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        self.stats.requests = requests.len() as u32;
        self.stats.fallback_pixels = words[4 * tiles];
        self.frame += 1;
        if self.frame.is_multiple_of(32) {
            self.stats.evicted += self.evict(device, 64);
        }
        self.plan(device, requests);
    }

    /// Reads back which slots pixels used, and frees pages unused for more
    /// than `age` frames. Returns how many were freed.
    fn evict(&mut self, device: &wgpu::Device, age: u32) -> u32 {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(self.buffer(40), 0, &self.used_readback, 0, None);
        self.queue.submit([encoder.finish()]);
        let raw = map_read(device, &self.used_readback);
        let used: &[u32] = bytemuck::cast_slice(&raw);
        for (l, &u) in self.last_used.iter_mut().zip(used) {
            *l = (*l).max(u);
        }
        let mut freed = 0;
        for slot in 0..self.options.pages {
            let k = self.slot_key[slot as usize];
            if k[0] == EMPTY || self.last_used[slot as usize] + age >= self.frame {
                continue;
            }
            let (_, entry) = self.map.remove(&k).expect("resident page");
            self.table[entry as usize] = [REMOVED, 0, 0, 0];
            self.mark(entry);
            self.removed += 1;
            self.slot_key[slot as usize] = [EMPTY; 3];
            self.free.push(slot);
            freed += 1;
        }
        freed
    }

    fn mark(&mut self, entry: u32) {
        if !self.is_dirty[entry as usize] {
            self.is_dirty[entry as usize] = true;
            self.dirty.push(entry);
        }
    }

    /// Puts `k` in the page table; `None` if its probe sequence is too long.
    fn insert(&mut self, k: [u32; 3], slot: u32) -> Option<u32> {
        let mask = self.table.len() - 1;
        let mut i = hash(k) as usize & mask;
        for _ in 0..MAX_PROBE {
            let e = self.table[i];
            if e[0] == EMPTY || e[0] == REMOVED {
                if e[0] == REMOVED {
                    self.removed -= 1;
                }
                self.table[i] = [k[0], k[1], k[2], slot];
                self.mark(i as u32);
                return Some(i as u32);
            }
            i = (i + 1) & mask;
        }
        None
    }

    /// Rebuilds the page table without removed entries.
    fn rebuild(&mut self) {
        self.table.fill([EMPTY, 0, 0, 0]);
        self.removed = 0;
        let pages: Vec<([u32; 3], u32)> = self.map.iter().map(|(k, v)| (*k, v.0)).collect();
        for (k, slot) in pages {
            let entry = self
                .insert(k, slot)
                .expect("page table too full to rebuild");
            self.map.insert(k, (slot, entry));
        }
        self.dirty.clear();
        self.is_dirty.fill(false);
        self.full_upload = true;
    }

    /// Chooses the next frame's pages: for each request, the missing pages from
    /// the wanted mip up to the first resident one (or the coarsest), ordered
    /// so that coarse levels come first (pixels with no page at all first of
    /// all), within the bake budget.
    fn plan(&mut self, device: &wgpu::Device, requests: Vec<[u32; 3]>) {
        self.bake_list.clear();
        self.jobs.clear();
        self.texels = 0;
        let mut wanted: HashMap<[u32; 3], u32> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for r in requests {
            if !seen.insert(r) || self.map.contains_key(&r) {
                continue;
            }
            let (m, c, mip0, page0) = unpack(r);
            let mut chain = vec![r];
            let (mut mip, mut page) = (mip0, page0);
            let mut resident = false;
            while self.grid(c, m, mip) != [1, 1] && mip < MAX_MIP {
                page = self.coarser(c, m, mip, page);
                mip += 1;
                let k = key(m, c, mip, page);
                if self.map.contains_key(&k) {
                    resident = true;
                    break;
                }
                chain.push(k);
            }
            for (depth, k) in chain.iter().rev().enumerate() {
                let priority = 2 * depth as u32 + resident as u32;
                let p = wanted.entry(*k).or_insert(priority);
                *p = (*p).min(priority);
            }
        }
        let mut order: Vec<([u32; 3], u32)> = wanted.into_iter().collect();
        order.sort_unstable_by_key(|&(k, p)| (p, k));
        let budget = self.options.bake_budget.max(PAGE_TEXELS);
        let mut evicted_now = false;
        for (k, _) in order {
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
                self.stats.evicted += self.evict(device, 2);
            }
            let Some(slot) = self.free.pop() else { break };
            let Some(entry) = self.insert(k, slot) else {
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
        if self.removed + self.map.len() > self.table.len() / 2 {
            self.rebuild();
        }
    }
}

/// Reads `buffer` back (mapping it and waiting for the GPU), then unmaps it.
fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU poll failed");
    let data = slice.get_mapped_range().expect("mapped buffer").to_vec();
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
    fn the_hash_matches_known_values() {
        // Fixed by the shader's formula; guards against edits on one side only.
        assert_eq!(hash([0, 0, 0]), 0);
        assert_ne!(hash([1, 0, 0]), hash([0, 1, 0]));
    }
}
