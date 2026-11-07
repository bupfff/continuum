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
use continuum::glam::{Vec2, Vec3};
use continuum::mesh::Surface;
use wgpu::util::DeviceExt;

use crate::InstancedScene;

/// Texels along a page's side (stored with a one-texel border).
pub const PAGE: u32 = 8;
const STRIDE: u32 = PAGE + 1;
pub(crate) const PAGE_TEXELS: u32 = STRIDE * STRIDE;
const EMPTY: u32 = u32::MAX;
/// Pages freed per frame at most from the eviction queue (the least recently
/// used first), so eviction never stalls a frame (M4 gate 11).
const EVICT_PER_FRAME: usize = 2_048;
/// Pages unused for this many frames are evicted.
const EVICT_AGE: u32 = 64;
/// The adaptive bake budget's floor and starting point (cones per frame).
const MIN_BUDGET: f32 = 1_000.0;
const START_BUDGET: f32 = 8_000.0;
/// Longest probe sequence the shader follows.
const MAX_PROBE: usize = 64;
/// Entries in the per-frame set of requested pages (a power of two, at least
/// twice the request tiles of a 1080p frame).
const REQUEST_SET: u64 = 1 << 18;
/// Movers (M9): pages the stale pass may append to a frame's bake at most
/// (fewer with many lights: at most 2^21 texel jobs), slots it may list as
/// invalid per frame, and how recently a stale page must have been read to
/// be re-baked rather than marked invalid (1: in the frame before; every
/// pixel marks the page it reads).
const APPEND_PAGES: u32 = 4096;
const APPEND_JOBS: u32 = 1 << 21;
const INVALID_CAP: u32 = 1 << 14;
/// Candidate pages (tested texel by texel) per frame at most; more are
/// marked invalid whole.
const CANDIDATES: u32 = 1 << 16;
const VISIBLE_WINDOW: u32 = 1;
/// The stale pass's counters (eight words) in each readback.
const MOVER_STATS_BYTES: u64 = 32;

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
    /// The most bake work per frame, in sun texels (one cone each; a texel
    /// with several lights counts once per light, a sky sample
    /// `sky_sample_cost`).
    pub bake_budget: u32,
    /// The frame period the bake aims for, in milliseconds (0: always bake
    /// `bake_budget`). Where the device has timestamp queries (M5 gate 5),
    /// the bake and the camera passes are timed on the GPU, bake time is fit
    /// as a latency plus a cost per unit of budget, and the budget is what
    /// fits the target after the camera passes, 20% more and less on
    /// alternate frames so the fit can tell latency from cost. Otherwise (M4
    /// gate 11),
    /// when the CPU had to wait for the GPU and the last frame took longer,
    /// the budget scales by target / period, and when a frame is quicker, it
    /// grows by 10%. Between 1,000 and `bake_budget`.
    pub frame_target_ms: f32,
    /// Pages the pool holds. Each texel holds one channel per light, then
    /// the sky's, so the pool takes 81 × 4 bytes × channels per page: 340 MB
    /// at the default 2^19 pages with one light, 680 MB with three. Fewer
    /// pages make flights evict while waiting for the GPU (M5 gate 6).
    pub pages: u32,
    /// With a sky: the spacing of sky samples on surfaces (meters). A page's
    /// sky channel is sampled every 1, 2, 4 or 8 texels along each axis, the
    /// most whose spacing stays within this, and interpolated bilinearly
    /// between samples; sky light varies slowly, and a sky sample costs as
    /// much as tens of sun texels (M5 gate 5). 0: every texel.
    pub sky_spacing: f32,
    /// What a sky sample counts against the bake budget, in sun texels (at
    /// least 1).
    pub sky_sample_cost: u32,
    /// With a sky: the spacing (in each geometry's own units) of the grid on
    /// which its own sky occlusion is baked once, for all its instances (M5
    /// gate 5). Sky texels then walk only the other objects. 0: off. Only
    /// geometries with several instances get masks; a geometry's spacing
    /// grows so it holds at most 65,536 texels, and all of them so a scene
    /// holds at most 2^20 (384 MB). A geometry whose grid would exceed 1,024
    /// texels along a chart walks itself.
    pub self_spacing: f32,
    /// Experiments only. Each light's texels hold the shadowed share of its
    /// unshadowed light,
    /// and shading multiplies that by the unshadowed light at the pixel, with
    /// the pixel's own normal (M5 gate 7); otherwise they hold irradiance,
    /// interpolated as it is (as before).
    #[doc(hidden)]
    pub light_ratio: bool,
}

impl Default for CacheOptions {
    fn default() -> Self {
        CacheOptions {
            base_texel: 0.005,
            pixels_per_texel: 2.0,
            request_tile: 4,
            bake_budget: 20_000,
            frame_target_ms: 14.0,
            pages: 1 << 19,
            sky_spacing: 0.0,
            sky_sample_cost: 10,
            self_spacing: 0.01,
            light_ratio: true,
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
    /// Sample pixels on a surface (one per request tile).
    pub samples: u32,
    /// Of `samples`, the ones that found the page of their wanted mip.
    pub at_wanted: u32,
    /// Pages baked in the frame.
    pub pages_baked: u32,
    /// Texels baked in the frame.
    pub texels_baked: u32,
    /// Sky samples baked in the frame.
    pub sky_samples: u32,
    /// Pages resident after the frame's bake.
    pub resident: u32,
    /// Pages evicted while planning the frame.
    pub evicted: u32,
    /// CPU time spent planning the frame's bake, including eviction (not
    /// counting `waiting`).
    pub planning: Duration,
    /// Of that work, eviction (merging slot use, freeing pages).
    pub evicting: Duration,
    /// The bake budget the frame planned with (sun texels, or their worth).
    pub budget: u32,
    /// Of `budget`, what the frame's plan used.
    pub spent: u32,
    /// GPU time of the frame's bake passes, in milliseconds, where the device
    /// has timestamp queries and the frame was not profiled (else 0).
    pub gpu_bake_ms: f32,
    /// GPU time of the frame's camera passes, likewise.
    pub gpu_render_ms: f32,
    /// Whether the frame's GPU timings count (not a profiled frame).
    pub timed: bool,
    /// Time spent waiting for the GPU to finish the frame two before, whose
    /// requests the plan reads.
    pub waiting: Duration,
    /// Movers (M9): objects moved this frame.
    pub moved: u32,
    /// Pages with stale texels re-baked in the frame (read in the frame
    /// before).
    pub rebaked_pages: u32,
    /// Texel-light pairs re-baked for movers (one cone each).
    pub rebaked_texels: u32,
    /// Pages whose boxes a move may touch, tested texel by texel.
    pub candidates: u32,
    /// Stale pages marked invalid instead (evicted when read back).
    pub invalidated: u32,
    /// Pages read in the frame before (counted when something moved).
    pub visible_pages: u32,
    /// Of the stale pages read recently, those that did not fit the frame's
    /// bake (marked invalid too).
    pub missed: u32,
    /// The bake's work, with `KernelOptions::count_work` (zero otherwise):
    /// object nodes tested, object nodes splatted, meshes entered,
    /// triangle-hierarchy nodes tested, of those splatted, triangles tested,
    /// triangles rasterized, cones traced; then object nodes and
    /// triangle-hierarchy nodes their packets visited (packet traversal).
    pub work: [u32; 12],
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
    sky: [u32; 4],
    movers: [u32; 4],
}

/// Per-chart tables shared by every object with a geometry: the chart's size
/// and cell grid, the triangles of each cell, the chart of each triangle and
/// its vertices' chart coordinates. Triangles are numbered as in the flattened
/// scene (geometries in order).
struct ChartTables {
    charts: Vec<VChartGpu>,
    /// Per cell: first entry in `cell_tris`, and how many.
    cells: Vec<[u32; 2]>,
    cell_tris: Vec<u32>,
    tri_chart: Vec<u32>,
    tri_uv: Vec<[f32; 2]>,
    /// First global chart of each geometry.
    first_chart: Vec<u32>,
    /// Per cell: the object-space box of its triangles (lo above hi: none).
    cell_bounds: Vec<[Vec3; 2]>,
}

fn chart_tables(geometries: &[Surface]) -> ChartTables {
    let mut t = ChartTables {
        charts: Vec::new(),
        cells: Vec::new(),
        cell_tris: Vec::new(),
        tri_chart: Vec::new(),
        tri_uv: Vec::new(),
        first_chart: Vec::new(),
        cell_bounds: Vec::new(),
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
        let first_cell = t.cells.len();
        for (size, grid) in g.charts.iter().zip(&g.chart_grid) {
            t.charts.push(VChartGpu {
                size: size.to_array(),
                grid: *grid,
                first_cell: t.cells.len() as u32,
                pad: 0,
            });
            t.cells
                .extend(std::iter::repeat_n([0, 0], (grid[0] * grid[1]) as usize));
        }
        // Each triangle goes in every cell its chart coordinates' bounds
        // meet, shrunk by a thousandth of a cell so a parametric quad's two
        // triangles stay in their own cell.
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); t.cells.len() - first_cell];
        for (k, (uv, &chart)) in g.uv.iter().zip(&g.chart).enumerate() {
            let c = &t.charts[(first_chart + chart) as usize];
            let grid = Vec2::new(c.grid[0] as f32, c.grid[1] as f32);
            let lo = (uv[0].min(uv[1]).min(uv[2]) * grid + Vec2::splat(1e-3)).floor();
            let hi = (uv[0].max(uv[1]).max(uv[2]) * grid - Vec2::splat(1e-3)).floor();
            let clamp = |v: f32, n: u32| (v.max(0.0) as u32).min(n - 1);
            let (x0, x1) = (clamp(lo.x, c.grid[0]), clamp(hi.x.max(lo.x), c.grid[0]));
            let (y0, y1) = (clamp(lo.y, c.grid[1]), clamp(hi.y.max(lo.y), c.grid[1]));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let cell = c.first_cell as usize + (y * c.grid[0] + x) as usize - first_cell;
                    lists[cell].push(first_tri + k as u32);
                }
            }
            t.tri_chart.push(first_chart + chart);
            t.tri_uv.extend(uv.iter().map(|v| v.to_array()));
        }
        for (cell, list) in lists.into_iter().enumerate() {
            t.cells[first_cell + cell] = [t.cell_tris.len() as u32, list.len() as u32];
            let mut b = [Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)];
            for &tri in &list {
                for v in g.triangles[(tri - first_tri) as usize] {
                    b = [b[0].min(v), b[1].max(v)];
                }
            }
            t.cell_bounds.push(b);
            t.cell_tris.extend(list);
        }
        first_tri += g.triangles.len() as u32;
    }
    assert!(t.charts.len() < 1 << 16, "too many charts for page keys");
    if t.cells.is_empty() {
        t.cells.push([0, 0]);
    }
    if t.cell_tris.is_empty() {
        t.cell_tris.push(0);
    }
    if t.tri_chart.is_empty() {
        t.tri_chart.push(0);
        t.tri_uv.push([0.0; 2]);
    }
    t
}

/// Own-object sky occlusion grids: per global chart `[first self texel, grid
/// u, grid v, 1 if baked]`, and the self texels to bake `[object, chart, x |
/// y << 16, 0]` (one instance of each geometry stands for all).
struct SelfTables {
    charts: Vec<[u32; 4]>,
    texels: Vec<[u32; 4]>,
    count: u32,
}

/// The most own-mask texels per geometry, and in a scene (384 bytes each).
const SELF_TEXELS_PER_GEOMETRY: u32 = 1 << 16;
const SELF_TEXELS: u32 = 1 << 20;

/// The most self texels along a chart; larger charts walk themselves.
const SELF_GRID_MAX: u32 = 1024;

fn self_tables(scene: &InstancedScene, tables: &ChartTables, spacing: f32) -> SelfTables {
    let mut own = SelfTables {
        charts: vec![[0; 4]; tables.charts.len().max(1)],
        texels: Vec::new(),
        count: 0,
    };
    if spacing > 0.0 {
        // Masks pay where a geometry has several instances (M6): a unique
        // object's texels walk it more cheaply than its whole surface is
        // baked. Each geometry's spacing grows so it holds at most
        // SELF_TEXELS_PER_GEOMETRY texels, and all spacings together so the
        // scene holds at most SELF_TEXELS.
        let mut instances = vec![0usize; tables.first_chart.len()];
        let mut first_instance = vec![usize::MAX; tables.first_chart.len()];
        for (k, i) in scene.instances.iter().enumerate() {
            instances[i.geometry] += 1;
            first_instance[i.geometry] = first_instance[i.geometry].min(k);
        }
        let charts_of = |g: usize| {
            let first = tables.first_chart[g];
            let end = tables
                .first_chart
                .get(g + 1)
                .copied()
                .unwrap_or(tables.charts.len() as u32);
            first..end
        };
        let area = |g: usize| -> f32 {
            charts_of(g)
                .map(|c| {
                    let size = tables.charts[c as usize].size;
                    size[0] * size[1]
                })
                .sum()
        };
        let candidates: Vec<usize> = (0..tables.first_chart.len())
            .filter(|&g| instances[g] >= 2)
            .collect();
        let mut spacings: Vec<f32> = candidates
            .iter()
            .map(|&g| spacing.max((area(g) / SELF_TEXELS_PER_GEOMETRY as f32).sqrt()))
            .collect();
        let total: f32 = candidates
            .iter()
            .zip(&spacings)
            .map(|(&g, s)| area(g) / (s * s))
            .sum();
        if total > SELF_TEXELS as f32 {
            let grow = (total / SELF_TEXELS as f32).sqrt();
            for s in &mut spacings {
                *s *= grow;
            }
        }
        for (&g, &spacing) in candidates.iter().zip(&spacings) {
            let grids: Vec<[u32; 2]> = charts_of(g)
                .map(|c| {
                    let size = Vec2::from_array(tables.charts[c as usize].size);
                    let n = (size / spacing).ceil();
                    [(n.x as u32).max(1), (n.y as u32).max(1)]
                })
                .collect();
            if grids
                .iter()
                .any(|n| n[0] > SELF_GRID_MAX || n[1] > SELF_GRID_MAX)
            {
                continue;
            }
            let m = first_instance[g];
            for (c, n) in charts_of(g).zip(grids) {
                own.charts[c as usize] = [own.texels.len() as u32, n[0], n[1], 1];
                for y in 0..n[1] {
                    for x in 0..n[0] {
                        own.texels.push([m as u32, c, x | y << 16, 0]);
                    }
                }
            }
        }
    }
    own.count = own.texels.len() as u32;
    if own.texels.is_empty() {
        own.texels.push([0; 4]);
    }
    own
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

/// A resident page as read back for diagnostics
/// ([`crate::InstancedRenderer::read_cache_pages`]).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct CachePage {
    pub object: u32,
    pub chart: u32,
    pub mip: u32,
    pub page: [u32; 2],
    /// Texels along u and v inside the chart's grid (those past it repeat
    /// the nearest inside).
    pub inside: [u32; 2],
    /// Texel spacing in world units.
    pub texel: f32,
    /// One channel, 9 × 9 texels in rows of 9: a light's shadowed share of
    /// its unshadowed light, or the sky's irradiance per unit radiance.
    pub texels: Vec<f32>,
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
/// GPU timestamps around a frame's bake passes (0, 1) and camera passes (2, 3).
pub(crate) struct GpuTimer {
    pub(crate) set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period: f32,
}

/// Bake time against budget spent, fit as `latency + cost × units` over
/// recent frames (least squares with exponential forgetting).
struct BakeModel {
    /// Decayed sums: weight, units, units², milliseconds, units × ms.
    sums: [f64; 5],
    latency: f32,
    cost: f32,
    /// The camera passes' time, smoothed.
    render: f32,
}

impl BakeModel {
    fn new() -> Self {
        BakeModel {
            sums: [0.0; 5],
            latency: 1.0,
            cost: 5e-4,
            render: 0.0,
        }
    }

    fn add(&mut self, units: f32, bake_ms: f32, render_ms: f32) {
        self.render = if self.render == 0.0 {
            render_ms
        } else {
            0.7 * self.render + 0.3 * render_ms
        };
        if units <= 0.0 {
            return;
        }
        let (u, y) = (units as f64, bake_ms as f64);
        for (s, x) in self.sums.iter_mut().zip([1.0, u, u * u, y, u * y]) {
            *s = 0.95 * *s + x;
        }
        let [w, su, suu, sy, suy] = self.sums;
        let (mu, my) = (su / w, sy / w);
        let var = suu / w - mu * mu;
        // Refit the cost only when the budget has varied enough to tell.
        if w >= 4.0 && var > (0.1 * mu) * (0.1 * mu) {
            let cost = (suy / w - mu * my) / var;
            self.cost = (cost as f32).clamp(1e-6, 1.0);
        }
        self.latency = ((my - self.cost as f64 * mu) as f32).clamp(0.0, 100.0);
    }

    /// The units that fit `target` milliseconds of GPU time with the camera
    /// passes.
    fn budget(&self, target: f32) -> f32 {
        (target - self.render - self.latency) / self.cost
    }
}

struct Readback {
    buffer: wgpu::Buffer,
    tiles: u32,
    pending: Option<(u32, wgpu::SubmissionIndex)>,
}

/// The shading and bake counters' size: four shading counters, then twelve
/// work counters (with `KernelOptions::count_work`).
const STATS_BYTES: u64 = 64;
/// Four GPU timestamps after the counters in each readback.
const TIMESTAMP_BYTES: u64 = 32;

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
    /// Each global chart's cell grid and first cell, and each cell's
    /// object-space box (for pages' boxes, M9).
    chart_cells: Vec<([u32; 2], u32)>,
    cell_bounds: Vec<[Vec3; 2]>,
    /// Per page in `bake_list`: its texels' object-space box (lo, hi).
    bake_bounds: Vec<[f32; 4]>,
    /// Movers (M9): the objects moved this frame (old box with the object's
    /// index, new box), the pages the stale pass may append, the frame each
    /// slot was last allocated in, and the appended bake work, smoothed (sun
    /// texels).
    moved: Vec<[f32; 4]>,
    moved_count: u32,
    append_pages: u32,
    alloc_frame: Vec<u32>,
    appended: f32,
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
    /// Whether the sky channel is baked; this frame's sky samples (page
    /// entry, texel) and each page's (first sample, count); and the budget
    /// the frame's plan used, in sun texels.
    sky: bool,
    /// Lights, each with a channel per texel.
    lights: u32,
    sky_samples: Vec<[u32; 2]>,
    page_sky: Vec<[u32; 2]>,
    spent: u32,
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
    /// Self texels whose masks are still to be baked (cleared once baked).
    pub(crate) self_texels: u32,
    params: wgpu::Buffer,
    /// Alternating per frame.
    readbacks: Vec<Readback>,
    used_readback: wgpu::Buffer,
    /// The frame and submission whose slot use `used_readback` holds.
    used_pending: Option<(u32, wgpu::SubmissionIndex)>,
    used_requested: bool,
    /// The current bake budget (cones per frame), and when the last frame
    /// began (for its period).
    budget: f32,
    last_begin: Option<Instant>,
    /// GPU timing, where the device has timestamp queries, and the bake
    /// model it feeds.
    pub(crate) timer: Option<GpuTimer>,
    model: BakeModel,
    /// Slots to free, least recently used last (popped first).
    evict_queue: Vec<u32>,
    /// A frame whose invalid slots overflowed the list: every slot's mark
    /// is read back before the next plan.
    overflowed: Option<u32>,
}

impl VirtualCache {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &InstancedScene,
        options: CacheOptions,
        sky: bool,
        lights: u32,
        width: u32,
        height: u32,
        movers: u32,
    ) -> Self {
        let channels = lights + 1;
        assert!(
            movers == 0 || !sky,
            "moving objects with a sky in the cache are not supported yet"
        );
        let append_pages = if movers > 0 {
            APPEND_PAGES.min(APPEND_JOBS / (PAGE_TEXELS * lights))
        } else {
            0
        };
        let append_jobs = (append_pages * PAGE_TEXELS * lights) as u64;
        // So a frame's sky samples never outnumber the budget (their buffers'
        // size).
        assert!(
            options.sky_sample_cost >= 1,
            "a sky sample costs at least 1"
        );
        let tables = chart_tables(&scene.geometries);
        let own = self_tables(scene, &tables, if sky { options.self_spacing } else { 0.0 });
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
                54,
                storage(
                    "cell triangles",
                    bytemuck::cast_slice(&tables.cell_tris),
                    copy_dst,
                ),
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
                // Per texel, a channel per light and the sky's.
                // Copied out only by diagnostics (`read_pages`).
                empty(
                    "page pool",
                    options.pages as u64 * PAGE_TEXELS as u64 * channels as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC,
                ),
            ),
            (
                38,
                empty("bake list", (budget + append_pages as u64) * 16, copy_dst),
            ),
            (
                63,
                empty("bake bounds", (budget + append_pages as u64) * 32, copy_dst),
            ),
            (
                60,
                empty(
                    "moved",
                    (2 * lights as u64 + 2 + movers as u64 * (4 + 4 * lights as u64)) * 16,
                    copy_dst,
                ),
            ),
            (
                61,
                empty(
                    "slot info",
                    options.pages as u64 * 48,
                    wgpu::BufferUsages::empty(),
                ),
            ),
            (
                62,
                empty(
                    "slot stale",
                    options.pages as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                64,
                empty(
                    "mover stats",
                    MOVER_STATS_BYTES,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                65,
                empty(
                    "invalid list",
                    INVALID_CAP as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC,
                ),
            ),
            (66, empty("dispatch args", 36, wgpu::BufferUsages::INDIRECT)),
            (
                67,
                empty(
                    "candidates",
                    if movers > 0 {
                        CANDIDATES as u64 * 8
                    } else {
                        16
                    },
                    wgpu::BufferUsages::empty(),
                ),
            ),
            // A plan never exceeds the budget but by its first page.
            (46, empty("sky samples", (budget + 81) * 8, copy_dst)),
            (
                47,
                empty(
                    "sky partials",
                    (budget + 81) * 6 * 4,
                    wgpu::BufferUsages::empty(),
                ),
            ),
            (48, empty("page sky", budget * 8, copy_dst)),
            (
                49,
                storage("self charts", bytemuck::cast_slice(&own.charts), copy_dst),
            ),
            (
                50,
                storage("self texels", bytemuck::cast_slice(&own.texels), copy_dst),
            ),
            (
                51,
                empty(
                    "self masks",
                    own.count as u64 * 6 * 16 * 4,
                    wgpu::BufferUsages::empty(),
                ),
            ),
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
                empty(
                    "cache stats",
                    STATS_BYTES,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                43,
                empty("bake jobs", (4 * (budget + 81) + append_jobs) * 8, copy_dst),
            ),
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
            chart_cells: tables
                .charts
                .iter()
                .map(|c| (c.grid, c.first_cell))
                .collect(),
            cell_bounds: tables.cell_bounds,
            bake_bounds: Vec::new(),
            moved: Vec::new(),
            moved_count: 0,
            append_pages,
            alloc_frame: vec![0; options.pages as usize],
            appended: 0.0,
            table,
            map: KeyMap::default(),
            slot_key: vec![[EMPTY; 3]; options.pages as usize],
            last_used: vec![0; options.pages as usize],
            free: (0..options.pages).rev().collect(),
            bake_list: Vec::new(),
            jobs: Vec::new(),
            texels: 0,
            sky,
            lights,
            sky_samples: Vec::new(),
            page_sky: Vec::new(),
            spent: 0,
            frame: 1,
            log: Vec::new(),
            latest: 0,
            evicted: 0,
            planning: Duration::ZERO,
            evicting: Duration::ZERO,
            waiting: Duration::ZERO,
            buffers,
            self_texels: own.count,
            params,
            readbacks: Vec::new(),
            used_readback,
            used_pending: None,
            used_requested: false,
            budget: if options.frame_target_ms > 0.0 {
                START_BUDGET.min(options.bake_budget as f32)
            } else {
                options.bake_budget as f32
            },
            last_begin: None,
            timer: device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| GpuTimer {
                    set: device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("cache timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 4,
                    }),
                    resolve: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("timestamp resolve"),
                        size: TIMESTAMP_BYTES,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    period: queue.get_timestamp_period(),
                }),
            model: BakeModel::new(),
            evict_queue: Vec::new(),
            overflowed: None,
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
                    size: tiles as u64 * 16
                        + STATS_BYTES
                        + TIMESTAMP_BYTES
                        + MOVER_STATS_BYTES
                        + INVALID_CAP as u64 * 4,
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

    /// Diagnostics: every resident page with channel `ch` of its texels, read
    /// back now (waiting for all GPU work).
    pub(crate) fn read_pages(&self, device: &wgpu::Device, ch: u32) -> Vec<CachePage> {
        let pool = self.buffer(37);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool readback"),
            size: pool.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(pool, 0, &readback, 0, None);
        let index = self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        let raw = map_wait(device, &readback, index);
        let values: &[f32] = bytemuck::cast_slice(&raw);
        let channels = (self.lights + self.sky as u32) as usize;
        let mut pages = Vec::new();
        for (slot, &k) in self.slot_key.iter().enumerate() {
            if k[0] == EMPTY {
                continue;
            }
            let (m, c, mip, page) = unpack(k);
            let n = self.grid(c, m, mip);
            let inside = [
                (n[0] - (page[0] * PAGE).min(n[0])).min(STRIDE),
                (n[1] - (page[1] * PAGE).min(n[1])).min(STRIDE),
            ];
            let first = slot * PAGE_TEXELS as usize;
            pages.push(CachePage {
                object: m,
                chart: c,
                mip,
                page,
                inside,
                texel: self.options.base_texel
                    * self.scale[m as usize]
                    * (1u64 << mip.min(40)) as f32,
                texels: (0..PAGE_TEXELS as usize)
                    .map(|t| values[(first + t) * channels + ch as usize])
                    .collect(),
            });
        }
        pages
    }

    /// The sky stride of pages on object `m` at `mip`: the largest of 1, 2,
    /// 4 and 8 whose spacing stays within `sky_spacing`.
    fn sky_stride(&self, m: u32, mip: u32) -> u32 {
        let texel = self.options.base_texel * self.scale[m as usize] * (1u64 << mip.min(40)) as f32;
        let mut stride = 1;
        while stride < 8 && (2 * stride) as f32 * texel <= self.options.sky_spacing {
            stride *= 2;
        }
        stride
    }

    /// Writes this frame's uniform and uploads its bake list, texel and sky
    /// sample lists and page table changes. Returns the passes' sizes: table
    /// updates, sun texels, sky face jobs, pages; then the objects moved (the
    /// stale pass runs if any).
    pub(crate) fn upload(
        &mut self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        vertical_fov_degrees: f32,
    ) -> [u32; 5] {
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
            queue.write_buffer(self.buffer(63), 0, bytemuck::cast_slice(&self.bake_bounds));
            queue.write_buffer(self.buffer(43), 0, bytemuck::cast_slice(&self.jobs));
            if self.sky {
                queue.write_buffer(self.buffer(48), 0, bytemuck::cast_slice(&self.page_sky));
            }
        }
        if !self.sky_samples.is_empty() {
            queue.write_buffer(self.buffer(46), 0, bytemuck::cast_slice(&self.sky_samples));
        }
        let moved = std::mem::take(&mut self.moved_count);
        if moved > 0 {
            queue.write_buffer(self.buffer(60), 0, bytemuck::cast_slice(&self.moved));
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
            sky: [
                self.sky_samples.len() as u32,
                self.sky as u32,
                self.lights,
                self.options.light_ratio as u32,
            ],
            movers: [moved, self.append_pages, self.append_jobs(), VISIBLE_WINDOW],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.log.push(CacheStats {
            frame: self.frame,
            pages_baked: self.bake_list.len() as u32,
            texels_baked: self.texels,
            sky_samples: self.sky_samples.len() as u32,
            resident: self.map.len() as u32,
            evicted: std::mem::take(&mut self.evicted),
            planning: std::mem::take(&mut self.planning),
            budget: self.budget as u32,
            spent: self.spent,
            timed: self.timer.is_some(),
            evicting: std::mem::take(&mut self.evicting),
            waiting: std::mem::take(&mut self.waiting),
            moved,
            ..Default::default()
        });
        self.moved.clear();
        [
            (updates.len() / 2) as u32,
            self.jobs.len() as u32,
            6 * self.sky_samples.len() as u32,
            self.bake_list.len() as u32,
            moved,
        ]
    }

    /// Objects moved since the last frame, each with its index and its old
    /// and new world boxes (relative to the origin, padded): the next
    /// frame's stale pass tests every page against them. With each light's
    /// box (`lights`) and a box holding every texel's apex (`scene`), it
    /// also lists, per light, each box's shadow region's box (empty if
    /// none) and their union, and the union of the moved boxes, so most
    /// pages are passed over with a few box tests.
    ///
    /// Layout (vec4s): per light its union's lo and hi; the moved boxes'
    /// union's lo and hi; then per object: old lo (with the object's index
    /// as bits in w), old hi, new lo, new hi, and per light the old and new
    /// shadow regions' lo and hi.
    pub(crate) fn set_moved(
        &mut self,
        moved: &[(u32, [Vec3; 2], [Vec3; 2])],
        lights: &[(Vec3, Vec3)],
        scene: (Vec3, Vec3),
    ) {
        let v4 = |v: Vec3, w: f32| [v.x, v.y, v.z, w];
        let empty = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let grow = |a: (Vec3, Vec3), b: (Vec3, Vec3)| (a.0.min(b.0), a.1.max(b.1));
        let mut unions = vec![empty; lights.len()];
        let mut all = empty;
        let mut entries = Vec::new();
        for (object, old, new) in moved {
            let (old, new) = ((old[0], old[1]), (new[0], new[1]));
            all = grow(grow(all, old), new);
            entries.push(v4(old.0, f32::from_bits(*object)));
            entries.push(v4(old.1, 0.0));
            entries.push(v4(new.0, 0.0));
            entries.push(v4(new.1, 0.0));
            for (l, light) in lights.iter().enumerate() {
                for b in [old, new] {
                    let r = crate::hierarchy::shadow_region(b, *light, scene).unwrap_or(empty);
                    if r.0.x <= r.1.x {
                        unions[l] = grow(unions[l], r);
                    }
                    entries.push(v4(r.0, 0.0));
                    entries.push(v4(r.1, 0.0));
                }
            }
        }
        self.moved.clear();
        for u in unions {
            self.moved.push(v4(u.0, 0.0));
            self.moved.push(v4(u.1, 0.0));
        }
        self.moved.push(v4(all.0, 0.0));
        self.moved.push(v4(all.1, 0.0));
        self.moved.extend(entries);
        self.moved_count = moved.len() as u32;
    }

    /// Texel jobs the stale passes may append to a frame's bake.
    fn append_jobs(&self) -> u32 {
        self.append_pages * PAGE_TEXELS * self.lights
    }

    /// Slots the pool holds.
    pub(crate) fn pages(&self) -> u32 {
        self.options.pages
    }

    /// Diagnostics: resident slots whose pages shading may read (not
    /// marked stale in `marks`).
    pub(crate) fn readable_pages(&self, marks: &[u32]) -> Vec<u32> {
        (0..self.options.pages)
            .filter(|&s| self.slot_key[s as usize][0] != EMPTY && marks[s as usize] == 0)
            .collect()
    }

    /// Diagnostics: pages one re-bake of `upload_rebake` holds at most.
    pub(crate) fn rebake_chunk(&self) -> usize {
        let budget = self.options.bake_budget.max(PAGE_TEXELS) as u64;
        let pages = budget + self.append_pages as u64;
        let jobs = 4 * (budget + 81) + (self.append_pages * PAGE_TEXELS * self.lights) as u64;
        pages.min(jobs / (PAGE_TEXELS * self.lights) as u64) as usize
    }

    /// Diagnostics: lists slots `slots` (resident) for a bake of every
    /// texel and light, as the frame's plan would, with no table updates
    /// and no sky samples. Returns the bake's counts.
    pub(crate) fn upload_rebake(&mut self, queue: &wgpu::Queue, slots: &[u32]) -> [u32; 5] {
        let mut list = Vec::new();
        let mut bounds = Vec::new();
        let mut jobs = Vec::new();
        for (index, &slot) in slots.iter().enumerate() {
            let k = self.slot_key[slot as usize];
            let (m, c, mip, page) = unpack(k);
            list.push([k[0], k[1], k[2], slot]);
            let b = self.page_bounds(m, c, mip, page);
            bounds.push([b[0], b[1], b[2], b[3]]);
            bounds.push([b[4], b[5], b[6], b[7]]);
            for l in 0..self.lights {
                for t in 0..PAGE_TEXELS {
                    jobs.push([index as u32, t | l << 8]);
                }
            }
        }
        queue.write_buffer(self.buffer(38), 0, bytemuck::cast_slice(&list));
        queue.write_buffer(self.buffer(63), 0, bytemuck::cast_slice(&bounds));
        queue.write_buffer(self.buffer(43), 0, bytemuck::cast_slice(&jobs));
        let mut params: VcParams = Zeroable::zeroed();
        params.counts = [self.table.entries.len() as u32 - 1, self.frame, 1, 1];
        params.jobs = [
            jobs.len() as u32,
            list.len() as u32,
            0,
            REQUEST_SET as u32 - 1,
        ];
        params.params = [self.inv_base, 0.0, self.options.pixels_per_texel, 0.0];
        params.sky = [
            0,
            self.sky as u32,
            self.lights,
            self.options.light_ratio as u32,
        ];
        params.movers = [0, self.append_pages, self.append_jobs(), VISIBLE_WINDOW];
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        [0, jobs.len() as u32, 0, list.len() as u32, 0]
    }

    /// The object-space box of the apexes of page `(m, c, mip, page)`'s
    /// texels: the boxes of the triangles of every chart cell its texel
    /// centers fall in (with a cell's margin for rounding). A texel's apex
    /// lies on one of its cell's triangles (`vc_surface`).
    fn page_bounds(&self, m: u32, c: u32, mip: u32, page: [u32; 2]) -> [f32; 8] {
        let n = self.grid(c, m, mip);
        let (g, first) = self.chart_cells[c as usize];
        let range = |p: u32, n: u32, g: u32| {
            let x0 = p * PAGE;
            let x1 = (p * PAGE + PAGE).min(n - 1);
            let cell = |x: u32| ((x as f32 + 0.5) / n as f32 * g as f32) as u32;
            (cell(x0).saturating_sub(1), (cell(x1) + 1).min(g - 1))
        };
        let (x0, x1) = range(page[0], n[0], g[0]);
        let (y0, y1) = range(page[1], n[1], g[1]);
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let b = self.cell_bounds[(first + y * g[0] + x) as usize];
                (lo, hi) = (lo.min(b[0]), hi.max(b[1]));
            }
        }
        if lo.x > hi.x {
            (lo, hi) = (Vec3::ZERO, Vec3::ZERO);
        }
        [lo.x, lo.y, lo.z, 0.0, hi.x, hi.y, hi.z, 0.0]
    }

    /// Clears the request tiles and counters before shading.
    pub(crate) fn encode_clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.buffer(42), 0, None);
        encoder.clear_buffer(self.buffer(45), 0, None);
        encoder.clear_buffer(self.buffer(64), 0, None);
    }

    /// Copies the frame's requests and counters (and every 32 frames, slot
    /// use) for reading back once the frame is done.
    pub(crate) fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let r = &self.readbacks[(self.frame % 2) as usize];
        let bytes = r.tiles as u64 * 16;
        encoder.copy_buffer_to_buffer(self.buffer(39), 0, &r.buffer, 0, bytes);
        encoder.copy_buffer_to_buffer(self.buffer(42), 0, &r.buffer, bytes, STATS_BYTES);
        if self.log.last().is_some_and(|l| l.moved > 0) {
            let at = bytes + STATS_BYTES + TIMESTAMP_BYTES;
            encoder.copy_buffer_to_buffer(self.buffer(64), 0, &r.buffer, at, MOVER_STATS_BYTES);
            encoder.copy_buffer_to_buffer(
                self.buffer(65),
                0,
                &r.buffer,
                at + MOVER_STATS_BYTES,
                INVALID_CAP as u64 * 4,
            );
        }
        self.used_requested = self.frame.is_multiple_of(32) && self.used_pending.is_none();
        if self.used_requested {
            encoder.copy_buffer_to_buffer(self.buffer(40), 0, &self.used_readback, 0, None);
        }
    }

    /// Marks the frame being prepared as not timed (its passes were split
    /// into submissions).
    pub(crate) fn untimed(&mut self) {
        if let Some(record) = self.log.last_mut() {
            record.timed = false;
        }
    }

    /// Copies the frame's GPU timestamps for reading back, after every pass
    /// that writes them.
    pub(crate) fn encode_timestamps(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(timer) = &self.timer else { return };
        let r = &self.readbacks[(self.frame % 2) as usize];
        encoder.resolve_query_set(&timer.set, 0..4, &timer.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &timer.resolve,
            0,
            &r.buffer,
            r.tiles as u64 * 16 + STATS_BYTES,
            TIMESTAMP_BYTES,
        );
    }

    /// After the frame's submission (`index`): starts its readbacks, without
    /// waiting.
    pub(crate) fn after_submit(&mut self, index: wgpu::SubmissionIndex) {
        let r = &mut self.readbacks[(self.frame % 2) as usize];
        // A failed map is ignored: it happens when the device goes away
        // (an engine shutting down), and a failure while running shows as a
        // panic when the range is read.
        r.buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        r.pending = Some((self.frame, index.clone()));
        if self.used_requested {
            self.used_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            self.used_pending = Some((self.frame, index));
            self.used_requested = false;
        }
        self.frame += 1;
    }

    /// Before the frame: reads back the requests of two frames ago (waiting
    /// only for that frame), adapts the bake budget to the frame time, queues
    /// stale pages when slot use arrives and frees a few, and plans the
    /// frame's bake.
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        let start = Instant::now();
        let period = self.last_begin.map(|t| start - t);
        self.last_begin = Some(start);
        let slot = (self.frame % 2) as usize;
        let requests = self.collect(device, slot);
        if self.overflowed.take().is_some() {
            self.evict_marked(device);
        }
        let target = self.options.frame_target_ms;
        if target > 0.0 && self.timer.is_some() {
            // Only once a frame has been timed. Alternate frames bake 20%
            // more and less, so the fit can tell latency from cost.
            if self.latest > 0 {
                let dither = if self.frame.is_multiple_of(2) {
                    1.2
                } else {
                    0.8
                };
                // Re-baking for movers comes out of the same time.
                self.budget = ((self.model.budget(target) - self.appended) * dither).clamp(
                    MIN_BUDGET.min(self.options.bake_budget as f32),
                    self.options.bake_budget as f32,
                );
            }
        } else if target > 0.0
            && let Some(period) = period
        {
            let period = period.as_secs_f32() * 1e3;
            let gpu_bound = self.waiting > Duration::from_millis(1);
            if gpu_bound && period > target {
                self.budget *= target / period;
            } else if period < target {
                self.budget *= 1.1;
            }
            self.budget = self.budget.clamp(
                MIN_BUDGET.min(self.options.bake_budget as f32),
                self.options.bake_budget as f32,
            );
        }
        let evicting = Instant::now();
        if let Some((frame, _)) = &self.used_pending
            && frame + 2 <= self.frame
        {
            let (_, index) = self.used_pending.take().expect("pending slot use");
            let raw = map_wait(device, &self.used_readback, index);
            self.merge_used(bytemuck::cast_slice(&raw));
            self.queue_stale(EVICT_AGE);
        }
        self.evicted += self.evict_queued(EVICT_PER_FRAME, EVICT_AGE);
        self.evicting += evicting.elapsed();
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
        let counters = &words[4 * tiles..4 * tiles + STATS_BYTES as usize / 4];
        let count = (counters[1] as usize).min(tiles);
        let requests: Vec<([u32; 3], u32)> = words[..4 * count]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|r| ([r[0], r[1], r[2]], r[3]))
            .collect();
        // Movers (M9): the stale pass's counters and invalid slots.
        let moved = self.log[frame as usize - 1].moved > 0;
        let mut mover = [0u32; 8];
        let mut invalid: Vec<u32> = Vec::new();
        if moved {
            let at = 4 * tiles + (STATS_BYTES + TIMESTAMP_BYTES) as usize / 4;
            mover.copy_from_slice(&words[at..at + 8]);
            let listed = (mover[1] as usize).min(INVALID_CAP as usize);
            invalid.extend_from_slice(&words[at + 8..at + 8 + listed]);
            if mover[1] as usize > listed {
                // More than the list holds: every slot's mark is read back.
                self.overflowed = Some(frame);
            }
        }
        let appended = mover[5].min(self.append_jobs());
        let record = &mut self.log[frame as usize - 1];
        record.rebaked_pages = mover[6];
        record.rebaked_texels = appended;
        record.invalidated = mover[1];
        record.visible_pages = mover[2];
        record.missed = mover[3];
        record.candidates = mover[4];
        record.spent += appended;
        if let Some(timer) = &self.timer
            && record.timed
        {
            let at = 4 * tiles + STATS_BYTES as usize / 4;
            let stamps: &[u64] =
                bytemuck::cast_slice(&raw[4 * at..4 * at + TIMESTAMP_BYTES as usize]);
            // A span, if both ends were written in order and within a second.
            let ms = |a: u64, b: u64| {
                let t = b.wrapping_sub(a) as f64 * timer.period as f64 * 1e-6;
                (a != 0 && b > a && t < 1000.0).then_some(t as f32)
            };
            let bake = if record.pages_baked > 0 || moved {
                ms(stamps[0], stamps[1])
            } else {
                Some(0.0)
            };
            if let (Some(bake), Some(render)) = (bake, ms(stamps[2], stamps[3])) {
                record.gpu_bake_ms = bake;
                record.gpu_render_ms = render;
                self.model.add(record.spent as f32, bake, render);
            }
        }
        record.fallback_pixels = counters[0];
        record.requests = count as u32;
        record.samples = counters[2];
        record.at_wanted = counters[3];
        record.work.copy_from_slice(&counters[4..16]);
        self.appended = 0.8 * self.appended + 0.2 * appended as f32;
        self.evict_invalid(&invalid, frame);
        self.latest = self.latest.max(frame);
        requests
    }

    /// Evicts the slots the stale pass marked invalid in `frame`, unless a
    /// later plan has reused them.
    fn evict_invalid(&mut self, slots: &[u32], frame: u32) {
        for &slot in slots {
            let k = self.slot_key[slot as usize];
            if k[0] == EMPTY || self.alloc_frame[slot as usize] >= frame {
                continue;
            }
            let (_, entry) = self.map.remove(&k).expect("resident page");
            let map = &mut self.map;
            self.table.remove(entry as usize, |moved, i| {
                map.get_mut(&moved).expect("resident page").1 = i;
            });
            self.slot_key[slot as usize] = [EMPTY; 3];
            self.free.push(slot);
            self.evicted += 1;
        }
    }

    /// Evicts every resident slot marked stale (waiting for all GPU work):
    /// when a frame marked more than the invalid list holds.
    fn evict_marked(&mut self, device: &wgpu::Device) {
        let marks = self.read_buffer(device, 62);
        let marks: &[u32] = bytemuck::cast_slice(&marks);
        let slots: Vec<u32> = (0..self.options.pages)
            .filter(|&s| marks[s as usize] != 0)
            .collect();
        self.evict_invalid(&slots, self.frame);
    }

    /// Binding `binding`'s buffer read back now (waiting for all GPU work).
    pub(crate) fn read_buffer(&self, device: &wgpu::Device, binding: u32) -> Vec<u8> {
        let source = self.buffer(binding);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cache readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &readback, 0, None);
        let index = self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        map_wait(device, &readback, index)
    }

    /// Merges slot use read back from the GPU into `last_used`.
    fn merge_used(&mut self, used: &[u32]) {
        for (l, &u) in self.last_used.iter_mut().zip(used) {
            *l = (*l).max(u);
        }
    }

    /// Queues every page unused for more than `age` frames, least recently
    /// used first (replacing the queue).
    fn queue_stale(&mut self, age: u32) {
        let mut stale: Vec<(u32, u32)> = (0..self.options.pages)
            .filter(|&slot| {
                self.slot_key[slot as usize][0] != EMPTY
                    && self.last_used[slot as usize] + age < self.frame
            })
            .map(|slot| (self.last_used[slot as usize], slot))
            .collect();
        // Most recently used first, so the least recently used pop first.
        stale.sort_unstable_by(|a, b| b.cmp(a));
        self.evict_queue = stale.into_iter().map(|(_, slot)| slot).collect();
    }

    /// Frees up to `count` queued pages that are still unused for more than
    /// `age` frames. Returns how many were freed.
    fn evict_queued(&mut self, count: usize, age: u32) -> u32 {
        let mut freed = 0;
        while freed < count {
            let Some(slot) = self.evict_queue.pop() else {
                break;
            };
            let k = self.slot_key[slot as usize];
            if k[0] == EMPTY || self.last_used[slot as usize] + age >= self.frame {
                continue;
            }
            let (_, entry) = self.map.remove(&k).expect("resident page");
            let map = &mut self.map;
            self.table.remove(entry as usize, |moved, i| {
                map.get_mut(&moved).expect("resident page").1 = i;
            });
            self.slot_key[slot as usize] = [EMPTY; 3];
            self.free.push(slot);
            freed += 1;
        }
        freed as u32
    }

    /// With the pool full: reads slot use back now (waiting for all GPU work)
    /// and frees pages unused for more than `age` frames, up to a frame's
    /// eviction count.
    fn evict_now(&mut self, device: &wgpu::Device, age: u32) -> u32 {
        let raw = if self.used_pending.is_some() {
            // The slot-use buffer is mapped or about to be: use it.
            let (_, index) = self.used_pending.take().expect("pending slot use");
            map_wait(device, &self.used_readback, index)
        } else {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(self.buffer(40), 0, &self.used_readback, 0, None);
            let index = self.queue.submit([encoder.finish()]);
            self.used_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            map_wait(device, &self.used_readback, index)
        };
        self.merge_used(bytemuck::cast_slice(&raw));
        self.queue_stale(age);
        self.evict_queued(EVICT_PER_FRAME, age)
    }

    /// Chooses the frame's pages from the requested ones (each listed once by
    /// the shader, with its priority), within the bake budget.
    fn plan(&mut self, device: &wgpu::Device, requests: Vec<([u32; 3], u32)>) {
        self.bake_list.clear();
        self.bake_bounds.clear();
        self.jobs.clear();
        self.sky_samples.clear();
        self.page_sky.clear();
        self.texels = 0;
        self.spent = 0;
        // Pages planned in the frame between are resident already.
        let mut order: Vec<(u32, [u32; 3])> = requests
            .into_iter()
            .filter(|(k, _)| !self.map.contains_key(k))
            .map(|(k, priority)| (priority, k))
            .collect();
        // Each page costs at least one texel: no more than `budget` can fit.
        let budget = (self.budget as u32).max(PAGE_TEXELS);
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
            // One texel past the grid on each side too, in case the shader's
            // grid is a rounding larger; it skips texels outside its own.
            let listed = [(inside[0] + 1).min(STRIDE), (inside[1] + 1).min(STRIDE)];
            let stride = self.sky_stride(m, mip);
            let samples = if self.sky {
                listed[0].div_ceil(stride) * listed[1].div_ceil(stride)
            } else {
                0
            };
            let cost = inside[0] * inside[1] * self.lights + samples * self.options.sky_sample_cost;
            // The first page always fits, so a page costlier than the whole
            // budget still gets baked.
            if self.spent + cost > budget && !self.bake_list.is_empty() {
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
            self.alloc_frame[slot as usize] = self.frame;
            let index = self.bake_list.len() as u32;
            self.bake_list.push([k[0], k[1], k[2], slot]);
            let b = self.page_bounds(m, c, mip, page);
            self.bake_bounds.push([b[0], b[1], b[2], b[3]]);
            self.bake_bounds.push([b[4], b[5], b[6], b[7]]);
            // Light by light, so neighbouring invocations trace one light.
            for l in 0..self.lights {
                for y in 0..listed[1] {
                    for x in 0..listed[0] {
                        self.jobs.push([index, (y * STRIDE + x) | l << 8]);
                    }
                }
            }
            if self.sky {
                let first = self.sky_samples.len() as u32;
                for y in (0..listed[1]).step_by(stride as usize) {
                    for x in (0..listed[0]).step_by(stride as usize) {
                        self.sky_samples.push([index, y * STRIDE + x]);
                    }
                }
                self.page_sky.push([first, samples | stride << 24]);
            }
            self.texels += inside[0] * inside[1];
            self.spent += cost;
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

    #[test]
    fn every_triangle_of_a_parametric_chart_lands_in_its_cell() {
        let s = Surface::parametric(6, 4, |u, v| Vec3::new(u, v, (u * v).sin()));
        let t = chart_tables(std::slice::from_ref(&s));
        let filled: usize = t.cells.iter().map(|c| c[1] as usize).sum();
        assert_eq!(filled, s.triangles.len());
        assert!(t.cells.iter().all(|c| c[1] <= 2));
        assert_eq!(t.cells.len(), 24);
    }

    #[test]
    fn deletion_keeps_every_remaining_page_findable() {
        let mut table = PageTable::new(1 << 10);
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
