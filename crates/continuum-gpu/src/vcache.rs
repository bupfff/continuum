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
/// Bounce light: samples per page, channels kept per sample (three lights
/// and the sky), slots listed incomplete per frame at most, and frames to
/// wait before re-baking an incomplete page's samples (its requested pages
/// arrive in about that).
const INDIRECT_SAMPLES: u32 = 9;
/// Bounce light's channels per sample: up to three lights, the sky, then
/// emission (M10 gate 11), which also takes the bounce of later lights at
/// their radiance (M10 gate 21).
const INDIRECT_CHANNELS: u64 = 5;
const INCOMPLETE_CAP: u32 = 1 << 14;
/// Pages whose bake met an absent lighting level (M14 gate 4), with that
/// geometry's object, listed per frame at most (pairs after the incomplete
/// list; the shader's `MISSING_CAP`).
const MISSING_CAP: u32 = 1 << 14;
/// Flags a missing list's slot as a bounce sample's want (the shader's
/// `BOUNCE_WANT`, M14 gate 8).
const BOUNCE_WANT: u32 = 0x8000_0000;
const INDIRECT_RETRY: u32 = 4;
/// Request priorities' flags (the shader's `SECONDARY`, `WANTS_SAMPLES`).
const SECONDARY: u32 = 0x8000_0000;
const WANTS_SAMPLES: u32 = 0x4000_0000;
/// The stale passes' and the ledger's counters (16 words) in each readback.
const MOVER_STATS_BYTES: u64 = 96;
/// Movers under a sky (M9 gate 3): slots listed for a sky re-bake per frame
/// at most, and frames a moved object's page waits (re-listed while it
/// moves) before its sky re-bakes.
const SKY_LIST_CAP: u32 = 1 << 14;
const SKY_SETTLE: u32 = 30;
/// Moving lights (M9 gate 2): frames of light poses kept (a page older than
/// this, or than 1,023 frames, re-bakes whole), and words of ledger record
/// per page and moving light.
const POSE_RING: u32 = 1024;
const LEDGER_WORDS: u64 = 48;

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
    /// Moving lights (M9 gate 2): the largest change of a texel's ratio
    /// (its light's shadowed share) the cache may show before re-baking it.
    pub ledger_epsilon: f32,
    /// Bounce light (M10 gate 1): one bounce of diffuse light, sampled on a
    /// 3×3 grid per page (192 rays each), from the direct light the cache
    /// holds at the hits. The first three lights' bounce re-colors free; later
    /// lights' is baked at their radiance, so re-coloring one re-bakes bounce
    /// light (M10 gate 21). Not with movers or moving lights. Off by default.
    pub indirect: bool,
    /// With bounce light (M10 gate 11; on by default): its samples also
    /// gather the emission of the surfaces they hit, so emissive geometry
    /// lights its surroundings (one bounce).
    pub emission_bounce: bool,
    /// What a bounce sample counts against the bake budget, in sun texels.
    pub indirect_sample_cost: u32,
    /// Anti-aliasing at silhouettes (M7 gates 3–4): pixels whose neighbours
    /// see another object or depth average four rays through their
    /// quarters. With hardware camera rays (and no movers) the rays are ray
    /// queries in their own passes: 1.29× the frame on a textured scene,
    /// 3.03× on the field from 150 m up, where half the pixels are edges
    /// (compute rays: 1.84× and 24×).
    pub edge_rays: bool,
    /// With `edge_rays` through ray queries: 4 quarter-rays per edge pixel,
    /// or 2 (one diagonal pair, alternating from pixel to pixel), averaged
    /// with the pixel's own ray.
    pub edge_samples: u32,
    /// Experiments only: glossy reflection rays for materials (M10 gate 2;
    /// on by default).
    #[doc(hidden)]
    pub glossy_rays: bool,
    /// With a sky: the spacing (in each geometry's own units) of the grid on
    /// which its own sky occlusion is baked once, for all its instances (M5
    /// gate 5). Sky texels then walk only the other objects. 0: off. Only
    /// geometries with several instances get masks; a geometry's spacing
    /// grows so it holds at most 65,536 texels, and all of them so a scene
    /// holds at most 2^20 (384 MB). A geometry whose grid would exceed 1,024
    /// texels along a chart walks itself.
    pub self_spacing: f32,
    /// With a sky (M5 gate 9; on by default): a page the pixels ask for bakes
    /// its lights first, its sky shown as the open sky over each pixel's
    /// normal until a later frame bakes the sky alone (up to half of each
    /// frame's budget before new pages, then what they leave; M9 gate 5). Views come within 5% of their warm
    /// image 10–20× sooner and warm in 0.6× the frames; the warm image is
    /// the same. Ignored with bounce light (its samples keep whatever
    /// coarser pages they find, so another bake order settles differently)
    /// or moving lights. With movers (M9 gate 5), a page whose sky is still
    /// a guess is not charged for its sky, and one whose light a mover may
    /// change is marked invalid and baked anew.
    pub sky_later: bool,
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
            ledger_epsilon: 0.01,
            indirect: false,
            emission_bounce: true,
            indirect_sample_cost: 8,
            edge_rays: false,
            edge_samples: 4,
            glossy_rays: true,
            self_spacing: 0.01,
            sky_later: true,
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
    /// Diagnostics: of `spent`, the units of the pages planned per mip level,
    /// from mip 6 (or finer) to mip 13 (or coarser).
    pub spent_by_mip: [u32; 8],
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
    /// Moving lights (M9 gate 2), in frames where one moved: pages with
    /// texels the ledger re-baked.
    pub ledger_pages: u32,
    /// Texel-light pairs the ledger re-baked.
    pub ledger_texels: u32,
    /// Pages the ledger marked invalid.
    pub ledger_invalidated: u32,
    /// Pages the ledger checked that were read in the frame before.
    pub ledger_checked: u32,
    /// Of those, pages with mixed texels.
    pub ledger_mixed: u32,
    /// Of those, pages passed whole (every texel fixed, clear or dark,
    /// within the guard).
    pub ledger_whole: u32,
    /// Bounce light (M10 gate 1): samples baked in the frame.
    pub indirect_samples: u32,
    /// Of those, samples that met hits with no page resident (re-baked
    /// later).
    pub indirect_incomplete: u32,
    /// Pixels shaded with four rays at silhouettes (M7 gate 3; not counted
    /// with `KernelOptions::count_work`).
    pub edge_pixels: u32,
    /// Movers under a sky (M9 gate 3): slots listed for a sky re-bake.
    pub sky_listed: u32,
    /// Pages read in the frame before that movers' sky charges touched.
    pub sky_charged: u32,
    /// Pages the sky ledger marked invalid.
    pub sky_invalidated: u32,
    /// Pages whose sky the plan re-baked (for movers).
    pub sky_rebaked: u32,
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
    ledger: [u32; 4],
    indirect: [u32; 4],
    extra: [u32; 4],
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

/// Geometries from `skip` on (streamed lighting levels, M14 gate 5) keep
/// their charts and cells, but list no triangles: their packages carry
/// those.
fn chart_tables(geometries: &[Surface], skip: usize) -> ChartTables {
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
    for (index, g) in geometries.iter().enumerate() {
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
        if index >= skip {
            // No triangles listed, and no boxes kept (`page_bounds` gives
            // such pages an empty box).
            for c in first_cell..t.cells.len() {
                t.cells[c] = [0, 0];
            }
            continue;
        }
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

/// Geometries from `streamed` on (lighting levels streamed, M14 gate 8) get
/// none: their levels are not resident to bake masks from, so their sky
/// walks cover their own level as any other's.
fn self_tables(
    scene: &InstancedScene,
    tables: &ChartTables,
    spacing: f32,
    streamed: usize,
) -> SelfTables {
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
        let candidates: Vec<usize> = (0..tables.first_chart.len().min(streamed))
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
    /// Whether pixels may read it: not marked stale, set aside, or baked
    /// against an absent lighting level (M14 gate 4).
    pub shown: bool,
    /// Whether its sky is baked (not the guess a page planned with the sky
    /// later shows until its sky is baked alone; M14 gate 8).
    pub sky_baked: bool,
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
    /// Moving lights (M9 gate 2): how many, the lights' poses now (written
    /// into the ring each frame), and whether one moved since the last
    /// frame.
    moving: u32,
    poses: Vec<crate::LightGpu>,
    lights_moved: bool,
    /// Bounce light: this frame's samples (entry in `bake_list`, sample),
    /// and slots whose samples to re-bake, each with the frame it is due.
    indirect_jobs: Vec<[u32; 2]>,
    retry: std::collections::VecDeque<(u32, u32)>,
    retrying: std::collections::HashSet<u32>,
    /// Pages baked for bounce samples' hits (secondary requests), with their
    /// keys: kept resident while any page's bounce samples wait for their
    /// hits. Bounce samples complete only when every hit finds a page at
    /// once; off-screen hit pages, read by no pixel, otherwise aged out
    /// before the rest arrived, and the waiting page retried forever (M10
    /// gate 23's view that never settled).
    pinned: std::collections::VecDeque<(u32, [u32; 3])>,
    /// Lighting levels streamed (M14 gate 4): the layout's cell part and
    /// the streamed geometries' table packages (taken by the renderer);
    /// pages whose bake met absent lighting levels, waiting for those
    /// objects (slot → key, objects); per object whether its lighting level
    /// is resident (as the frame began); and this frame's wanted and pinned
    /// objects.
    lighting: bool,
    lit_cells: Option<crate::stream::LitCells>,
    lit_waiting: HashMap<u32, ([u32; 3], Vec<u32>, u32)>,
    /// Objects of pages released from waiting, pinned until the frame given
    /// (their re-bake comes two frames later).
    lit_released: Vec<(u32, u32)>,
    /// Pages asked for on objects whose lighting level was absent, oldest
    /// first, with their priority and the frame first asked for: planned as
    /// soon as the level is resident (a pixel asks for a missing page only
    /// now and then, so waiting for it to ask again wastes the load).
    lit_pending: std::collections::VecDeque<([u32; 3], u32, u32)>,
    lit_pending_keys: std::collections::HashSet<[u32; 3]>,
    /// Every object each page's bakes have found absent, kept across its
    /// re-bakes (a bake reports only the first): it waits for all of them.
    lit_needs: HashMap<[u32; 3], Vec<u32>>,
    lit_resident: Vec<bool>,
    lit_wants: Vec<u32>,
    /// Levels bounce samples met absent (M14 gate 8), wanted until their
    /// retries are due (and a few frames more).
    lit_bounce: Vec<(u32, u32)>,
    lit_pins: Vec<u32>,
    sky_rebaked: u32,
    /// Stable indices of objects that have moved (for the sky ledger).
    mover_ids: HashMap<u32, u32>,
    /// Movers under a sky: slots whose sky to re-bake, each with the frame
    /// it is due.
    sky_due: HashMap<u32, u32>,
    /// Silhouette rays through ray queries (M7 gate 4): `hw_edges` lists
    /// edge pixels for `vc_shade_edges` (buffer 81; counted in the stats).
    hardware_edges: bool,
    /// The extra samples' list (buffer 81): silhouette rays through ray
    /// queries, or layers behind alpha-blended surfaces (M10 gate 16).
    extra_samples: bool,
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
    spent_by_mip: [u32; 8],
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
        moving: &[usize],
        poses: &[crate::LightGpu],
        pool: u32,
        lighting: Option<&crate::stream::LightingLayout>,
    ) -> Self {
        let channels = lights + 1;
        assert!(
            moving.is_empty() || movers == 0,
            "moving lights and moving objects together in the cache are not supported yet"
        );
        assert!(
            moving.is_empty() || !sky,
            "moving lights with a sky in the cache are not supported yet"
        );
        assert!(
            !options.indirect || (movers == 0 && moving.is_empty()),
            "bounce light with moving objects or lights is not supported yet"
        );
        let mut moving_index = vec![u32::MAX; lights as usize];
        for (j, &l) in moving.iter().enumerate() {
            moving_index[l] = j as u32;
        }
        let append_pages = if movers > 0 || !moving.is_empty() {
            APPEND_PAGES.min(APPEND_JOBS / (PAGE_TEXELS * lights))
        } else {
            0
        };
        let append_jobs = (append_pages * PAGE_TEXELS * lights) as u64;
        let sky_append = if movers > 0 && sky {
            append_pages as u64 * PAGE_TEXELS as u64
        } else {
            0
        };
        // So a frame's sky samples never outnumber the budget (their buffers'
        // size).
        assert!(
            options.sky_sample_cost >= 1,
            "a sky sample costs at least 1"
        );
        let mut tables = chart_tables(&scene.geometries, lighting.map_or(usize::MAX, |l| l.first));
        let own = self_tables(
            scene,
            &tables,
            if sky { options.self_spacing } else { 0.0 },
            lighting.map_or(usize::MAX, |l| l.first),
        );
        // Lighting levels streamed (M14 gates 4–5): the streamed geometries'
        // chart coordinates, charts and cell lists come in packages; their
        // cells stay, empty until a level loads.
        let lit_cells = lighting.map(|lay| {
            let static_cells = tables.cell_tris.len();
            let t_s = lay.static_tris as usize;
            let slots = (lay.slots * lay.slot_tris) as usize;
            tables.tri_uv.truncate(3 * t_s);
            tables.tri_uv.resize(3 * (t_s + slots), [0.0; 2]);
            tables.tri_chart.truncate(t_s);
            tables.tri_chart.resize(t_s + slots, 0);
            tables
                .cell_tris
                .resize(static_cells + (lay.slots * lay.slot_cells) as usize, 0);
            crate::stream::LitCells {
                static_cells: static_cells as u32,
                slot_cells: lay.slot_cells,
            }
        });
        // Camera levels' pool (M14 gate 1): their charts and chart
        // coordinates follow the lighting level's triangles'.
        if pool > 0 {
            let lit: usize = lighting.map_or_else(
                || scene.geometries.iter().map(|g| g.triangles.len()).sum(),
                |l| l.lit_tris() as usize,
            );
            tables.tri_chart.resize(lit + pool as usize, 0);
            tables.tri_uv.resize(3 * (lit + pool as usize), [0.0; 2]);
        }
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
                // Copied only by diagnostics (`read_pages`, the checks).
                empty(
                    "page pool",
                    options.pages as u64 * PAGE_TEXELS as u64 * channels as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
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
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
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
            (66, empty("dispatch args", 48, wgpu::BufferUsages::INDIRECT)),
            (
                68,
                empty("light poses", (POSE_RING * lights) as u64 * 64, copy_dst),
            ),
            (
                69,
                empty(
                    "ledger",
                    options.pages as u64 * moving.len() as u64 * LEDGER_WORDS * 4,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                70,
                storage(
                    "moving index",
                    bytemuck::cast_slice(&moving_index),
                    copy_dst,
                ),
            ),
            (
                72,
                empty(
                    "ledger bound",
                    options.pages as u64 * moving.len() as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                73,
                empty(
                    "bounce samples",
                    if options.indirect {
                        options.pages as u64 * INDIRECT_SAMPLES as u64 * INDIRECT_CHANNELS * 8
                    } else {
                        16
                    },
                    // Cleared when a light past the third is re-colored
                    // (M10 gate 21).
                    wgpu::BufferUsages::COPY_DST,
                ),
            ),
            (
                74,
                empty(
                    "bounce jobs",
                    (budget + 81) * INDIRECT_SAMPLES as u64 * 8,
                    copy_dst,
                ),
            ),
            (
                75,
                empty(
                    "incomplete",
                    INCOMPLETE_CAP as u64 * 4 + MISSING_CAP as u64 * 8,
                    wgpu::BufferUsages::COPY_SRC,
                ),
            ),
            // Pairs listed missing this frame, a hash bitset (M14 gate 8).
            (83, empty("lit seen", 1 << 13, copy_dst)),
            (
                77,
                empty(
                    "sky ledger",
                    if movers > 0 && sky {
                        options.pages as u64 * 32
                    } else {
                        32
                    },
                    wgpu::BufferUsages::COPY_SRC | copy_dst,
                ),
            ),
            (
                78,
                empty(
                    "sky list",
                    SKY_LIST_CAP as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC,
                ),
            ),
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
            // With movers, the stale pass appends up to `sky_append` more.
            (
                46,
                empty("sky samples", (budget + 81 + sky_append) * 8, copy_dst),
            ),
            (
                47,
                empty(
                    "sky partials",
                    (budget + 81 + sky_append) * 6 * 4,
                    wgpu::BufferUsages::empty(),
                ),
            ),
            (
                48,
                empty("page sky", (budget + append_pages as u64) * 8, copy_dst),
            ),
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
            moving: moving.len() as u32,
            poses: poses.to_vec(),
            lights_moved: false,
            indirect_jobs: Vec::new(),
            retry: Default::default(),
            retrying: Default::default(),
            pinned: Default::default(),
            lighting: lit_cells.is_some(),
            lit_cells,
            lit_waiting: HashMap::new(),
            lit_released: Vec::new(),
            lit_pending: Default::default(),
            lit_pending_keys: Default::default(),
            lit_needs: HashMap::new(),
            lit_resident: Vec::new(),
            lit_wants: Vec::new(),
            lit_bounce: Vec::new(),
            lit_pins: Vec::new(),
            sky_due: HashMap::new(),
            hardware_edges: false,
            extra_samples: false,
            sky_rebaked: 0,
            mover_ids: HashMap::new(),
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
            spent_by_mip: [0; 8],
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
        if self.extra_samples {
            self.buffers.retain(|(b, _)| *b != 81);
            self.buffers.push((
                81,
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("extra samples"),
                    size: (width as u64 * height as u64 * 80).max(80),
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
            ));
        }
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
                        + INVALID_CAP as u64 * 4
                        + INCOMPLETE_CAP as u64 * 4
                        + SKY_LIST_CAP as u64 * 4
                        + MISSING_CAP as u64 * 8,
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
        let stale = self.buffer(62);
        let stale_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stale readback"),
            size: stale.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(pool, 0, &readback, 0, None);
        encoder.copy_buffer_to_buffer(stale, 0, &stale_readback, 0, None);
        let index = self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        stale_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        let raw = map_wait(device, &readback, index.clone());
        let stale_raw = map_wait(device, &stale_readback, index);
        let stale_marks: &[u32] = bytemuck::cast_slice(&stale_raw);
        let values: &[f32] = bytemuck::cast_slice(&raw);
        // The pool keeps a sky channel after the lights' whether or not the
        // sky is on (the shader's `vc_channels`).
        let channels = (self.lights + 1) as usize;
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
                shown: stale_marks.get(slot).is_none_or(|&m| m == 0 || m == 3),
                sky_baked: stale_marks.get(slot).is_none_or(|&m| m == 0),
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
    /// stale pass runs if any), and 1 if a moving light moved (the ledger
    /// runs).
    pub(crate) fn upload(
        &mut self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        vertical_fov_degrees: f32,
    ) -> [u32; 6] {
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
        if !self.indirect_jobs.is_empty() {
            queue.write_buffer(
                self.buffer(74),
                0,
                bytemuck::cast_slice(&self.indirect_jobs),
            );
        }
        let moved = std::mem::take(&mut self.moved_count);
        if moved > 0 {
            queue.write_buffer(self.buffer(60), 0, bytemuck::cast_slice(&self.moved));
        }
        let lit = std::mem::take(&mut self.lights_moved) && self.moving > 0;
        if self.moving > 0 {
            queue.write_buffer(
                self.buffer(68),
                (self.frame % POSE_RING) as u64 * self.lights as u64 * 64,
                bytemuck::cast_slice(&self.poses),
            );
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
            ledger: [
                self.moving,
                POSE_RING,
                self.options.ledger_epsilon.to_bits(),
                // A limit on the ledger's re-bakes: the appended jobs'
                // capacity (M9 gate 4: a limit of eight times the frame's
                // budget bound in gate 2 and set visible pages aside); the
                // top bit tells whether a moving light moved.
                self.append_jobs().min(0x7fff_ffff) | (lit as u32) << 31,
            ],
            indirect: [
                self.indirect_jobs.len() as u32,
                self.options.indirect as u32,
                !self.options.glossy_rays as u32,
                if self.hardware_edges {
                    if self.options.edge_samples == 2 { 3 } else { 2 }
                } else {
                    self.options.edge_rays as u32
                },
            ],
            // x: 1 if bounce samples gather emission (M10 gate 11).
            extra: [
                (self.options.indirect && self.options.emission_bounce) as u32,
                0,
                0,
                0,
            ],
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
            spent_by_mip: self.spent_by_mip,
            timed: self.timer.is_some(),
            evicting: std::mem::take(&mut self.evicting),
            waiting: std::mem::take(&mut self.waiting),
            moved: moved.max(lit as u32),
            indirect_samples: self.indirect_jobs.len() as u32,
            sky_rebaked: self.sky_rebaked,
            ..Default::default()
        });
        self.moved.clear();
        [
            (updates.len() / 2) as u32,
            self.jobs.len() as u32,
            6 * self.sky_samples.len() as u32,
            self.bake_list.len() as u32,
            moved,
            // The ledger runs every frame with moving lights (it re-bakes
            // pages it set aside once wanted).
            (self.moving > 0) as u32,
        ]
    }

    /// Moving lights: light `index` is now `pose` (from the next frame).
    pub(crate) fn set_light_pose(&mut self, index: usize, pose: crate::LightGpu) {
        self.poses[index] = pose;
        self.lights_moved = true;
    }

    /// Bounce samples to bake this frame (after `upload`).
    pub(crate) fn indirect_jobs(&self) -> u32 {
        self.indirect_jobs.len() as u32
    }

    /// Moving lights tracked by the ledger.
    pub(crate) fn moving_lights(&self) -> u32 {
        self.moving
    }

    /// The ledger's ε.
    /// Diagnostics: the frame slot `slot` was allocated, and its key.
    pub(crate) fn slot_debug(&self, slot: u32) -> String {
        format!(
            "alloc {} of {}, key {:?}",
            self.alloc_frame[slot as usize], self.frame, self.slot_key[slot as usize]
        )
    }

    /// Whether the cache holds a sky channel.
    pub(crate) fn has_sky(&self) -> bool {
        self.sky
    }

    pub(crate) fn epsilon(&self) -> f32 {
        self.options.ledger_epsilon
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
    /// as bits in w), old hi (with a stable index among movers as bits in
    /// w, for the sky ledger), new lo, new hi, and per light the old and new
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
            let next = self.mover_ids.len() as u32;
            let id = *self.mover_ids.entry(*object).or_insert(next);
            entries.push(v4(old.0, f32::from_bits(*object)));
            entries.push(v4(old.1, f32::from_bits(id)));
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
    pub(crate) fn upload_rebake(&mut self, queue: &wgpu::Queue, slots: &[u32]) -> [u32; 6] {
        let mut list = Vec::new();
        let mut bounds = Vec::new();
        let mut jobs = Vec::new();
        self.sky_samples.clear();
        self.page_sky.clear();
        for (index, &slot) in slots.iter().enumerate() {
            let k = self.slot_key[slot as usize];
            let (m, c, mip, page) = unpack(k);
            list.push([k[0], k[1], k[2], slot]);
            let b = self.page_bounds(m, c, mip, page);
            bounds.push([b[0], b[1], b[2], b[3]]);
            bounds.push([b[4], b[5], b[6], b[7]]);
            if self.sky {
                self.push_sky_samples(index as u32, k);
            }
            for l in 0..self.lights {
                for t in 0..PAGE_TEXELS {
                    jobs.push([index as u32, t | l << 8]);
                }
            }
        }
        queue.write_buffer(self.buffer(38), 0, bytemuck::cast_slice(&list));
        queue.write_buffer(self.buffer(63), 0, bytemuck::cast_slice(&bounds));
        queue.write_buffer(self.buffer(43), 0, bytemuck::cast_slice(&jobs));
        if self.sky && !self.sky_samples.is_empty() {
            queue.write_buffer(self.buffer(46), 0, bytemuck::cast_slice(&self.sky_samples));
            queue.write_buffer(self.buffer(48), 0, bytemuck::cast_slice(&self.page_sky));
        }
        let sky_samples = self.sky_samples.len() as u32;
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
            sky_samples,
            self.sky as u32,
            self.lights,
            self.options.light_ratio as u32,
        ];
        params.movers = [0, self.append_pages, self.append_jobs(), VISIBLE_WINDOW];
        params.ledger = [
            self.moving,
            POSE_RING,
            self.options.ledger_epsilon.to_bits(),
            0x7fff_ffff,
        ];
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        [
            0,
            jobs.len() as u32,
            6 * sky_samples,
            list.len() as u32,
            0,
            0,
        ]
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
                if let Some(b) = self.cell_bounds.get((first + y * g[0] + x) as usize) {
                    (lo, hi) = (lo.min(b[0]), hi.max(b[1]));
                }
            }
        }
        if lo.x > hi.x {
            (lo, hi) = (Vec3::ZERO, Vec3::ZERO);
        }
        [lo.x, lo.y, lo.z, 0.0, hi.x, hi.y, hi.z, 0.0]
    }

    /// Edge rays through ray queries (M7 gate 4), with `edge_rays` on: the
    /// renderer's `hw_edges` and `vc_shade_edges` passes trace and shade
    /// them instead of the shading pass's compute rays. Before the first
    /// `resize`, which makes their buffers.
    /// Bounce light is on.
    pub(crate) fn indirect(&self) -> bool {
        self.options.indirect
    }

    pub(crate) fn set_hardware_edges(&mut self, blend: bool) {
        self.hardware_edges = self.options.edge_rays;
        self.extra_samples = self.hardware_edges || blend;
    }

    /// Clears the request tiles and counters before shading.
    pub(crate) fn encode_clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.buffer(42), 0, None);
        encoder.clear_buffer(self.buffer(45), 0, None);
        encoder.clear_buffer(self.buffer(64), 0, None);
        if self.lighting {
            encoder.clear_buffer(self.buffer(83), 0, None);
        }
    }

    /// Copies the frame's requests and counters (and every 32 frames, slot
    /// use) for reading back once the frame is done.
    pub(crate) fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let r = &self.readbacks[(self.frame % 2) as usize];
        let bytes = r.tiles as u64 * 16;
        encoder.copy_buffer_to_buffer(self.buffer(39), 0, &r.buffer, 0, bytes);
        encoder.copy_buffer_to_buffer(self.buffer(42), 0, &r.buffer, bytes, STATS_BYTES);
        if self.log.last().is_some_and(|l| l.moved > 0) || self.options.indirect || self.lighting {
            let at = bytes + STATS_BYTES + TIMESTAMP_BYTES;
            if self.lighting {
                encoder.copy_buffer_to_buffer(
                    self.buffer(75),
                    INCOMPLETE_CAP as u64 * 4,
                    &r.buffer,
                    at + MOVER_STATS_BYTES
                        + (INVALID_CAP + INCOMPLETE_CAP + SKY_LIST_CAP) as u64 * 4,
                    MISSING_CAP as u64 * 8,
                );
            }
            encoder.copy_buffer_to_buffer(self.buffer(64), 0, &r.buffer, at, MOVER_STATS_BYTES);
            encoder.copy_buffer_to_buffer(
                self.buffer(65),
                0,
                &r.buffer,
                at + MOVER_STATS_BYTES,
                INVALID_CAP as u64 * 4,
            );
            if self.options.indirect {
                encoder.copy_buffer_to_buffer(
                    self.buffer(75),
                    0,
                    &r.buffer,
                    at + MOVER_STATS_BYTES + INVALID_CAP as u64 * 4,
                    INCOMPLETE_CAP as u64 * 4,
                );
            }
            if self.sky {
                encoder.copy_buffer_to_buffer(
                    self.buffer(78),
                    0,
                    &r.buffer,
                    at + MOVER_STATS_BYTES + (INVALID_CAP + INCOMPLETE_CAP) as u64 * 4,
                    SKY_LIST_CAP as u64 * 4,
                );
            }
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
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device, lit_resident: Option<&[bool]>) {
        let start = Instant::now();
        let period = self.last_begin.map(|t| start - t);
        self.last_begin = Some(start);
        let slot = (self.frame % 2) as usize;
        if let Some(r) = lit_resident {
            self.lit_resident.clear();
            self.lit_resident.extend_from_slice(r);
        }
        self.lit_wants.clear();
        self.lit_pins.clear();
        let requests = self.collect(device, slot);
        self.release_waiting();
        let now = self.frame;
        self.lit_bounce.retain(|&(_, until)| until >= now);
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
        self.refresh_pinned();
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
        // Bounce samples' wants last: the slots serve direct light first.
        if self.lighting {
            self.lit_wants
                .extend(self.lit_bounce.iter().map(|&(o, _)| o));
        }
        if self.lighting && trace_lighting() {
            let resident = self.lit_resident.iter().filter(|&&r| r).count();
            eprintln!(
                "lit frame {}: requests {}, planned {}, wants {:?} ({} in all), pins {}, waiting {} (needing at most {} levels), objects resident {}",
                self.frame,
                self.log.last().map_or(0, |l| l.requests),
                self.bake_list.len(),
                &self.lit_wants[..self.lit_wants.len().min(12)],
                self.lit_wants.len(),
                self.lit_pins.len(),
                self.lit_waiting.len(),
                self.lit_waiting
                    .values()
                    .map(|w| w.1.len() + 1)
                    .max()
                    .unwrap_or(0),
                resident
            );
        }
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
        let mut mover = [0u32; 24];
        let mut invalid: Vec<u32> = Vec::new();
        let mut incomplete: Vec<u32> = Vec::new();
        let mut sky_listed: Vec<u32> = Vec::new();
        let mut missing: Vec<(u32, u32)> = Vec::new();
        if moved || self.options.indirect || self.lighting {
            let at = 4 * tiles + (STATS_BYTES + TIMESTAMP_BYTES) as usize / 4;
            mover.copy_from_slice(&words[at..at + 24]);
            let listed = (mover[1] as usize).min(INVALID_CAP as usize);
            let first = at + MOVER_STATS_BYTES as usize / 4;
            invalid.extend_from_slice(&words[first..first + listed]);
            if mover[1] as usize > listed {
                // More than the list holds: every slot's mark is read back.
                self.overflowed = Some(frame);
            }
            if self.options.indirect {
                let first = first + INVALID_CAP as usize;
                let listed = (mover[15] as usize).min(INCOMPLETE_CAP as usize);
                incomplete.extend_from_slice(&words[first..first + listed]);
            }
            if self.sky {
                let first = first + (INVALID_CAP + INCOMPLETE_CAP) as usize;
                let listed = (mover[7] as usize).min(SKY_LIST_CAP as usize);
                sky_listed.extend_from_slice(&words[first..first + listed]);
            }
            if self.lighting {
                let first = first + (INVALID_CAP + INCOMPLETE_CAP + SKY_LIST_CAP) as usize;
                let listed = (mover[20] as usize).min(MISSING_CAP as usize);
                missing.extend(
                    words[first..first + 2 * listed]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|p| (p[0], p[1])),
                );
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
        record.ledger_pages = mover[8];
        record.ledger_texels = mover[9];
        record.ledger_invalidated = mover[10];
        record.ledger_checked = mover[11];
        record.ledger_mixed = mover[12];
        record.ledger_whole = mover[13];
        record.indirect_incomplete = mover[15];
        record.sky_listed = mover[7];
        record.sky_rebaked += mover[19];
        record.sky_charged = mover[16];
        record.sky_invalidated = mover[17];
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
        record.edge_pixels = counters[15];
        self.appended = 0.8 * self.appended + 0.2 * appended as f32;
        self.evict_invalid(&invalid, frame);
        // Sky re-bakes for movers (M9 gate 3): a receiver's now; a moved
        // object's page (listed once per sky bake) a while later.
        for entry in sky_listed {
            let (slot, own) = (entry & 0x7fff_ffff, entry >> 31 == 1);
            let due = if own { frame + SKY_SETTLE } else { frame };
            self.sky_due.insert(slot, due);
        }
        // Incomplete bounce samples: re-bake once the requested pages have
        // had time to arrive.
        for slot in incomplete {
            if self.retrying.insert(slot) {
                self.retry.push_back((slot, frame + INDIRECT_RETRY));
            }
        }
        // Pages baked against absent lighting levels (M14 gate 4): shown as
        // missing until re-baked; they wait for those objects.
        for (entry, object) in missing {
            // Flagged: a bounce sample's want (M14 gate 8): the page stays
            // shown and its samples retry, with the level wanted and pinned
            // for the retry.
            let bounce = entry & BOUNCE_WANT != 0;
            let slot = entry & !BOUNCE_WANT;
            let k = self.slot_key[slot as usize];
            if k[0] == EMPTY {
                continue;
            }
            // Bounce's wants stay apart from direct light's needs: a page's
            // direct light never waits on what its bounce rays reach (more
            // levels, often, than the slots hold).
            if bounce {
                self.lit_bounce.push((object, frame + INDIRECT_RETRY + 8));
                if self.retrying.insert(slot) {
                    self.retry.push_back((slot, frame + INDIRECT_RETRY));
                }
                continue;
            }
            if self.lit_needs.len() > 1 << 16 {
                self.lit_needs.clear();
            }
            let needs = self.lit_needs.entry(k).or_default();
            if !needs.contains(&object) {
                needs.push(object);
            }
            let needs = needs.clone();
            let entry = self
                .lit_waiting
                .entry(slot)
                .or_insert((k, Vec::new(), frame));
            if entry.0 != k {
                *entry = (k, Vec::new(), frame);
            }
            entry.1 = needs;
        }
        self.latest = self.latest.max(frame);
        requests
    }

    /// Pages waiting for lighting levels (M14 gate 4): once all their
    /// objects' levels are resident, they are evicted, so pixels ask for
    /// them again and they bake whole (their objects pinned meanwhile);
    /// until then their objects are wanted, the longest waiting first. A
    /// waiting page pins nothing: pins held by pages that cannot progress
    /// would deadlock a small budget.
    fn release_waiting(&mut self) {
        let frame = self.frame;
        self.lit_released.retain(|&(_, until)| until >= frame);
        let mut waiting: Vec<(u32, u32, [u32; 3], Vec<u32>)> = self
            .lit_waiting
            .iter()
            .map(|(&s, (k, o, since))| (*since, s, *k, o.clone()))
            .collect();
        waiting.sort_unstable_by_key(|w| (w.0, w.1));
        for (_, slot, k, objects) in waiting {
            if self.slot_key[slot as usize] != k {
                self.lit_waiting.remove(&slot);
                continue;
            }
            let ready = objects
                .iter()
                .chain(std::iter::once(&k[0]))
                .all(|&o| self.lit_resident.get(o as usize).copied().unwrap_or(true));
            if ready {
                self.lit_waiting.remove(&slot);
                if let Some((_, entry)) = self.map.remove(&k) {
                    let map = &mut self.map;
                    self.table.remove(entry as usize, |moved, i| {
                        map.get_mut(&moved).expect("resident page").1 = i;
                    });
                    self.slot_key[slot as usize] = [EMPTY; 3];
                    self.sky_due.remove(&slot);
                    self.free.push(slot);
                    self.evicted += 1;
                }
                for &o in objects.iter().chain(std::iter::once(&k[0])) {
                    self.lit_released.push((o, frame + 4));
                }
                if self.lit_pending_keys.insert(k) {
                    self.lit_pending.push_front((k, 1, frame));
                }
            } else {
                self.lit_wants.push(k[0]);
                self.lit_wants.extend(&objects);
            }
        }
        self.lit_pins
            .extend(self.lit_released.iter().map(|&(o, _)| o));
    }

    /// This frame's lighting-level wants (absent levels pages wait for or
    /// are asked on) and pins (levels pages wait for or bake on now), M14
    /// gate 4.
    pub(crate) fn lighting_requests(&self) -> (&[u32], &[u32]) {
        (&self.lit_wants, &self.lit_pins)
    }

    /// Takes the cell part of the layout (once).
    pub(crate) fn take_lit_cells(&mut self) -> Option<crate::stream::LitCells> {
        self.lit_cells.take()
    }

    /// Pages waiting for lighting levels.
    pub(crate) fn waiting_pages(&self) -> usize {
        self.lit_waiting.len()
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

    /// Keeps pages baked for bounce hits in use while bounce samples wait
    /// (see `pinned`), at most a quarter of the pool, the latest first;
    /// releases them once none wait.
    fn refresh_pinned(&mut self) {
        if self.retry.is_empty() {
            self.pinned.clear();
            return;
        }
        let cap = (self.options.pages / 4) as usize;
        while self.pinned.len() > cap {
            self.pinned.pop_front();
        }
        let (keys, used, frame) = (&self.slot_key, &mut self.last_used, self.frame);
        self.pinned.retain(|&(slot, k)| {
            let still = keys[slot as usize] == k;
            if still {
                used[slot as usize] = frame;
            }
            still
        });
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
            self.sky_due.remove(&slot);
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
    /// Lists skies due to be re-baked alone (for movers, and pages planned
    /// with the sky later), oldest first, until the frame's work reaches
    /// `limit`. A listed page counts as used this frame, so the frame's
    /// eviction keeps it.
    fn plan_due_skies(&mut self, limit: u32) {
        if !self.sky || self.sky_due.is_empty() {
            return;
        }
        let mut due: Vec<(u32, u32)> = self
            .sky_due
            .iter()
            .filter(|&(_, &d)| d <= self.frame)
            .map(|(&s, &d)| (d, s))
            .collect();
        // Oldest first, then by slot: the same order every run.
        due.sort_unstable();
        for (_, slot) in due {
            if self.spent >= limit || self.bake_list.len() >= limit.max(PAGE_TEXELS) as usize {
                break;
            }
            let k = self.slot_key[slot as usize];
            if k[0] == EMPTY {
                self.sky_due.remove(&slot);
                continue;
            }
            // Only within the limit (M5 gate 9: a sky that does not fit
            // waits, as a new page's would).
            let cost = self.sky_layout(k).2 * self.options.sky_sample_cost;
            if self.spent + cost > limit && !self.bake_list.is_empty() {
                break;
            }
            if !self.lighting_ready(k) {
                continue;
            }
            self.sky_due.remove(&slot);
            self.last_used[slot as usize] = self.frame;
            let (m, c, mip, page) = unpack(k);
            let index = self.bake_list.len() as u32;
            self.bake_list.push([k[0], k[1], k[2], slot]);
            let b = self.page_bounds(m, c, mip, page);
            self.bake_bounds.push([b[0], b[1], b[2], b[3]]);
            // w 3: the sky alone (the fill keeps an invalid mark).
            self.bake_bounds.push([b[4], b[5], b[6], 3.0]);
            let samples = self.push_sky_samples(index, k);
            self.spent += samples * self.options.sky_sample_cost;
            self.spent_by_mip[(mip.clamp(6, 13) - 6) as usize] +=
                samples * self.options.sky_sample_cost;
            self.sky_rebaked += 1;
        }
    }

    /// With streamed lighting levels, whether page `k` can bake now (M14
    /// gate 8: skies baked alone and bounce retries): its object's level and
    /// those its earlier bakes found absent all resident. If so they are
    /// pinned for the frame; if not they are wanted, and nothing is pinned
    /// (pins held by pages that cannot progress deadlock the slots).
    fn lighting_ready(&mut self, k: [u32; 3]) -> bool {
        if !self.lighting {
            return true;
        }
        let needs = self.lit_needs.get(&k).cloned().unwrap_or_default();
        let resident = |o: u32| self.lit_resident.get(o as usize).copied().unwrap_or(true);
        if resident(k[0]) && needs.iter().all(|&o| resident(o)) {
            self.lit_pins.push(k[0]);
            self.lit_pins.extend(&needs);
            true
        } else {
            self.lit_wants.push(k[0]);
            self.lit_wants.extend(&needs);
            false
        }
    }

    /// Page `k`'s texels listed along each axis, its sky stride, and its sky
    /// samples, as the plan lists them.
    fn sky_layout(&self, k: [u32; 3]) -> ([u32; 2], u32, u32) {
        let (m, c, mip, page) = unpack(k);
        let n = self.grid(c, m, mip).map(|x| x + 1);
        let inside = [
            (n[0] - (page[0] * PAGE).min(n[0])).min(STRIDE),
            (n[1] - (page[1] * PAGE).min(n[1])).min(STRIDE),
        ];
        let listed = [(inside[0] + 1).min(STRIDE), (inside[1] + 1).min(STRIDE)];
        let stride = self.sky_stride(m, mip);
        (
            listed,
            stride,
            listed[0].div_ceil(stride) * listed[1].div_ceil(stride),
        )
    }

    /// Lists page `k` (entry `index`) for its sky samples, as the plan
    /// does; their count.
    fn push_sky_samples(&mut self, index: u32, k: [u32; 3]) -> u32 {
        let (listed, stride, samples) = self.sky_layout(k);
        let first = self.sky_samples.len() as u32;
        for y in (0..listed[1]).step_by(stride as usize) {
            for x in (0..listed[0]).step_by(stride as usize) {
                self.sky_samples.push([index, y * STRIDE + x]);
            }
        }
        self.page_sky.push([first, samples | stride << 24]);
        samples
    }

    fn plan(&mut self, device: &wgpu::Device, requests: Vec<([u32; 3], u32)>) {
        self.sky_rebaked = 0;
        self.bake_list.clear();
        self.bake_bounds.clear();
        self.indirect_jobs.clear();
        self.jobs.clear();
        self.sky_samples.clear();
        self.page_sky.clear();
        self.texels = 0;
        self.spent = 0;
        self.spent_by_mip = [0; 8];
        // Resident pages shown without bounce samples: schedule them.
        for (k, priority) in &requests {
            if priority & WANTS_SAMPLES != 0
                && let Some(&(slot, _)) = self.map.get(k)
                && self.retrying.insert(slot)
            {
                self.retry.push_front((slot, self.frame));
            }
        }
        // Pages planned in the frame between are resident already. Pixels'
        // requests sort first; secondary ones (for light at hits) last.
        let mut order: Vec<(u32, [u32; 3])> = requests
            .into_iter()
            .filter(|(k, p)| p & WANTS_SAMPLES == 0 && !self.map.contains_key(k))
            .map(|(k, priority)| (priority, k))
            .collect();
        // Pages pending for lighting levels (M14 gate 4): those whose level
        // is now resident join the order; the rest are wanted, oldest first
        // (pending more than 240 frames: no longer asked for, dropped).
        if self.lighting {
            let frame = self.frame;
            let mut keep = std::collections::VecDeque::with_capacity(self.lit_pending.len());
            while let Some((k, priority, since)) = self.lit_pending.pop_front() {
                if self.map.contains_key(&k) || since + 240 < frame {
                    self.lit_pending_keys.remove(&k);
                } else if self
                    .lit_resident
                    .get(k[0] as usize)
                    .copied()
                    .unwrap_or(true)
                {
                    self.lit_pending_keys.remove(&k);
                    order.push((priority, k));
                } else {
                    self.lit_wants.push(k[0]);
                    keep.push_back((k, priority, since));
                }
            }
            self.lit_pending = keep;
        }
        // Each page costs at least one texel: no more than `budget` can fit.
        let budget = (self.budget as u32).max(PAGE_TEXELS);
        if order.len() > budget as usize {
            order.select_nth_unstable(budget as usize);
            order.truncate(budget as usize);
        }
        order.sort_unstable();
        // Skies to re-bake alone (for movers, and pages planned with the sky
        // later), once due: up to half the budget before new pages, so a
        // moving view cannot starve them (M9 gate 5), and then whatever new
        // pages leave.
        self.plan_due_skies(budget / 2);
        let mut evicted_now = false;
        // The sky later (M5 gate 9; with movers, M9 gate 5): not with moving
        // lights, whose ledger re-bakes would clear the sky's guess, nor with
        // bounce light, whose samples keep the coarser pages they find, so a
        // different bake order would settle to a different image.
        let sky_later =
            self.sky && self.options.sky_later && self.moving == 0 && !self.options.indirect;
        let mut lit_reserved = 0u32;
        for (priority, k) in order {
            let secondary = priority & SECONDARY != 0;
            let later = sky_later && !secondary;
            // A fingerprint collision can list a page twice.
            if self.map.contains_key(&k) {
                continue;
            }
            let (m, c, mip, page) = unpack(k);
            // The shader's grid (f32) can be one texel larger than this
            // one: a page starting just past this grid is still asked for
            // (M10 gate 6), and its texels outside the shader's are skipped.
            let n = self.grid(c, m, mip).map(|x| x + 1);
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
            let samples = if self.sky && !later {
                listed[0].div_ceil(stride) * listed[1].div_ceil(stride)
            } else {
                0
            };
            let bounce = if self.options.indirect && !secondary {
                INDIRECT_SAMPLES * self.options.indirect_sample_cost
            } else {
                0
            };
            let cost = inside[0] * inside[1] * self.lights
                + samples * self.options.sky_sample_cost
                + bounce;
            // The first page always fits, so a page costlier than the whole
            // budget still gets baked.
            if self.spent + lit_reserved + cost > budget
                && !(self.bake_list.is_empty() && lit_reserved == 0)
            {
                break;
            }
            // A page on an object whose lighting level is absent waits for
            // it (M14 gate 4), keeping its share of the budget, so only as
            // many levels are asked for as the bake can use.
            if self.lighting
                && !self
                    .lit_resident
                    .get(k[0] as usize)
                    .copied()
                    .unwrap_or(true)
            {
                self.lit_wants.push(k[0]);
                lit_reserved += cost;
                if self.lit_pending.len() < 1 << 16 && self.lit_pending_keys.insert(k) {
                    self.lit_pending.push_back((k, priority, self.frame));
                }
                continue;
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
            if self.lighting {
                self.lit_pins.push(k[0]);
            }
            if secondary && self.options.indirect {
                self.pinned.push_back((slot, k));
            }
            let index = self.bake_list.len() as u32;
            self.bake_list.push([k[0], k[1], k[2], slot]);
            let b = self.page_bounds(m, c, mip, page);
            self.bake_bounds.push([b[0], b[1], b[2], b[3]]);
            // w 1: no bounce samples planned (the fill clears the slot's);
            // w 2: lights only, the sky a guess until it is baked alone.
            let bare = self.options.indirect && secondary;
            let kind = if later { 2.0 } else { bare as u32 as f32 };
            self.bake_bounds.push([b[4], b[5], b[6], kind]);
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
                if !later {
                    for y in (0..listed[1]).step_by(stride as usize) {
                        for x in (0..listed[0]).step_by(stride as usize) {
                            self.sky_samples.push([index, y * STRIDE + x]);
                        }
                    }
                }
                self.page_sky.push([first, samples | stride << 24]);
            }
            // Its sky alone from the next frame: in this one the fill would
            // race the guess's mark against the sky's clearing it.
            if later {
                self.sky_due.insert(slot, self.frame + 1);
            }
            if self.options.indirect && !secondary {
                for k in 0..INDIRECT_SAMPLES {
                    self.indirect_jobs.push([index, k]);
                }
            }
            if trace_bakes() {
                eprintln!(
                    "bake frame {} page m{m} c{c} mip{mip} {page:?} slot {slot} priority {priority:#x}",
                    self.frame
                );
            }
            self.texels += inside[0] * inside[1];
            self.spent += cost;
            self.spent_by_mip[(mip.clamp(6, 13) - 6) as usize] += cost;
        }
        // Then skies to re-bake alone with what is left.
        self.plan_due_skies(budget);
        // Bounce samples of pages that met missing hits, once due, within
        // what is left of the budget.
        let bounce = INDIRECT_SAMPLES * self.options.indirect_sample_cost;
        while self.options.indirect
            && self.spent + bounce <= budget
            && self.bake_list.len() < budget as usize
            && self
                .retry
                .front()
                .is_some_and(|&(_, due)| due <= self.frame)
        {
            let (slot, _) = self.retry.pop_front().expect("a retry");
            let k = self.slot_key[slot as usize];
            if k[0] == EMPTY {
                self.retrying.remove(&slot);
                continue;
            }
            if !self.lighting_ready(k) {
                // Again later, its levels wanted meanwhile.
                self.retry.push_back((slot, self.frame + INDIRECT_RETRY));
                continue;
            }
            self.retrying.remove(&slot);
            let (m, c, mip, page) = unpack(k);
            if trace_bakes() {
                eprintln!(
                    "retry frame {} page m{m} c{c} mip{mip} {page:?} slot {slot}",
                    self.frame
                );
            }
            let index = self.bake_list.len() as u32;
            self.bake_list.push([k[0], k[1], k[2], slot]);
            let b = self.page_bounds(m, c, mip, page);
            self.bake_bounds.push([b[0], b[1], b[2], b[3]]);
            // w 5: bounce samples alone (the fill keeps the page's marks).
            self.bake_bounds.push([b[4], b[5], b[6], 5.0]);
            if self.sky {
                let stride = self.sky_stride(m, mip);
                let first = self.sky_samples.len() as u32;
                self.page_sky.push([first, stride << 24]);
            }
            for s in 0..INDIRECT_SAMPLES {
                self.indirect_jobs.push([index, s]);
            }
            self.spent += bounce;
        }
    }
}

/// Diagnostics: `CONTINUUM_TRACE_LIT` set prints, each frame, what the plan
/// did with streamed lighting levels (M14 gate 4; stderr).
fn trace_lighting() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CONTINUUM_TRACE_LIT").is_some())
}

/// Diagnostics: `CONTINUUM_TRACE_BAKES` set prints every page planned and
/// every bounce retry (stderr).
fn trace_bakes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CONTINUUM_TRACE_BAKES").is_some())
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
        let t = chart_tables(std::slice::from_ref(&s), usize::MAX);
        let filled: usize = t.cells.iter().map(|c| c[1] as usize).sum();
        assert_eq!(filled, s.triangles.len());
        assert!(t.cells.iter().all(|c| c[1] <= 2));
        assert_eq!(t.cells.len(), 24);
    }

    /// A streamed lighting level's package built alone (M14 gate 5) holds
    /// exactly what the whole scene's buffers hold for it, rebased.
    #[test]
    fn a_lighting_package_built_alone_matches_the_scene_buffers() {
        use crate::hierarchy::COVERAGE_SAMPLES as CS;
        let a = Surface::parametric(7, 5, |u, v| Vec3::new(u, v, (9.0 * u).sin() * 0.1));
        let b = Surface::parametric(32, 16, |u, v| {
            let (t, p) = (std::f32::consts::TAU * u, std::f32::consts::PI * v);
            let r = 1.0 + 0.1 * (5.0 * t).sin() * (3.0 * p).cos();
            Vec3::new(p.sin() * t.cos(), p.sin() * t.sin(), -p.cos()) * r
        });
        let scene = crate::InstancedScene {
            geometries: vec![a.clone(), b.clone()],
            ..Default::default()
        };
        let places = crate::stream::chart_places(&scene);
        let (p, r) = crate::stream::lit_package(&b, &places[1], None, false);
        let flat = crate::hierarchy::flatten(&[a.triangles.clone(), b.triangles.clone()]);
        let [first, n, root, _] = flat.ranges[1];
        let (f, n, root) = (first as usize, n as usize, root as usize);
        let next = flat.blas.len();
        let bytes = |x: &[crate::hierarchy::NodeGpu]| bytemuck::cast_slice::<_, u8>(x).to_vec();
        let rebased: Vec<_> = flat.blas[root..next]
            .iter()
            .map(|x| x.rebased(-(root as i64), -(f as i64)))
            .collect();
        assert_eq!(bytes(&p.nodes), bytes(&rebased));
        let items: Vec<u32> = flat.blas_items[f..f + n]
            .iter()
            .map(|&i| i - first)
            .collect();
        assert_eq!(p.items, items);
        let cov = &flat.blas_cov[root * CS..next * CS];
        assert!(
            p.cov
                .iter()
                .zip(cov)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert_eq!(p.cov.len(), cov.len());
        // The stub's root carries the root's coverage.
        assert!(
            r.stub_cov[..CS]
                .iter()
                .zip(cov)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert_eq!(p.tris, flat.tris[3 * f..3 * (f + n)]);
        assert_eq!(r.bounds, crate::hierarchy::mesh_bounds(&b.triangles));
        let t = chart_tables(&scene.geometries, usize::MAX);
        assert_eq!(p.uv, t.tri_uv[3 * f..3 * (f + n)]);
        assert_eq!(p.chart, t.tri_chart[f..f + n]);
        let c0 = t.charts[t.first_chart[1] as usize].first_cell as usize;
        let e0 = t.cells[c0][0];
        let cells: Vec<[u32; 3]> = (c0..t.cells.len())
            .map(|c| [c as u32, t.cells[c][0] - e0, t.cells[c][1]])
            .collect();
        assert_eq!(p.cells, cells);
        let list: Vec<u32> = t.cell_tris[e0 as usize..]
            .iter()
            .map(|&i| i - first)
            .collect();
        assert_eq!(p.cell_tris, list);
        // Skipped from the tables: cells kept, empty; no triangles listed.
        let s = chart_tables(&scene.geometries, 1);
        assert_eq!(s.cells.len(), t.cells.len());
        assert!(s.cells[c0..].iter().all(|c| *c == [0, 0]));
        assert_eq!(s.tri_chart.len(), a.triangles.len());
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
