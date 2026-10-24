//! The incremental renderer shared by the ledger experiments (gates 2–5).
//!
//! Receivers sit on a grid of pixel centers. Each receiver asks one cone per
//! target (a light, or one sector of the sky). Results are cached per (target,
//! receiver), and every (target, tile) has an error ledger. Moved occluders charge
//! ledgers with worst-case bounds, and a policy picks what to recompute so every
//! receiver's certified bound stays within ε. A full recompute runs alongside as
//! the reference, and every receiver is checked every frame.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use continuum::{ConeMask, CoverageMask, Ledger};

use crate::anim::Scenario;
use crate::bounds::{
    Aabb, Circle, convex_distance, convex_hull, convex_overlap, max_segment_angle,
};
use crate::geom::{Cone, Segment, Vec2, v2};
use crate::img::{self, Image};
use crate::methods;

pub const TILE: usize = 8;
const TILE_PIXELS: usize = TILE * TILE;
/// Absolute slack for float rounding in the soundness check.
const FLOAT_SLACK: f32 = 1e-5;

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub size: usize,
}

impl Grid {
    pub fn pixels(self) -> usize {
        self.size * self.size
    }

    pub fn tiles_per_side(self) -> usize {
        self.size / TILE
    }

    pub fn tiles(self) -> usize {
        self.tiles_per_side() * self.tiles_per_side()
    }

    pub fn pixel_center(self, px: usize) -> Vec2 {
        let (i, j) = (px % self.size, px / self.size);
        let n = self.size as f32;
        v2((i as f32 + 0.5) / n, 1.0 - (j as f32 + 0.5) / n)
    }

    pub fn tile_of(self, px: usize) -> usize {
        let (i, j) = (px % self.size, px / self.size);
        (j / TILE) * self.tiles_per_side() + i / TILE
    }

    pub fn tile_pixels(self, t: usize) -> impl Iterator<Item = usize> {
        let (tx, ty) = (t % self.tiles_per_side(), t / self.tiles_per_side());
        (0..TILE).flat_map(move |dy| {
            (0..TILE).map(move |dx| (ty * TILE + dy) * self.size + tx * TILE + dx)
        })
    }
}

/// How a cone's visibility is estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// [`ConeMask`]: a bin is blocked when an occluder covers its center.
    Binary,
    /// [`CoverageMask`]: partially covered bins keep a fraction.
    Continuous,
}

impl Visibility {
    pub fn name(self) -> &'static str {
        match self {
            Visibility::Binary => "binary masks",
            Visibility::Continuous => "continuous masks",
        }
    }

    fn visible_fraction(self, cone: &Cone, occluders: &[Segment]) -> f32 {
        match self {
            Visibility::Binary => methods::mask(cone, occluders),
            Visibility::Continuous => methods::coverage(cone, occluders),
        }
    }
}

/// How recomputation is scheduled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Every target may use an equal share of ε.
    EqualShare,
    /// Per tile, recompute the largest weighted balances until the rest fit in ε.
    TileBudget,
}

/// How moved occluders charge ledgers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Charging {
    /// Geometric worst cases per tile: the swept region's angle, or the endpoint
    /// paths' angles, from the tile's nearest point.
    Bounds,
    /// Project the mover's old and new positions into every receiver's cone in
    /// the tile and charge the larger of the coverage it gains and loses (or the
    /// geometric bound, if smaller). Costs two projections per receiver.
    Projected,
}

impl Charging {
    pub fn name(self) -> &'static str {
        match self {
            Charging::Bounds => "geometric bounds",
            Charging::Projected => "projected charges",
        }
    }
}

/// Maps `f` over `0..n` on `threads` threads in chunks of `chunk`, preserving order.
pub fn par_map<R: Send>(
    n: usize,
    threads: usize,
    chunk: usize,
    f: impl Fn(usize) -> R + Sync,
) -> Vec<R> {
    if threads <= 1 || n <= chunk {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<(usize, Vec<R>)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let start = next.fetch_add(chunk, Ordering::Relaxed);
                        if start >= n {
                            break;
                        }
                        out.push((start, (start..(start + chunk).min(n)).map(&f).collect()));
                    }
                    out
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    });
    parts.sort_unstable_by_key(|p| p.0);
    parts.into_iter().flat_map(|p| p.1).collect()
}

/// Everything that depends only on static geometry and targets. Arrays are
/// indexed `[target * pixels + pixel]` or `[target * tiles + tile]`.
pub struct Precomputed {
    pub grid: Grid,
    pub visibility: Visibility,
    pub targets: usize,
    pub threads: usize,
    cones: Vec<Option<Cone>>,
    /// Binary only: static occluders' masks, reused by recomputes.
    static_masks: Vec<ConeMask>,
    /// Continuous only: static occluders' coverage, reused by recomputes
    /// (264 bytes per cone on the CPU).
    static_coverage: Vec<CoverageMask>,
    tile_boxes: Vec<Aabb>,
    /// Hull of a tile and its targets: contains every receiver's cone in the tile.
    frustums: Vec<Vec<Vec2>>,
    frustum_boxes: Vec<Aabb>,
    /// Hull of the targets' far edges as seen from the tile.
    far_regions: Vec<Vec<Vec2>>,
    /// Largest cone angle of any receiver in the tile.
    theta_max: Vec<f32>,
    /// Statics hide the target from every receiver in the tile, so nothing can change there.
    static_blocked: Vec<bool>,
}

impl Precomputed {
    pub fn new(s: &Scenario, grid: Grid, visibility: Visibility, threads: usize) -> Self {
        let targets = s.targets();
        let pixels = grid.pixels();
        let cones: Vec<Option<Cone>> = par_map(targets.len() * pixels, threads, 256, |k| {
            let p = grid.pixel_center(k % pixels);
            Cone::toward(p, targets[k / pixels].at(p))
        });
        let statics = s.statics.clone();
        let static_masks = match visibility {
            Visibility::Binary => par_map(cones.len(), threads, 256, |k| {
                cones[k].map_or(ConeMask::FULL, |c| methods::mask_bits(&c, &statics))
            }),
            Visibility::Continuous => Vec::new(),
        };
        let static_coverage = match visibility {
            Visibility::Binary => Vec::new(),
            Visibility::Continuous => par_map(cones.len(), threads, 256, |k| {
                let mut m = CoverageMask::EMPTY;
                if let Some(c) = cones[k] {
                    methods::add_coverage(&mut m, &c, &statics);
                } else {
                    m.add_interval(0.0, 1.0);
                }
                m
            }),
        };
        let static_visible: Vec<f32> = (0..cones.len())
            .map(|k| match visibility {
                Visibility::Binary => static_masks[k].visible_fraction(),
                Visibility::Continuous => static_coverage[k].visible_fraction(),
            })
            .collect();
        let tile_boxes: Vec<Aabb> = (0..grid.tiles())
            .map(|t| {
                Aabb::of(
                    &grid
                        .tile_pixels(t)
                        .map(|px| grid.pixel_center(px))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut pre = Precomputed {
            grid,
            visibility,
            targets: targets.len(),
            threads,
            cones,
            static_masks,
            static_coverage,
            tile_boxes,
            frustums: Vec::new(),
            frustum_boxes: Vec::new(),
            far_regions: Vec::new(),
            theta_max: Vec::new(),
            static_blocked: Vec::new(),
        };
        for (q, target) in targets.iter().enumerate() {
            for t in 0..grid.tiles() {
                let corners = pre.tile_boxes[t].corners();
                let far: Vec<Vec2> = corners
                    .iter()
                    .flat_map(|&c| {
                        let s = target.at(c);
                        [s.a, s.b]
                    })
                    .collect();
                let mut pts = corners.to_vec();
                pts.extend(&far);
                let hull = convex_hull(&pts);
                pre.frustum_boxes.push(Aabb::of(&hull));
                pre.frustums.push(hull);
                pre.far_regions.push(convex_hull(&far));
                let in_tile = || grid.tile_pixels(t).map(|px| q * pixels + px);
                pre.theta_max.push(
                    in_tile()
                        .filter_map(|k| pre.cones[k].map(|c| c.angle))
                        .fold(0.0, f32::max),
                );
                pre.static_blocked
                    .push(in_tile().all(|k| static_visible[k] == 0.0));
            }
        }
        pre
    }

    /// Visible angle (radians) of cone `k` with the movers in place, as a recompute does it.
    fn visible_angle(&self, k: usize, movers: &[Segment]) -> f32 {
        let Some(cone) = self.cones[k] else {
            return 0.0;
        };
        match self.visibility {
            Visibility::Binary => {
                let s = self.static_masks[k];
                if s.is_full() {
                    0.0
                } else {
                    cone.angle * (s | methods::mask_bits(&cone, movers)).visible_fraction()
                }
            }
            Visibility::Continuous => {
                let mut m = self.static_coverage[k];
                if m.is_full() {
                    return 0.0;
                }
                methods::add_coverage(&mut m, &cone, movers);
                cone.angle * m.visible_fraction()
            }
        }
    }

    /// How much coverage cone `k` gains and loses (in radians) when one occluder
    /// moves from `old` to `new` with everything else fixed. Bins the statics
    /// already block fully cannot change, so they are left out (occlusion-aware).
    ///
    /// When everything visible through the cone has the same radiance, the
    /// visible angle changes by at most the larger of the two; when radiance
    /// varies by direction, by at most their sum times the radiance range.
    fn coverage_change(&self, k: usize, old: &Segment, new: &Segment) -> (f32, f32) {
        let Some(cone) = self.cones[k] else {
            return (0.0, 0.0);
        };
        let (a, b) = (cone.project(old), cone.project(new));
        match self.visibility {
            Visibility::Binary => {
                let mask =
                    |i: Option<(f32, f32)>| i.map_or(0, |(x, y)| ConeMask::from_interval(x, y).0);
                let (ma, mb) = (mask(a), mask(b));
                let free = !self.static_masks[k].0;
                let bin = cone.angle / ConeMask::BINS as f32;
                let gained = (mb & !ma & free).count_ones() as f32 * bin;
                let lost = (ma & !mb & free).count_ones() as f32 * bin;
                (gained, lost)
            }
            Visibility::Continuous => {
                let statics = &self.static_coverage[k];
                let open = |x: Option<(f32, f32)>, y: Option<(f32, f32)>| -> f32 {
                    difference(x, y)
                        .iter()
                        .map(|&(lo, hi)| statics.open_length(lo, hi))
                        .sum()
                };
                (open(b, a) * cone.angle, open(a, b) * cone.angle)
            }
        }
    }

    /// Conventional baseline: every cone against every occluder.
    pub fn full_recompute(&self, occluders: &[Segment]) -> Vec<f32> {
        par_map(self.cones.len(), self.threads, 256, |k| {
            self.cones[k].map_or(0.0, |c| {
                c.angle * self.visibility.visible_fraction(&c, occluders)
            })
        })
    }

    /// Every cone with the recompute path (static masks reused in binary mode).
    pub fn recompute_all(&self, movers: &[Segment]) -> Vec<f32> {
        par_map(self.cones.len(), self.threads, 256, |k| {
            self.visible_angle(k, movers)
        })
    }

    /// Visible angle of every cone under any per-cone estimator (e.g. the exact answer).
    pub fn evaluate(&self, fraction: impl Fn(&Cone) -> f32 + Sync) -> Vec<f32> {
        par_map(self.cones.len(), self.threads, 256, |k| {
            self.cones[k].map_or(0.0, |c| c.angle * fraction(&c))
        })
    }

    /// Fluence per receiver: visible angles weighted by each target's radiance.
    pub fn compose(&self, visible: &[f32], weights: &[f32]) -> Vec<f32> {
        let pixels = self.grid.pixels();
        (0..pixels)
            .map(|px| {
                weights
                    .iter()
                    .enumerate()
                    .map(|(q, &w)| w * visible[q * pixels + px])
                    .sum()
            })
            .collect()
    }
}

/// `x \ y` for intervals, as up to two pieces (empty pieces have zero length).
fn difference(x: Option<(f32, f32)>, y: Option<(f32, f32)>) -> [(f32, f32); 2] {
    match (x, y) {
        (None, _) => [(0.0, 0.0); 2],
        (Some(x), None) => [x, (0.0, 0.0)],
        (Some(x), Some(y)) => {
            let left = (x.0, x.1.min(y.0));
            let right = (x.0.max(y.1), x.1);
            [left, right].map(|(lo, hi)| (lo, hi.max(lo)))
        }
    }
}

/// Conservative region a moving occluder can have touched during one frame step.
pub struct MovedHull {
    prev: Segment,
    now: Segment,
    hull: Vec<Vec2>,
    bbox: Aabb,
    circle: Circle,
    /// Farthest either endpoint moved.
    max_displacement: f32,
}

pub fn moved_hulls(prev: &[Segment], now: &[Segment]) -> Vec<MovedHull> {
    prev.iter()
        .zip(now)
        .filter(|(p, n)| p != n)
        .map(|(p, n)| {
            // Linearly moving endpoints sweep a region inside the hull of both positions.
            let hull = convex_hull(&[p.a, p.b, n.a, n.b]);
            MovedHull {
                prev: *p,
                now: *n,
                bbox: Aabb::of(&hull),
                circle: Circle::enclosing(&hull),
                max_displacement: (n.a - p.a).length().max((n.b - p.b).length()),
                hull,
            }
        })
        .collect()
}

pub struct Incremental {
    /// Absolute fluence error allowed per receiver.
    pub epsilon: f32,
    pub visible: Vec<f32>,
    pub ledgers: Vec<Ledger>,
    pub recomputed_tiles: Vec<bool>,
    pub recomputed_cones: u64,
    /// Charging plus recomputing, excluding the reference and the soundness check.
    pub time: Duration,
}

impl Incremental {
    pub fn new(pre: &Precomputed, epsilon: f32, visible: Vec<f32>) -> Self {
        Incremental {
            epsilon,
            visible,
            ledgers: vec![Ledger::default(); pre.targets * pre.grid.tiles()],
            recomputed_tiles: vec![false; pre.grid.tiles()],
            recomputed_cones: 0,
            time: Duration::ZERO,
        }
    }

    /// One frame: charge moved occluders, then recompute what `policy` selects.
    /// Returns the number of cones recomputed.
    pub fn step(
        &mut self,
        pre: &Precomputed,
        policy: Policy,
        charging: Charging,
        hulls: &[MovedHull],
        weights: &[f32],
        movers: &[Segment],
    ) -> u64 {
        let start = Instant::now();
        self.charge(pre, charging, hulls);
        let keys = self.select(pre, policy, weights);
        let (tiles, pixels) = (pre.grid.tiles(), pre.grid.pixels());
        let results = par_map(keys.len(), pre.threads, 4, |i| {
            let (q, t) = (keys[i] / tiles, keys[i] % tiles);
            let mut values = [0.0f32; TILE_PIXELS];
            for (v, px) in values.iter_mut().zip(pre.grid.tile_pixels(t)) {
                *v = pre.visible_angle(q * pixels + px, movers);
            }
            values
        });
        self.recomputed_tiles.fill(false);
        for (&k, values) in keys.iter().zip(results) {
            let (q, t) = (k / tiles, k % tiles);
            for (v, px) in values.into_iter().zip(pre.grid.tile_pixels(t)) {
                self.visible[q * pixels + px] = v;
            }
            self.ledgers[k].settle();
            self.recomputed_tiles[t] = true;
        }
        self.time += start.elapsed();
        let cones = (keys.len() * TILE_PIXELS) as u64;
        self.recomputed_cones += cones;
        cones
    }

    /// Charges every (target, tile) whose cones a moved occluder could reach.
    ///
    /// The geometric charge is the smaller of two worst-case bounds on the change
    /// in visible angle:
    ///
    /// - Region bound: only directions into the swept hull can change, so at most
    ///   the hull's subtended angle.
    /// - Motion bound: a direction can only change if an endpoint's path crosses
    ///   it (or the occluder slides past the target's far edge, which is checked),
    ///   so at most the two endpoint paths' angles.
    ///
    /// Binary masks add one bin per term, because an edge crossing a bin center
    /// by a hair flips the whole bin; continuous masks need no such term. Neither
    /// bound may exceed the whole cone.
    ///
    /// The projected charge is the largest per-receiver
    /// [`Precomputed::coverage_change`] in the tile. Both are valid bounds, so the
    /// smaller is used whenever both are computed.
    fn charge(&mut self, pre: &Precomputed, charging: Charging, hulls: &[MovedHull]) {
        let (tiles, pixels) = (pre.grid.tiles(), pre.grid.pixels());
        // Charges per (target, tile) are independent, so compute them in parallel,
        // then apply them in hull order.
        let charges = par_map(pre.targets * tiles, pre.threads, 16, |k| {
            let mut out = Vec::new();
            if pre.static_blocked[k] {
                return out;
            }
            let (q, t) = (k / tiles, k % tiles);
            for h in hulls {
                if !h.bbox.overlaps(&pre.frustum_boxes[k])
                    || !convex_overlap(&pre.frustums[k], &h.hull)
                {
                    continue;
                }
                let theta = pre.theta_max[k];
                let bin = match pre.visibility {
                    Visibility::Binary => theta / ConeMask::BINS as f32,
                    Visibility::Continuous => 0.0,
                };
                let region = h.circle.max_subtended_angle(&pre.tile_boxes[t]) + bin;
                let d = convex_distance(&pre.tile_boxes[t].corners(), &h.hull);
                let motion = if d > 0.0 && !convex_overlap(&h.hull, &pre.far_regions[k]) {
                    2.0 * (max_segment_angle(h.max_displacement, d) + bin)
                } else {
                    f32::INFINITY
                };
                let geometric = region.min(motion).min(theta);
                let charge = if charging == Charging::Projected {
                    pre.grid
                        .tile_pixels(t)
                        .map(|px| {
                            let (gained, lost) =
                                pre.coverage_change(q * pixels + px, &h.prev, &h.now);
                            gained.max(lost)
                        })
                        .fold(0.0, f32::max)
                        .min(geometric)
                } else {
                    geometric
                };
                out.push(charge);
            }
            out
        });
        for (ledger, charges) in self.ledgers.iter_mut().zip(charges) {
            for c in charges {
                ledger.charge(c);
            }
        }
    }

    /// The (target, tile) keys to recompute this frame.
    fn select(&self, pre: &Precomputed, policy: Policy, weights: &[f32]) -> Vec<usize> {
        let tiles = pre.grid.tiles();
        match policy {
            Policy::EqualShare => {
                let share = self.epsilon / pre.targets as f32;
                (0..pre.targets * tiles)
                    .filter(|&k| self.ledgers[k].exceeds(weights[k / tiles], share))
                    .collect()
            }
            Policy::TileBudget => {
                let mut keys = Vec::new();
                let mut weighted: Vec<(f32, usize)> = Vec::with_capacity(pre.targets);
                for t in 0..tiles {
                    weighted.clear();
                    weighted.extend(
                        (0..pre.targets)
                            .map(|q| {
                                (
                                    weights[q] * self.ledgers[q * tiles + t].balance(),
                                    q * tiles + t,
                                )
                            })
                            .filter(|&(b, _)| b > 0.0),
                    );
                    // Smallest first, so a running sum gives exactly what stays deferred.
                    weighted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                    let mut kept = 0.0;
                    let mut keep = 0;
                    while keep < weighted.len() && kept + weighted[keep].0 <= self.epsilon {
                        kept += weighted[keep].0;
                        keep += 1;
                    }
                    keys.extend(weighted[keep..].iter().map(|&(_, k)| k));
                }
                keys.sort_unstable();
                keys
            }
        }
    }

    /// Composed fluence and certified bound per receiver.
    pub fn fluence_and_bound(&self, pre: &Precomputed, weights: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (tiles, pixels) = (pre.grid.tiles(), pre.grid.pixels());
        (0..pixels)
            .map(|px| {
                let t = pre.grid.tile_of(px);
                weights
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(f, b), (q, &w)| {
                        (
                            f + w * self.visible[q * pixels + px],
                            b + w * self.ledgers[q * tiles + t].balance(),
                        )
                    })
            })
            .unzip()
    }
}

pub struct RunConfig {
    pub grid: Grid,
    pub visibility: Visibility,
    pub policy: Policy,
    pub charging: Charging,
    pub threads: usize,
    /// ε values, as fractions of the scenario's mean fluence at frame 0.
    pub epsilons: Vec<f32>,
    /// Indices into `epsilons` whose runs get an image sheet.
    pub sheets: Vec<usize>,
    /// Also time (and check) a full recompute that reuses cached static coverage.
    pub time_static_cache: bool,
}

#[derive(Default)]
pub struct RunStats {
    pub violations: u64,
    pub max_error: f32,
    pub error_sum: f64,
    pub bound_sum: f64,
    /// Receiver-frames whose output carried a nonzero certified bound, i.e. where
    /// the ledger actually deferred work instead of recomputing.
    pub deferred: u64,
    pub phase_cones: Vec<u64>,
    pub max_error_by_frame: Vec<f32>,
}

pub struct ScenarioResult {
    pub mean_fluence: f32,
    /// Absolute ε per run.
    pub epsilons: Vec<f32>,
    pub runs: Vec<(Incremental, RunStats)>,
    pub full_time: Duration,
    pub static_cached_time: Option<Duration>,
    pub frames_timed: u32,
    /// Cone evaluations a full recompute does per frame.
    pub cones_per_frame: u64,
    /// One list of rows per entry in `RunConfig::sheets`.
    pub sheets: Vec<Vec<[Image; 4]>>,
}

pub fn run(s: &Scenario, cfg: &RunConfig) -> ScenarioResult {
    let pre = Precomputed::new(s, cfg.grid, cfg.visibility, cfg.threads);
    let pixels = cfg.grid.pixels();
    let occluders_at = |f: u32| {
        let mut v = s.statics.clone();
        v.extend((s.movers)(f));
        v
    };

    let visible0 = pre.full_recompute(&occluders_at(0));
    let fluence0 = pre.compose(&visible0, &s.weights(0));
    let mean_fluence = fluence0.iter().sum::<f32>() / pixels as f32;
    let epsilons: Vec<f32> = cfg.epsilons.iter().map(|e| e * mean_fluence).collect();
    let mut runs: Vec<(Incremental, RunStats)> = epsilons
        .iter()
        .map(|&epsilon| {
            let stats = RunStats {
                phase_cones: vec![0; s.phases.len()],
                max_error_by_frame: vec![0.0; s.frames as usize],
                ..Default::default()
            };
            (Incremental::new(&pre, epsilon, visible0.clone()), stats)
        })
        .collect();

    let mut full_time = Duration::ZERO;
    let mut static_cached_time = cfg.time_static_cache.then_some(Duration::ZERO);
    let mut sheets: Vec<Vec<[Image; 4]>> = cfg.sheets.iter().map(|_| Vec::new()).collect();
    for f in 1..s.frames {
        let movers_prev = (s.movers)(f - 1);
        let movers_now = (s.movers)(f);
        let weights = s.weights(f);

        let start = Instant::now();
        let reference_visible = pre.full_recompute(&occluders_at(f));
        full_time += start.elapsed();
        let reference = pre.compose(&reference_visible, &weights);

        if let Some(time) = static_cached_time.as_mut() {
            // Static caching alone, to separate its gain from the ledger's.
            let start = Instant::now();
            let cached = pre.recompute_all(&movers_now);
            *time += start.elapsed();
            assert_eq!(
                cached, reference_visible,
                "static-mask caching must not change results"
            );
        }

        let hulls = moved_hulls(&movers_prev, &movers_now);
        for (i, (state, stats)) in runs.iter_mut().enumerate() {
            let cones = state.step(
                &pre,
                cfg.policy,
                cfg.charging,
                &hulls,
                &weights,
                &movers_now,
            );
            if let Some(p) = s.phases.iter().position(|(_, r)| r.contains(&f)) {
                stats.phase_cones[p] += cones;
            }

            let (fluence, bound) = state.fluence_and_bound(&pre, &weights);
            let mut frame_max = 0.0f32;
            for px in 0..pixels {
                let err = (fluence[px] - reference[px]).abs();
                if err > bound[px] + FLOAT_SLACK {
                    stats.violations += 1;
                }
                frame_max = frame_max.max(err);
                if bound[px] > 0.0 {
                    stats.deferred += 1;
                }
                stats.error_sum += err as f64;
                stats.bound_sum += bound[px] as f64;
            }
            stats.max_error = stats.max_error.max(frame_max);
            stats.max_error_by_frame[f as usize] = frame_max;

            if let Some(sheet) = cfg.sheets.iter().position(|&e| e == i)
                && s.key_frames.contains(&f)
            {
                sheets[sheet].push(sheet_row(s, &pre, f, state, &fluence, &bound, &reference));
            }
        }
    }
    ScenarioResult {
        mean_fluence,
        epsilons,
        runs,
        full_time,
        static_cached_time,
        frames_timed: s.frames - 1,
        cones_per_frame: (pre.targets * pixels) as u64,
        sheets,
    }
}

/// Lit image, recomputed tiles, certified bound, and actual error for one frame.
fn sheet_row(
    s: &Scenario,
    pre: &Precomputed,
    f: u32,
    state: &Incremental,
    fluence: &[f32],
    bound: &[f32],
    reference: &[f32],
) -> [Image; 4] {
    let size = pre.grid.size;
    let overlay = |im: &mut Image| {
        img::draw_segments(im, &s.statics, img::OCCLUDER_COLOR);
        img::draw_segments(im, &(s.movers)(f), [255, 120, 200]);
        let lights: Vec<Segment> = s.lights.iter().map(|l| l.segment).collect();
        img::draw_segments(im, &lights, img::LIGHT_COLOR);
    };
    let exposed: Vec<f32> = fluence.iter().map(|v| v * s.exposure).collect();
    let mut lit = img::tonemap(&exposed, size);
    overlay(&mut lit);
    let mut tiles = img::tonemap(&exposed, size);
    for px in (0..pre.grid.pixels()).filter(|&px| state.recomputed_tiles[pre.grid.tile_of(px)]) {
        let (x, y) = (px % size, px / size);
        let c = tiles.get(x, y);
        tiles.set(x, y, [c[0] / 3 + 170, c[1] / 3 + 90, c[2] / 3 + 20]);
    }
    overlay(&mut tiles);
    let scale = state.epsilon.max(1e-6);
    let mut bound_img = img::magnitude(bound, size, scale);
    overlay(&mut bound_img);
    let mut error_img = img::signed_error(fluence, reference, size, scale);
    overlay(&mut error_img);
    [lit, tiles, bound_img, error_img]
}

pub fn save_sheet(rows: &[[Image; 4]], path: &Path) -> std::io::Result<()> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    let (size, gap) = (first[0].w, 8);
    let mut sheet = Image::new(4 * size + 5 * gap, rows.len() * (size + gap) + gap, [28; 3]);
    for (r, row) in rows.iter().enumerate() {
        for (c, im) in row.iter().enumerate() {
            sheet.blit(im, gap + c * (size + gap), gap + r * (size + gap));
        }
    }
    sheet.save(path)
}

fn ms_per_frame(d: Duration, frames: u32) -> f64 {
    d.as_secs_f64() * 1000.0 / frames as f64
}

impl ScenarioResult {
    /// Fraction of a full recompute's cone evaluations that run `i` performed.
    pub fn recomputed_fraction(&self, i: usize) -> f64 {
        self.runs[i].0.recomputed_cones as f64
            / (self.cones_per_frame * self.frames_timed as u64) as f64
    }
}

/// Appends a results section. Runs from `stress_from` on are labeled stress tests.
pub fn report(
    out: &mut String,
    title: &str,
    s: &Scenario,
    cfg: &RunConfig,
    r: &ScenarioResult,
    stress_from: usize,
) {
    let frames = r.frames_timed;
    let full_ms = ms_per_frame(r.full_time, frames);
    writeln!(out, "## {title}\n").unwrap();
    write!(
        out,
        "Mean fluence at frame 0: {:.4}. ε is a fraction of it. Full recompute: {:.2} ms/frame",
        r.mean_fluence, full_ms
    )
    .unwrap();
    if let Some(t) = r.static_cached_time {
        write!(
            out,
            "; full recompute reusing static masks (no ledger): {:.2} ms/frame",
            ms_per_frame(t, frames)
        )
        .unwrap();
    }
    writeln!(out, ".\n").unwrap();
    writeln!(out, "| ε | Gate 2 violations | Deferred receiver-frames | Max error | Max error / ε | Bound use | Cones recomputed | Skipped | Incremental ms/frame | Speedup vs full |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|").unwrap();
    let receiver_frames = (frames as u64 * cfg.grid.pixels() as u64) as f64;
    for (i, ((state, stats), (&frac, &eps))) in r
        .runs
        .iter()
        .zip(cfg.epsilons.iter().zip(&r.epsilons))
        .enumerate()
    {
        let stress = if i >= stress_from {
            " (stress test)"
        } else {
            ""
        };
        let recomputed = r.recomputed_fraction(i);
        let inc_ms = ms_per_frame(state.time, frames);
        let ratio = if eps > 0.0 {
            format!("{:.2}", stats.max_error / eps)
        } else {
            "—".to_string()
        };
        let bound_use = if stats.bound_sum > 0.0 {
            format!("{:.1}%", 100.0 * stats.error_sum / stats.bound_sum)
        } else {
            "—".to_string()
        };
        writeln!(
            out,
            "| {:.0}%{} | {} | {:.1}% | {:.5} | {} | {} | {:.1}% | {:.1}% | {:.2} | {:.1}× |",
            frac * 100.0,
            stress,
            stats.violations,
            100.0 * stats.deferred as f64 / receiver_frames,
            stats.max_error,
            ratio,
            bound_use,
            recomputed * 100.0,
            (1.0 - recomputed) * 100.0,
            inc_ms,
            full_ms / inc_ms
        )
        .unwrap();
    }
    if s.phases.len() > 1 {
        writeln!(out, "\nCones recomputed per frame, by phase:\n").unwrap();
        let header: Vec<String> = s
            .phases
            .iter()
            .map(|(n, r)| format!("{n} (frames {}–{})", r.start, r.end - 1))
            .collect();
        writeln!(out, "| ε | {} |", header.join(" | ")).unwrap();
        writeln!(out, "|---|{}", "---|".repeat(s.phases.len())).unwrap();
        for ((_, stats), &frac) in r.runs.iter().zip(&cfg.epsilons) {
            let cells: Vec<String> = s
                .phases
                .iter()
                .zip(&stats.phase_cones)
                .map(|((_, range), &c)| {
                    format!(
                        "{:.1}%",
                        100.0 * c as f64 / (r.cones_per_frame as f64 * range.len() as f64)
                    )
                })
                .collect();
            writeln!(out, "| {:.0}% | {} |", frac * 100.0, cells.join(" | ")).unwrap();
        }
    }
    writeln!(out).unwrap();
}
