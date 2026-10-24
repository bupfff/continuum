//! Flatland gate 6: one-bounce indirect light with charges that propagate.
//!
//! Static surfaces reflect their direct light; receivers gather that bounce
//! light through 8 wide cones. Cached gathers are charged from two sources:
//!
//! - **Geometry**: a mover changes the nearest surface only in directions its
//!   own coverage changes, so the charge is (coverage gained + lost) times the
//!   brightest patch radiance. If a mover's swept hull touches a static surface
//!   (depth order could flip), the charge falls back to the hull's whole angle.
//! - **Propagation**: when a patch's radiance changes (a shadow sweeps across a
//!   wall), every cached gather that can see that patch is charged the change
//!   times the angle the patch can subtend there.
//!
//! Charges per frame follow the triangle inequality: radiance changes first
//! (geometry fixed), then geometry changes (new radiance). Patch radiance itself
//! is recomputed exactly every frame; it is small and cheap.
//!
//! Pass criterion, fixed before the first run: zero soundness violations at every
//! ε and frame, and at ε = 1% of mean indirect fluence at most half the cone
//! recomputes of ε = 0.
//!
//! Attempt 1 priced geometry charges at the brightest patch anywhere and failed
//! (76.1%). Attempt 2 prices them at the brightest patch side the mover could
//! reveal or hide from that tile (patches inside the beam from the tile through
//! the mover's swept hull, sides facing the tile) and failed (68.1%). Attempt 3
//! adds that movers are black: directions they newly cover can only lose light
//! and directions they uncover can only gain it, so the change is at most the
//! larger of the two, not their sum. Same criterion throughout.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::Ledger;
use flatland::anim::{self, Scenario, sky_targets};
use flatland::bounce::{self, LIGHTS, Surface};
use flatland::bounds::{Aabb, Circle, convex_hull, convex_overlap};
use flatland::geom::Segment;
use flatland::img::{self, Image};
use flatland::incremental::{
    self, Grid, MovedHull, Precomputed, TILE, Visibility, difference, moved_hulls, par_map,
};

const SIZE: usize = 128;
const SECTORS: usize = 8;
const PATCH_LENGTH: f32 = 0.04;
const EPSILONS: [f32; 4] = [0.0, 0.01, 0.04, 0.16];
const STRESS_FROM: usize = 3;
const HEADLINE: usize = 1;
const FLOAT_SLACK: f32 = 1e-5;

type PerLight = [f32; LIGHTS];

/// How a geometry charge prices the directions a mover covers or uncovers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RadianceBound {
    /// (gained + lost) × the brightest patch side anywhere (attempt 1).
    Global,
    /// (gained + lost) × the brightest patch side the tile could see through the
    /// mover's sweep (attempt 2).
    Local,
    /// max(gained, lost) × the same local radiance (attempt 3): valid because
    /// movers are black, so gains and losses have opposite signs.
    LocalBlack,
    /// Diagnostic, not implementable: charge each key the true change of its
    /// cached values since the previous frame, taken from the reference. The best
    /// any charging scheme at this granularity could do.
    Oracle,
}

impl RadianceBound {
    fn name(self) -> &'static str {
        match self {
            RadianceBound::Global => "attempt 1: global radiance bound",
            RadianceBound::Local => "attempt 2: local radiance bound",
            RadianceBound::LocalBlack => "attempt 3: local radiance bound, black movers",
            RadianceBound::Oracle => {
                "diagnostic: oracle charges (true per-frame change, not implementable)"
            }
        }
    }
}

/// Beyond everything in the unit square, for beams cast from tiles through movers.
const BEAM_LENGTH: f32 = 4.0;

struct World {
    grid: Grid,
    threads: usize,
    /// Sector cones per receiver; `Precomputed` supplies cones, tiles and frustums.
    pre: Precomputed,
    statics: Vec<Segment>,
    patches: Vec<Segment>,
    patch_circles: Vec<(Circle, Aabb)>,
    lights: Vec<Segment>,
}

impl World {
    fn new(s: &Scenario, grid: Grid, threads: usize) -> Self {
        let pre = Precomputed::with_targets(
            &sky_targets(SECTORS),
            &[],
            grid,
            Visibility::Binary,
            threads,
        );
        let patches = bounce::patches(&s.statics, PATCH_LENGTH);
        let patch_circles = patches
            .iter()
            .map(|p| (Circle::enclosing(&[p.a, p.b]), Aabb::of(&[p.a, p.b])))
            .collect();
        World {
            grid,
            threads,
            pre,
            statics: s.statics.clone(),
            patches,
            patch_circles,
            lights: s.lights.iter().map(|l| l.segment).collect(),
        }
    }

    fn keys(&self) -> usize {
        SECTORS * self.grid.tiles()
    }

    fn radiance(&self, movers: &[Segment]) -> Vec<f32> {
        let mut occluders = self.statics.clone();
        occluders.extend_from_slice(movers);
        bounce::patch_radiance(&self.patches, &occluders, &self.lights)
    }

    fn surfaces(&self, movers: &[Segment]) -> Vec<Surface> {
        let mut out: Vec<Surface> = self
            .patches
            .iter()
            .enumerate()
            .map(|(j, &segment)| Surface {
                segment,
                patch: Some(j),
            })
            .collect();
        out.extend(movers.iter().map(|&segment| Surface {
            segment,
            patch: None,
        }));
        out
    }

    fn gather_one(&self, k: usize, surfaces: &[Surface], radiance: &[f32]) -> PerLight {
        self.pre
            .cone(k)
            .map_or([0.0; LIGHTS], |c| bounce::gather(&c, surfaces, radiance))
    }

    /// Conventional baseline: every (sector, receiver) gathered from scratch.
    fn full_gather(&self, surfaces: &[Surface], radiance: &[f32]) -> Vec<PerLight> {
        par_map(SECTORS * self.grid.pixels(), self.threads, 256, |k| {
            self.gather_one(k, surfaces, radiance)
        })
    }

    fn compose(&self, cache: &[PerLight], weights: &[f32]) -> Vec<f32> {
        let pixels = self.grid.pixels();
        (0..pixels)
            .map(|px| {
                (0..SECTORS)
                    .map(|q| dot(&cache[q * pixels + px], weights))
                    .sum()
            })
            .collect()
    }
}

fn dot(v: &PerLight, w: &[f32]) -> f32 {
    v.iter().zip(w).map(|(a, b)| a * b).sum()
}

struct State {
    epsilon: f32,
    cache: Vec<PerLight>,
    ledgers: Vec<[Ledger; LIGHTS]>,
    recomputed_tiles: Vec<bool>,
    recomputed_cones: u64,
    time: Duration,
    /// Total charge by source, summed over keys and lights (for the breakdown).
    geometry_charged: f64,
    propagated_charged: f64,
}

impl State {
    /// Charges per key: (geometry, propagation), each per light.
    fn charges(
        world: &World,
        bound: RadianceBound,
        hulls: &[MovedHull],
        crossing: &[bool],
        radiance: &[f32],
        changed_patches: &[(usize, PerLight)],
    ) -> Vec<(PerLight, PerLight)> {
        let (tiles, pixels) = (world.grid.tiles(), world.grid.pixels());
        let mut global = [0.0f32; LIGHTS];
        for (i, &r) in radiance.iter().enumerate() {
            global[i % LIGHTS] = global[i % LIGHTS].max(r);
        }
        // Brightest radiance per (tile, hull) that a mover's sweep can reveal or hide.
        let brightest: Vec<PerLight> = match bound {
            RadianceBound::Global => vec![global; tiles * hulls.len()],
            RadianceBound::Local | RadianceBound::LocalBlack | RadianceBound::Oracle => {
                par_map(tiles * hulls.len(), world.threads, 8, |i| {
                    local_brightest(
                        world,
                        world.pre.tile_box(i / hulls.len()),
                        &hulls[i % hulls.len()],
                        radiance,
                        global,
                    )
                })
            }
        };
        par_map(world.keys(), world.threads, 16, |k| {
            let (q, t) = (k / tiles, k % tiles);
            let theta = world.pre.theta_max(k);
            let tile_box = world.pre.tile_box(t);
            let mut geometry = [0.0f32; LIGHTS];
            for (hi, (h, &crosses)) in hulls.iter().zip(crossing).enumerate() {
                if !world.pre.frustum_touches(k, &h.hull, &h.bbox) {
                    continue;
                }
                let bright = brightest[t * hulls.len() + hi];
                let region = h.circle.max_subtended_angle(tile_box).min(theta);
                let angle = if crosses {
                    region
                } else {
                    let projected = world
                        .grid
                        .tile_pixels(t)
                        .filter_map(|px| world.pre.cone(q * pixels + px))
                        .map(|c| {
                            let (a, b) = (c.project(&h.prev), c.project(&h.now));
                            let len = |x, y| {
                                difference(x, y)
                                    .iter()
                                    .map(|&(lo, hi)| hi - lo)
                                    .sum::<f32>()
                            };
                            let (gained, lost) = (len(b, a), len(a, b));
                            let changed = if bound == RadianceBound::LocalBlack {
                                gained.max(lost)
                            } else {
                                gained + lost
                            };
                            changed * c.angle
                        })
                        .fold(0.0, f32::max);
                    region.min(projected)
                };
                for l in 0..LIGHTS {
                    geometry[l] += angle * bright[l];
                }
            }
            let mut propagated = [0.0f32; LIGHTS];
            for &(j, delta) in changed_patches {
                let (circle, bbox) = &world.patch_circles[j];
                let p = world.patches[j];
                if !world.pre.frustum_touches(k, &[p.a, p.b], bbox) {
                    continue;
                }
                let angle = circle.max_subtended_angle(tile_box).min(theta);
                for l in 0..LIGHTS {
                    propagated[l] += angle * delta[l];
                }
            }
            (geometry, propagated)
        })
    }

    fn apply(&mut self, charges: &[(PerLight, PerLight)]) {
        for (ledgers, (geometry, propagated)) in self.ledgers.iter_mut().zip(charges) {
            // Radiance first (geometry fixed), then geometry (new radiance).
            for l in 0..LIGHTS {
                ledgers[l].charge(propagated[l]);
                ledgers[l].charge(geometry[l]);
                self.propagated_charged += propagated[l] as f64;
                self.geometry_charged += geometry[l] as f64;
            }
        }
    }

    /// Per tile, recompute the largest weighted balances until the rest fit in ε.
    fn select(&self, world: &World, weights: &[f32]) -> Vec<usize> {
        let tiles = world.grid.tiles();
        let mut keys = Vec::new();
        let mut weighted: Vec<(f32, usize)> = Vec::with_capacity(SECTORS);
        for t in 0..tiles {
            weighted.clear();
            weighted.extend(
                (0..SECTORS)
                    .map(|q| q * tiles + t)
                    .map(|k| {
                        (
                            self.ledgers[k]
                                .iter()
                                .zip(weights)
                                .map(|(b, w)| b.balance() * w)
                                .sum::<f32>(),
                            k,
                        )
                    })
                    .filter(|&(b, _)| b > 0.0),
            );
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

    fn recompute(&mut self, world: &World, keys: &[usize], surfaces: &[Surface], radiance: &[f32]) {
        let (tiles, pixels) = (world.grid.tiles(), world.grid.pixels());
        let results = par_map(keys.len(), world.threads, 2, |i| {
            let (q, t) = (keys[i] / tiles, keys[i] % tiles);
            world
                .grid
                .tile_pixels(t)
                .map(|px| world.gather_one(q * pixels + px, surfaces, radiance))
                .collect::<Vec<_>>()
        });
        self.recomputed_tiles.fill(false);
        for (&k, values) in keys.iter().zip(results) {
            let (q, t) = (k / tiles, k % tiles);
            for (v, px) in values.into_iter().zip(world.grid.tile_pixels(t)) {
                self.cache[q * pixels + px] = v;
            }
            for ledger in &mut self.ledgers[k] {
                ledger.settle();
            }
            self.recomputed_tiles[t] = true;
        }
        self.recomputed_cones += (keys.len() * TILE * TILE) as u64;
    }

    fn fluence_and_bound(&self, world: &World, weights: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (tiles, pixels) = (world.grid.tiles(), world.grid.pixels());
        (0..pixels)
            .map(|px| {
                let t = world.grid.tile_of(px);
                (0..SECTORS).fold((0.0, 0.0), |(f, b), q| {
                    let balances = self.ledgers[q * tiles + t].map(|l| l.balance());
                    (
                        f + dot(&self.cache[q * pixels + px], weights),
                        b + dot(&balances, weights),
                    )
                })
            })
            .unzip()
    }
}

/// The brightest patch side a receiver in `tile` could see in a direction a
/// mover's sweep covers or uncovers. Such a direction passes through the swept
/// hull, so its nearest static surface lies in the beam from the tile through the
/// hull; only patch sides facing the tile can be seen from it.
fn local_brightest(
    world: &World,
    tile: &Aabb,
    h: &MovedHull,
    radiance: &[f32],
    global: PerLight,
) -> PerLight {
    let corners = tile.corners();
    if convex_overlap(&corners, &h.hull) {
        return global;
    }
    let mut pts: Vec<_> = corners.to_vec();
    for &v in &h.hull {
        pts.push(v);
        pts.extend(
            corners
                .iter()
                .map(|&c| v + (v - c).normalized() * BEAM_LENGTH),
        );
    }
    let beam = convex_hull(&pts);
    let beam_box = Aabb::of(&beam);
    let mut out = [0.0f32; LIGHTS];
    for (j, p) in world.patches.iter().enumerate() {
        if !world.patch_circles[j].1.overlaps(&beam_box) || !convex_overlap(&beam, &[p.a, p.b]) {
            continue;
        }
        let sides = corners.map(|c| bounce::side_of(p, c));
        for side in 0..2 {
            if sides.iter().all(|&s| s != side) {
                continue;
            }
            for (l, o) in out.iter_mut().enumerate() {
                *o = o.max(radiance[bounce::radiance_index(j, side, l)]);
            }
        }
    }
    out
}

/// Per key and light, the largest true change of any cached value in the tile.
fn oracle_charges(world: &World, now: &[PerLight], prev: &[PerLight]) -> Vec<(PerLight, PerLight)> {
    let (tiles, pixels) = (world.grid.tiles(), world.grid.pixels());
    par_map(world.keys(), world.threads, 16, |k| {
        let (q, t) = (k / tiles, k % tiles);
        let mut change = [0.0f32; LIGHTS];
        for px in world.grid.tile_pixels(t) {
            let i = q * pixels + px;
            for l in 0..LIGHTS {
                change[l] = change[l].max((now[i][l] - prev[i][l]).abs());
            }
        }
        (change, [0.0; LIGHTS])
    })
}

#[derive(Default)]
struct Stats {
    violations: u64,
    deferred: u64,
    max_error: f32,
    error_sum: f64,
    bound_sum: f64,
    phase_cones: Vec<u64>,
    max_error_by_frame: Vec<f32>,
}

fn sheet_row(
    s: &Scenario,
    world: &World,
    f: u32,
    state: &State,
    fluence: &[f32],
    bound: &[f32],
    reference: &[f32],
) -> [Image; 4] {
    let size = world.grid.size;
    let overlay = |im: &mut Image| {
        img::draw_segments(im, &world.statics, img::OCCLUDER_COLOR);
        img::draw_segments(im, &(s.movers)(f), [255, 120, 200]);
        img::draw_segments(im, &world.lights, img::LIGHT_COLOR);
    };
    let exposed: Vec<f32> = fluence.iter().map(|v| v * s.exposure).collect();
    let mut lit = img::tonemap(&exposed, size);
    overlay(&mut lit);
    let mut tiles = img::tonemap(&exposed, size);
    for px in (0..world.grid.pixels()).filter(|&px| state.recomputed_tiles[world.grid.tile_of(px)])
    {
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

fn ms(d: Duration, frames: u32) -> f64 {
    d.as_secs_f64() * 1000.0 / frames as f64
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let s = anim::bounce();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let world = World::new(&s, Grid { size: SIZE }, threads);
    let (pixels, keys) = (world.grid.pixels(), world.keys());

    let movers0 = (s.movers)(0);
    let radiance0 = world.radiance(&movers0);
    let cache0 = world.full_gather(&world.surfaces(&movers0), &radiance0);
    let fluence0 = world.compose(&cache0, &s.weights(0));
    let mean = fluence0.iter().sum::<f32>() / pixels as f32;
    let bounds = [
        RadianceBound::Global,
        RadianceBound::Local,
        RadianceBound::LocalBlack,
        RadianceBound::Oracle,
    ];
    let mut states: Vec<(RadianceBound, State, Stats)> = bounds
        .iter()
        .flat_map(|&b| EPSILONS.iter().map(move |&e| (b, e)))
        .map(|(b, e)| {
            let state = State {
                epsilon: e * mean,
                cache: cache0.clone(),
                ledgers: vec![[Ledger::default(); LIGHTS]; keys],
                recomputed_tiles: vec![false; world.grid.tiles()],
                recomputed_cones: 0,
                time: Duration::ZERO,
                geometry_charged: 0.0,
                propagated_charged: 0.0,
            };
            let stats = Stats {
                phase_cones: vec![0; s.phases.len()],
                max_error_by_frame: vec![0.0; s.frames as usize],
                ..Default::default()
            };
            (b, state, stats)
        })
        .collect();

    let (mut radiance_time, mut full_time) = (Duration::ZERO, Duration::ZERO);
    let mut crossings = 0u32;
    let mut changed_patch_frames = 0u64;
    let mut sheet: Vec<[Image; 4]> = Vec::new();
    let mut radiance_prev = radiance0;
    let mut reference_prev = cache0.clone();
    for f in 1..s.frames {
        let (movers_prev, movers_now) = ((s.movers)(f - 1), (s.movers)(f));
        let weights = s.weights(f);

        let start = Instant::now();
        let radiance = world.radiance(&movers_now);
        radiance_time += start.elapsed();
        let surfaces = world.surfaces(&movers_now);

        let start = Instant::now();
        let reference_cache = world.full_gather(&surfaces, &radiance);
        full_time += start.elapsed();
        let reference = world.compose(&reference_cache, &weights);

        let hulls = moved_hulls(&movers_prev, &movers_now);
        let crossing: Vec<bool> = hulls
            .iter()
            .map(|h| {
                world
                    .statics
                    .iter()
                    .any(|st| convex_overlap(&h.hull, &[st.a, st.b]))
            })
            .collect();
        crossings += crossing.iter().filter(|&&c| c).count() as u32;
        let changed: Vec<(usize, PerLight)> = (0..world.patches.len())
            .filter_map(|j| {
                let mut delta = [0.0f32; LIGHTS];
                for side in 0..2 {
                    for (l, d) in delta.iter_mut().enumerate() {
                        let i = bounce::radiance_index(j, side, l);
                        *d = d.max((radiance[i] - radiance_prev[i]).abs());
                    }
                }
                delta.iter().any(|&d| d > 0.0).then_some((j, delta))
            })
            .collect();
        changed_patch_frames += changed.len() as u64;

        for (i, (bound, state, stats)) in states.iter_mut().enumerate() {
            let start = Instant::now();
            let charges = if *bound == RadianceBound::Oracle {
                oracle_charges(&world, &reference_cache, &reference_prev)
            } else {
                State::charges(&world, *bound, &hulls, &crossing, &radiance, &changed)
            };
            state.apply(&charges);
            let selected = state.select(&world, &weights);
            state.recompute(&world, &selected, &surfaces, &radiance);
            state.time += start.elapsed();
            let cones = (selected.len() * TILE * TILE) as u64;
            if let Some(p) = s.phases.iter().position(|(_, r)| r.contains(&f)) {
                stats.phase_cones[p] += cones;
            }

            let (fluence, certified) = state.fluence_and_bound(&world, &weights);
            let mut frame_max = 0.0f32;
            for px in 0..pixels {
                let err = (fluence[px] - reference[px]).abs();
                if err > certified[px] + FLOAT_SLACK {
                    stats.violations += 1;
                }
                if certified[px] > 0.0 {
                    stats.deferred += 1;
                }
                frame_max = frame_max.max(err);
                stats.error_sum += err as f64;
                stats.bound_sum += certified[px] as f64;
            }
            stats.max_error = stats.max_error.max(frame_max);
            stats.max_error_by_frame[f as usize] = frame_max;
            if i == 2 * EPSILONS.len() + HEADLINE && s.key_frames.contains(&f) {
                sheet.push(sheet_row(
                    &s, &world, f, state, &fluence, &certified, &reference,
                ));
            }
        }
        radiance_prev = radiance;
        reference_prev = reference_cache;
    }

    let frames = s.frames - 1;
    let full_ms = ms(full_time + radiance_time, frames);
    let total_cones = frames as f64 * (SECTORS * pixels) as f64;
    let mut out = String::new();
    writeln!(out, "# Flatland gate 6 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} receivers, {TILE}×{TILE} tiles, {SECTORS} gather cones per receiver, {} patches \
         (albedo {}), 2 lights, {} frames, {threads} threads for reference and incremental alike. \
         Times include recomputing patch radiance ({:.2} ms/frame) for both.\n",
        world.patches.len(),
        bounce::ALBEDO,
        s.frames,
        ms(radiance_time, frames)
    )
    .unwrap();
    writeln!(
        out,
        "Mean indirect fluence at frame 0: {mean:.4}; ε is a fraction of it. Full recompute: {full_ms:.2} ms/frame. \
         Patches changing radiance: {:.1} per frame on average. Mover sweeps touching a surface: {crossings}.\n",
        changed_patch_frames as f64 / frames as f64
    )
    .unwrap();
    let receiver_frames = frames as f64 * pixels as f64;
    let mut ratios = Vec::new();
    for (bi, bound) in bounds.iter().enumerate() {
        let runs = &states[bi * EPSILONS.len()..(bi + 1) * EPSILONS.len()];
        writeln!(out, "## {}\n", bound.name()).unwrap();
        writeln!(out, "| ε | Violations | Deferred receiver-frames | Max error | Max error / ε | Bound use | Cones recomputed | Skipped | Charge share: propagated | Incremental ms/frame | Speedup vs full |").unwrap();
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|").unwrap();
        for (i, ((_, state, stats), &frac)) in runs.iter().zip(&EPSILONS).enumerate() {
            let recomputed = state.recomputed_cones as f64 / total_cones;
            let inc_ms = ms(state.time + radiance_time, frames);
            let ratio = if state.epsilon > 0.0 {
                format!("{:.2}", stats.max_error / state.epsilon)
            } else {
                "—".into()
            };
            let bound_use = if stats.bound_sum > 0.0 {
                format!("{:.1}%", 100.0 * stats.error_sum / stats.bound_sum)
            } else {
                "—".into()
            };
            let charged = state.geometry_charged + state.propagated_charged;
            writeln!(
                out,
                "| {:.0}%{} | {} | {:.1}% | {:.5} | {} | {} | {:.1}% | {:.1}% | {:.0}% | {:.2} | {:.1}× |",
                frac * 100.0,
                if i >= STRESS_FROM { " (stress test)" } else { "" },
                stats.violations,
                100.0 * stats.deferred as f64 / receiver_frames,
                stats.max_error,
                ratio,
                bound_use,
                recomputed * 100.0,
                (1.0 - recomputed) * 100.0,
                if charged > 0.0 { 100.0 * state.propagated_charged / charged } else { 0.0 },
                inc_ms,
                full_ms / inc_ms
            )
            .unwrap();
        }
        writeln!(out, "\nCones recomputed per frame, by phase:\n").unwrap();
        let header: Vec<String> = s
            .phases
            .iter()
            .map(|(n, r)| format!("{n} (frames {}–{})", r.start, r.end - 1))
            .collect();
        writeln!(out, "| ε | {} |", header.join(" | ")).unwrap();
        writeln!(out, "|---|{}", "---|".repeat(s.phases.len())).unwrap();
        for ((_, _, stats), &frac) in runs.iter().zip(&EPSILONS) {
            let cells: Vec<String> = s
                .phases
                .iter()
                .zip(&stats.phase_cones)
                .map(|((_, r), &c)| {
                    format!(
                        "{:.1}%",
                        100.0 * c as f64 / ((SECTORS * pixels) as f64 * r.len() as f64)
                    )
                })
                .collect();
            writeln!(out, "| {:.0}% | {} |", frac * 100.0, cells.join(" | ")).unwrap();
        }
        let switch = s.phases[1].1.start as usize;
        let (_, headline, headline_stats) = &runs[HEADLINE];
        writeln!(
            out,
            "\nLight B switches on at frame {switch}: max indirect error that frame {:.5} (ε = {:.5}).\n",
            headline_stats.max_error_by_frame[switch], headline.epsilon
        )
        .unwrap();
        ratios.push(runs[HEADLINE].1.recomputed_cones as f64 / runs[0].1.recomputed_cones as f64);
    }

    let violations: u64 = states.iter().map(|(_, _, st)| st.violations).sum();
    let verdict = |r: f64| {
        if violations == 0 && r <= 0.5 {
            "PASS"
        } else {
            "FAIL"
        }
    };
    writeln!(out, "## Verdict\n").unwrap();
    writeln!(
        out,
        "- Soundness: {violations} violations across every configuration, ε and frame.\n\
         - Cone recomputes at ε = 1% as a fraction of ε = 0 (target ≤ 50%): attempt 1 {:.1}% → {}; \
         attempt 2 {:.1}% → {}; attempt 3 {:.1}% → {}.",
        ratios[0] * 100.0,
        verdict(ratios[0]),
        ratios[1] * 100.0,
        verdict(ratios[1]),
        ratios[2] * 100.0,
        verdict(ratios[2])
    )
    .unwrap();
    writeln!(
        out,
        "- Diagnostic (not a verdict): oracle charges reach {:.1}% — the floor for any charging scheme at \
         this tile granularity and ε.",
        ratios[3] * 100.0
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("gate6_results.md"), &out)?;
    incremental::save_sheet(&sheet, &out_dir.join("gate6_bounce.png"))?;
    Ok(())
}
