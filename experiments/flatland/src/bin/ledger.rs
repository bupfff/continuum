//! Flatland gates 2 and 3: is the error ledger sound, and how much work does it skip?
//!
//! Animates scenarios where occluders move and lights switch on and flicker.
//! Each frame, an incremental renderer reuses cached visibility unless a tile's
//! ledger says it could be more than ε off, while a full recompute runs
//! alongside as the reference.
//!
//! - Gate 2 (soundness): the certified bound is never below the actual error.
//! - Gate 3 (the edge): the fraction of cone evaluations the ledger skips.
//!
//! Caches hold visible angle per light, so light intensity changes are exact and
//! free (lighting is linear in emitted power); only geometry motion is charged.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::{ConeMask, Ledger};
use flatland::anim::{self, Scenario};
use flatland::bounds::{
    Aabb, Circle, convex_distance, convex_hull, convex_overlap, max_segment_angle,
};
use flatland::geom::{Cone, Segment, Vec2, v2};
use flatland::img::{self, Image};
use flatland::methods;

const SIZE: usize = 256;
const PIXELS: usize = SIZE * SIZE;
const TILE: usize = 8;
const TILES_PER_SIDE: usize = SIZE / TILE;
const TILES: usize = TILES_PER_SIDE * TILES_PER_SIDE;
/// Absolute slack for float rounding in the soundness check.
const FLOAT_SLACK: f32 = 1e-5;
/// ε values, as fractions of the scenario's mean fluence at frame 0. Values from
/// `STRESS_FROM` on are far too large to use; they exist to force deferral so the
/// magnitude bounds, not just the dependency regions, get tested.
const EPSILONS: [f32; 5] = [0.0, 0.01, 0.04, 0.16, 0.64];
const STRESS_FROM: usize = 3;
/// Index into `EPSILONS` used for the gate 3 verdict and the main image sheet.
const HEADLINE: usize = 1;
/// Image sheets to write, as (index into `EPSILONS`, file-name suffix).
const SHEETS: [(usize, &str); 2] = [(HEADLINE, ""), (STRESS_FROM, "_stress")];

fn pixel_center(px: usize) -> Vec2 {
    let (i, j) = (px % SIZE, px / SIZE);
    v2(
        (i as f32 + 0.5) / SIZE as f32,
        1.0 - (j as f32 + 0.5) / SIZE as f32,
    )
}

fn tile_of(px: usize) -> usize {
    let (i, j) = (px % SIZE, px / SIZE);
    (j / TILE) * TILES_PER_SIDE + i / TILE
}

fn tile_pixels(t: usize) -> impl Iterator<Item = usize> {
    let (tx, ty) = (t % TILES_PER_SIDE, t / TILES_PER_SIDE);
    (0..TILE).flat_map(move |dy| (0..TILE).map(move |dx| (ty * TILE + dy) * SIZE + tx * TILE + dx))
}

/// Everything that depends only on static geometry and light positions. Arrays
/// are indexed `[light * PIXELS + pixel]` or `[light * TILES + tile]`.
struct Precomputed {
    lights: usize,
    light_segments: Vec<Segment>,
    cones: Vec<Option<Cone>>,
    static_masks: Vec<ConeMask>,
    tile_boxes: Vec<Aabb>,
    /// Hull of a tile and a light: contains every receiver's cone in the tile.
    frustums: Vec<Vec<Vec2>>,
    frustum_boxes: Vec<Aabb>,
    /// Largest light angle seen from any receiver in the tile.
    theta_max: Vec<f32>,
    /// Statics hide the light from every receiver in the tile, so nothing can change there.
    static_blocked: Vec<bool>,
}

impl Precomputed {
    fn new(s: &Scenario) -> Self {
        let lights = s.lights.len();
        let mut cones = Vec::with_capacity(lights * PIXELS);
        let mut static_masks = Vec::with_capacity(lights * PIXELS);
        for light in &s.lights {
            for px in 0..PIXELS {
                let cone = Cone::toward(pixel_center(px), light.segment);
                static_masks
                    .push(cone.map_or(ConeMask::FULL, |c| methods::mask_bits(&c, &s.statics)));
                cones.push(cone);
            }
        }
        let tile_boxes: Vec<Aabb> = (0..TILES)
            .map(|t| Aabb::of(&tile_pixels(t).map(pixel_center).collect::<Vec<_>>()))
            .collect();
        let mut pre = Precomputed {
            lights,
            light_segments: s.lights.iter().map(|l| l.segment).collect(),
            cones,
            static_masks,
            tile_boxes,
            frustums: Vec::new(),
            frustum_boxes: Vec::new(),
            theta_max: Vec::new(),
            static_blocked: Vec::new(),
        };
        for (li, light) in s.lights.iter().enumerate() {
            for t in 0..TILES {
                let mut pts = pre.tile_boxes[t].corners().to_vec();
                pts.extend([light.segment.a, light.segment.b]);
                let hull = convex_hull(&pts);
                pre.frustum_boxes.push(Aabb::of(&hull));
                pre.frustums.push(hull);
                let in_tile = || tile_pixels(t).map(|px| li * PIXELS + px);
                pre.theta_max.push(
                    in_tile()
                        .filter_map(|k| pre.cones[k].map(|c| c.angle))
                        .fold(0.0, f32::max),
                );
                pre.static_blocked
                    .push(in_tile().all(|k| pre.static_masks[k].is_full()));
            }
        }
        pre
    }

    /// Visible angle (radians) of light `li` from pixel `px`, reusing the static mask.
    fn visible_angle(&self, li: usize, px: usize, movers: &[Segment]) -> f32 {
        let k = li * PIXELS + px;
        match self.cones[k] {
            Some(cone) if !self.static_masks[k].is_full() => {
                let m = self.static_masks[k] | methods::mask_bits(&cone, movers);
                cone.angle * m.visible_fraction()
            }
            _ => 0.0,
        }
    }
}

/// Conservative region a moving occluder can have touched during one frame step.
struct MovedHull {
    hull: Vec<Vec2>,
    bbox: Aabb,
    circle: Circle,
    /// Farthest either endpoint moved.
    max_displacement: f32,
    /// Per light: whether the hull touches the light itself.
    touches_light: Vec<bool>,
}

fn moved_hulls(prev: &[Segment], now: &[Segment], lights: &[Segment]) -> Vec<MovedHull> {
    prev.iter()
        .zip(now)
        .filter(|(p, n)| p != n)
        .map(|(p, n)| {
            // Linearly moving endpoints sweep a region inside the hull of both positions.
            let hull = convex_hull(&[p.a, p.b, n.a, n.b]);
            MovedHull {
                bbox: Aabb::of(&hull),
                circle: Circle::enclosing(&hull),
                max_displacement: (n.a - p.a).length().max((n.b - p.b).length()),
                touches_light: lights
                    .iter()
                    .map(|l| convex_overlap(&hull, &[l.a, l.b]))
                    .collect(),
                hull,
            }
        })
        .collect()
}

struct Incremental {
    /// Absolute fluence error allowed per receiver.
    epsilon: f32,
    visible: Vec<f32>,
    ledgers: Vec<Ledger>,
    recomputed_tiles: Vec<bool>,
    recomputed_cones: u64,
    time: Duration,
}

impl Incremental {
    /// Charges every (light, tile) whose cones a moved occluder could reach, with
    /// the smaller of two worst-case bounds on the change in visible angle:
    ///
    /// - Region bound: only directions into the swept hull can change, so at most
    ///   the hull's subtended angle plus one bin (a bin flips when its center is
    ///   crossed, so even a sliver can flip one).
    /// - Motion bound: a direction can only change if an endpoint's path crosses
    ///   it (or the occluder slides past the light, excluded via `touches_light`),
    ///   so at most the two endpoint paths' angles plus one bin each.
    ///
    /// Neither may exceed the whole light.
    fn charge(&mut self, pre: &Precomputed, hulls: &[MovedHull]) {
        for k in 0..pre.lights * TILES {
            if pre.static_blocked[k] {
                continue;
            }
            let (li, t) = (k / TILES, k % TILES);
            for h in hulls {
                if !h.bbox.overlaps(&pre.frustum_boxes[k])
                    || !convex_overlap(&pre.frustums[k], &h.hull)
                {
                    continue;
                }
                let theta = pre.theta_max[k];
                let bin = theta / ConeMask::BINS as f32;
                let region = h.circle.max_subtended_angle(&pre.tile_boxes[t]) + bin;
                let d = convex_distance(&pre.tile_boxes[t].corners(), &h.hull);
                let motion = if d > 0.0 && !h.touches_light[li] {
                    2.0 * (max_segment_angle(h.max_displacement, d) + bin)
                } else {
                    f32::INFINITY
                };
                self.ledgers[k].charge(region.min(motion).min(theta));
            }
        }
    }

    /// Recomputes every (light, tile) whose weighted balance exceeds its share of ε.
    fn recompute(&mut self, pre: &Precomputed, radiance: &[f32], movers: &[Segment]) -> u64 {
        self.recomputed_tiles.fill(false);
        let share = self.epsilon / pre.lights as f32;
        let mut cones = 0;
        for k in 0..pre.lights * TILES {
            let (li, t) = (k / TILES, k % TILES);
            if self.ledgers[k].exceeds(radiance[li], share) {
                for px in tile_pixels(t) {
                    self.visible[li * PIXELS + px] = pre.visible_angle(li, px, movers);
                }
                self.ledgers[k].settle();
                self.recomputed_tiles[t] = true;
                cones += (TILE * TILE) as u64;
            }
        }
        cones
    }

    fn fluence_and_bound(&self, radiance: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (0..PIXELS)
            .map(|px| {
                let t = tile_of(px);
                radiance
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(f, b), (li, &r)| {
                        (
                            f + r * self.visible[li * PIXELS + px],
                            b + r * self.ledgers[li * TILES + t].balance(),
                        )
                    })
            })
            .unzip()
    }
}

fn compose(visible: &[f32], radiance: &[f32]) -> Vec<f32> {
    (0..PIXELS)
        .map(|px| {
            radiance
                .iter()
                .enumerate()
                .map(|(li, &r)| r * visible[li * PIXELS + px])
                .sum()
        })
        .collect()
}

/// Conventional baseline: every cone against every occluder, every frame.
fn full_recompute(pre: &Precomputed, occluders: &[Segment]) -> Vec<f32> {
    pre.cones
        .iter()
        .map(|c| c.map_or(0.0, |c| c.angle * methods::mask(&c, occluders)))
        .collect()
}

#[derive(Default)]
struct RunStats {
    violations: u64,
    max_error: f32,
    error_sum: f64,
    bound_sum: f64,
    /// Receiver-frames whose output carried a nonzero certified bound, i.e. where
    /// the ledger actually deferred work instead of recomputing.
    deferred: u64,
    phase_cones: Vec<u64>,
    max_error_by_frame: Vec<f32>,
}

struct ScenarioResult {
    mean_fluence: f32,
    epsilons: Vec<f32>,
    runs: Vec<(Incremental, RunStats)>,
    full_time: Duration,
    static_cached_time: Duration,
    frames_timed: u32,
    /// One list of rows per entry in `SHEETS`.
    sheets: Vec<Vec<[Image; 4]>>,
}

fn run(s: &Scenario) -> ScenarioResult {
    let pre = Precomputed::new(s);
    let occluders_at = |f: u32| {
        let mut v = s.statics.clone();
        v.extend((s.movers)(f));
        v
    };
    let radiance_at = |f: u32| {
        s.lights
            .iter()
            .map(|l| (l.radiance)(f))
            .collect::<Vec<f32>>()
    };

    let visible0 = full_recompute(&pre, &occluders_at(0));
    let fluence0 = compose(&visible0, &radiance_at(0));
    let mean_fluence = fluence0.iter().sum::<f32>() / PIXELS as f32;
    let epsilons: Vec<f32> = EPSILONS.iter().map(|e| e * mean_fluence).collect();
    let mut runs: Vec<(Incremental, RunStats)> = epsilons
        .iter()
        .map(|&epsilon| {
            let state = Incremental {
                epsilon,
                visible: visible0.clone(),
                ledgers: vec![Ledger::default(); pre.lights * TILES],
                recomputed_tiles: vec![false; TILES],
                recomputed_cones: 0,
                time: Duration::ZERO,
            };
            let stats = RunStats {
                phase_cones: vec![0; s.phases.len()],
                max_error_by_frame: vec![0.0; s.frames as usize],
                ..Default::default()
            };
            (state, stats)
        })
        .collect();

    let (mut full_time, mut static_cached_time) = (Duration::ZERO, Duration::ZERO);
    let mut sheets: Vec<Vec<[Image; 4]>> = SHEETS.iter().map(|_| Vec::new()).collect();
    for f in 1..s.frames {
        let movers_prev = (s.movers)(f - 1);
        let movers_now = (s.movers)(f);
        let occluders = occluders_at(f);
        let radiance = radiance_at(f);

        let start = Instant::now();
        let reference_visible = full_recompute(&pre, &occluders);
        full_time += start.elapsed();
        let reference = compose(&reference_visible, &radiance);

        // Static caching alone, to separate its gain from the ledger's.
        let start = Instant::now();
        let cached: Vec<f32> = (0..pre.lights * PIXELS)
            .map(|k| pre.visible_angle(k / PIXELS, k % PIXELS, &movers_now))
            .collect();
        static_cached_time += start.elapsed();
        assert_eq!(
            cached, reference_visible,
            "static-mask caching must not change results"
        );

        let hulls = moved_hulls(&movers_prev, &movers_now, &pre.light_segments);
        for (i, (state, stats)) in runs.iter_mut().enumerate() {
            let start = Instant::now();
            state.charge(&pre, &hulls);
            let cones = state.recompute(&pre, &radiance, &movers_now);
            state.time += start.elapsed();
            state.recomputed_cones += cones;
            if let Some(p) = s.phases.iter().position(|(_, r)| r.contains(&f)) {
                stats.phase_cones[p] += cones;
            }

            let (fluence, bound) = state.fluence_and_bound(&radiance);
            let mut frame_max = 0.0f32;
            for px in 0..PIXELS {
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

            if let Some(sheet) = SHEETS.iter().position(|&(e, _)| e == i)
                && s.key_frames.contains(&f)
            {
                sheets[sheet].push(sheet_row(s, f, state, &fluence, &bound, &reference));
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
        sheets,
    }
}

/// Lit image, recomputed tiles, certified bound, and actual error for one frame.
fn sheet_row(
    s: &Scenario,
    f: u32,
    state: &Incremental,
    fluence: &[f32],
    bound: &[f32],
    reference: &[f32],
) -> [Image; 4] {
    let overlay = |im: &mut Image| {
        img::draw_segments(im, &s.statics, img::OCCLUDER_COLOR);
        img::draw_segments(im, &(s.movers)(f), [255, 120, 200]);
        let lights: Vec<Segment> = s.lights.iter().map(|l| l.segment).collect();
        img::draw_segments(im, &lights, img::LIGHT_COLOR);
    };
    let mut lit = img::tonemap(fluence, SIZE);
    overlay(&mut lit);
    let mut tiles = img::tonemap(fluence, SIZE);
    for px in (0..PIXELS).filter(|&px| state.recomputed_tiles[tile_of(px)]) {
        let (x, y) = (px % SIZE, px / SIZE);
        let c = tiles.get(x, y);
        tiles.set(x, y, [c[0] / 3 + 170, c[1] / 3 + 90, c[2] / 3 + 20]);
    }
    overlay(&mut tiles);
    let scale = state.epsilon.max(1e-6);
    let mut bound_img = img::magnitude(bound, SIZE, scale);
    overlay(&mut bound_img);
    let mut error_img = img::signed_error(fluence, reference, SIZE, scale);
    overlay(&mut error_img);
    [lit, tiles, bound_img, error_img]
}

fn save_sheet(rows: &[[Image; 4]], path: &Path) -> std::io::Result<()> {
    let gap = 8;
    let mut sheet = Image::new(4 * SIZE + 5 * gap, rows.len() * (SIZE + gap) + gap, [28; 3]);
    for (r, row) in rows.iter().enumerate() {
        for (c, im) in row.iter().enumerate() {
            sheet.blit(im, gap + c * (SIZE + gap), gap + r * (SIZE + gap));
        }
    }
    sheet.save(path)
}

fn ms_per_frame(d: Duration, frames: u32) -> f64 {
    d.as_secs_f64() * 1000.0 / frames as f64
}

fn report(out: &mut String, s: &Scenario, r: &ScenarioResult) {
    let frames = r.frames_timed;
    let total_cones = (frames as u64) * (r.runs[0].0.ledgers.len() / TILES * PIXELS) as u64;
    let full_ms = ms_per_frame(r.full_time, frames);
    writeln!(out, "## {}\n", s.name).unwrap();
    writeln!(
        out,
        "Mean fluence at frame 0: {:.4}. ε is a fraction of it. Full recompute: {:.2} ms/frame; \
         full recompute reusing static masks (no ledger): {:.2} ms/frame.\n",
        r.mean_fluence,
        full_ms,
        ms_per_frame(r.static_cached_time, frames)
    )
    .unwrap();
    writeln!(out, "| ε | Gate 2 violations | Deferred receiver-frames | Max error | Max error / ε | Bound use | Cones recomputed | Skipped | Incremental ms/frame | Speedup vs full |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|").unwrap();
    let total_receiver_frames = (frames as u64 * PIXELS as u64) as f64;
    for (i, ((state, stats), (&frac, &eps))) in r
        .runs
        .iter()
        .zip(EPSILONS.iter().zip(&r.epsilons))
        .enumerate()
    {
        let stress = if i >= STRESS_FROM {
            " (stress test)"
        } else {
            ""
        };
        let recomputed = state.recomputed_cones as f64 / total_cones as f64;
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
            100.0 * stats.deferred as f64 / total_receiver_frames,
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
        let per_frame = (r.runs[0].0.ledgers.len() / TILES * PIXELS) as f64;
        for ((_, stats), &frac) in r.runs.iter().zip(&EPSILONS) {
            let cells: Vec<String> = s
                .phases
                .iter()
                .zip(&stats.phase_cones)
                .map(|((_, range), &c)| {
                    format!(
                        "{:.1}%",
                        100.0 * c as f64 / (per_frame * range.len() as f64)
                    )
                })
                .collect();
            writeln!(out, "| {:.0}% | {} |", frac * 100.0, cells.join(" | ")).unwrap();
        }
    }
    writeln!(out).unwrap();
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut out = String::new();
    writeln!(out, "# Flatland gates 2 and 3 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} receivers, {TILE}×{TILE}-receiver tiles, 2 lights, 240 frames, \
         single-threaded release build. Speedups count charging and recomputing; \
         the soundness check is not timed.\n"
    )
    .unwrap();

    let mut gate2_violations = 0;
    let mut typical_skip = 0.0;
    for s in [anim::typical(), anim::everything_moves()] {
        let r = run(&s);
        report(&mut out, &s, &r);
        gate2_violations += r.runs.iter().map(|(_, st)| st.violations).sum::<u64>();
        if s.slug == "typical" {
            let (state, stats) = &r.runs[HEADLINE];
            let total = (r.frames_timed as u64) * (state.ledgers.len() / TILES * PIXELS) as u64;
            typical_skip = 1.0 - state.recomputed_cones as f64 / total as f64;
            let switch = s.phases[1].1.start as usize;
            writeln!(
                out,
                "Light switch (frame {switch}) at ε = {:.0}%: max error {:.5} vs ε {:.5}.\n",
                EPSILONS[HEADLINE] * 100.0,
                stats.max_error_by_frame[switch],
                r.epsilons[HEADLINE]
            )
            .unwrap();
        }
        for (rows, (_, suffix)) in r.sheets.iter().zip(SHEETS) {
            save_sheet(
                rows,
                &out_dir.join(format!("ledger_{}{suffix}.png", s.slug)),
            )?;
        }
    }

    writeln!(out, "## Verdicts\n").unwrap();
    writeln!(
        out,
        "- Gate 2 (soundness): {} violations across every scenario, ε and frame → {}.",
        gate2_violations,
        if gate2_violations == 0 {
            "PASS"
        } else {
            "FAIL"
        }
    )
    .unwrap();
    writeln!(
        out,
        "- Gate 3 (edge): typical scenario at ε = {:.0}% skips {:.1}% of cone evaluations (target ≥ 80%) → {}.",
        EPSILONS[HEADLINE] * 100.0,
        typical_skip * 100.0,
        if typical_skip >= 0.8 { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("ledger_results.md"), &out)?;
    Ok(())
}
