//! Flatland gate 7: the ledger's own job, a moving light.
//!
//! When a light drifts (a sun on a day-night cycle), every cone toward it changes
//! a little every frame, so exact invalidation must recompute 100% of cones every
//! frame. Only error bounds can skip that work. The light here drifts sideways
//! while the door, walker and fan move as in the typical scenario.
//!
//! Charges per tile follow the triangle inequality, light first (movers fixed),
//! then movers (light at its new position):
//!
//! - **Light motion**: directions enter or leave a receiver's cone only as the
//!   light's endpoints move, so the visible angle changes by at most the angles
//!   the two endpoint paths subtend. Directions inside both cones keep their
//!   blocked status unless an occluder lies between the old and new light, which
//!   is checked (and then the whole cone is charged).
//! - **Movers**: the geometric region/motion bounds from gates 2–3.
//!
//! Visibility uses the exact 2D estimator, so the bounds are tested without
//! bin-alignment effects (a mask's bins move with the cone).
//!
//! Pass criterion, fixed before the first run: at the realistic drift speed (1×,
//! a 24-minute day: about 0.25° of sun motion per second) and ε = 1%, zero
//! soundness violations and at most 25% of cones recomputed. Faster speeds are
//! informational.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::Ledger;
use flatland::anim;
use flatland::bounds::{
    Aabb, Circle, convex_distance, convex_hull, convex_overlap, max_segment_angle,
};
use flatland::geom::{Cone, Segment, Vec2, seg};
use flatland::incremental::{Grid, TILE, moved_hulls, par_map};
use flatland::methods;

const SIZE: usize = 256;
const FRAMES: u32 = 240;
const EPSILONS: [f32; 3] = [0.0, 0.01, 0.04];
const HEADLINE: usize = 1;
/// Light drift per frame at 1×: 0.25°/s at 60 fps seen from about 0.5 away.
const BASE_DRIFT: f32 = 0.5 * (0.25f32 / 60.0) * (std::f32::consts::PI / 180.0);
const SPEEDS: [f32; 4] = [1.0, 4.0, 16.0, 64.0];
const FLOAT_SLACK: f32 = 1e-5;

fn light_at(f: u32, speed: f32) -> Segment {
    let dx = f as f32 * BASE_DRIFT * speed;
    seg(0.38 + dx, 0.95, 0.62 + dx, 0.95)
}

fn visible_angle(p: Vec2, light: Segment, occluders: &[Segment]) -> f32 {
    Cone::toward(p, light).map_or(0.0, |c| {
        c.angle * methods::exact(&c, occluders, &mut Vec::new())
    })
}

struct Run {
    recomputed: u64,
    violations: u64,
    max_error: f32,
    time: Duration,
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let grid = Grid { size: SIZE };
    let (pixels, tiles) = (grid.pixels(), grid.tiles());
    let typical = anim::typical();
    let statics = typical.statics.clone();
    let movers_at = |f: u32| (typical.movers)(f);
    let tile_boxes: Vec<Aabb> = (0..tiles)
        .map(|t| {
            Aabb::of(
                &grid
                    .tile_pixels(t)
                    .map(|px| grid.pixel_center(px))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    let mut out = String::new();
    writeln!(out, "# Flatland gate 7 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} receivers, {TILE}×{TILE} tiles, one drifting light plus the door, walker and fan, \
         {FRAMES} frames, exact 2D visibility, {threads} threads for both paths. 1× drift = {:.2e} of the \
         room per frame (a 24-minute day). Exact invalidation recomputes 100% of cones whenever the light moves.\n",
        BASE_DRIFT
    )
    .unwrap();
    writeln!(out, "| Drift speed | ε | Violations | Max error / ε | Cones recomputed | Incremental ms/frame | Full ms/frame | Speedup |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|").unwrap();

    let mut headline = (0.0f64, 0u64);
    for &speed in &SPEEDS {
        let occluders_at = |f: u32| {
            let mut v = statics.clone();
            v.extend(movers_at(f));
            v
        };
        let occluders0 = occluders_at(0);
        let reference0: Vec<f32> = par_map(pixels, threads, 256, |px| {
            visible_angle(grid.pixel_center(px), light_at(0, speed), &occluders0)
        });
        let mean = reference0.iter().sum::<f32>() / pixels as f32;
        let mut caches: Vec<Vec<f32>> = EPSILONS.iter().map(|_| reference0.clone()).collect();
        let mut ledgers: Vec<Vec<Ledger>> = EPSILONS
            .iter()
            .map(|_| vec![Ledger::default(); tiles])
            .collect();
        let mut runs: Vec<Run> = EPSILONS
            .iter()
            .map(|_| Run {
                recomputed: 0,
                violations: 0,
                max_error: 0.0,
                time: Duration::ZERO,
            })
            .collect();
        let mut full_time = Duration::ZERO;

        for f in 1..FRAMES {
            let (light_prev, light_now) = (light_at(f - 1, speed), light_at(f, speed));
            let (movers_prev, movers_now) = (movers_at(f - 1), movers_at(f));
            let occluders = occluders_at(f);

            let start = Instant::now();
            let reference: Vec<f32> = par_map(pixels, threads, 256, |px| {
                visible_angle(grid.pixel_center(px), light_now, &occluders)
            });
            full_time += start.elapsed();

            for (i, &e) in EPSILONS.iter().enumerate() {
                let epsilon = e * mean;
                let start = Instant::now();
                // Step 1: the light moves, movers still at their previous positions.
                let sweep = convex_hull(&[light_prev.a, light_prev.b, light_now.a, light_now.b]);
                let mut before = statics.clone();
                before.extend_from_slice(&movers_prev);
                let light_crosses = before.iter().any(|s| convex_overlap(&sweep, &[s.a, s.b]));
                let (da, db) = (
                    (light_now.a - light_prev.a).length(),
                    (light_now.b - light_prev.b).length(),
                );
                // Step 2: movers move with the light at its new position.
                let hulls = moved_hulls(&movers_prev, &movers_now);
                let light_circle =
                    Circle::enclosing(&[light_now.a, light_now.b, light_prev.a, light_prev.b]);
                let charges: Vec<f32> = par_map(tiles, threads, 16, |t| {
                    let tile = &tile_boxes[t];
                    let corners = tile.corners();
                    let theta = light_circle.max_subtended_angle(tile);
                    let mut charge = 0.0;
                    if da > 0.0 || db > 0.0 {
                        let d0 = convex_distance(&corners, &[light_prev.a, light_now.a]);
                        let d1 = convex_distance(&corners, &[light_prev.b, light_now.b]);
                        charge += if light_crosses || d0 <= 0.0 || d1 <= 0.0 {
                            theta
                        } else {
                            (max_segment_angle(da, d0) + max_segment_angle(db, d1)).min(theta)
                        };
                    }
                    let mut frustum_pts = corners.to_vec();
                    frustum_pts.extend([light_now.a, light_now.b]);
                    let frustum = convex_hull(&frustum_pts);
                    for h in &hulls {
                        if !convex_overlap(&frustum, &h.hull) {
                            continue;
                        }
                        let region = h.circle.max_subtended_angle(tile);
                        let d = convex_distance(&corners, &h.hull);
                        let motion =
                            if d > 0.0 && !convex_overlap(&h.hull, &[light_now.a, light_now.b]) {
                                2.0 * max_segment_angle(h.max_displacement, d)
                            } else {
                                f32::INFINITY
                            };
                        charge += region.min(motion).min(theta);
                    }
                    charge
                });
                let selected: Vec<usize> = (0..tiles)
                    .filter(|&t| {
                        ledgers[i][t].charge(charges[t]);
                        ledgers[i][t].exceeds(1.0, epsilon)
                    })
                    .collect();
                let values: Vec<Vec<f32>> = par_map(selected.len(), threads, 2, |s| {
                    grid.tile_pixels(selected[s])
                        .map(|px| visible_angle(grid.pixel_center(px), light_now, &occluders))
                        .collect()
                });
                for (&t, vals) in selected.iter().zip(values) {
                    for (px, v) in grid.tile_pixels(t).zip(vals) {
                        caches[i][px] = v;
                    }
                    ledgers[i][t].settle();
                }
                runs[i].time += start.elapsed();
                runs[i].recomputed += (selected.len() * TILE * TILE) as u64;

                for px in 0..pixels {
                    let err = (caches[i][px] - reference[px]).abs();
                    if err > ledgers[i][grid.tile_of(px)].balance() + FLOAT_SLACK {
                        runs[i].violations += 1;
                    }
                    runs[i].max_error = runs[i].max_error.max(err);
                }
            }
        }

        let frames = (FRAMES - 1) as f64;
        let full_ms = full_time.as_secs_f64() * 1000.0 / frames;
        for (i, (run, &e)) in runs.iter().zip(&EPSILONS).enumerate() {
            let recomputed = run.recomputed as f64 / (frames * pixels as f64);
            let inc_ms = run.time.as_secs_f64() * 1000.0 / frames;
            let ratio = if e > 0.0 {
                format!("{:.2}", run.max_error / (e * mean))
            } else {
                "—".into()
            };
            writeln!(
                out,
                "| {speed}× | {:.0}% | {} | {} | {:.1}% | {:.2} | {:.2} | {:.1}× |",
                e * 100.0,
                run.violations,
                ratio,
                recomputed * 100.0,
                inc_ms,
                full_ms,
                full_ms / inc_ms
            )
            .unwrap();
            if speed == 1.0 && i == HEADLINE {
                headline = (recomputed, runs.iter().map(|r| r.violations).sum());
            }
        }
    }
    let pass = headline.1 == 0 && headline.0 <= 0.25;
    writeln!(
        out,
        "\n## Verdict\n\nAt 1× drift and ε = 1%: {:.1}% of cones recomputed (target ≤ 25%), {} violations → {}.",
        headline.0 * 100.0,
        headline.1,
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("gate7_results.md"), &out)?;
    Ok(())
}
