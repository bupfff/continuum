//! Flatland gate 6b: exact incremental indirect light with cone G-buffers.
//!
//! Gate 6 found that bounds on how much indirect light can change are about 3×
//! too loose to beat exact invalidation by much. This gate removes the bounds for
//! the two kinds of change in that scene:
//!
//! - Each gather cone caches a G-buffer: per bin, the nearest static patch at the
//!   bin's center direction and its depth (`bounce::StaticLayer`).
//! - A black mover is composited on top by depth: project it into the bins and
//!   compare depths. Only cones whose frustum the mover's swept hull touches are
//!   updated.
//! - A patch whose radiance changed is resolved by re-summing only the cones whose
//!   G-buffer actually shows that patch.
//!
//! Both are exact updates of the binned estimator, so the incremental image must
//! equal a full recompute.
//!
//! Pass criterion, fixed before the first run: (1) the incremental output equals
//! a full recompute with the same estimator at every receiver and frame, within
//! float rounding (1e-5); (2) the incremental path is at least 4× faster than the
//! full recompute at the same thread count.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use flatland::anim::{self, sky_targets};
use flatland::bounce::{self, LIGHTS, NO_SURFACE, StaticLayer, Surface};
use flatland::bounds::Aabb;
use flatland::geom::Segment;
use flatland::incremental::{Grid, Precomputed, TILE, Visibility, moved_hulls, par_map};

const SIZE: usize = 128;
const SECTORS: usize = 8;
const PATCH_LENGTH: f32 = 0.04;
const TOLERANCE: f32 = 1e-5;

type PerLight = [f32; LIGHTS];

fn dot(v: &PerLight, w: &[f32]) -> f32 {
    v.iter().zip(w).map(|(a, b)| a * b).sum()
}

fn ms(d: Duration, frames: u32) -> f64 {
    d.as_secs_f64() * 1000.0 / frames as f64
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let s = anim::bounce();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let grid = Grid { size: SIZE };
    let (pixels, tiles) = (grid.pixels(), grid.tiles());
    let cones_total = SECTORS * pixels;
    let pre = Precomputed::with_targets(
        &sky_targets(SECTORS),
        &[],
        grid,
        Visibility::Binary,
        threads,
    );
    let patches = bounce::patches(&s.statics, PATCH_LENGTH);
    let patch_boxes: Vec<Aabb> = patches.iter().map(|p| Aabb::of(&[p.a, p.b])).collect();
    let lights: Vec<Segment> = s.lights.iter().map(|l| l.segment).collect();
    let radiance_at = |movers: &[Segment]| {
        let mut occluders = s.statics.clone();
        occluders.extend_from_slice(movers);
        bounce::patch_radiance(&patches, &occluders, &lights)
    };
    let empty = StaticLayer {
        id: [NO_SURFACE; 64],
        depth: [f32::INFINITY; 64],
    };
    let gather =
        |k: usize, layer: &StaticLayer, movers: &[Segment], radiance: &[f32]| -> PerLight {
            pre.cone(k).map_or([0.0; LIGHTS], |c| {
                bounce::binned_gather(&c, layer, &patches, movers, radiance)
            })
        };
    let compose = |cache: &[PerLight], weights: &[f32]| -> Vec<f32> {
        (0..pixels)
            .map(|px| {
                (0..SECTORS)
                    .map(|q| dot(&cache[q * pixels + px], weights))
                    .sum()
            })
            .collect()
    };

    // Build the G-buffers once (static geometry never changes).
    let start = Instant::now();
    let layers: Vec<StaticLayer> = par_map(cones_total, threads, 256, |k| {
        pre.cone(k)
            .map_or(empty, |c| bounce::static_layer(&c, &patches))
    });
    let layer_time = start.elapsed();

    let movers0 = (s.movers)(0);
    let radiance0 = radiance_at(&movers0);
    let mut cache: Vec<PerLight> = par_map(cones_total, threads, 256, |k| {
        gather(k, &layers[k], &movers0, &radiance0)
    });

    // Accuracy of the binned estimator against the exact sweep at frame 0.
    let mut surfaces: Vec<Surface> = patches
        .iter()
        .enumerate()
        .map(|(j, &segment)| Surface {
            segment,
            patch: Some(j),
        })
        .collect();
    surfaces.extend(movers0.iter().map(|&segment| Surface {
        segment,
        patch: None,
    }));
    let exact: Vec<PerLight> = par_map(cones_total, threads, 256, |k| {
        pre.cone(k)
            .map_or([0.0; LIGHTS], |c| bounce::gather(&c, &surfaces, &radiance0))
    });
    let weights0 = s.weights(0);
    let (exact_f, binned_f) = (compose(&exact, &weights0), compose(&cache, &weights0));
    let mean = exact_f.iter().sum::<f32>() / pixels as f32;
    let rmse = (exact_f
        .iter()
        .zip(&binned_f)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / pixels as f64)
        .sqrt()
        / mean as f64;

    let (mut radiance_time, mut full_time, mut incremental_time) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut mover_cones, mut radiance_cones) = (0u64, 0u64);
    let (mut max_diff, mut mismatches) = (0.0f32, 0u64);
    let mut radiance_prev = radiance0;
    for f in 1..s.frames {
        let (movers_prev, movers_now) = ((s.movers)(f - 1), (s.movers)(f));
        let weights = s.weights(f);

        let start = Instant::now();
        let radiance = radiance_at(&movers_now);
        radiance_time += start.elapsed();

        // Conventional baseline: rebuild every cone's nearest-surface layer and gather.
        let start = Instant::now();
        let reference: Vec<PerLight> = par_map(cones_total, threads, 256, |k| {
            pre.cone(k).map_or([0.0; LIGHTS], |c| {
                bounce::binned_gather(
                    &c,
                    &bounce::static_layer(&c, &patches),
                    &patches,
                    &movers_now,
                    &radiance,
                )
            })
        });
        full_time += start.elapsed();

        let start = Instant::now();
        let hulls = moved_hulls(&movers_prev, &movers_now);
        let mut changed = vec![false; patches.len()];
        let mut any_changed = Vec::new();
        for (j, flag) in changed.iter_mut().enumerate() {
            let range = bounce::radiance_index(j, 0, 0)..bounce::radiance_index(j + 1, 0, 0);
            if radiance[range.clone()] != radiance_prev[range] {
                *flag = true;
                any_changed.push(j);
            }
        }
        // Per (sector, tile): does a mover's sweep reach it, or can a changed patch?
        let reach: Vec<(bool, bool)> = par_map(SECTORS * tiles, threads, 32, |k| {
            let mover = hulls
                .iter()
                .any(|h| pre.frustum_touches(k, &h.hull, &h.bbox));
            let patch = !mover
                && any_changed.iter().any(|&j| {
                    pre.frustum_touches(k, &[patches[j].a, patches[j].b], &patch_boxes[j])
                });
            (mover, patch)
        });
        let work: Vec<(usize, bool)> = reach
            .iter()
            .enumerate()
            .filter(|(_, r)| r.0 || r.1)
            .map(|(k, r)| (k, r.0))
            .collect();
        let updates: Vec<Vec<(usize, PerLight)>> = par_map(work.len(), threads, 2, |i| {
            let (k, mover) = work[i];
            let (q, t) = (k / tiles, k % tiles);
            grid.tile_pixels(t)
                .map(|px| q * pixels + px)
                .filter(|&c| {
                    mover
                        || layers[c]
                            .id
                            .iter()
                            .any(|&id| id != NO_SURFACE && changed[id as usize])
                })
                .map(|c| (c, gather(c, &layers[c], &movers_now, &radiance)))
                .collect()
        });
        for (&(_, mover), batch) in work.iter().zip(&updates) {
            *(if mover {
                &mut mover_cones
            } else {
                &mut radiance_cones
            }) += batch.len() as u64;
            for &(c, v) in batch {
                cache[c] = v;
            }
        }
        incremental_time += start.elapsed();

        let (inc, full) = (compose(&cache, &weights), compose(&reference, &weights));
        for (a, b) in inc.iter().zip(&full) {
            let d = (a - b).abs();
            max_diff = max_diff.max(d);
            if d > TOLERANCE {
                mismatches += 1;
            }
        }
        radiance_prev = radiance;
    }

    let frames = s.frames - 1;
    let full_ms = ms(full_time + radiance_time, frames);
    let inc_ms = ms(incremental_time + radiance_time, frames);
    let total = frames as f64 * cones_total as f64;
    let speedup = full_ms / inc_ms;
    let pass = mismatches == 0 && speedup >= 4.0;
    let mut out = String::new();
    writeln!(out, "# Flatland gate 6b results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} receivers, {SECTORS} gather cones each, {TILE}×{TILE} tiles, {} patches, {} frames, \
         {threads} threads for both paths. Times include recomputing patch radiance ({:.2} ms/frame) for both.\n",
        patches.len(),
        s.frames,
        ms(radiance_time, frames)
    )
    .unwrap();
    writeln!(
        out,
        "- G-buffer build (once): {:.1} ms; memory {:.1} MB ({} bytes per cone).\n\
         - Binned estimator vs exact sweep at frame 0: {:.2}% relative RMSE.\n\
         - Full recompute (rebuild every G-buffer and gather): {:.2} ms/frame.\n\
         - Incremental (G-buffer reuse, exact updates): {:.2} ms/frame → {:.1}× faster.\n\
         - Cones updated: {:.1}% for mover sweeps, {:.1}% for radiance changes, {:.1}% untouched.\n\
         - Exactness: max difference from the full recompute {:.2e}; receiver-frames over {TOLERANCE:e}: {mismatches}.",
        layer_time.as_secs_f64() * 1000.0,
        (layers.len() * std::mem::size_of::<StaticLayer>()) as f64 / 1e6,
        std::mem::size_of::<StaticLayer>(),
        rmse * 100.0,
        full_ms,
        inc_ms,
        speedup,
        100.0 * mover_cones as f64 / total,
        100.0 * radiance_cones as f64 / total,
        100.0 * (1.0 - (mover_cones + radiance_cones) as f64 / total),
        max_diff,
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nExact and at least 4× faster → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("gate6b_results.md"), &out)?;
    Ok(())
}
