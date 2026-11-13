//! M14 gate 15: coarser levels for meshes that come without them
//! (DESIGN §3.8 item 2.3).
//!
//! Camera levels so far come only from callers: the rocks' levels are
//! generated from their analytic shapes (M14 gates 1–13; since gate 13 they
//! carry the finest surface's normals at their vertices). An imported mesh
//! has one level. `continuum::simplify::levels` makes coarser levels of any
//! surface by half-edge collapses ordered by quadric error: each keeps a
//! subset of the finest's vertices with their normals, chart and texture
//! coordinates, attribute seams and borders in place, and states its
//! largest distance to the finest, measured both ways from sample points
//! (vertices, edge midpoints, centroids).
//!
//! Test: each rock's finest level (65,536 triangles) treated as an imported
//! mesh. Its camera levels 1 and 2 are replaced by simplified ones of the
//! same triangle counts (4,096 and 16,384), stating their own errors; the
//! lighting level and the finest stay.
//!
//! Scene and renderers are gate 13's (256 unique rocks, sun only, a
//! 2.6-million-triangle streaming budget against every finest level
//! resident, fixed bake budget, the three flights' last views averaged over
//! 4×4 sub-pixel offsets, a first pass discarded).
//!
//! Pass criteria, fixed before the implementation:
//! 1. **Shading against the finest:** the colour difference (relative RMSE)
//!    streamed against resident with simplified camera levels at most 1.25×
//!    that with the analytic levels carrying the finest's normals (gate 13:
//!    1.25–1.48%), at each of the three views.
//! 2. **Stated errors hold:** for every simplified level of every rock, the
//!    largest distance to the finest measured from 64 random points per
//!    triangle both ways at most 1.1× its stated error.
//!
//! Reported: simplification time per rock; the simplified levels' stated
//! errors against the analytic levels'; with the lighting level simplified
//! too (to 1,024 triangles), its streamed image against the analytic
//! resident one.
//!
//! **Changed before the run, after development runs (`M14_DEV=1`, 36
//! rocks, and `simplify_dev`; disclosed):** the stated error was the
//! largest distance found from 7 sample points a triangle. The simplified
//! levels then stated about a third of the analytic levels' errors at the
//! same triangle counts, so the streamer showed them where their triangles
//! were several pixels across, and their colour difference was 2.4–3.4× the
//! analytic levels' (criterion 1 failed); the dense check found up to 1.12×
//! the stated distance (criterion 2 failed). Now a level states a bound on
//! the distance (grids over each triangle plus the farthest any point lies
//! from its grid) and its error is the larger of that and half its mean
//! edge, so it shows where its triangles are about a pixel across. The
//! development mode then passed: 0.372–0.656× and 0.37×; simplification
//! 1.5 s a rock. The criteria are unchanged.
//!
//! `M14_DEV=1`: a development check on 36 rocks; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m14_gate15`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum::mesh::Surface;
use continuum::simplify;
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, Level, MaskConeKernel, StreamOptions, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::rocks::{Detail, Rocks};

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until the cache and streaming are quiet for 16 frames (at most
/// 4,000), as M14 gate 1.
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    for f in 1..=4000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        let cache_quiet = latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0);
        let log = r.stream_log();
        let stream_quiet = log.len() < 16
            || log[log.len() - 16..]
                .iter()
                .all(|s| s.loads == 0 && s.evictions == 0);
        if cache_quiet && stream_quiet && f >= 16 {
            r.finish();
            return f;
        }
    }
    r.finish();
    4000
}

/// 4×4 sub-pixel offsets averaged, one frame each (as M14 gate 1).
fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> Vec<[f32; 3]> {
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        r.frame(camera, target);
        r.finish();
        for (c, s) in color.iter_mut().zip(r.read_colors()) {
            for i in 0..3 {
                c[i] += s[i] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    color
}

/// Each rock's simplified levels: (level 1, level 2, lighting level), each
/// with its stated error.
type Simplified = Vec<[simplify::Level; 3]>;

/// The rocks' levels simplified from their finest (CPU threads).
fn simplify_rocks(rocks: &Rocks) -> Simplified {
    let finest: Vec<&Surface> = rocks.scene.geometry_levels[1..]
        .iter()
        .map(|l| {
            &l.as_ref()
                .expect("levels")
                .finer
                .last()
                .expect("a finest")
                .surface
        })
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = finest.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = finest
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|s| {
                            let mut l = simplify::levels(s, &[16_384, 4_096, 1_024]);
                            let lighting = l.pop().unwrap();
                            let one = l.pop().unwrap();
                            let two = l.pop().unwrap();
                            [one, two, lighting]
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a thread"))
            .collect()
    })
}

/// The rocks with simplified camera levels 1 and 2 (and with `lighting`,
/// a simplified lighting level in the analytic one's chart grid).
fn apply(rocks: &mut Rocks, simplified: &Simplified, lighting: bool) {
    for (g, s) in simplified.iter().enumerate() {
        let levels = rocks.scene.geometry_levels[g + 1].as_mut().expect("levels");
        for (k, l) in s[..2].iter().enumerate() {
            levels.finer[k] = Level {
                surface: l.surface.clone(),
                error: l.error,
                triangles: l.surface.triangles.len() as u32,
            };
        }
        if lighting {
            let analytic = &rocks.scene.geometries[g + 1];
            let surface = Surface {
                charts: analytic.charts.clone(),
                chart_grid: analytic.chart_grid.clone(),
                ..s[2].surface.clone()
            };
            levels.base_error = s[2].error;
            rocks.scene.geometries[g + 1] = surface;
        }
    }
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M14_DEV").is_some();
    let side = if dev { 6 } else { 16 };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say("DEVELOPMENT RUN: 36 rocks; not the gate's numbers".into());
    }
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_sky(Vec3::ZERO);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let detail = Detail {
        finest_normals: true,
        ..Default::default()
    };
    let build = || Rocks::with_detail(side, 6.0, detail);

    // The simplified levels: per-rock time on one thread (a sample of four
    // rocks), then all of them on every thread.
    let rocks = build();
    let started = Instant::now();
    let sample = 4.min(rocks.positions.len());
    for levels in rocks.scene.geometry_levels[1..=sample].iter().flatten() {
        let finest = &levels.finer[2].surface;
        std::hint::black_box(simplify::levels(finest, &[16_384, 4_096, 1_024]));
    }
    let per_rock = started.elapsed().as_secs_f64() * 1e3 / sample as f64;
    let simplified = simplify_rocks(&rocks);
    let analytic: Vec<[f32; 3]> = rocks.scene.geometry_levels[1..]
        .iter()
        .map(|l| {
            let l = l.as_ref().unwrap();
            [l.finer[0].error, l.finer[1].error, l.base_error]
        })
        .collect();
    let n = simplified.len() as f32;
    let mean = |f: &dyn Fn(usize) -> f32| (0..simplified.len()).map(f).sum::<f32>() / n;
    say(format!(
        "simplified in {per_rock:.0} ms a rock (one thread; 65,536 to 16,384, 4,096 and 1,024 triangles); stated errors on average: level 1 {:.4} m (analytic {:.4}), level 2 {:.4} (analytic {:.4}), lighting {:.4} (analytic {:.4}); triangles {:.0}, {:.0}, {:.0}",
        mean(&|g| simplified[g][0].error),
        mean(&|g| analytic[g][0]),
        mean(&|g| simplified[g][1].error),
        mean(&|g| analytic[g][1]),
        mean(&|g| simplified[g][2].error),
        mean(&|g| analytic[g][2]),
        mean(&|g| simplified[g][0].surface.triangles.len() as f32),
        mean(&|g| simplified[g][1].surface.triangles.len() as f32),
        mean(&|g| simplified[g][2].surface.triangles.len() as f32),
    ));

    // 1. Shading against the finest; reported, the lighting level too.
    let mut images: Vec<Vec<Vec<[f32; 3]>>> = Vec::new();
    let mut working: Vec<Vec<u64>> = Vec::new();
    // A first pass, discarded (as gate 13).
    let passes: &[(bool, bool)] = &[(false, false), (false, false), (true, false), (true, true)];
    for &(simple, lighting) in passes {
        let mut rocks = build();
        if simple {
            apply(&mut rocks, &simplified, lighting);
        }
        let half = rocks.half as f32;
        let make = |options: StreamOptions| {
            InstancedRenderer::with_streaming(
                &kernel,
                format,
                W,
                H,
                1,
                &[rocks.sun],
                &rocks.scene,
                DVec3::ZERO,
                Some(CacheOptions {
                    frame_target_ms: 0.0,
                    ..Default::default()
                }),
                options,
            )
        };
        let mut streamed = make(StreamOptions {
            budget_triangles: 5 << 19,
            ..Default::default()
        });
        let mut resident = make(StreamOptions {
            resident: true,
            ..Default::default()
        });
        let mut set = Vec::new();
        let mut triangles = Vec::new();
        for k in 0..3 {
            let last = room3d::rocks::flight(k, 1.0, half, 50.0);
            settle(&mut resident, last, &target);
            set.push(supersampled(&mut resident, last, &target));
            settle(&mut streamed, last, &target);
            set.push(supersampled(&mut streamed, last, &target));
            triangles.push(
                streamed
                    .stream_log()
                    .last()
                    .map_or(0, |s| s.resident_triangles),
            );
        }
        images.push(set);
        working.push(triangles);
    }
    images.remove(0);
    working.remove(0);
    let mut c1 = true;
    for k in 0..3 {
        let (a_res, a_str) = (&images[0][2 * k], &images[0][2 * k + 1]);
        let (s_res, s_str) = (&images[1][2 * k], &images[1][2 * k + 1]);
        let l_str = &images[2][2 * k + 1];
        let analytic = relative_rmse(a_str, a_res);
        let ours = relative_rmse(s_str, s_res);
        let ok = ours <= 1.25 * analytic;
        c1 &= ok;
        say(format!(
            "1. view {k}: streamed against resident {:.3}% with simplified levels, {:.3}% with the analytic ones ({:.3}×; at most 1.25×): {}; resident images {:.6}% apart; with the lighting level simplified too, {:.3}% from the analytic resident image; streamed working set {} triangles simplified, {} analytic",
            100.0 * ours,
            100.0 * analytic,
            ours / analytic,
            if ok { "PASS" } else { "FAIL" },
            100.0 * relative_rmse(s_res, a_res),
            100.0 * relative_rmse(l_str, a_res),
            working[1][k],
            working[0][k]
        ));
    }
    say(format!(
        "1. shading against the finest: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));

    // 2. The stated errors, measured densely (64 random points a triangle,
    // both ways), every level of every rock, on every thread.
    let finest: Vec<&Surface> = rocks.scene.geometry_levels[1..]
        .iter()
        .map(|l| &l.as_ref().unwrap().finer[2].surface)
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = finest.len().div_ceil(threads).max(1);
    let worst: Vec<(usize, usize, f32, f32)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..finest.len())
            .collect::<Vec<_>>()
            .chunks(chunk)
            .map(|part| {
                let part = part.to_vec();
                let (finest, simplified) = (&finest, &simplified);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for g in part {
                        for (k, l) in simplified[g].iter().enumerate() {
                            let d = simplify::sampled_distance(
                                &l.surface.triangles,
                                &finest[g].triangles,
                                64,
                                (g * 3 + k) as u64,
                            );
                            out.push((g, k, d, l.error));
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a thread"))
            .collect()
    });
    let levels_only: Vec<&(usize, usize, f32, f32)> = worst.iter().filter(|w| w.1 < 2).collect();
    let ratio = levels_only
        .iter()
        .map(|w| w.2 / w.3.max(1e-12))
        .fold(0.0f32, f32::max);
    let c2 = ratio <= 1.1;
    let lighting_ratio = worst
        .iter()
        .filter(|w| w.1 == 2)
        .map(|w| w.2 / w.3.max(1e-12))
        .fold(0.0f32, f32::max);
    say(format!(
        "2. stated errors: dense distance at most {ratio:.4}× the stated error over {} levels (at most 1.1×): {}; lighting levels at most {lighting_ratio:.4}× (reported)",
        levels_only.len(),
        if c2 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m14_gate15_dev.txt"
        } else {
            "m14_gate15.txt"
        }),
        report,
    )?;
    Ok(())
}
