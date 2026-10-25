//! M4 gate 11: baking to a frame-time target.
//!
//! M4 gate 10 took the page manager's walk off the CPU (planning 1.9 ms per
//! frame), which left its flight bound by two things:
//! - GPU work per frame. Rendering plus a fixed 8,000 cones took 11.5–18.6 ms
//!   per 100-frame stretch on a hot machine. The cost per cone varies with
//!   what cones see and with the machine's heat, so a fixed count cannot
//!   hold a frame time.
//! - Eviction rounds. Every 32 frames, up to 32,768 pages were freed at once
//!   (12–26 ms of CPU), which made 12 of the 17 frames over 33 ms in flight 1.
//!
//! Changes (pages, mips, requests, baking and shading are unchanged):
//! - The bake budget follows the frame time: the frame budget of the quality
//!   contract (DESIGN §3.3). Each frame measures its period P (between
//!   successive frames) and its wait for the GPU.
//!   - If the CPU waited more than 1 ms (GPU-bound) and P exceeds the target
//!     T = 14 ms, the budget scales by T / P.
//!   - If P is under T, it grows by 10%.
//!   - Otherwise it holds.
//!   - Bounds: 1,000 to 20,000 cones; start: 8,000.
//! - Eviction is spread over frames: when slot use arrives (every 32 frames),
//!   pages unused for 64 frames are queued, least recently used first, and
//!   at most 2,048 are freed per frame.
//!
//! Scene and settings: gate 10's (1,000,000 boulders, 1280×720, prefiltered
//! cones, base texel 5 mm, 2 px per texel, 4×4 request tiles, pool of 2^19
//! pages), with the faster of compute or hardware camera rays.
//!
//! Pass criterion, fixed before the first run (as gate 10's):
//! 1. Same images: warm, each viewpoint's relative RMSE against per-pixel
//!    exact cones is within 0.2 points of gate 8's (14.49, 25.87, 30.88,
//!    38.80% with hardware rays; 14.49, 25.86, 30.88, 38.80% with compute).
//! 2. Speed: warm, each viewpoint takes at most 16.7 ms per frame (wall time
//!    of 60 frames ending with a GPU wait, divided by 60).
//! 3. Flight: gate 8's 600-frame flight from an empty cache, run three times
//!    in a row; in every run, the mean frame is at most 16.7 ms, the 99th
//!    percentile frame at most 33.3 ms, and the mean fallback at most 10%.
//!
//! Reported: frames until warm, the worst frame, CPU planning and GPU wait
//! per frame, the bake budget over the flight, and the fraction of pixel
//! samples in flight that found their wanted mip.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::to_rgb;

const W: u32 = 1280;
const H: u32 = 720;
const SS: u32 = 4;

fn save_png(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
}

struct Warm {
    name: &'static str,
    frames_until_warm: Option<usize>,
    frame_time: Duration,
    rmse: f64,
    gate8: f64,
    cached_vs_truth: f64,
    exact_vs_truth: f64,
    resident: u32,
    cache_bytes: u64,
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // A smaller field for development runs; the gate is 1,000,000.
    let count = args
        .iter()
        .position(|a| a == "--boulders")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(1_000_000usize);
    let official = count == 1_000_000;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let field = Field::new(count, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |ahead: f32, height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(0.0, 1.7, 0.0)),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ];
    // Gate 8's relative RMSE against per-pixel exact cones (hardware, compute).
    let gate8 = [
        (0.1449, 0.1449),
        (0.2587, 0.2586),
        (0.3088, 0.3088),
        (0.3880, 0.3880),
    ];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
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
    let options = CacheOptions::default();
    let cached = |hardware: bool| {
        kernel.set_hardware_rays(hardware);
        kernel.set_prefilter(1.0);
        InstancedRenderer::with_cache(
            &kernel,
            format,
            W,
            H,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            options,
        )
    };
    let per_pixel = |camera: Camera, width: u32, height: u32| {
        kernel.set_prefilter(0.0);
        kernel
            .render_instanced(
                camera,
                width,
                height,
                light,
                &field.scene,
                origin,
                field.radiance,
                1,
            )
            .0
    };

    let modes: Vec<bool> = if kernel.has_hardware_rays() {
        vec![false, true]
    } else {
        vec![false]
    };
    let mut results: Vec<(bool, Vec<Warm>)> = Vec::new();
    for &hardware in &modes {
        let mut rows = Vec::new();
        for (k, (name, camera)) in views.into_iter().enumerate() {
            let mut r = cached(hardware);
            // Warm: 16 consecutive frames (as read back) with no request and
            // nothing baked.
            let mut frames = 0usize;
            let mut warm_from = None;
            while frames < 5000 {
                r.frame(camera, &target);
                frames += 1;
                let log = r.cache_log();
                let latest = r.cache_stats().unwrap().frame as usize;
                if latest >= 16 {
                    let quiet = log[latest - 16..latest]
                        .iter()
                        .all(|s| s.requests == 0 && s.pages_baked == 0);
                    if quiet {
                        warm_from = Some(latest - 16);
                        break;
                    }
                }
            }
            r.finish();
            let start = Instant::now();
            for _ in 0..60 {
                r.frame(camera, &target);
            }
            r.finish();
            let frame_time = start.elapsed() / 60;
            let baked: u32 = r.cache_log()[r.cache_log().len() - 60..]
                .iter()
                .map(|s| s.pages_baked)
                .sum();
            if baked > 0 {
                println!("note: {baked} pages baked during the timed warm frames");
            }
            let image = r.read_colors();
            let stats = r.cache_stats().unwrap();
            let cache_bytes = r.cache_bytes();
            drop(r);
            let exact = per_pixel(camera, W, H);
            let big = per_pixel(camera, W * SS, H * SS);
            let mut truth = vec![[0.0f32; 3]; (W * H) as usize];
            for y in 0..H * SS {
                for x in 0..W * SS {
                    let p = big[(y * W * SS + x) as usize];
                    let t = &mut truth[((y / SS) * W + x / SS) as usize];
                    for c in 0..3 {
                        t[c] += p[c] / (SS * SS) as f32;
                    }
                }
            }
            drop(big);
            let tag = format!(
                "{}_{}",
                if hardware { "hw" } else { "compute" },
                name.replace(' ', "_")
            );
            save_png(&out_dir.join(format!("m4_gate11_{tag}_cached.png")), &image)?;
            let row = Warm {
                name,
                frames_until_warm: warm_from,
                frame_time,
                rmse: relative_rmse(&image, &exact),
                gate8: if hardware { gate8[k].0 } else { gate8[k].1 },
                cached_vs_truth: relative_rmse(&image, &truth),
                exact_vs_truth: relative_rmse(&exact, &truth),
                resident: stats.resident,
                cache_bytes,
            };
            println!(
                "{} {name}: warm after {:?} frames, {:.2} ms per frame, RMSE {:.2}% (gate 8 {:.2}%), vs supersampled: cached {:.2}%, per-pixel {:.2}%, {} pages, {:.0} MB",
                if hardware { "hardware" } else { "compute" },
                row.frames_until_warm,
                ms(row.frame_time),
                100.0 * row.rmse,
                100.0 * row.gate8,
                100.0 * row.cached_vs_truth,
                100.0 * row.exact_vs_truth,
                row.resident,
                cache_bytes as f64 / 1e6
            );
            rows.push(row);
        }
        results.push((hardware, rows));
    }
    // The faster ray mode: lower total of the warm frame times.
    let (hardware, rows) = results
        .iter()
        .min_by_key(|(_, rows)| rows.iter().map(|r| r.frame_time).sum::<Duration>())
        .unwrap();
    let hardware = *hardware;

    // The flight, from an empty cache, three times in a row.
    struct Flight {
        mean: Duration,
        p99: Duration,
        worst: Duration,
        worst_at: usize,
        fallback: f64,
        worst_fallback: f64,
        planning: f64,
        max_planning: f64,
        waiting: f64,
        at_wanted: f64,
        budget: (f64, u32, u32),
        bytes: u64,
        csv: String,
    }
    let mut flights = Vec::new();
    for _ in 0..3 {
        let mut r = cached(hardware);
        let mut times = Vec::new();
        let start = Instant::now();
        for f in 0..600 {
            let t = f as f32 / 599.0;
            let camera = view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t);
            let call = Instant::now();
            r.frame(camera, &target);
            times.push(call.elapsed());
        }
        r.finish();
        let total = start.elapsed();
        let log = r.cache_log().to_vec();
        let bytes = r.cache_bytes();
        drop(r);
        let mut sorted = times.clone();
        sorted.sort();
        let worst = *sorted.last().unwrap();
        let fallback: Vec<f64> = log
            .iter()
            .map(|s| s.fallback_pixels as f64 / (W * H) as f64)
            .collect();
        let mean_of = |f: &dyn Fn(&continuum_gpu::CacheStats) -> f64| {
            log.iter().map(f).sum::<f64>() / log.len() as f64
        };
        let samples: u64 = log.iter().map(|s| s.samples as u64).sum();
        let at_wanted: u64 = log.iter().map(|s| s.at_wanted as u64).sum();
        let csv: String = times
            .iter()
            .zip(&log)
            .enumerate()
            .map(|(f, (t, s))| {
                format!(
                    "{f},{:.3},{:.5},{},{},{},{:.3},{:.3}\n",
                    ms(*t),
                    s.fallback_pixels as f64 / (W * H) as f64,
                    s.requests,
                    s.budget,
                    s.texels_baked,
                    ms(s.planning),
                    ms(s.waiting)
                )
            })
            .collect();
        flights.push(Flight {
            mean: total / 600,
            p99: sorted[(sorted.len() * 99) / 100 - 1],
            worst,
            worst_at: times.iter().position(|&t| t == worst).unwrap(),
            fallback: fallback.iter().sum::<f64>() / fallback.len() as f64,
            worst_fallback: fallback.iter().copied().fold(0.0, f64::max),
            planning: mean_of(&|s| ms(s.planning)),
            max_planning: log.iter().map(|s| ms(s.planning)).fold(0.0, f64::max),
            waiting: mean_of(&|s| ms(s.waiting)),
            at_wanted: at_wanted as f64 / samples.max(1) as f64,
            budget: (
                mean_of(&|s| s.budget as f64),
                log.iter().map(|s| s.budget).min().unwrap_or(0),
                log.iter().map(|s| s.budget).max().unwrap_or(0),
            ),
            bytes,
            csv,
        });
        let f = flights.last().unwrap();
        println!(
            "flight {}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, planning {:.2} ms (max {:.2}), waiting {:.2} ms, at wanted mip {:.1}%, budget mean {:.0} ({}-{})",
            flights.len(),
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            100.0 * f.fallback,
            f.planning,
            f.max_planning,
            f.waiting,
            100.0 * f.at_wanted,
            f.budget.0,
            f.budget.1,
            f.budget.2
        );
    }

    let same = rows.iter().all(|r| (r.rmse - r.gate8).abs() <= 0.002)
        && results
            .iter()
            .all(|(_, rows)| rows.iter().all(|r| (r.rmse - r.gate8).abs() <= 0.002));
    let warm_all = results
        .iter()
        .all(|(_, rows)| rows.iter().all(|r| r.frames_until_warm.is_some()));
    let speed = warm_all && rows.iter().all(|r| ms(r.frame_time) <= 16.7);
    let flight = flights
        .iter()
        .all(|f| ms(f.mean) <= 16.7 && ms(f.p99) <= 33.3 && f.fallback <= 0.10);
    let pass = same && speed && flight;

    let mut out = String::new();
    writeln!(out, "# M4 gate 11 results\n").unwrap();
    writeln!(
        out,
        "{}, {} boulders, {W}×{H}, prefiltered cones, base texel {} m, {} px per texel, bake budget adaptive (frame target {} ms, at most {} cones), pool {} pages.\n",
        kernel.adapter_name,
        field.boulders(),
        options.base_texel,
        options.pixels_per_texel,
        options.frame_target_ms,
        options.bake_budget,
        options.pages
    )
    .unwrap();
    writeln!(
        out,
        "| Rays | Viewpoint | Frames until warm | Warm frame | RMSE vs per-pixel exact (gate 8) | Cached vs supersampled | Per-pixel exact vs supersampled | Pages | Cache memory |\n|---|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for (hw, rows) in &results {
        for r in rows {
            writeln!(
                out,
                "| {} | {} | {} | {:.2} ms | {:.2}% ({:.2}%) | {:.2}% | {:.2}% | {} | {:.0} MB |",
                if *hw { "hardware" } else { "compute" },
                r.name,
                r.frames_until_warm
                    .map_or("not warm in 5000".to_string(), |f| f.to_string()),
                ms(r.frame_time),
                100.0 * r.rmse,
                100.0 * r.gate8,
                100.0 * r.cached_vs_truth,
                100.0 * r.exact_vs_truth,
                r.resident,
                r.cache_bytes as f64 / 1e6
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "\nFlights ({} rays, 600 frames from an empty cache, three runs):\n\n| Run | Mean frame | 99th percentile | Worst (frame) | Fallback mean | Worst fallback | Planning mean (max) | GPU wait mean | Samples at wanted mip | Bake budget mean (range) | Cache |\n|---|---|---|---|---|---|---|---|---|---|---|",
        if hardware { "hardware" } else { "compute" }
    )
    .unwrap();
    for (i, f) in flights.iter().enumerate() {
        writeln!(
            out,
            "| {} | {:.2} ms | {:.2} ms | {:.2} ms ({}) | {:.2}% | {:.2}% | {:.2} ms ({:.2}) | {:.2} ms | {:.1}% | {:.0} ({}–{}) | {:.0} MB |",
            i + 1,
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            f.worst_at,
            100.0 * f.fallback,
            100.0 * f.worst_fallback,
            f.planning,
            f.max_planning,
            f.waiting,
            100.0 * f.at_wanted,
            f.budget.0,
            f.budget.1,
            f.budget.2,
            f.bytes as f64 / 1e6
        )
        .unwrap();
        std::fs::write(
            out_dir.join(format!("m4_gate11_flight{}.csv", i + 1)),
            format!(
                "frame,ms,fallback,requests,budget,texels_baked,planning_ms,waiting_ms\n{}",
                f.csv
            ),
        )?;
    }
    writeln!(
        out,
        "\n## Verdict\n\n1. Same images (within 0.2 points of gate 8): {}\n2. Speed (warm, at most 16.7 ms per frame): {}\n3. Flight (in each of three runs: mean at most 16.7 ms, 99th percentile at most 33.3 ms, fallback at most 10%): {}\n\n→ {}",
        if same { "pass" } else { "fail" },
        if speed { "pass" } else { "fail" },
        if flight { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    if official {
        std::fs::write(out_dir.join("m4_gate11_results.md"), &out)?;
    }
    Ok(())
}
