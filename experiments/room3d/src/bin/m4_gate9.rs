//! M4 gate 9: the lighting cache at speed.
//!
//! M4 gate 8's cache cut warm shading to 2–3 ms, but frames stayed at 20–43
//! ms. Its diagnostics found three costs outside the idea itself:
//! - wgpu-core spent 15–17 ms of CPU per `queue.submit` walking one TLAS
//!   dependency per instance (a million). A patched copy keeps one per
//!   distinct BLAS (`third_party/`). Measured before this gate was written:
//!   hardware visibility at 720p then takes 2.2–3.2 ms.
//! - Each frame waited for the GPU, then planned the next bake on the CPU
//!   (up to 30 ms in busy frames), so CPU and GPU never overlapped.
//! - Baking 20,000 cones took 12–23 ms of GPU time.
//!
//! Changes (all engineering; the cache's design is unchanged):
//! - the wgpu-core patch;
//! - pipelined requests: frame N plans its bake from frame N−2's requests,
//!   read back asynchronously, so the CPU plans while the GPU renders; slot-use
//!   readbacks for eviction are asynchronous too;
//! - a faster page manager: a cheap hash for its maps, and walks to coarser
//!   pages that stop at a page another request already listed;
//! - a bake budget of 8,000 cones per frame (gate 8 measured 2,500–40,000
//!   cones at 3.5–26.5 ms; fallback in gate 8's flight was 1.8% at 20,000).
//!
//! Scene: gate 8's (1,000,000 boulders, 1280×720, prefiltered cones, base
//! texel 5 mm, 2 px per texel, 4×4 request tiles, pool of 2^19 pages), the
//! faster of compute or hardware camera rays (lower total of warm times).
//!
//! Pass criterion, fixed before the first run:
//! 1. Same images: warm, each viewpoint's relative RMSE against per-pixel
//!    exact cones is within 0.2 points of gate 8's (14.49, 25.87, 30.88 and
//!    38.80% with hardware rays; 14.49, 25.86, 30.88, 38.80% with compute).
//! 2. Speed: warm, each viewpoint (standing; 10, 40, 150 m up) takes at most
//!    16.7 ms per frame: wall time of 60 consecutive frames, ending with a
//!    wait for the GPU, divided by 60.
//! 3. Flight: gate 8's 600-frame flight from an empty cache. The mean frame
//!    (wall time of all 600 frames ending with a GPU wait, divided by 600) is
//!    at most 16.7 ms; the 99th percentile of per-frame wall time (one call
//!    of `frame`) is at most 33.3 ms; and on average at most 10% of pixels
//!    fall back to unshadowed light.
//!
//! Warm: as gate 8 (16 consecutive frames with no request and nothing
//! baked). Reported: frames until warm, worst frame, memory, CPU planning
//! time, and both the cached and the per-pixel exact image against a 4×4
//! supersampled exact render (the area-integrated image).

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
    // Diagnostics only: the flight alone, with hardware rays, timing each
    // part of the CPU's work (no verdict, no results file).
    let flight_only = args.iter().any(|a| a == "--flight-only");
    let official = count == 1_000_000 && !flight_only;
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

    if flight_only {
        let mut r = cached(true);
        let mut rows =
            String::from("frame,ms,waiting_ms,planning_ms,evicting_ms,texels_baked,fallback\n");
        let mut times = Vec::new();
        for f in 0..600 {
            let t = f as f32 / 599.0;
            let camera = view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t);
            let call = Instant::now();
            r.frame(camera, &target);
            times.push(call.elapsed());
        }
        r.finish();
        let log = r.cache_log();
        for (f, (t, s)) in times.iter().zip(log).enumerate() {
            rows += &format!(
                "{f},{:.3},{:.3},{:.3},{:.3},{},{:.5}\n",
                ms(*t),
                ms(s.waiting),
                ms(s.planning),
                ms(s.evicting),
                s.texels_baked,
                s.fallback_pixels as f64 / (W * H) as f64
            );
        }
        std::fs::write(out_dir.join("m4_gate9_flight_diagnostic.csv"), rows)?;
        let mean = |f: &dyn Fn(&continuum_gpu::CacheStats) -> Duration| {
            log.iter().map(|s| ms(f(s))).sum::<f64>() / log.len() as f64
        };
        println!(
            "flight: mean call {:.2} ms; waiting {:.2} ms, planning {:.2} ms (evicting {:.2} ms)",
            times.iter().map(|t| ms(*t)).sum::<f64>() / 600.0,
            mean(&|s| s.waiting),
            mean(&|s| s.planning),
            mean(&|s| s.evicting)
        );
        return Ok(());
    }
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
            save_png(&out_dir.join(format!("m4_gate9_{tag}_cached.png")), &image)?;
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

    // The flight, from an empty cache.
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
    let flight_bytes = r.cache_bytes();
    drop(r);
    let mean_frame = total / 600;
    let mut sorted = times.clone();
    sorted.sort();
    let p99 = sorted[(sorted.len() * 99) / 100 - 1];
    let worst = *sorted.last().unwrap();
    let worst_at = times.iter().position(|&t| t == worst).unwrap();
    let fallback: Vec<f64> = log
        .iter()
        .map(|s| s.fallback_pixels as f64 / (W * H) as f64)
        .collect();
    let mean_fallback = fallback.iter().sum::<f64>() / fallback.len() as f64;
    let worst_fallback = fallback.iter().copied().fold(0.0, f64::max);
    let planning: Vec<f64> = log.iter().map(|s| ms(s.planning)).collect();
    let mean_planning = planning.iter().sum::<f64>() / planning.len() as f64;
    let max_planning = planning.iter().copied().fold(0.0, f64::max);
    let baked: u64 = log.iter().map(|s| s.texels_baked as u64).sum();

    let same = rows.iter().all(|r| (r.rmse - r.gate8).abs() <= 0.002)
        && results
            .iter()
            .all(|(_, rows)| rows.iter().all(|r| (r.rmse - r.gate8).abs() <= 0.002));
    let warm_all = results
        .iter()
        .all(|(_, rows)| rows.iter().all(|r| r.frames_until_warm.is_some()));
    let speed = warm_all && rows.iter().all(|r| ms(r.frame_time) <= 16.7);
    let flight = ms(mean_frame) <= 16.7 && ms(p99) <= 33.3 && mean_fallback <= 0.10;
    let pass = same && speed && flight;

    let mut out = String::new();
    writeln!(out, "# M4 gate 9 results\n").unwrap();
    writeln!(
        out,
        "{}, {} boulders, {W}×{H}, prefiltered cones, base texel {} m, {} px per texel, budget {} cones per frame, pool {} pages.\n",
        kernel.adapter_name,
        field.boulders(),
        options.base_texel,
        options.pixels_per_texel,
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
        "\nFlight ({} rays, 600 frames from an empty cache): mean {:.2} ms per frame (wall time / 600), 99th percentile {:.2} ms, worst {:.2} ms (frame {worst_at}); fallback pixels mean {:.2}%, worst frame {:.2}%; planning on the CPU mean {:.2} ms, max {:.2} ms; {} texels baked; {:.0} MB of cache.",
        if hardware { "hardware" } else { "compute" },
        ms(mean_frame),
        ms(p99),
        ms(worst),
        100.0 * mean_fallback,
        100.0 * worst_fallback,
        mean_planning,
        max_planning,
        baked,
        flight_bytes as f64 / 1e6
    )
    .unwrap();
    let flight_csv: String = times
        .iter()
        .zip(&log)
        .enumerate()
        .map(|(f, (t, s))| {
            format!(
                "{f},{:.3},{:.5},{},{:.3}\n",
                ms(*t),
                s.fallback_pixels as f64 / (W * H) as f64,
                s.texels_baked,
                ms(s.planning)
            )
        })
        .collect();
    std::fs::write(
        out_dir.join("m4_gate9_flight.csv"),
        format!("frame,ms,fallback,texels_baked,planning_ms\n{flight_csv}"),
    )?;
    writeln!(
        out,
        "\n## Verdict\n\n1. Same images (within 0.2 points of gate 8): {}\n2. Speed (warm, at most 16.7 ms per frame): {}\n3. Flight (mean at most 16.7 ms, 99th percentile at most 33.3 ms, fallback at most 10%): {}\n\n→ {}",
        if same { "pass" } else { "fail" },
        if speed { "pass" } else { "fail" },
        if flight { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    if official {
        std::fs::write(out_dir.join("m4_gate9_results.md"), &out)?;
    }
    Ok(())
}
