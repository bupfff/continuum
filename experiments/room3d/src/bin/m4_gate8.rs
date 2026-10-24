//! M4 gate 8: a view-dependent lighting cache (real-time at scale).
//!
//! M4 gate 7 showed per-pixel cones every frame cannot run a million-object
//! field in real time. Static lighting only needs computing once per visible
//! patch of surface. The cache keeps pages of 8×8 texels (plus a one-texel
//! border) on each object's charts, at power-of-two resolutions chosen per
//! pixel (a texel about two pixels across). The shading pass reports the pages
//! it needed; a CPU page manager allocates them (coarse first) and evicts pages
//! unseen for a while; new pages are baked with mask cones within a per-frame
//! budget; pixels without their page use a coarser resident page or,
//! failing that, unshadowed light.
//!
//! Scene: the 1,000,000-boulder field at 1280×720, prefiltered cones, the
//! faster of compute or hardware camera rays.
//!
//! Pass criterion, fixed before the first run:
//! 1. Quality: warm (camera still until no page is missing), each of gate 6's
//!    four viewpoints (standing; 10, 40, 150 m up) is within 3% relative RMSE
//!    of per-pixel exact cones with the same rays.
//! 2. Speed: warm, each viewpoint renders in at most 16.7 ms (visibility,
//!    cached shading and the request readback; nothing left to bake).
//! 3. Flight: along a 600-frame scripted flight from standing height up to 150
//!    m, with a bake budget per frame, the mean frame takes at most 16.7 ms
//!    and on average at most 10% of pixels per frame fall back to unshadowed
//!    light.
//!
//! Reported: memory, the worst frame, frames until warm.
//!
//! Details fixed before the first run (added with the cache's first commit,
//! before any measurement):
//! - Cache settings: base texel 5 mm (texels at mip m are 5 mm × 2^m × the
//!   object's scale), mip chosen so a texel spans about 2 pixels (log2
//!   rounded), one request per 4×4-pixel tile per frame, bake budget 20,000
//!   cones per frame (all phases), pool of 2^19 pages.
//! - Warm: a viewpoint is warm once 16 consecutive frames (every pixel of every
//!   tile sampled once) report no missing page. Warm-up uses the same budget;
//!   frames until warm are counted from an empty cache.
//! - Speed statistic: the median of 30 warm frames, wall clock from the
//!   frame's start until its request readback is mapped (the fastest is also
//!   reported).
//! - Reference: per-pixel exact cones (prefilter 0) through `render_instanced`
//!   with the same ray mode as the cached render; relative RMSE = RMSE / mean
//!   of the reference over all pixels and channels.
//! - Flight: frame f of 600 at t = f / 599 has the eye 300·t m ahead of the
//!   standing eye along the view's flat direction, at height 1.7 + 148.3·t m,
//!   looking 45·t degrees down, from an empty cache. A pixel falls back when
//!   no page of any mip is resident for it.

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

fn save_png(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
}

struct Warm {
    name: &'static str,
    frames_until_warm: usize,
    median: Duration,
    fastest: Duration,
    rmse: f64,
    /// Per-pixel prefiltered cones against exact: the prefilter's share.
    prefilter_rmse: f64,
    resident: u32,
    cache_bytes: u64,
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // A smaller field for development runs; the gate is 1,000,000.
    let count = args
        .iter()
        .position(|a| a == "--boulders")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(1_000_000usize);
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
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel.device().create_texture(&wgpu::TextureDescriptor {
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
    });
    let target = target.create_view(&Default::default());
    // Development diagnostics only: the gate runs with the defaults.
    let ppt = args
        .iter()
        .position(|a| a == "--pixels-per-texel")
        .and_then(|i| args.get(i + 1)?.parse().ok());
    let options = CacheOptions {
        pixels_per_texel: ppt.unwrap_or(CacheOptions::default().pixels_per_texel),
        ..Default::default()
    };
    let official = count == 1_000_000 && ppt.is_none();
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

    let modes: Vec<bool> = if kernel.has_hardware_rays() {
        vec![false, true]
    } else {
        vec![false]
    };
    let mut results: Vec<(bool, Vec<Warm>)> = Vec::new();
    for &hardware in &modes {
        let mut rows = Vec::new();
        for (name, camera) in views {
            let mut r = cached(hardware);
            let (mut frames, mut quiet) = (0usize, 0usize);
            while quiet < 16 && frames < 5000 {
                r.frame(camera, &target);
                frames += 1;
                let s = r.cache_stats().unwrap();
                quiet = if s.requests == 0 && s.pages_baked == 0 {
                    quiet + 1
                } else {
                    0
                };
            }
            let warm = quiet == 16;
            let mut times = Vec::new();
            for _ in 0..30 {
                let start = Instant::now();
                r.frame(camera, &target);
                times.push(start.elapsed());
                let s = r.cache_stats().unwrap();
                assert!(s.pages_baked == 0, "baked while warm");
            }
            let image = r.read_colors();
            let stats = r.cache_stats().unwrap();
            let cache_bytes = r.cache_bytes();
            drop(r);
            kernel.set_prefilter(0.0);
            let (reference, _, _) = kernel.render_instanced(
                camera,
                W,
                H,
                light,
                &field.scene,
                origin,
                field.radiance,
                1,
            );
            let rmse = relative_rmse(&image, &reference);
            kernel.set_prefilter(1.0);
            let (prefiltered, _, _) = kernel.render_instanced(
                camera,
                W,
                H,
                light,
                &field.scene,
                origin,
                field.radiance,
                1,
            );
            let prefilter_rmse = relative_rmse(&prefiltered, &reference);
            let tag = format!(
                "{}_{}",
                if hardware { "hw" } else { "compute" },
                name.replace(' ', "_")
            );
            save_png(&out_dir.join(format!("m4_gate8_{tag}_cached.png")), &image)?;
            save_png(
                &out_dir.join(format!("m4_gate8_{tag}_exact.png")),
                &reference,
            )?;
            let fastest = *times.iter().min().unwrap();
            let row = Warm {
                name,
                frames_until_warm: if warm { frames - 16 } else { usize::MAX },
                median: median(&mut times),
                fastest,
                rmse,
                prefilter_rmse,
                resident: stats.resident,
                cache_bytes,
            };
            println!(
                "{} {name}: warm after {} frames, median {:.2} ms (fastest {:.2}), RMSE {:.2}% (per-pixel prefiltered {:.2}%), {} pages, {:.0} MB",
                if hardware { "hardware" } else { "compute" },
                row.frames_until_warm,
                ms(row.median),
                ms(row.fastest),
                100.0 * rmse,
                100.0 * prefilter_rmse,
                row.resident,
                cache_bytes as f64 / 1e6
            );
            rows.push(row);
        }
        results.push((hardware, rows));
    }
    // The faster ray mode: lower total of the warm medians.
    let (hardware, rows) = results
        .iter()
        .min_by_key(|(_, rows)| rows.iter().map(|r| r.median).sum::<Duration>())
        .unwrap();
    let hardware = *hardware;

    // The flight, from an empty cache.
    let mut r = cached(hardware);
    let mut times = Vec::new();
    let mut fallback = Vec::new();
    for f in 0..600 {
        let t = f as f32 / 599.0;
        let camera = view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t);
        let start = Instant::now();
        r.frame(camera, &target);
        times.push(start.elapsed());
        let s = r.cache_stats().unwrap();
        fallback.push(s.fallback_pixels as f64 / (W * H) as f64);
    }
    let flight_bytes = r.cache_bytes();
    let flight_resident = r.cache_stats().unwrap().resident;
    drop(r);
    let mean_frame = times.iter().sum::<Duration>() / times.len() as u32;
    let worst = times.iter().copied().max().unwrap();
    let worst_at = times.iter().position(|&t| t == worst).unwrap();
    let mean_fallback = fallback.iter().sum::<f64>() / fallback.len() as f64;
    let worst_fallback = fallback.iter().copied().fold(0.0, f64::max);
    let mut sorted = times.clone();
    let flight_median = median(&mut sorted);
    let first_half = times[..300].iter().sum::<Duration>() / 300;
    let second_half = times[300..].iter().sum::<Duration>() / 300;

    let quality = rows
        .iter()
        .all(|r| r.rmse <= 0.03 && r.frames_until_warm != usize::MAX);
    let speed = rows.iter().all(|r| ms(r.median) <= 16.7);
    let flight = ms(mean_frame) <= 16.7 && mean_fallback <= 0.10;
    let pass = quality && speed && flight;

    let mut out = String::new();
    writeln!(out, "# M4 gate 8 results\n").unwrap();
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
        "| Rays | Viewpoint | Frames until warm | Warm median | Fastest | Rel. RMSE vs exact | Per-pixel prefiltered vs exact | Pages | Cache memory |\n|---|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for (hw, rows) in &results {
        for r in rows {
            writeln!(
                out,
                "| {} | {} | {} | {:.2} ms | {:.2} ms | {:.2}% | {:.2}% | {} | {:.0} MB |",
                if *hw { "hardware" } else { "compute" },
                r.name,
                if r.frames_until_warm == usize::MAX {
                    "not warm in 5000".to_string()
                } else {
                    r.frames_until_warm.to_string()
                },
                ms(r.median),
                ms(r.fastest),
                100.0 * r.rmse,
                100.0 * r.prefilter_rmse,
                r.resident,
                r.cache_bytes as f64 / 1e6
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "\nFlight ({} rays, 600 frames from an empty cache): mean {:.2} ms, median {:.2} ms, worst {:.2} ms (frame {worst_at}); first half {:.2} ms, second half {:.2} ms; fallback pixels mean {:.2}%, worst frame {:.2}%; {} pages resident at the end ({:.0} MB).",
        if hardware { "hardware" } else { "compute" },
        ms(mean_frame),
        ms(flight_median),
        ms(worst),
        ms(first_half),
        ms(second_half),
        100.0 * mean_fallback,
        100.0 * worst_fallback,
        flight_resident,
        flight_bytes as f64 / 1e6
    )
    .unwrap();
    let flight_csv: String = times
        .iter()
        .zip(&fallback)
        .enumerate()
        .map(|(f, (t, fb))| format!("{f},{:.3},{:.5}\n", ms(*t), fb))
        .collect();
    std::fs::write(
        out_dir.join("m4_gate8_flight.csv"),
        format!("frame,ms,fallback\n{flight_csv}"),
    )?;
    writeln!(
        out,
        "\n## Verdict\n\n1. Quality (warm, every viewpoint within 3%): {}\n2. Speed (warm median at most 16.7 ms): {}\n3. Flight (mean at most 16.7 ms, fallback at most 10%): {}\n\n→ {}",
        if quality { "pass" } else { "fail" },
        if speed { "pass" } else { "fail" },
        if flight { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    if official {
        std::fs::write(out_dir.join("m4_gate8_results.md"), &out)?;
    }
    Ok(())
}
