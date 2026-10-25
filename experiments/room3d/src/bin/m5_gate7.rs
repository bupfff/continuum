//! M5 gate 7: lights cached as ratios, shaded with each pixel's normal.
//!
//! What was built since gate 6:
//! - A cache texel's light channel holds the light's shadowed share of its
//!   unshadowed light at the texel (`CacheOptions::light_ratio`, on by
//!   default), and shading multiplies the interpolated share by the
//!   unshadowed light at the pixel, from the pixel's own position and
//!   normal: a ratio estimator (Heitz et al. 2018 combine analytic
//!   unshadowed light with a stochastic shadow ratio). Before, texels held
//!   irradiance, so the cosine at each texel's normal was interpolated with
//!   the shadow, and normal detail finer than a texel was lost.
//! - Where the light is below a texel's horizon, the texel's share is one
//!   (its own light is zero either way), so pixels nearby that face the
//!   light are not darkened, as on a convex object's terminator.
//! - Light cones whose window crosses the receiver's horizon now clip the
//!   window and each covered bin to it, as sky faces do. Before, every
//!   bin's unclipped weight was summed, so bins below the horizon counted
//!   as light (development check against gate 6's Cycles references: the
//!   three lights 1.22% before and after, the lamp alone 4.69% → 4.72%).
//!
//! Scene: the million-boulder field, gate 8's four viewpoints at 1280×720,
//! hardware rays, prefilter 1 bin with object-level prefiltering, sky
//! prefilter ×8, cache defaults; the sun alone (`Field::relative`) and gate
//! 6's three colored lights (`Field::colored_lights(origin, 1 km)`). "Cache
//! error" is the warm cached image against the per-pixel image with the same
//! lights and prefiltering (1 sample per pixel), relative RMSE.
//!
//! Development runs, disclosed: sun alone at 1280×720, cache error with
//! irradiance texels 14.49 / 25.87 / 30.88 / 38.80% (gate 5's figures) and
//! with ratios 6.21 / 10.63 / 14.78 / 21.75% (standing, 10 m, 40 m, 150 m
//! up); three lights at 640×360, 2.12 / 28.27 / 39.05 / 46.74% against
//! 0.94 / 14.98 / 21.66 / 31.77%. Timings could not be taken (another
//! application was loading the GPU).
//!
//! Control, before anything else: the sun-only warm frame at the standing
//! viewpoint, with ratios, at most 3.3 ms (twice gate 6's 1.65 ms with
//! irradiance texels), or the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. Sun alone: at each viewpoint, the cache error with ratios at most 0.6
//!    times the cache error with irradiance texels (`light_ratio: false`),
//!    measured in the same run.
//! 2. Three lights: at each viewpoint, at most 0.75 times, likewise.
//! 3. Per-pixel accuracy is kept: the three lights per pixel, 16×16 samples
//!    per pixel at 320×180, within 1.30% of gate 6's Cycles reference (4,096
//!    samples per pixel) over surfaces, as gate 6 scored it (1.22% there).
//!    The reference is reused when its inputs are byte-identical, else
//!    rendered again.
//! 4. Speed with the three lights and ratios: warm, each viewpoint at most
//!    16.7 ms per frame; gate 8's flight from an empty cache, three runs,
//!    each with mean at most 16.7 ms, 99th percentile at most 33.3 ms and
//!    fallback at most 10%.
//!
//! Reported, not gated: warm frame times and frames until warm for both
//! schemes, and each light alone against Cycles.
//! `cargo run --release -p room3d --bin m5_gate7`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;

struct Warm {
    frames: Option<usize>,
    frame_time: Duration,
    image: Vec<[f32; 3]>,
}

fn warm(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Warm {
    let mut frames = 0usize;
    let mut warm_from = None;
    while frames < 40_000 {
        r.frame(camera, target);
        frames += 1;
        let latest = r.cache_stats().unwrap().frame as usize;
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            warm_from = Some(latest - 16);
            break;
        }
    }
    r.finish();
    let start = Instant::now();
    for _ in 0..60 {
        r.frame(camera, target);
    }
    r.finish();
    let frame_time = start.elapsed() / 60;
    Warm {
        frames: warm_from,
        frame_time,
        image: r.read_colors(),
    }
}

/// Per pixel at `ss × ss` samples, box filtered to `w × h`, and which pixels
/// (with their eight neighbours) have a sample that sees a light.
fn supersampled(
    kernel: &MaskConeKernel,
    camera: Camera,
    (w, h): (u32, u32),
    ss: u32,
    lights: &[Light],
    field: &Field,
    origin: DVec3,
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let (big, _, _) =
        kernel.render_instanced_lights(camera, w * ss, h * ss, lights, &field.scene, origin, 1);
    let mut px = vec![[0.0f32; 3]; (w * h) as usize];
    let mut sees = vec![false; (w * h) as usize];
    let n = (ss * ss) as f32;
    for y in 0..(h * ss) as usize {
        for x in 0..(w * ss) as usize {
            let p = big[y * (w * ss) as usize + x];
            let i = (y / ss as usize) * w as usize + x / ss as usize;
            if lights.iter().any(|l| l.radiance.to_array() == p) {
                sees[i] = true;
            }
            for c in 0..3 {
                px[i][c] += p[c] / n;
            }
        }
    }
    let mut apart = sees.clone();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !sees[(y * w as i32 + x) as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                        apart[(ny * w as i32 + nx) as usize] = true;
                    }
                }
            }
        }
    }
    (px, apart)
}

/// Relative RMSE over the pixels not set apart.
fn surfaces_rmse(a: &[[f32; 3]], b: &[[f32; 3]], apart: &[bool]) -> f64 {
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(apart)
            .filter(|(_, x)| !**x)
            .map(|(p, _)| *p)
            .collect()
    };
    relative_rmse(&keep(a), &keep(b))
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_sky(Vec3::ZERO);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let sun = [Light::white(sun, field.radiance)];
    let lights = field.colored_lights(origin, 1000.0);
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
    let cached = |lights: &[Light], ratio: bool| {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            lights,
            &field.scene,
            origin,
            CacheOptions {
                light_ratio: ratio,
                ..Default::default()
            },
        )
    };

    // Control: the sun-only warm frame at the standing viewpoint.
    let control = {
        let mut r = cached(&sun, true);
        warm(&mut r, ground, &target)
    };
    let fit = ms(control.frame_time) <= 3.3;
    println!(
        "control: sun-only warm frame {:.2} ms (at most 3.3 ms) after {:?} frames: {}",
        ms(control.frame_time),
        control.frames,
        if fit {
            "fit"
        } else {
            "NOT FIT: the run is void"
        }
    );
    if !fit {
        return Ok(());
    }

    // 1–2. Cache error with ratios and with irradiance texels, at gate 8's
    // viewpoints, for the sun alone and the three lights.
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |ahead: f32, height: f32, down: f32| {
        let d = down.to_radians();
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
    struct Case {
        error: f64,
        frames: Option<usize>,
        frame_time: Duration,
    }
    struct Row {
        name: &'static str,
        // Sun alone and three lights; each with ratios, then irradiance.
        cases: [[Case; 2]; 2],
    }
    let mut rows = Vec::new();
    for (name, camera) in views {
        let mut cases = Vec::new();
        for set in [&sun[..], &lights[..]] {
            let exact = kernel
                .render_instanced_lights(camera, W, H, set, &field.scene, origin, 1)
                .0;
            let mut pair = Vec::new();
            for ratio in [true, false] {
                let mut r = cached(set, ratio);
                let w = warm(&mut r, camera, &target);
                drop(r);
                pair.push(Case {
                    error: relative_rmse(&w.image, &exact),
                    frames: w.frames,
                    frame_time: w.frame_time,
                });
            }
            let [a, b]: [Case; 2] = pair.try_into().ok().unwrap();
            cases.push([a, b]);
        }
        let [sun_cases, three_cases]: [[Case; 2]; 2] = cases.try_into().ok().unwrap();
        let row = Row {
            name,
            cases: [sun_cases, three_cases],
        };
        for (label, c) in [("sun", &row.cases[0]), ("three lights", &row.cases[1])] {
            println!(
                "{name}, {label}: cache error {:.2}% with ratios ({:.2} ms warm, {:?} frames), {:.2}% with irradiance ({:.2} ms, {:?} frames): ×{:.2}",
                100.0 * c[0].error,
                ms(c[0].frame_time),
                c[0].frames,
                100.0 * c[1].error,
                ms(c[1].frame_time),
                c[1].frames,
                c[0].error / c[1].error
            );
        }
        rows.push(row);
    }

    // 3. Per pixel against gate 6's Cycles references.
    kernel.set_hardware_rays(false);
    let small = (320, 180);
    let reference = |lights: &[Light], tag: &str| {
        render_instanced_lights(
            &field.scene,
            origin,
            lights,
            Vec3::ZERO,
            ground,
            small,
            4096,
            0,
            1,
            &out_dir.join(format!("m5_gate6_cycles_{tag}")),
        )
    };
    let (ours, apart) = supersampled(&kernel, ground, small, 16, &lights, &field, origin);
    let cycles = reference(&lights, "lights")?;
    let accuracy = surfaces_rmse(&ours, &cycles.pixels, &apart);
    println!(
        "3. three lights vs Cycles: {:.2}% over surfaces (reference: {}, {:.1} s)",
        100.0 * accuracy,
        cycles.device,
        cycles.seconds
    );
    let mut alone = Vec::new();
    for (k, name) in ["sun", "fill", "lamp"].iter().enumerate() {
        let one = [lights[k]];
        let (o, a) = supersampled(&kernel, ground, small, 16, &one, &field, origin);
        let c = reference(&one, name)?;
        let e = surfaces_rmse(&o, &c.pixels, &a);
        println!("   {name} alone vs Cycles: {:.2}%", 100.0 * e);
        alone.push((*name, e));
    }
    kernel.set_hardware_rays(true);

    // 4. Flights with the three lights.
    struct Flight {
        mean: Duration,
        p99: Duration,
        worst: Duration,
        fallback: f64,
        budget: f64,
        gpu_bake: f64,
        gpu_camera: f64,
    }
    let fly = || {
        let mut r = cached(&lights, true);
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
        drop(r);
        let mut sorted = times.clone();
        sorted.sort();
        let n = log.len() as f64;
        Flight {
            mean: total / 600,
            p99: sorted[(sorted.len() * 99) / 100 - 1],
            worst: *sorted.last().unwrap(),
            fallback: log
                .iter()
                .map(|s| s.fallback_pixels as f64 / (W * H) as f64)
                .sum::<f64>()
                / n,
            budget: log.iter().map(|s| s.budget as f64).sum::<f64>() / n,
            gpu_bake: log.iter().map(|s| s.gpu_bake_ms as f64).sum::<f64>() / n,
            gpu_camera: log.iter().map(|s| s.gpu_render_ms as f64).sum::<f64>() / n,
        }
    };
    let flights: Vec<Flight> = (0..3).map(|_| fly()).collect();
    for f in &flights {
        println!(
            "flight: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, budget {:.0}, GPU bake {:.2} ms, camera {:.2} ms",
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            100.0 * f.fallback,
            f.budget,
            f.gpu_bake,
            f.gpu_camera
        );
    }

    let c1 = rows
        .iter()
        .all(|r| r.cases[0][0].error <= 0.6 * r.cases[0][1].error);
    let c2 = rows
        .iter()
        .all(|r| r.cases[1][0].error <= 0.75 * r.cases[1][1].error);
    let c3 = accuracy <= 0.013;
    let c4 = rows
        .iter()
        .all(|r| r.cases[1][0].frames.is_some() && ms(r.cases[1][0].frame_time) <= 16.7)
        && flights
            .iter()
            .all(|f| ms(f.mean) <= 16.7 && ms(f.p99) <= 33.3 && f.fallback <= 0.10);
    let pass = c1 && c2 && c3 && c4;

    let mut out = String::new();
    let fr = |x: &Option<usize>| x.map_or("not warm".to_string(), |v| v.to_string());
    writeln!(out, "# M5 gate 7 results\n").unwrap();
    writeln!(
        out,
        "{}, 999,986 boulders, gate 8's viewpoints at {W}×{H}, hardware rays, prefilter 1 bin with object-level prefiltering, cache defaults.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "Control: sun-only warm frame at the standing viewpoint with ratios {:.2} ms (at most 3.3 ms).\n",
        ms(control.frame_time)
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | Lights | Cache error, ratios | Cache error, irradiance | Ratio | Warm frame, ratios / irradiance | Frames until warm, ratios / irradiance |\n|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for r in &rows {
        for (label, c) in [("sun", &r.cases[0]), ("three", &r.cases[1])] {
            writeln!(
                out,
                "| {} | {label} | **{:.2}%** | {:.2}% | ×{:.2} | {:.2} / {:.2} ms | {} / {} |",
                r.name,
                100.0 * c[0].error,
                100.0 * c[1].error,
                c[0].error / c[1].error,
                ms(c[0].frame_time),
                ms(c[1].frame_time),
                fr(&c[0].frames),
                fr(&c[1].frames)
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "\n3. Three lights per pixel (16×16, 320×180) vs Cycles (4,096 spp; {}), surfaces: **{:.2}%**. Alone: {}.",
        cycles.device,
        100.0 * accuracy,
        alone
            .iter()
            .map(|(n, e)| format!("{n} {:.2}%", 100.0 * e))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        out,
        "\nFlights with the three lights and ratios, 600 frames from an empty cache:\n\n| Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |\n|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    for (i, f) in flights.iter().enumerate() {
        writeln!(
            out,
            "| {} | {:.2} ms | {:.2} ms | {:.2} ms | {:.2}% | {:.0} | {:.2} ms | {:.2} ms |",
            i + 1,
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            100.0 * f.fallback,
            f.budget,
            f.gpu_bake,
            f.gpu_camera
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n## Verdict\n\n1. Sun alone, ratios at most 0.6× irradiance at every viewpoint: {}\n2. Three lights, at most 0.75×: {}\n3. Per pixel within 1.30% of Cycles: {}\n4. Speed with three lights (warm and three flights): {}\n\n→ {}",
        if c1 { "pass" } else { "fail" },
        if c2 { "pass" } else { "fail" },
        if c3 { "pass" } else { "fail" },
        if c4 { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    std::fs::write(out_dir.join("m5_gate7_results.md"), &out)?;
    println!("{out}");
    Ok(())
}
