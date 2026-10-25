//! M5 gate 8: point and spot lights, and cheaper shading of lights.
//!
//! What was built since gate 7:
//! - Sphere lights (`Emitter::Sphere`): a point light with a size, or a spot
//!   with glTF's falloff (the square of a ramp in the cosine to its axis
//!   between inner and outer half-angles), per pixel and in the view cache.
//!   - A sphere's cone is the square around its disk, its bins weighted by
//!     their part inside the disk.
//!   - Its unshadowed light is the projected solid angle of the cap,
//!     clipped to the horizon (Snyder 1996; Lagarde and de Rousiers 2014).
//!   - Camera rays see spheres; a spot's falloff applies per receiver.
//!   - glTF files bring their punctual lights (`Imported::lights`).
//!   - Cycles references model a sphere as a finely divided mesh with the
//!     true silhouette area, and a spot through an emission node graph.
//! - Shading evaluates a rectangle's unshadowed light from its corners
//!   (Lambert's formula) instead of building its cone, and skips it for
//!   fully shadowed pixels. This is gate 7's failed speed criterion,
//!   retested.
//!
//! Scenes:
//! - **Night.** The textured scene at night, lit by `Textured::night_lights`:
//!   a point light 10 cm across, 2 m up; a spot 5 cm across, 4 m up.
//! - **Field.** The million-boulder field with the sun and
//!   `Field::sphere_lights` (a point light 3 m up and a spot 8 m up, each
//!   giving the sun's ground irradiance at its brightest), at gate 8's four
//!   viewpoints, 1280×720.
//!
//! Prefilter 1 bin, object-level prefiltering, cache defaults, hardware
//! rays except where stated. "Over surfaces": pixels where a sample sees a
//! light, and their eight neighbours, are scored apart. "Cache error": the
//! warm cached image against the per-pixel image with the same lights and
//! prefiltering (1 sample per pixel).
//!
//! Development runs, disclosed:
//! - Night scene at 320×180, over surfaces: 13.71% against Cycles at 256
//!   samples per pixel (4×4 ours), 3.53% at 4,096 (8×8 ours), means within
//!   0.3%. The fourfold drop suggests Cycles' noise dominates.
//! - Field, sun and sphere lights, cache error: 5.44 / 9.76 / 14.68 /
//!   21.75% (standing, 10 m, 40 m, 150 m up).
//! - One flight with them: mean 14.34 ms, 99th percentile 20.69 ms.
//! - Flights with gate 6's three lights and the cheaper shading: means
//!   15.29 and 15.43 ms, 99th percentiles 28.5 and 30.4 ms.
//!
//! Control, before anything else: the sun-only warm frame at the standing
//! viewpoint at most 3.3 ms, or the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. Accuracy: the night scene per pixel, 16×16 samples per pixel at
//!    320×180 (compute rays), within 3% relative RMSE of instanced Cycles at
//!    16,384 samples per pixel, over surfaces.
//! 2. The cache on the night scene: at 1280×720, the warm cached image
//!    within 2% of its per-pixel image, over surfaces.
//! 3. The cache on the field with the sun and sphere lights: at each
//!    viewpoint, cache error at most gate 7's sun-only figure with ratios
//!    plus 1 point (6.21, 10.63, 14.78, 21.75%).
//! 4. Speed with the sun and sphere lights: warm, each viewpoint at most
//!    16.7 ms per frame; three of gate 8's flights from an empty cache, each
//!    with mean at most 16.7 ms, 99th percentile at most 33.3 ms and
//!    fallback at most 10%.
//! 5. Speed with gate 6's three lights (gate 7's criterion 4 again): three
//!    flights, each with mean at most 16.7 ms, 99th percentile at most
//!    33.3 ms and fallback at most 10%.
//!
//! Reported, not gated: each night light alone against Cycles at 4,096
//! samples per pixel; frames until warm; flights' bake budgets and GPU
//! times.
//! `cargo run --release -p room3d --bin m5_gate8`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::field::Field;
use room3d::textured::Textured;

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

/// Which pixels of a `w × h` image whose samples (`ss × ss` per pixel) are
/// `big` see a light (any channel above 1% of the brightest light's
/// radiance), with their eight neighbours; and the box-filtered image.
fn apart_and_filtered(
    big: &[[f32; 3]],
    (w, h): (u32, u32),
    ss: u32,
    lights: &[Light],
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let peak = lights
        .iter()
        .map(|l| l.radiance.max_element())
        .fold(0.0f32, f32::max);
    let mut px = vec![[0.0f32; 3]; (w * h) as usize];
    let mut sees = vec![false; (w * h) as usize];
    let n = (ss * ss) as f32;
    for y in 0..(h * ss) as usize {
        for x in 0..(w * ss) as usize {
            let p = big[y * (w * ss) as usize + x];
            let i = (y / ss as usize) * w as usize + x / ss as usize;
            if p.iter().any(|&c| c > 0.01 * peak) {
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

struct Flight {
    mean: Duration,
    p99: Duration,
    worst: Duration,
    fallback: f64,
    budget: f64,
    gpu_bake: f64,
    gpu_camera: f64,
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
    let sun = Light::white(sun, field.radiance);
    let mut field_lights = vec![sun];
    field_lights.extend(field.sphere_lights(origin));
    let colored = field.colored_lights(origin, 1000.0);
    let night = Textured::new();
    let night_lights = Textured::night_lights();
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
    let cached = |lights: &[Light], scene: &InstancedScene, origin: DVec3| {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            lights,
            scene,
            origin,
            CacheOptions::default(),
        )
    };

    // Control: the sun-only warm frame at the standing viewpoint.
    let control = {
        let mut r = cached(&[sun], &field.scene, origin);
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

    // 1. The night scene per pixel against Cycles.
    kernel.set_hardware_rays(false);
    let small = (320, 180);
    let night_render = |lights: &[Light], ss: u32| {
        let big = kernel
            .render_instanced_lights(
                night.camera,
                small.0 * ss,
                small.1 * ss,
                lights,
                &night.scene,
                night.origin,
                1,
            )
            .0;
        apart_and_filtered(&big, small, ss, lights)
    };
    let reference = |lights: &[Light], spp: u32, tag: &str| {
        render_instanced_lights(
            &night.scene,
            night.origin,
            lights,
            Vec3::ZERO,
            night.camera,
            small,
            spp,
            0,
            1,
            &out_dir.join(format!("m5_gate8_cycles_{tag}")),
        )
    };
    let (ours, apart) = night_render(&night_lights, 16);
    let cycles = reference(&night_lights, 16_384, "night")?;
    let accuracy = surfaces_rmse(&ours, &cycles.pixels, &apart);
    println!(
        "1. night scene vs Cycles (16,384 spp, {}, {:.1} s): {:.2}% over surfaces ({} pixels apart)",
        cycles.device,
        cycles.seconds,
        100.0 * accuracy,
        apart.iter().filter(|a| **a).count()
    );
    let mut alone = Vec::new();
    for (k, name) in ["point", "spot"].iter().enumerate() {
        let one = [night_lights[k]];
        let (o, a) = night_render(&one, 16);
        let c = reference(&one, 4096, name)?;
        let e = surfaces_rmse(&o, &c.pixels, &a);
        println!("   {name} alone vs Cycles (4,096 spp): {:.2}%", 100.0 * e);
        alone.push((*name, e));
    }
    kernel.set_hardware_rays(true);

    // 2. The cache on the night scene.
    let night_cache = {
        let mut r = cached(&night_lights, &night.scene, night.origin);
        let w = warm(&mut r, night.camera, &target);
        drop(r);
        let big = kernel
            .render_instanced_lights(
                night.camera,
                W,
                H,
                &night_lights,
                &night.scene,
                night.origin,
                1,
            )
            .0;
        let (exact, apart) = apart_and_filtered(&big, (W, H), 1, &night_lights);
        (surfaces_rmse(&w.image, &exact, &apart), w)
    };
    println!(
        "2. night scene, cached vs per-pixel over surfaces: {:.2}% (warm after {:?} frames, {:.2} ms)",
        100.0 * night_cache.0,
        night_cache.1.frames,
        ms(night_cache.1.frame_time)
    );

    // 3–4. The field with the sun and sphere lights, at gate 8's viewpoints.
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
        ("standing", view(0.0, 1.7, 0.0), 0.0621),
        ("10 m up", view(0.0, 10.0, 15.0), 0.1063),
        ("40 m up", view(0.0, 40.0, 30.0), 0.1478),
        ("150 m up", view(0.0, 150.0, 45.0), 0.2175),
    ];
    struct Row {
        name: &'static str,
        error: f64,
        bound: f64,
        warm: Warm,
    }
    let mut rows = Vec::new();
    for (name, camera, gate7) in views {
        let mut r = cached(&field_lights, &field.scene, origin);
        let w = warm(&mut r, camera, &target);
        drop(r);
        let exact = kernel
            .render_instanced_lights(camera, W, H, &field_lights, &field.scene, origin, 1)
            .0;
        let row = Row {
            name,
            error: relative_rmse(&w.image, &exact),
            bound: gate7 + 0.01,
            warm: w,
        };
        println!(
            "{name}: cache error {:.2}% (at most {:.2}%), warm {:.2} ms after {:?} frames",
            100.0 * row.error,
            100.0 * row.bound,
            ms(row.warm.frame_time),
            row.warm.frames
        );
        rows.push(row);
    }

    // Flights from an empty cache along gate 8's path.
    let fly = |lights: &[Light]| {
        let mut r = cached(lights, &field.scene, origin);
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
    let sphere_flights: Vec<Flight> = (0..3).map(|_| fly(&field_lights)).collect();
    let colored_flights: Vec<Flight> = (0..3).map(|_| fly(&colored)).collect();
    for (label, set) in [
        ("sun and sphere lights", &sphere_flights),
        ("three lights", &colored_flights),
    ] {
        for f in set.iter() {
            println!(
                "flight, {label}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, budget {:.0}, GPU bake {:.2} ms, camera {:.2} ms",
                ms(f.mean),
                ms(f.p99),
                ms(f.worst),
                100.0 * f.fallback,
                f.budget,
                f.gpu_bake,
                f.gpu_camera
            );
        }
    }

    let flights_ok = |set: &[Flight]| {
        set.iter()
            .all(|f| ms(f.mean) <= 16.7 && ms(f.p99) <= 33.3 && f.fallback <= 0.10)
    };
    let c1 = accuracy <= 0.03;
    let c2 = night_cache.0 <= 0.02;
    let c3 = rows.iter().all(|r| r.error <= r.bound);
    let c4 = rows
        .iter()
        .all(|r| r.warm.frames.is_some() && ms(r.warm.frame_time) <= 16.7)
        && flights_ok(&sphere_flights);
    let c5 = flights_ok(&colored_flights);
    let pass = c1 && c2 && c3 && c4 && c5;

    let mut out = String::new();
    let fr = |x: &Option<usize>| x.map_or("not warm".to_string(), |v| v.to_string());
    writeln!(out, "# M5 gate 8 results\n").unwrap();
    writeln!(
        out,
        "{}, prefilter 1 bin with object-level prefiltering, cache defaults, hardware rays at {W}×{H} except for criterion 1.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "Control: sun-only warm frame at the standing viewpoint {:.2} ms (at most 3.3 ms).\n",
        ms(control.frame_time)
    )
    .unwrap();
    writeln!(
        out,
        "1. Night scene per pixel (16×16, 320×180) vs Cycles (16,384 spp; {}, {:.1} s), over surfaces: **{:.2}%**. Alone (4,096 spp): {}.\n",
        cycles.device,
        cycles.seconds,
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
        "2. Night scene cached vs per-pixel at {W}×{H}, over surfaces: **{:.2}%** (warm after {} frames, warm frame {:.2} ms).\n",
        100.0 * night_cache.0,
        fr(&night_cache.1.frames),
        ms(night_cache.1.frame_time)
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | 3. Cache error, sun and sphere lights | Bound | 4. Warm frame | Frames until warm |\n|---|---|---|---|---|"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            out,
            "| {} | **{:.2}%** | {:.2}% | {:.2} ms | {} |",
            r.name,
            100.0 * r.error,
            100.0 * r.bound,
            ms(r.warm.frame_time),
            fr(&r.warm.frames)
        )
        .unwrap();
    }
    for (label, set) in [
        ("the sun and sphere lights (criterion 4)", &sphere_flights),
        ("gate 6's three lights (criterion 5)", &colored_flights),
    ] {
        writeln!(
            out,
            "\nFlights with {label}, 600 frames from an empty cache:\n\n| Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |\n|---|---|---|---|---|---|---|---|"
        )
        .unwrap();
        for (i, f) in set.iter().enumerate() {
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
    }
    let verdict = |c: bool| if c { "pass" } else { "fail" };
    writeln!(
        out,
        "\n## Verdict\n\n1. Night scene within 3% of Cycles: {}\n2. Night scene cached within 2% of per-pixel: {}\n3. Field cache errors within gate 7's plus 1 point: {}\n4. Speed with the sun and sphere lights: {}\n5. Speed with gate 6's three lights: {}\n\n→ {}",
        verdict(c1),
        verdict(c2),
        verdict(c3),
        verdict(c4),
        verdict(c5),
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    std::fs::write(out_dir.join("m5_gate8_results.md"), &out)?;
    println!("{out}");
    Ok(())
}
