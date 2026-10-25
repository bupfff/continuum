//! M5 gate 6: several colored lights.
//!
//! What was built since gate 5:
//! - Lights are a list of rectangles with RGB radiance (`Light`, at most
//!   `MAX_LIGHTS`), in per-pixel rendering and in the view cache. Camera rays
//!   record which light they see. A cache texel holds one channel per light
//!   (irradiance per unit radiance) plus the sky's, so re-coloring a light
//!   (`InstancedRenderer::set_light_radiance`) re-weights stored channels and
//!   bakes nothing: lighting is linear in emitted power.
//! - Light cones are culled against a circular cone bounding them (a light
//!   seen at a grazing angle made the window's planes, tested one at a time,
//!   pass boxes metres off its line), walk objects nearest first along their
//!   axis, and treat coverage within 1e-5 of a whole bin as whole, so an
//!   occluded cone stops early. Sun-only images move by 4e-8 (relative).
//!   Development flights with a grazing lamp went from 579 ms to 14 ms.
//! - The cache's pool keeps 2^19 pages with any number of lights (fewer made
//!   flights evict while waiting for the GPU).
//!
//! Scene: the million-boulder field with `Field::colored_lights(origin, 1 km)`:
//! the sun, warm, 1 km from the camera in its usual direction; a cool fill
//! (4° across, 20° up, opposite azimuth, a quarter of the sun's irradiance);
//! and an orange lamp (1.5 m square, 6 m up, facing down, 25 m ahead of the
//! camera). Prefilter 1 bin with object-level prefiltering, sky prefilter ×8,
//! cache defaults, hardware rays at 1280×720, gate 8's four viewpoints.
//!
//! Control, before anything else: gate 11's sun-only warm frame at the
//! standing viewpoint at most 8.03 ms, or the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. Accuracy: the three lights per pixel, 16×16 samples per pixel (box
//!    filtered) at 320×180, at the field's camera, within 3% relative RMSE of
//!    instanced Cycles at 4,096 samples per pixel, over surfaces: pixels where
//!    a sample sees a light, and their eight neighbours, are scored apart
//!    (emitters are hundreds of times brighter than surfaces; development
//!    runs found 16 samples per pixel too few for small lamp-lit boulders).
//! 2. Re-coloring is exact and free: a warm cache with the three lights and
//!    the sky, given new RGB radiance for all three, bakes no page in the
//!    next 4 frames, and its image is bit-identical to a fresh warm cache
//!    built with the new colors.
//! 3. Speed with the three lights: warm, each viewpoint at most 16.7 ms per
//!    frame; gate 8's flight from an empty cache, three runs, each with mean
//!    at most 16.7 ms, 99th percentile at most 33.3 ms, fallback at most 10%.
//!
//! Reported, not gated: each light alone against Cycles; the three lights and
//! the sky against Cycles; the cached image against per-pixel exact cones at
//! each viewpoint, with the warm cache of the sun alone for comparison;
//! frames until warm; and flights with the three lights and the sky
//! (development runs measured a 99th percentile of about 34 ms there, from
//! sky pages on the ground seen from above, gate 5's open problem).

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
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0);
    let sky = field.sky();
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
    let cached = |lights: &[Light], sky: Vec3| {
        kernel.set_sky(sky);
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            lights,
            &field.scene,
            origin,
            CacheOptions::default(),
        )
    };

    // Control: gate 11's sun-only warm frame at the standing viewpoint.
    let control = {
        let mut r = cached(&[Light::white(sun, field.radiance)], Vec3::ZERO);
        warm(&mut r, ground, &target)
    };
    let fit = ms(control.frame_time) <= 8.03;
    println!(
        "control: sun-only warm frame {:.2} ms (gate 11: 8.03 ms) after {:?} frames: {}",
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

    // 1. Accuracy against Cycles, and each light alone (reported).
    kernel.set_hardware_rays(false);
    kernel.set_sky(Vec3::ZERO);
    let small = (320, 180);
    let reference = |lights: &[Light], sky: Vec3, tag: &str| {
        render_instanced_lights(
            &field.scene,
            origin,
            lights,
            sky,
            ground,
            small,
            4096,
            0,
            1,
            &out_dir.join(format!("m5_gate6_cycles_{tag}")),
        )
    };
    let (ours, apart) = supersampled(&kernel, ground, small, 16, &lights, &field, origin);
    let cycles = reference(&lights, Vec3::ZERO, "lights")?;
    let accuracy = surfaces_rmse(&ours, &cycles.pixels, &apart);
    let apart_count = apart.iter().filter(|x| **x).count();
    println!(
        "1. three lights vs Cycles: {:.2}% over surfaces ({apart_count} pixels apart, Cycles {:.1} s)",
        100.0 * accuracy,
        cycles.seconds
    );
    let mut alone = Vec::new();
    for (k, name) in ["sun", "fill", "lamp"].iter().enumerate() {
        let one = [lights[k]];
        let (o, a) = supersampled(&kernel, ground, small, 16, &one, &field, origin);
        let c = reference(&one, Vec3::ZERO, name)?;
        let e = surfaces_rmse(&o, &c.pixels, &a);
        println!("   {name} alone vs Cycles: {:.2}%", 100.0 * e);
        alone.push((*name, e));
    }
    kernel.set_sky(sky);
    let (with_sky, apart_sky) = supersampled(&kernel, ground, small, 16, &lights, &field, origin);
    let cycles_sky = reference(&lights, sky, "lights_sky")?;
    let accuracy_sky = surfaces_rmse(&with_sky, &cycles_sky.pixels, &apart_sky);
    println!(
        "   three lights and sky vs Cycles: {:.2}%",
        100.0 * accuracy_sky
    );
    kernel.set_hardware_rays(true);

    // 2. Re-coloring a warm cache with the three lights and the sky.
    let recolor = [
        Vec3::new(0.7, 0.8, 1.0) * field.radiance,
        Vec3::new(1.0, 0.5, 0.3) * 40.0,
        Vec3::new(0.2, 1.0, 0.4) * 300.0,
    ];
    let mut r = cached(&lights, sky);
    let both = warm(&mut r, ground, &target);
    for (k, rgb) in recolor.iter().enumerate() {
        r.set_light_radiance(k, *rgb);
    }
    let before = r.cache_log().len();
    for _ in 0..4 {
        r.frame(ground, &target);
    }
    r.finish();
    let baked: u32 = r.cache_log()[before..].iter().map(|s| s.pages_baked).sum();
    let recolored = r.read_colors();
    drop(r);
    let new_lights: Vec<Light> = lights
        .iter()
        .zip(recolor)
        .map(|(l, radiance)| Light { radiance, ..*l })
        .collect();
    let mut r = cached(&new_lights, sky);
    warm(&mut r, ground, &target);
    let fresh = r.read_colors();
    drop(r);
    let identical = recolored == fresh;
    println!(
        "2. re-colored warm cache (warm after {:?} frames with sky): {baked} pages baked in 4 frames, bit-identical to a fresh cache: {identical}",
        both.frames
    );

    // 3. Speed with the three lights, at gate 8's viewpoints; the cache
    // against per-pixel exact cones (reported), with the sun alone.
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
    struct Row {
        name: &'static str,
        warm: Warm,
        sun_frames: Option<usize>,
        cached_vs_exact: f64,
        sun_cached_vs_exact: f64,
    }
    let mut rows = Vec::new();
    for (name, camera) in views {
        let mut r = cached(&lights, Vec3::ZERO);
        let three = warm(&mut r, camera, &target);
        drop(r);
        let mut r = cached(&lights[..1], Vec3::ZERO);
        let one = warm(&mut r, camera, &target);
        drop(r);
        kernel.set_sky(Vec3::ZERO);
        kernel.set_prefilter(0.0);
        let exact = kernel
            .render_instanced_lights(camera, W, H, &lights, &field.scene, origin, 1)
            .0;
        let exact_sun = kernel
            .render_instanced_lights(camera, W, H, &lights[..1], &field.scene, origin, 1)
            .0;
        kernel.set_prefilter(1.0);
        let row = Row {
            name,
            cached_vs_exact: relative_rmse(&three.image, &exact),
            sun_cached_vs_exact: relative_rmse(&one.image, &exact_sun),
            sun_frames: one.frames,
            warm: three,
        };
        println!(
            "{name}: warm {:.2} ms after {:?} frames (sun alone {:?}); cached vs per-pixel exact {:.2}% (sun alone {:.2}%)",
            ms(row.warm.frame_time),
            row.warm.frames,
            row.sun_frames,
            100.0 * row.cached_vs_exact,
            100.0 * row.sun_cached_vs_exact
        );
        rows.push(row);
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
    let fly = |sky: Vec3| {
        let mut r = cached(&lights, sky);
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
    let flights: Vec<Flight> = (0..3).map(|_| fly(Vec3::ZERO)).collect();
    let sky_flights: Vec<Flight> = (0..3).map(|_| fly(sky)).collect();
    for (label, set) in [
        ("three lights", &flights),
        ("three lights and sky", &sky_flights),
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

    let c1 = accuracy <= 0.03;
    let c2 = baked == 0 && identical;
    let c3 = rows
        .iter()
        .all(|r| r.warm.frames.is_some() && ms(r.warm.frame_time) <= 16.7)
        && flights
            .iter()
            .all(|f| ms(f.mean) <= 16.7 && ms(f.p99) <= 33.3 && f.fallback <= 0.10);
    let pass = c1 && c2 && c3;

    let mut out = String::new();
    let fr = |x: &Option<usize>| x.map_or("not warm".to_string(), |v| v.to_string());
    writeln!(out, "# M5 gate 6 results\n").unwrap();
    writeln!(
        out,
        "{}, 999,986 boulders, three colored lights (`Field::colored_lights`, 1 km), prefilter 1 bin with object-level prefiltering, sky prefilter ×8, cache defaults, hardware rays at {W}×{H}.\n",
        kernel.adapter_name
    )
    .unwrap();
    writeln!(
        out,
        "Control: sun-only warm frame at the standing viewpoint {:.2} ms (gate 11: 8.03 ms).\n",
        ms(control.frame_time)
    )
    .unwrap();
    writeln!(
        out,
        "1. Three lights per pixel (16×16, 320×180) vs Cycles (4,096 spp, {:.1} s), surfaces: **{:.2}%** ({apart_count} pixels apart). Alone: {}. With the sky: {:.2}%.\n",
        cycles.seconds,
        100.0 * accuracy,
        alone
            .iter()
            .map(|(n, e)| format!("{n} {:.2}%", 100.0 * e))
            .collect::<Vec<_>>()
            .join(", "),
        100.0 * accuracy_sky
    )
    .unwrap();
    writeln!(
        out,
        "2. Re-colored warm cache (three lights and sky, warm after {} frames): {baked} pages baked in 4 frames; bit-identical to a fresh cache: {identical}.\n",
        fr(&both.frames)
    )
    .unwrap();
    writeln!(
        out,
        "| Viewpoint | 3. Warm, three lights | Frames until warm: three lights / sun alone | Cached vs per-pixel exact: three lights / sun alone |\n|---|---|---|---|"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            out,
            "| {} | {:.2} ms | {} / {} | {:.2}% / {:.2}% |",
            r.name,
            ms(r.warm.frame_time),
            fr(&r.warm.frames),
            fr(&r.sun_frames),
            100.0 * r.cached_vs_exact,
            100.0 * r.sun_cached_vs_exact
        )
        .unwrap();
    }
    for (label, set) in [
        ("three lights (criterion 3)", &flights),
        ("three lights and sky (reported)", &sky_flights),
    ] {
        writeln!(
            out,
            "\nFlights, {label}, 600 frames from an empty cache:\n\n| Run | Mean | 99th percentile | Worst | Fallback | Bake budget mean | GPU bake | GPU camera |\n|---|---|---|---|---|---|---|---|"
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
    writeln!(
        out,
        "\n## Verdict\n\n1. Three lights within 3% of Cycles: {}\n2. Re-coloring exact and free: {}\n3. Speed with three lights (warm and three flights): {}\n\n→ {}",
        if c1 { "pass" } else { "fail" },
        if c2 { "pass" } else { "fail" },
        if c3 { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m5_gate6_results.md"), &out)?;
    Ok(())
}
