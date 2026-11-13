//! M5 gate 13: the sky through hardware rays: warm in a few hundred frames
//! (Landmark 2), as accurate.
//!
//! A view with a sky warms slowly: with the library's defaults (a bake
//! budget that follows GPU time, capped by the frame period), the
//! million-boulder field's view from 10 m up took 12,058 frames to warm in
//! development, because a sky sample (six cube faces of mask cones) costs
//! about 10 µs of GPU time, 17 sun texels' worth.
//!
//! Now, where camera rays use hardware ray queries, a sky sample traces the
//! 192 directions of a hemicube around the texel's normal (one ray through
//! each bin's center, the first hit ending it) and adds the projected solid
//! angle of the open bins above the horizon: about 0.7 µs, counted as one
//! sun texel (`CacheOptions::sky_rays`, `sky_ray_cost`). Movers, streamed
//! levels and alpha-blended objects keep cones.
//!
//! Pass criteria, fixed before the first run:
//! 1. **Speed in flight** (run first): three pairs of flights over the field
//!    with sun and sky (M9's flight, 600 frames from standing height to 150
//!    m, 1280×720, default options), cones then rays: the rays' mean frame
//!    at most 1.05× the cones' (over the three flights each) and at most
//!    16.7 ms. Reported: 99th percentiles, fallback.
//! 2. **Warm-up:** from an empty cache with default options, at each of M5
//!    gate 9's four viewpoints (standing, 10, 40 and 150 m up; 1280×720),
//!    the view warms (16 frames in a row requesting and baking nothing) in
//!    at most 600 frames. Reported: the cones' frames with the same options.
//! 3. **Accuracy against Cycles:** at each of the four viewpoints, the field
//!    with its sun at 1 km, sun and sky, 640×360, the cache settled at 4×4
//!    sub-pixel offsets against instanced Cycles at 1,024 samples per pixel:
//!    the rays' relative RMSE at most 1.05× the cones', and at most 3% at the
//!    standing viewpoint (where the sky's estimate, not the cache's
//!    resolution at distance, decides). Reported: sky only.
//!
//! **Disclosed (before the run):** the ray sky was developed with `sky_dev`
//! (warm-up and pass times) and `sky_rays_dev` (against Cycles) on the full
//! field: per sky sample 0.72 µs with rays against 10.3 µs with cones at
//! 10 m up; against Cycles (sun and sky) rays 2.55, 10.04, 14.59, 20.43% at
//! the four viewpoints, cones 4.68, 11.99, 18.08, 20.02% (sky only: rays
//! 1.66, 6.92, 14.31, 21.75%, cones 4.31, 10.26, 18.67, 19.01%); with
//! default options rays warmed the four views in 176, 373, 426 and 287
//! frames. A readback every 20th frame held the frame-period cap near its
//! floor (up to 1,895 frames to warm), so warm-ups here run undisturbed. The
//! development mode passed: flights 9.71 against 9.70 ms; warm in 165 and
//! 353 frames against 1,904 and 12,347; sun and sky 2.14% against 4.40%
//! standing, 7.80% against 10.22% at 10 m.
//!
//! `M5_DEV=1`: a development check on 100,000 boulders, two viewpoints,
//! one flight pair of 200 frames; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m5_gate13`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const CW: u32 = 640;
const CH: u32 = 360;
const FLIGHT: usize = 600;

fn target(kernel: &MaskConeKernel, w: u32, h: u32) -> wgpu::TextureView {
    kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Frames until 16 in a row request and bake nothing (at most 60,000).
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Option<usize> {
    for _ in 0..60_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return Some(latest - 16);
        }
    }
    None
}

/// The cache settled at 4×4 sub-pixel offsets, averaged.
fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> Vec<[f32; 3]> {
    let mut sum = vec![[0.0f32; 3]; (CW * CH) as usize];
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        settle(r, camera, target);
        r.finish();
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    sum
}

struct Flight {
    mean: Duration,
    p99: Duration,
    fallback: f64,
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M5_DEV").is_some();
    if dev {
        println!(
            "DEVELOPMENT RUN: 100,000 boulders, two viewpoints, one flight pair of 200 frames; not the gate's numbers"
        );
    }
    let boulders = if dev { 100_000 } else { 1_000_000 };
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
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let big = target(&kernel, W, H);
    let small = target(&kernel, CW, CH);
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let options = |rays: bool| CacheOptions {
        sky_rays: rays,
        ..Default::default()
    };
    let label = |rays: bool| if rays { "rays" } else { "cones" };
    let views_for = |ground: Camera| {
        let forward = (ground.target - ground.eye).normalize();
        let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
        let view = move |ahead: f32, height: f32, down: f32| {
            let d = down.to_radians();
            let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
            Camera {
                eye,
                target: eye + flat * d.cos() - Vec3::Z * d.sin(),
                vertical_fov_degrees: ground.vertical_fov_degrees,
            }
        };
        (
            [
                ("standing", view(0.0, 1.7, 0.0)),
                ("10 m up", view(0.0, 10.0, 15.0)),
                ("40 m up", view(0.0, 40.0, 30.0)),
                ("150 m up", view(0.0, 150.0, 45.0)),
            ],
            move |f: usize| {
                let t = f as f32 / (FLIGHT - 1) as f32;
                view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t)
            },
        )
    };
    let views_n = if dev { 2 } else { 4 };

    // 1 and 2: M5 gate 9's scene (the field's own sun).
    let field = Field::new(boulders, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    let (views, flight_camera) = views_for(ground);
    let renderer = |rays: bool| {
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
            options(rays),
        )
    };

    // 1. Flights, first.
    let frames = if dev { 200 } else { FLIGHT };
    let fly = |rays: bool| {
        let mut r = renderer(rays);
        let mut times = Vec::with_capacity(frames);
        let begin = Instant::now();
        for f in 0..frames {
            let call = Instant::now();
            r.frame(flight_camera(f * FLIGHT / frames), &big);
            times.push(call.elapsed());
        }
        r.finish();
        let total = begin.elapsed();
        times.sort();
        let log = r.cache_log();
        Flight {
            mean: total / frames as u32,
            p99: times[(times.len() * 99) / 100 - 1],
            fallback: log.iter().map(|s| s.fallback_pixels as f64).sum::<f64>()
                / (log.len() as f64 * (W * H) as f64),
        }
    };
    let mut flights: [Vec<Flight>; 2] = [Vec::new(), Vec::new()];
    for _ in 0..if dev { 1 } else { 3 } {
        for rays in [false, true] {
            let f = fly(rays);
            say(format!(
                "1. flight, {}: mean {:.2} ms, p99 {:.2} ms, fallback {:.2}%",
                label(rays),
                ms(f.mean),
                ms(f.p99),
                100.0 * f.fallback
            ));
            flights[rays as usize].push(f);
        }
    }
    let mean = |v: &[Flight]| v.iter().map(|f| ms(f.mean)).sum::<f64>() / v.len() as f64;
    let (cones_mean, rays_mean) = (mean(&flights[0]), mean(&flights[1]));
    let c1 = rays_mean <= 1.05 * cones_mean && rays_mean <= 16.7;
    say(format!(
        "1. speed in flight: rays {rays_mean:.2} ms against cones {cones_mean:.2} ms ({:.3}×; at most 1.05× and 16.7 ms): {}",
        rays_mean / cones_mean,
        if c1 { "PASS" } else { "FAIL" }
    ));

    // 2. Warm-up from empty, undisturbed.
    let mut c2 = true;
    for &(name, camera) in views.iter().take(views_n) {
        let mut row = format!("2. {name}:");
        for rays in [true, false] {
            let mut r = renderer(rays);
            let frames = settle(&mut r, camera, &big);
            let samples: u64 = r.cache_log().iter().map(|s| s.sky_samples as u64).sum();
            if rays {
                c2 &= frames.is_some_and(|f| f <= 600);
            }
            row += &format!(
                " {} warm in {} frames ({samples} sky samples);",
                label(rays),
                frames.map_or("never (60,000)".to_string(), |f| f.to_string())
            );
        }
        say(row);
    }
    say(format!(
        "2. warm-up: every view warm with rays within 600 frames: {}",
        if c2 { "PASS" } else { "FAIL" }
    ));
    drop(field);

    // 3. Accuracy against Cycles: the sun at 1 km.
    let near = Field::new(boulders, DVec3::ZERO).with_sun_at(1000.0);
    let n_origin = near.origin_near_camera();
    let (n_ground, n_light) = near.relative(n_origin);
    let n_sky = near.sky();
    kernel.set_sky(n_sky);
    let (n_views, _) = views_for(n_ground);
    let mut c3 = true;
    for (k, &(name, camera)) in n_views.iter().take(views_n).enumerate() {
        for (what, radiance) in [("sky only", 0.0f32), ("sun and sky", near.radiance)] {
            let reference = render_instanced(
                &near.scene,
                n_origin,
                n_light,
                radiance,
                n_sky,
                camera,
                (CW, CH),
                1024,
                0,
                1,
                &out_dir.join(if dev {
                    "m5_gate13_dev_cycles"
                } else {
                    "m5_gate13_cycles"
                }),
            )?;
            let mut errors = [0.0f64; 2];
            for rays in [false, true] {
                let mut r = InstancedRenderer::with_cache(
                    &kernel,
                    format,
                    CW,
                    CH,
                    1,
                    n_light,
                    &near.scene,
                    n_origin,
                    radiance,
                    options(rays),
                );
                let image = supersampled(&mut r, camera, &small);
                errors[rays as usize] = relative_rmse(&image, &reference.pixels);
            }
            let gated = what == "sun and sky";
            let pass = errors[1] <= 1.05 * errors[0] && (k != 0 || errors[1] <= 0.03);
            if gated {
                c3 &= pass;
            }
            say(format!(
                "3. {name}, {what}: rays {:.2}%, cones {:.2}% ({:.3}×){}",
                100.0 * errors[1],
                100.0 * errors[0],
                errors[1] / errors[0],
                if gated {
                    if pass { ": PASS" } else { ": FAIL" }
                } else {
                    " (reported)"
                }
            ));
        }
    }
    say(format!(
        "3. accuracy against Cycles: {}",
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m5_gate13_dev.txt"
        } else {
            "m5_gate13.txt"
        }),
        report,
    )?;
    Ok(())
}
