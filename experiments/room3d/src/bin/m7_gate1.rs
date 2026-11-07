//! M7 gate 1: pixel cones for primary visibility (anti-aliasing).
//!
//! Today each pixel casts one ray and shows the surface at its center. The
//! area-integrated image (what Cycles renders) averages everything inside
//! the pixel's square. One ray per pixel is 13–24% from the supersampled
//! image on the boulder field (M4 gate 8), the largest visible quality gap,
//! and distant boulders smaller than a pixel flicker as the camera moves.
//!
//! Method: each pixel traces a mask cone through its square (8×8 bins of
//! equal image-plane area), front to back through the object and triangle
//! hierarchies, splatting prefiltered coverage below a bin (as light cones
//! do). Each object it meets becomes a fragment: its coverage of the pixel,
//! a binary mask of its covered bins, and its depth. Fragments are sorted by
//! depth and composited by mask (nearer fragments hide farther ones bin by
//! bin); up to four are shaded, each at a ray through its visible bins, and
//! averaged by visible coverage. What no fragment covers shows the sky.
//!
//! Pass criterion, fixed before the first run:
//! 1. Quality: in each of three views at 1280×720 (the textured scene of M6
//!    from its camera; the million-boulder field from M4 gate 8's standing
//!    and 150 m viewpoints), the pixel-cone image's relative RMSE from the
//!    area-integrated reference is at most 0.35× the one-ray image's.
//! 2. Stability: from the field's 150 m viewpoint, moving the camera half a
//!    pixel sideways changes the pixel-cone image (relative RMS of the
//!    difference) by at most 1.5× as much as it changes the reference.
//! 3. Speed: with the lighting cache, warm, on the textured scene at
//!    1280×720, a frame with pixel cones takes at most 16.7 ms (median of 30
//!    warm frames). Measured first, on a cool machine, with the one-ray warm
//!    frame (M6 gate 1: 1.53 ms) as the control.
//!
//! Reported: the one-ray errors and changes; the field's warm frames with
//! pixel cones; fragments per pixel; coverage left to the sky.
//!
//! Details fixed before the first run:
//! - Lighting is per pixel (mask cones, no cache) in every image of
//!   criteria 1 and 2, so only primary visibility differs: the textured
//!   scene's sun and sky, the field's sun (M4 gate 8's light).
//! - Reference: the one-ray renderer at 16×16 samples per pixel (rendered at
//!   16 times the resolution), box filtered. Relative RMSE = RMSE / mean of
//!   the reference over all pixels and channels.
//! - Pixels where a sample of the reference sees a light, and their eight
//!   neighbours, are set apart in criterion 1 and 2 (as in M5 gate 6).
//! - Warm: 16 consecutive frames with no page missing; the camera still.
//! - Timing: release build, `ps` checked for stray processes first.
//!
//! Changed before the first full-size run, after development runs at
//! 320×180 and 160×90 (disclosed in the results):
//! - The reference cannot be rendered at 16 times the resolution (20480 ×
//!   11520 exceeds the 16384-pixel texture limit). It averages 256 frames
//!   at 1280×720 whose camera rays cross each pixel at the centers of a
//!   16×16 grid, sampling textures for 1/16 of a pixel: the same box filter
//!   and texture footprint as the 16× render.
//! - "Moving the camera half a pixel sideways" is a turn by half a pixel's
//!   angle at the image's center.
//! - Fragments per pixel and coverage left to the sky are not reported; the
//!   work per pixel (object nodes, meshes, triangle-hierarchy nodes) is.
//! - The field's warm frames with pixel cones are not measured (a cold
//!   frame takes seconds); its per-pixel frame time is reported.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::textured::Textured;
use room3d::to_rgb;

const W: u32 = 1280;
const H: u32 = 720;
const SS: u32 = 16;

type Pixels = Vec<[f32; 3]>;

fn target(kernel: &MaskConeKernel) -> wgpu::TextureView {
    kernel
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// A view: its scene, lights, sky and camera.
struct View<'a> {
    name: &'static str,
    scene: &'a InstancedScene,
    origin: DVec3,
    lights: Vec<Light>,
    sky: Vec3,
}

fn per_pixel(kernel: &MaskConeKernel, view: &View, pixel_cones: bool) -> InstancedRenderer {
    kernel.set_sky(view.sky);
    kernel.set_pixel_cones(pixel_cones);
    let r = InstancedRenderer::new_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        &view.lights,
        view.scene,
        view.origin,
    );
    kernel.set_pixel_cones(false);
    r
}

/// One frame, its colors and its time (submit to readback).
fn render_once(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Pixels, Duration) {
    r.wait();
    let start = Instant::now();
    r.frame(camera, target);
    r.wait();
    let time = start.elapsed();
    (r.read_colors(), time)
}

/// The area-integrated reference: 16×16 sub-pixel samples averaged, and
/// which pixels (with their neighbours) have a sample that sees a light.
fn area_reference(
    kernel: &MaskConeKernel,
    view: &View,
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Pixels, Vec<bool>) {
    let mut r = per_pixel(kernel, view, false);
    r.set_footprint(1.0 / SS as f32);
    let n = (W * H) as usize;
    let mut sum = vec![[0.0f64; 3]; n];
    let mut sees = vec![false; n];
    for j in 0..SS {
        for i in 0..SS {
            let offset = |k: u32| (k as f32 + 0.5) / SS as f32 - 0.5;
            r.set_pixel_offset(offset(i), offset(j));
            r.frame(camera, target);
            let colors = r.read_colors();
            for (p, c) in colors.iter().enumerate() {
                // Lit surfaces stay far below 100; a seen light is far above.
                if c.iter().any(|&v| v > 100.0) {
                    sees[p] = true;
                }
                for k in 0..3 {
                    sum[p][k] += c[k] as f64;
                }
            }
        }
    }
    let samples = (SS * SS) as f64;
    let image = sum
        .iter()
        .map(|s| s.map(|v| (v / samples) as f32))
        .collect();
    let mut apart = sees.clone();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if !sees[(y * W as i32 + x) as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < W as i32 && ny < H as i32 {
                        apart[(ny * W as i32 + nx) as usize] = true;
                    }
                }
            }
        }
    }
    (image, apart)
}

fn keep(image: &[[f32; 3]], apart: &[bool]) -> Pixels {
    image
        .iter()
        .zip(apart)
        .filter(|(_, a)| !**a)
        .map(|(p, _)| *p)
        .collect()
}

/// Relative RMS of `b − a`, over the mean of `a` (pixels not set apart).
fn change(a: &[[f32; 3]], b: &[[f32; 3]], apart: &[bool]) -> f64 {
    let (a, b) = (keep(a, apart), keep(b, apart));
    let n = (3 * a.len()) as f64;
    let mean = a.iter().flatten().map(|&v| v as f64).sum::<f64>() / n;
    let err = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| ((y - x) as f64).powi(2))
        .sum::<f64>();
    (err / n).sqrt() / mean
}

fn save(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
}

/// Warm timing with the cache: frames until warm (16 frames with nothing
/// requested or baked), then 30 frames each waited for (median), and 60
/// pipelined frames (mean, as M6 gate 1 timed).
struct Warm {
    frames: Option<usize>,
    median: Duration,
    pipelined: Duration,
}

fn warm(
    kernel: &MaskConeKernel,
    view: &View,
    camera: Camera,
    target: &wgpu::TextureView,
    pixel_cones: bool,
    cap: usize,
) -> Warm {
    kernel.set_sky(view.sky);
    kernel.set_pixel_cones(pixel_cones);
    let mut r = InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        &view.lights,
        view.scene,
        view.origin,
        CacheOptions::default(),
    );
    kernel.set_pixel_cones(false);
    let mut frames = 0usize;
    let mut warm_from = None;
    while frames < cap {
        r.frame(camera, target);
        frames += 1;
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
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
    let mut times: Vec<Duration> = (0..30)
        .map(|_| {
            let start = Instant::now();
            r.frame(camera, target);
            r.wait();
            start.elapsed()
        })
        .collect();
    times.sort();
    let start = Instant::now();
    for _ in 0..60 {
        r.frame(camera, target);
    }
    r.finish();
    Warm {
        frames: warm_from,
        median: times[15],
        pipelined: start.elapsed() / 60,
    }
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    // As `Renderer` sets them (M5 gates 3–4).
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let target = target(&kernel);
    let mut out = String::new();

    // Criterion 3 first, on a cool machine.
    let t = Textured::new();
    let textured = View {
        name: "textured",
        scene: &t.scene,
        origin: t.origin,
        lights: t.lights.clone(),
        sky: t.sky,
    };
    let one_warm = warm(&kernel, &textured, t.camera, &target, false, 40_000);
    println!(
        "control, one ray: warm after {:?} frames, median {:.2} ms, pipelined {:.2} ms",
        one_warm.frames,
        ms(one_warm.median),
        ms(one_warm.pipelined)
    );
    let cones_warm = warm(&kernel, &textured, t.camera, &target, true, 6_000);
    println!(
        "pixel cones: warm after {:?} frames, median {:.2} ms, pipelined {:.2} ms",
        cones_warm.frames,
        ms(cones_warm.median),
        ms(cones_warm.pipelined)
    );

    // Criteria 1 and 2.
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let field_view = |height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let field_lights = vec![Light::white(light, field.radiance)];
    let fv = |name| View {
        name,
        scene: &field.scene,
        origin,
        lights: field_lights.clone(),
        sky: Vec3::ZERO,
    };
    let views = [
        (textured, t.camera),
        (fv("field, standing"), field_view(1.7, 0.0)),
        (fv("field, 150 m up"), field_view(150.0, 45.0)),
    ];
    struct Quality {
        name: &'static str,
        one: f64,
        cones: f64,
        one_ms: f64,
        cones_ms: f64,
        work: [f64; 3],
        apart: usize,
    }
    let mut quality = Vec::new();
    let mut stability = None;
    for (view, camera) in &views {
        let start = Instant::now();
        let (reference_image, apart) = area_reference(&kernel, view, *camera, &target);
        let reference = reference_image;
        let ref_secs = start.elapsed().as_secs_f64();
        let mut r = per_pixel(&kernel, view, false);
        let (one, one_time) = render_once(&mut r, *camera, &target);
        drop(r);
        let mut r = per_pixel(&kernel, view, true);
        let (cones, cones_time) = render_once(&mut r, *camera, &target);
        drop(r);
        // The pixel cones' work per pixel (profile stage 9).
        kernel.set_profile_stage(9);
        let mut r = per_pixel(&kernel, view, true);
        let (work, _) = render_once(&mut r, *camera, &target);
        drop(r);
        kernel.set_profile_stage(0);
        let mean = |c: usize| work.iter().map(|p| p[c] as f64).sum::<f64>() / work.len() as f64;
        let q = Quality {
            name: view.name,
            one: relative_rmse(&keep(&one, &apart), &keep(&reference, &apart)),
            cones: relative_rmse(&keep(&cones, &apart), &keep(&reference, &apart)),
            one_ms: ms(one_time),
            cones_ms: ms(cones_time),
            work: [mean(0), mean(1), mean(2)],
            apart: apart.iter().filter(|a| **a).count(),
        };
        println!(
            "{}: reference {ref_secs:.0} s; one ray {:.2}% ({:.1} ms), pixel cones {:.2}% ({:.1} ms) = {:.3}×; work per pixel {:.0} object nodes, {:.1} meshes, {:.0} triangle-hierarchy nodes; {} pixels set apart",
            q.name,
            100.0 * q.one,
            q.one_ms,
            100.0 * q.cones,
            q.cones_ms,
            q.cones / q.one,
            q.work[0],
            q.work[1],
            q.work[2],
            q.apart
        );
        let slug = view.name.replace([',', ' '], "_").replace("__", "_");
        save(
            &out_dir.join(format!("m7_gate1_{slug}_reference.png")),
            &reference,
        )?;
        save(&out_dir.join(format!("m7_gate1_{slug}_one_ray.png")), &one)?;
        save(
            &out_dir.join(format!("m7_gate1_{slug}_pixel_cones.png")),
            &cones,
        )?;
        quality.push(q);

        if view.name == "field, 150 m up" {
            // Half a pixel's angle sideways, at the image's center.
            let half =
                0.5 * 2.0 * (camera.vertical_fov_degrees.to_radians() * 0.5).tan() / H as f32;
            let f = (camera.target - camera.eye).normalize();
            let right = f.cross(Vec3::Z).normalize();
            let moved = Camera {
                target: camera.eye + f + right * half,
                ..*camera
            };
            let (reference2, apart2) = area_reference(&kernel, view, moved, &target);
            let both: Vec<bool> = apart.iter().zip(&apart2).map(|(a, b)| *a || *b).collect();
            let mut r = per_pixel(&kernel, view, false);
            let (one2, _) = render_once(&mut r, moved, &target);
            drop(r);
            let mut r = per_pixel(&kernel, view, true);
            let (cones2, _) = render_once(&mut r, moved, &target);
            drop(r);
            let s = (
                change(&reference, &reference2, &both),
                change(&one, &one2, &both),
                change(&cones, &cones2, &both),
            );
            println!(
                "stability: reference changes {:.2}%, one ray {:.2}%, pixel cones {:.2}% ({:.2}× the reference's)",
                100.0 * s.0,
                100.0 * s.1,
                100.0 * s.2,
                s.2 / s.0
            );
            stability = Some(s);
        }
    }

    let c1 = quality.iter().all(|q| q.cones <= 0.35 * q.one);
    let (s_ref, s_one, s_cones) = stability.unwrap();
    let c2 = s_cones <= 1.5 * s_ref;
    let c3 = cones_warm.frames.is_some() && ms(cones_warm.median) <= 16.7;
    let warm_frames = |w: &Warm| w.frames.map_or("not warm".to_string(), |f| f.to_string());
    writeln!(out, "# M7 gate 1 results (raw)\n").unwrap();
    writeln!(out, "{}, 1280×720.\n", kernel.adapter_name).unwrap();
    writeln!(
        out,
        "3. Textured scene with the cache, warm: one ray (control) {:.2} ms median, {:.2} ms pipelined (warm after {} frames); pixel cones **{:.2} ms** median, {:.2} ms pipelined (warm after {}).",
        ms(one_warm.median),
        ms(one_warm.pipelined),
        warm_frames(&one_warm),
        ms(cones_warm.median),
        ms(cones_warm.pipelined),
        warm_frames(&cones_warm),
    )
    .unwrap();
    writeln!(
        out,
        "\n1. Against the 16×16 reference (per-pixel lighting):\n"
    )
    .unwrap();
    writeln!(out, "| View | One ray | Pixel cones | Ratio | One ray ms | Pixel cones ms | Object nodes | Meshes | Triangle-hierarchy nodes | Set apart |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|").unwrap();
    for q in &quality {
        writeln!(
            out,
            "| {} | {:.2}% | {:.2}% | {:.3}× | {:.1} | {:.0} | {:.0} | {:.1} | {:.0} | {} |",
            q.name,
            100.0 * q.one,
            100.0 * q.cones,
            q.cones / q.one,
            q.one_ms,
            q.cones_ms,
            q.work[0],
            q.work[1],
            q.work[2],
            q.apart
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n2. Half a pixel sideways from 150 m: the reference changes {:.2}%, one ray {:.2}%, pixel cones {:.2}% ({:.2}×).",
        100.0 * s_ref,
        100.0 * s_one,
        100.0 * s_cones,
        s_cones / s_ref
    )
    .unwrap();
    writeln!(
        out,
        "\nVerdict: 1 {}, 2 {}, 3 {} → {}.",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m7_gate1_results.md"), &out)?;
    Ok(())
}
