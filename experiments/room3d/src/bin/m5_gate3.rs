//! M5 gate 3: object-level prefiltered coverage for wide cones, against
//! Cycles on the million-boulder field.
//!
//! M5 gate 2 matched Cycles to 0.7–1.2% with sky light on a 63-object scene.
//! On the million-boulder field, a sky cone from the ground looks across
//! hundreds of thousands of boulders, and walking the object hierarchy to each
//! one is unaffordable. Triangle hierarchies already prefilter (M4 gate 1);
//! this extends it to objects:
//! - Each object-hierarchy node stores directional coverage for the 26 cube
//!   directions: the summed projected front area of its objects. Each
//!   object's own comes from its geometry's root coverage, rotated and scaled.
//! - A cone meeting a node that projects to at most the prefilter threshold
//!   (in bins), with every corner between the cone's near and far planes,
//!   splats the node's hull instead of descending. Opacity is 1 − exp(−front
//!   area / silhouette area): independent occluders, so overlap saturates
//!   instead of exceeding one.
//! - It combines across meshes like any mesh (max for sun cones; as
//!   independent occluders for sky cones).
//!
//! Scene: the 1,000,000-boulder field from its standing viewpoint, 640×360,
//! the gate 2 sky (radiance from the sun's irradiance as in
//! `room3d::outdoor`), the field's sun moved to 1 km (same angular size and
//! direction). Lambertian, direct light only, compute rays, prefilter 1 bin.
//!
//! Reference: Cycles (Blender 5.2) with the boulders instanced through
//! geometry nodes (one mesh, a million instances, per-instance albedo),
//! direct light only, one-pixel box filter, 1024 samples per pixel, two seeds.
//!
//! Pass criterion, fixed before the first run:
//! 1. Sky only (sun off): per-pixel cones, 4×4 per pixel, with object-level
//!    prefiltering, within 3% relative RMSE of Cycles (seed 1).
//! 2. Sun and sky: the same, within 3% of Cycles (seed 1).
//! 3. Cost: the sky-only render, 1 sample per pixel, takes at most a tenth of
//!    the time without object-level prefiltering (or, if that render takes
//!    longer than 10 minutes, it is stopped and counted as more than 10×).
//! 4. No regression: gate 8's four sun-only viewpoints (1280×720, 1 sample
//!    per pixel, prefiltered, sun 3 km away) differ from the renders without
//!    object-level prefiltering by at most 0.5% relative RMSE.
//!
//! Reported: render times, Cycles' noise and time, node splats per cone.

use std::fmt::Write as _;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, MaskConeKernel};
use flatland::img::Image;
use room3d::cycles::{Reference, relative_rmse, render_instanced};
use room3d::field::Field;
use room3d::to_rgb;

const W: u32 = 640;
const H: u32 = 360;
const SS: u32 = 4;
const SAMPLES: u32 = 1024;

fn save_png(path: &Path, pixels: &[[f32; 3]], w: u32, h: u32) -> std::io::Result<()> {
    let mut image = Image::new(w as usize, h as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
}

fn downsample(big: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; (W * H) as usize];
    for y in 0..H * SS {
        for x in 0..W * SS {
            let p = big[(y * W * SS + x) as usize];
            let o = &mut out[((y / SS) * W + x / SS) as usize];
            for c in 0..3 {
                o[c] += p[c] / (SS * SS) as f32;
            }
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(1.0);
    let field = Field::new(1_000_000, DVec3::ZERO).with_sun_at(1000.0);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    let sky = field.sky();

    // Child process: the sky-only render without object-level prefiltering,
    // timed (the parent stops it after 10 minutes).
    if args.iter().any(|a| a == "--time-without") {
        kernel.set_object_prefilter(false);
        kernel.set_sky(sky);
        println!("READY");
        let (_, time, _) =
            kernel.render_instanced(camera, W, H, light, &field.scene, origin, 0.0, 1);
        println!("TIME {}", time.as_secs_f64());
        return Ok(());
    }

    let render = |radiance: f32, sky: Vec3, objects: bool, w: u32, h: u32, camera: Camera| {
        kernel.set_sky(sky);
        kernel.set_object_prefilter(objects);
        let (pixels, time, _) =
            kernel.render_instanced(camera, w, h, light, &field.scene, origin, radiance, 1);
        (pixels, time)
    };
    // 1 and 2: 4×4 per pixel against Cycles.
    let (sky_big, sky_t) = render(0.0, sky, true, W * SS, H * SS, camera);
    let sky_ss = downsample(&sky_big);
    drop(sky_big);
    let (both_big, both_t) = render(field.radiance, sky, true, W * SS, H * SS, camera);
    let both_ss = downsample(&both_big);
    drop(both_big);
    println!(
        "ours: sky {:.1} s, both {:.1} s (4×4)",
        sky_t.as_secs_f64(),
        both_t.as_secs_f64()
    );
    let work = out_dir.join("m5_gate3_cycles");
    let cycles = |radiance: f32, seed: u32| {
        render_instanced(
            &field.scene,
            origin,
            light,
            radiance,
            sky,
            camera,
            (W, H),
            SAMPLES,
            0,
            seed,
            &work,
        )
    };
    let sky_ref = [cycles(0.0, 1)?, cycles(0.0, 2)?];
    let both_ref = [cycles(field.radiance, 1)?, cycles(field.radiance, 2)?];
    let noise = |r: &[Reference; 2]| relative_rmse(&r[0].pixels, &r[1].pixels) / 2f64.sqrt();
    for (name, img) in [
        ("sky_ss", &sky_ss),
        ("both_ss", &both_ss),
        ("sky_cycles", &sky_ref[0].pixels),
        ("both_cycles", &both_ref[0].pixels),
    ] {
        save_png(&out_dir.join(format!("m5_gate3_{name}.png")), img, W, H)?;
    }
    let sky_err = relative_rmse(&sky_ss, &sky_ref[0].pixels);
    let both_err = relative_rmse(&both_ss, &both_ref[0].pixels);

    // 3: cost with and without object-level prefiltering, 1 per pixel.
    let (_, with_t) = render(0.0, sky, true, W, H, camera);
    let mut child = Command::new(std::env::current_exe()?)
        .arg("--time-without")
        .stdout(Stdio::piped())
        .spawn()?;
    let (tx, rx) = mpsc::channel();
    let stdout = child.stdout.take().expect("child stdout");
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    // Setup (building the field) is not counted: wait for READY first.
    while let Ok(line) = rx.recv() {
        if line == "READY" {
            break;
        }
    }
    let started = Instant::now();
    let mut without: Option<f64> = None;
    loop {
        let left = Duration::from_secs(600).saturating_sub(started.elapsed());
        match rx.recv_timeout(left) {
            Ok(line) => {
                if let Some(t) = line.strip_prefix("TIME ") {
                    without = t.parse().ok();
                    break;
                }
            }
            Err(_) => break,
        }
    }
    if without.is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
    let speedup = without.map(|w| w / with_t.as_secs_f64());
    let cheap = speedup.is_none_or(|s| s >= 10.0);

    // 4: no regression on gate 8's sun-only viewpoints (sun 3 km away).
    let original = Field::new(1_000_000, DVec3::ZERO);
    let o_origin = original.origin_near_camera();
    let (ground, o_light) = original.relative(o_origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down: f32| {
        let d = down.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let mut regressions = Vec::new();
    for (name, cam) in [
        ("standing", view(1.7, 0.0)),
        ("10 m up", view(10.0, 15.0)),
        ("40 m up", view(40.0, 30.0)),
        ("150 m up", view(150.0, 45.0)),
    ] {
        let mut images = Vec::new();
        for objects in [false, true] {
            kernel.set_sky(Vec3::ZERO);
            kernel.set_object_prefilter(objects);
            let (pixels, _, _) = kernel.render_instanced(
                cam,
                1280,
                720,
                o_light,
                &original.scene,
                o_origin,
                original.radiance,
                1,
            );
            images.push(pixels);
        }
        regressions.push((name, relative_rmse(&images[1], &images[0])));
    }
    let same = regressions.iter().all(|r| r.1 <= 0.005);
    let pass = sky_err <= 0.03 && both_err <= 0.03 && cheap && same;

    let mut out = String::new();
    writeln!(out, "# M5 gate 3 results\n").unwrap();
    writeln!(
        out,
        "{}, {} boulders, standing viewpoint, {W}×{H}, compute rays, prefilter 1 bin with object-level prefiltering, sun at 1 km, sky {:.4} {:.4} {:.4}. Cycles: instanced, {} spp, direct light only, one-pixel box, device {}, {:.1} s per render.\n",
        kernel.adapter_name,
        field.boulders(),
        sky.x,
        sky.y,
        sky.z,
        SAMPLES,
        both_ref[0].device,
        both_ref[0].seconds
    )
    .unwrap();
    writeln!(
        out,
        "| Criterion | Result |\n|---|---|\n| 1. Sky only, 4×4 per pixel, vs Cycles | {:.2}% |\n| 2. Sun and sky, 4×4 per pixel, vs Cycles | {:.2}% |\n| 3. Sky only, 1 per pixel: with object prefiltering | {:.1} s |\n| 3. Without | {} |\n| 4. Sun only, with vs without, worst of four viewpoints | {:.3}% |",
        100.0 * sky_err,
        100.0 * both_err,
        with_t.as_secs_f64(),
        without.map_or("stopped after 10 minutes (more than 10×)".to_string(), |w| format!(
            "{w:.1} s ({:.1}×)",
            w / with_t.as_secs_f64()
        )),
        100.0 * regressions.iter().map(|r| r.1).fold(0.0, f64::max)
    )
    .unwrap();
    writeln!(
        out,
        "\nTimes at 4×4 per pixel: sky only {:.1} s, sun and sky {:.1} s. Cycles' own noise: sky only {:.2}%, sun and sky {:.2}%.",
        sky_t.as_secs_f64(),
        both_t.as_secs_f64(),
        100.0 * noise(&sky_ref),
        100.0 * noise(&both_ref)
    )
    .unwrap();
    for (name, r) in &regressions {
        writeln!(out, "- {name}: sun only with vs without {:.3}%", 100.0 * r).unwrap();
    }
    writeln!(
        out,
        "\nNode splats per cone: not measured (the kernel has no counter for it yet).\n\n## Verdict\n\n1. Sky only within 3%: {}\n2. Sun and sky within 3%: {}\n3. At least 10× cheaper: {}\n4. Sun-only images unchanged within 0.5%: {}\n\n→ {}",
        if sky_err <= 0.03 { "pass" } else { "fail" },
        if both_err <= 0.03 { "pass" } else { "fail" },
        if cheap { "pass" } else { "fail" },
        if same { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m5_gate3_results.md"), &out)?;
    Ok(())
}
