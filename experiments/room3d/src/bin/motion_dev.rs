//! Development (M21): motion blur against the shutter sampled outright in
//! our renderer and against Cycles' motion blur, on the textured scene (sun
//! and sky). Not a gate.
//!
//! `MB_VIEW`: `truck` (the camera moving sideways 0.4 m a frame), `pan`
//! (turning 2° a frame), `movers` (six boulders crossing at 0.5 m a frame,
//! the camera still). `MB_W`/`MB_H` (640×360), `MB_SAMPLES` Cycles' samples
//! (1024), `MB_SHUTTER` (0.5), `MB_MAX` the longest smear (0.1 of the
//! height), `MB_TRUTH` the shutter's samples in our
//! renderer (64; 0: none), cached in `out/`. Ours and the still image at
//! 4×4 sub-pixel offsets, each settled. `time`: the cost at 1280×720.

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, MotionBlur, wgpu};
use room3d::cycles::{CyclesMotion, relative_rmse, render_instanced_motion};
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return;
        }
    }
    eprintln!("WARNING: not settled");
}

/// The camera turned about the vertical through its eye by `angle`.
fn turned(c: Camera, angle: f32) -> Camera {
    let r = Mat3::from_rotation_z(angle);
    Camera {
        target: c.eye + r * (c.target - c.eye),
        ..c
    }
}

fn main() -> std::io::Result<()> {
    let timing = std::env::args().any(|a| a == "time");
    let (w, h): (u32, u32) = if timing {
        (1280, 720)
    } else {
        (env("MB_W", 640), env("MB_H", 360))
    };
    let samples: u32 = env("MB_SAMPLES", 1024);
    let shutter: f32 = env("MB_SHUTTER", 0.5);
    let view: String = env("MB_VIEW", "truck".to_string());
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
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
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let t = Textured::new();
    kernel.set_sky(t.sky);
    let camera = t.camera;
    let forward = (camera.target - camera.eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    // The motion over one frame, as a function of time (0: now, -1: before).
    let movers: Vec<usize> = if view == "movers" {
        (1..=6).collect()
    } else {
        Vec::new()
    };
    let speed = DVec3::new(0.5, 0.0, 0.0);
    let camera_at = |time: f32| match view.as_str() {
        "truck" => Camera {
            eye: camera.eye + right * 0.4 * time,
            target: camera.target + right * 0.4 * time,
            ..camera
        },
        "pan" => turned(camera, 2f32.to_radians() * time),
        _ => camera,
    };
    let place = |r: &mut InstancedRenderer, time: f32| {
        for &k in &movers {
            let i = &t.scene.instances[k];
            r.set_instance_transform(k, i.rotation, i.position + speed * time as f64);
        }
    };
    let mut r = if movers.is_empty() {
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        )
    } else {
        InstancedRenderer::with_movers(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            Some(CacheOptions::default()),
            &movers,
            &[],
        )
    };
    let blur = MotionBlur {
        shutter,
        max_blur: env("MB_MAX", 0.1),
    };
    r.set_previous_camera(Some(camera_at(-1.0)));
    for &k in &movers {
        let i = &t.scene.instances[k];
        r.set_instance_previous_transform(k, i.rotation, i.position - speed);
    }
    settle(&mut r, camera, &target);
    if timing {
        let timed = |r: &mut InstancedRenderer| {
            let mut v: Vec<f64> = (0..40)
                .map(|_| {
                    let s = Instant::now();
                    r.frame(camera, &target);
                    r.wait();
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[20]
        };
        for _ in 0..3 {
            r.set_motion_blur(None);
            let plain = timed(&mut r);
            r.set_motion_blur(Some(blur));
            let blurred = timed(&mut r);
            println!(
                "1280x720 {view}: {plain:.3} ms, with motion blur {blurred:.3} ms (+{:.3})",
                blurred - plain
            );
        }
        return Ok(());
    }
    let supersampled = |r: &mut InstancedRenderer| {
        let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        for k in 0..16 {
            let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
            r.set_pixel_offset(dx, dy);
            settle(r, camera, &target);
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
    };
    let still = supersampled(&mut r);
    r.set_motion_blur(Some(blur));
    let ours = supersampled(&mut r);
    r.set_motion_blur(None);
    // The shutter sampled outright: n instants evenly across it, each
    // settled, sub-pixel offsets cycling through 4 × 4.
    let truth_n = env("MB_TRUTH", 64u32);
    let truth_file = out_dir.join(format!(
        "motion_dev_truth_{view}_{w}x{h}_{shutter}_{truth_n}.f32"
    ));
    let cached: Option<Vec<[f32; 3]>> = std::fs::read(&truth_file).ok().map(|b| {
        b.as_chunks::<12>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))
            })
            .collect()
    });
    let truth = match truth_n {
        0 => None,
        _ if cached.is_some() => cached,
        n => {
            let mut sum = vec![[0.0f64; 3]; (w * h) as usize];
            r.set_footprint(0.25);
            for k in 0..n {
                let time = ((k as f32 + 0.5) / n as f32 - 0.5) * shutter;
                let cell = (k * 7 + k / 16) % 16;
                r.set_pixel_offset(
                    (cell % 4) as f32 * 0.25 - 0.375,
                    (cell / 4) as f32 * 0.25 - 0.375,
                );
                place(&mut r, time);
                settle(&mut r, camera_at(time), &target);
                r.finish();
                for (s, p) in sum.iter_mut().zip(r.read_colors()) {
                    for c in 0..3 {
                        s[c] += p[c] as f64;
                    }
                }
            }
            place(&mut r, 0.0);
            let image: Vec<[f32; 3]> = sum
                .iter()
                .map(|p| p.map(|v| (v / n as f64) as f32))
                .collect();
            let bytes: Vec<u8> = image
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(&truth_file, bytes)?;
            Some(image)
        }
    };
    let motion = CyclesMotion {
        shutter,
        camera_before: camera_at(-1.0),
        movers: movers
            .iter()
            .map(|&k| {
                let i = &t.scene.instances[k];
                (k, i.rotation, i.position - speed)
            })
            .collect(),
    };
    let reference = render_instanced_motion(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        &motion,
        camera,
        (w, h),
        samples,
        1,
        1,
        &out_dir.join(format!("motion_dev_cycles_{view}")),
    )?;
    let mean = |px: &[[f32; 3]]| {
        px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>() / (3 * px.len()) as f64
    };
    if let Some(truth) = &truth {
        let e = |a: &[[f32; 3]]| relative_rmse(a, truth);
        println!(
            "  against the shutter sampled in our renderer: ours {:.2}%, still {:.2}% ({:.3}x); means: ours {:+.3}%; the sampled shutter against Cycles {:.2}%",
            100.0 * e(&ours),
            100.0 * e(&still),
            e(&ours) / e(&still),
            100.0 * (mean(&ours) / mean(truth) - 1.0),
            100.0 * relative_rmse(truth, &reference.pixels)
        );
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, (a, b)) in ours.iter().zip(truth.iter()).enumerate() {
            let v = |k: usize| ((0.5 + 4.0 * (a[k] - b[k])).clamp(0.0, 1.0) * 255.0) as u8;
            image.set(i % w as usize, i / w as usize, [v(0), v(1), v(2)]);
        }
        image.save(&out_dir.join(format!("motion_dev_{view}_truth_diff.png")))?;
    }
    println!(
        "{view} {w}x{h} (shutter {shutter}), Cycles {samples} samples ({:.0} s, {}): ours {:.2}%, still {:.2}% ({:.3}x); means: ours {:+.2}%, still {:+.2}%",
        reference.seconds,
        reference.device,
        100.0 * relative_rmse(&ours, &reference.pixels),
        100.0 * relative_rmse(&still, &reference.pixels),
        relative_rmse(&ours, &reference.pixels) / relative_rmse(&still, &reference.pixels),
        100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&still) / mean(&reference.pixels) - 1.0)
    );
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("motion_dev_{view}_{name}.png")))
    };
    save("ours", &ours)?;
    save("still", &still)?;
    save("cycles", &reference.pixels)?;
    if let Some(truth) = &truth {
        save("truth", truth)?;
    }
    Ok(())
}
