//! M20 gate 1: depth of field.
//!
//! Method (DESIGN §3.4, "Depth of field"): `DepthOfField`, a thin lens
//! gathered each frame from the pinhole image and its depths: levels of
//! (colour, signed radius) merged toward the sharper side, highlights'
//! excess splatted exactly from lists per tile, a gather per pixel of a
//! centre tap, the pixels within 3.5 px and rings of 16 taps from coarser
//! levels in three groups (in front, its own surface, behind), every fourth
//! pixel gathered and the rest interpolated where the blur is wide and
//! smooth.
//!
//! Views (640×360, aperture radius 0.1 m, the largest blur 4% of the
//! height): (a) M6's textured scene with sun and sky, focused on the brick
//! block (9.1 m; the ground in front blurred); (b) the same focused at 5 m
//! (the distance blurred); (c) not used in development: the scene at night
//! with the glowing panel, foliage and the night lights (the point light
//! seen), focused at 3 m. Ours and the pinhole at 4×4 sub-pixel offsets,
//! each settled.
//!
//! References: (1) the lens sampled outright in our renderer: 16 × 16
//! aperture points by the concentric map, for each the eye moved to it and
//! the image plane sheared through the pixel offset so the plane in focus
//! stays put, settled, sub-pixel offsets cycling through 4 × 4; (2) Cycles'
//! thin lens (f-stop the focal length over twice the radius), 4,096
//! samples, one bounce.
//!
//! The bars were set with development numbers known (DESIGN §3.4: from the
//! sampled lens 0.129× and 0.134× the pinhole's error in views (b) and
//! (a); from Cycles 1.055× and 1.033× the sampled lens's error; +1.77 and
//! +1.71 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost:** depth of field adds at most 2.0 ms to the warm frame at
//!    1280×720 in views (a) and (b) (medians of 40 frames, five interleaved
//!    rounds; timed first).
//! 2. **Against the sampled lens:** in each view, ours within 0.2× the
//!    pinhole's error (relative RMSE).
//! 3. **Against Cycles:** in each view, ours at most 1.12× the sampled
//!    lens's error.
//!
//! Reported: each view's errors, means, highlights; the cost in view (c).
//!
//! Development mode (`M20_DEV=1`): 160×90, 2 × 2 aperture points, Cycles
//! at 16 samples, 10 frames a round: it checks the binary runs.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Camera, DepthOfField, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_dof};
use room3d::textured::Textured;

type Image = Vec<[f32; 3]>;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> bool {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return true;
        }
    }
    false
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct View {
    name: &'static str,
    scene: Textured,
    lens: DepthOfField,
}

fn views() -> Vec<View> {
    let lens = |focus_distance| DepthOfField {
        focus_distance,
        aperture_radius: 0.1,
        max_blur: 0.04,
    };
    vec![
        View {
            name: "(a) focused on the brick block",
            scene: Textured::new(),
            lens: lens(9.1),
        },
        View {
            name: "(b) focused at 5 m",
            scene: Textured::new(),
            lens: lens(5.0),
        },
        View {
            name: "(c) night, focused at 3 m",
            scene: Textured::new().with_foliage().with_glow(),
            lens: lens(3.0),
        },
    ]
}

fn target(kernel: &MaskConeKernel, (w, h): (u32, u32)) -> wgpu::TextureView {
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

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M20_DEV").is_some();
    let (w, h) = if dev { (160u32, 90u32) } else { (640, 360) };
    let timed_size = if dev { (320u32, 180u32) } else { (1280, 720) };
    let truth_n: u32 = if dev { 2 } else { 16 };
    let samples: u32 = if dev { 16 } else { 4096 };
    let frames = if dev { 10 } else { 40 };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        std::io::stdout().flush().ok();
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    say(format!(
        "M20 gate 1{}: {w}x{h}, timed at {}x{}, adapter {}",
        if dev { " (development mode)" } else { "" },
        timed_size.0,
        timed_size.1,
        kernel.adapter_name
    ));
    let make = |t: &Textured, size: (u32, u32), target: &wgpu::TextureView| {
        kernel.set_sky(t.sky);
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            size.0,
            size.1,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        );
        settle(&mut r, t.camera, target);
        r
    };
    let views = views();

    // 1. Cost, first (the machine coolest).
    let big = target(&kernel, timed_size);
    let mut c1 = true;
    for (k, v) in views.iter().enumerate() {
        let mut r = make(&v.scene, timed_size, &big);
        let timed = |r: &mut InstancedRenderer| {
            let mut t: Vec<f64> = (0..frames)
                .map(|_| {
                    let s = Instant::now();
                    r.frame(v.scene.camera, &big);
                    r.wait();
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            median(&mut t)
        };
        let (mut added, mut rounds) = (Vec::new(), String::new());
        for _ in 0..5 {
            r.set_depth_of_field(None);
            let plain = timed(&mut r);
            r.set_depth_of_field(Some(v.lens));
            let lensed = timed(&mut r);
            added.push(lensed - plain);
            rounds.push_str(&format!(" {plain:.3}/{lensed:.3}"));
        }
        let add = median(&mut added);
        let judged = k < 2;
        let ok = add <= 2.0;
        if judged {
            c1 &= ok;
        }
        say(format!(
            "1. cost, {}: +{add:.3} ms{}; highlights {:?} (rounds, without/with ms:{rounds})",
            v.name,
            if judged {
                format!(" (at most 2.0): {}", verdict(ok))
            } else {
                " (reported)".to_string()
            },
            r.depth_of_field_highlights()
        ));
    }

    // 2. and 3., each view.
    let small = target(&kernel, (w, h));
    let (mut c2, mut c3) = (true, true);
    let mut unsettled = 0;
    for v in &views {
        let t = &v.scene;
        let mut r = make(t, (w, h), &small);
        let started = Instant::now();
        let mut supersampled = |r: &mut InstancedRenderer| -> Image {
            let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
            r.set_footprint(0.25);
            for k in 0..16 {
                let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
                r.set_pixel_offset(dx, dy);
                if !settle(r, t.camera, &small) {
                    unsettled += 1;
                }
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
        let pinhole = supersampled(&mut r);
        r.set_depth_of_field(Some(v.lens));
        let ours = supersampled(&mut r);
        r.set_depth_of_field(None);
        // The lens sampled outright.
        let forward = (t.camera.target - t.camera.eye).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward);
        let f = h as f32 / (2.0 * (t.camera.vertical_fov_degrees.to_radians() * 0.5).tan());
        let mut sum = vec![[0.0f64; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        let n = truth_n;
        for k in 0..n * n {
            let (a, b) = (k / n, k % n);
            let cell = (k * 7 + k / 16) % 16;
            let (dx, dy) = (
                (cell % 4) as f32 * 0.25 - 0.375,
                (cell / 4) as f32 * 0.25 - 0.375,
            );
            let (u, s) = (
                2.0 * (a as f32 + 0.5) / n as f32 - 1.0,
                2.0 * (b as f32 + 0.5) / n as f32 - 1.0,
            );
            let (rad, phi) = if u.abs() > s.abs() {
                (u, std::f32::consts::FRAC_PI_4 * s / u)
            } else {
                (
                    s,
                    std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * u / s,
                )
            };
            let (ux, uy) = (
                rad * phi.cos() * v.lens.aperture_radius,
                rad * phi.sin() * v.lens.aperture_radius,
            );
            let shift = right * ux + up * uy;
            let camera = Camera {
                eye: t.camera.eye + shift,
                target: t.camera.target + shift,
                ..t.camera
            };
            r.set_pixel_offset(
                dx - ux * f / v.lens.focus_distance,
                dy + uy * f / v.lens.focus_distance,
            );
            if !settle(&mut r, camera, &small) {
                unsettled += 1;
            }
            r.finish();
            for (s, p) in sum.iter_mut().zip(r.read_colors()) {
                for c in 0..3 {
                    s[c] += p[c] as f64;
                }
            }
        }
        let truth: Image = sum
            .iter()
            .map(|p| p.map(|x| (x / (n * n) as f64) as f32))
            .collect();
        let ours_seconds = started.elapsed().as_secs_f64();
        let reference = render_instanced_dof(
            &t.scene,
            t.origin,
            &t.lights,
            t.sky,
            &v.lens,
            t.camera,
            (w, h),
            samples,
            1,
            1,
            &out_dir.join(format!(
                "m20_gate1_cycles_{}{}",
                v.name.chars().nth(1).unwrap_or('x'),
                if dev { "_dev" } else { "" }
            )),
        )?;
        let e_ours = relative_rmse(&ours, &truth);
        let e_pin = relative_rmse(&pinhole, &truth);
        let ok2 = e_ours <= 0.2 * e_pin;
        let cy_ours = relative_rmse(&ours, &reference.pixels);
        let cy_truth = relative_rmse(&truth, &reference.pixels);
        let cy_pin = relative_rmse(&pinhole, &reference.pixels);
        let ok3 = cy_ours <= 1.12 * cy_truth;
        c2 &= ok2;
        c3 &= ok3;
        let mean = |px: &[[f32; 3]]| {
            px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>() / (3 * px.len()) as f64
        };
        say(format!(
            "2. {}: from the sampled lens {:.2}%, the pinhole {:.2}% ({:.3}x, at most 0.2x): {}",
            v.name,
            100.0 * e_ours,
            100.0 * e_pin,
            e_ours / e_pin,
            verdict(ok2)
        ));
        say(format!(
            "3. {}: from Cycles ({:.0} s, {}) {:.2}%, the sampled lens {:.2}% ({:.3}x, at most 1.12x), the pinhole {:.2}%; means against Cycles: ours {:+.2}%, sampled lens {:+.2}%, pinhole {:+.2}% (ours took {ours_seconds:.0} s): {}",
            v.name,
            reference.seconds,
            reference.device,
            100.0 * cy_ours,
            100.0 * cy_truth,
            cy_ours / cy_truth,
            100.0 * cy_pin,
            100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0),
            100.0 * (mean(&truth) / mean(&reference.pixels) - 1.0),
            100.0 * (mean(&pinhole) / mean(&reference.pixels) - 1.0),
            verdict(ok3)
        ));
        let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
            let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                image.set(
                    i % w as usize,
                    i / w as usize,
                    room3d::to_rgb(Vec3::from_array(*p)),
                );
            }
            image.save(&out_dir.join(format!(
                "m20_gate1_{}_{name}{}.png",
                v.name.chars().nth(1).unwrap_or('x'),
                if dev { "_dev" } else { "" }
            )))
        };
        save("ours", &ours)?;
        save("pinhole", &pinhole)?;
        save("lens", &truth)?;
        save("cycles", &reference.pixels)?;
    }
    say(format!(
        "views that did not settle within 20,000 frames: {unsettled}"
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {}: {}",
        verdict(c1),
        verdict(c2),
        verdict(c3),
        verdict(c1 && c2 && c3)
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m20_gate1_dev.txt"
        } else {
            "m20_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
