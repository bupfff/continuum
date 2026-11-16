//! M21 gate 1: motion blur.
//!
//! Method (DESIGN §3.4, "Motion blur"): `MotionBlur`, a shutter centred on
//! the frame: each pixel's segment from its hit point's motion since the
//! previous camera and movers' transforms; a gather along the tile's
//! fastest motion and the pixel's own, a tap covering the pixel when the
//! pixel lies on its segment, in three groups by depth.
//!
//! Views (640×360, the textured scene with sun and sky, shutter 0.5, the
//! longest smear 10% of the height): (a) the camera trucking sideways 0.4 m
//! a frame; (b) turning 2° a frame; (c) six boulders crossing at 0.5 m a
//! frame, the camera still; (d), not used in development: with foliage,
//! dollying forward 0.5 m a frame while turning 1°. Ours and the still
//! image at 4×4 sub-pixel offsets, each settled.
//!
//! References: (1) the shutter sampled outright in our renderer: 64
//! instants evenly across it, the camera and movers placed at each,
//! settled, sub-pixel offsets cycling through 4 × 4; (2) Cycles' motion
//! blur (the camera and moving instances keyed before, now and as far
//! again, linearly), 4,096 samples, one bounce.
//!
//! The bars were set with development numbers known (DESIGN §3.4: from the
//! sampled shutter 0.072×, 0.189× and 0.271× the still image's error; from
//! Cycles 0.998×, 1.029× and 1.025× the sampled shutter's error; means
//! within 0.06%; +1.44, +1.30 and +0.54 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost:** motion blur adds at most 1.7 ms to the warm frame at
//!    1280×720 in views (a) and (b), and at most 0.8 ms in view (c)
//!    (medians of 40 frames, five interleaved rounds; timed first).
//! 2. **Against the sampled shutter:** in each view, ours within 0.35× the
//!    still image's error (relative RMSE).
//! 3. **Against Cycles:** in each view, ours at most 1.12× the sampled
//!    shutter's error.
//! 4. **Light:** in each view, the mean of ours within 1% of the sampled
//!    shutter's.
//!
//! Reported: each view's errors and means against Cycles; the cost in view
//! (d).
//!
//! Development mode (`M21_DEV=1`): 160×90, 4 instants, Cycles at 16
//! samples, 10 frames a round: it checks the binary runs.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, MotionBlur, wgpu};
use room3d::cycles::{CyclesMotion, relative_rmse, render_instanced_motion};
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

/// The camera turned about the vertical through its eye by `angle`.
fn turned(c: Camera, angle: f32) -> Camera {
    let r = Mat3::from_rotation_z(angle);
    Camera {
        target: c.eye + r * (c.target - c.eye),
        ..c
    }
}

#[derive(Clone, Copy)]
enum Motion {
    /// Sideways by this many metres a frame.
    Truck(f32),
    /// Turning by this many degrees a frame.
    Turn(f32),
    /// Boulders 1 to 6 crossing at 0.5 m a frame along +x.
    Movers,
    /// Forward by this many metres a frame while turning by this many
    /// degrees.
    DollyTurn(f32, f32),
}

struct View {
    name: &'static str,
    scene: Textured,
    motion: Motion,
}

impl View {
    /// The camera at `time` frames from now.
    fn camera_at(&self, time: f32) -> Camera {
        let c = self.scene.camera;
        let forward = (c.target - c.eye).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        match self.motion {
            Motion::Truck(d) => Camera {
                eye: c.eye + right * d * time,
                target: c.target + right * d * time,
                ..c
            },
            Motion::Turn(a) => turned(c, a.to_radians() * time),
            Motion::Movers => c,
            Motion::DollyTurn(d, a) => turned(
                Camera {
                    eye: c.eye + forward * d * time,
                    target: c.target + forward * d * time,
                    ..c
                },
                a.to_radians() * time,
            ),
        }
    }

    fn movers(&self) -> Vec<usize> {
        match self.motion {
            Motion::Movers => (1..=6).collect(),
            _ => Vec::new(),
        }
    }
}

const SPEED: DVec3 = DVec3::new(0.5, 0.0, 0.0);

fn views() -> Vec<View> {
    vec![
        View {
            name: "(a) trucking sideways",
            scene: Textured::new(),
            motion: Motion::Truck(0.4),
        },
        View {
            name: "(b) turning",
            scene: Textured::new(),
            motion: Motion::Turn(2.0),
        },
        View {
            name: "(c) boulders crossing",
            scene: Textured::new(),
            motion: Motion::Movers,
        },
        View {
            name: "(d) dollying while turning, foliage",
            scene: Textured::new().with_foliage(),
            motion: Motion::DollyTurn(0.5, 1.0),
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
    let dev = std::env::var_os("M21_DEV").is_some();
    let (w, h) = if dev { (160u32, 90u32) } else { (640, 360) };
    let timed_size = if dev { (320u32, 180u32) } else { (1280, 720) };
    let instants: u32 = if dev { 4 } else { 64 };
    let samples: u32 = if dev { 16 } else { 4096 };
    let frames = if dev { 10 } else { 40 };
    let shutter = 0.5f32;
    let blur = MotionBlur {
        shutter,
        max_blur: 0.1,
    };
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
        "M21 gate 1{}: {w}x{h}, timed at {}x{}, adapter {}",
        if dev { " (development mode)" } else { "" },
        timed_size.0,
        timed_size.1,
        kernel.adapter_name
    ));
    // A renderer for `v` at `size`, its previous poses set, settled now.
    let make = |v: &View, size: (u32, u32), target: &wgpu::TextureView| {
        let t = &v.scene;
        kernel.set_sky(t.sky);
        let movers = v.movers();
        let mut r = if movers.is_empty() {
            InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                size.0,
                size.1,
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
                size.0,
                size.1,
                1,
                &t.lights,
                &t.scene,
                t.origin,
                Some(CacheOptions::default()),
                &movers,
                &[],
            )
        };
        r.set_previous_camera(Some(v.camera_at(-1.0)));
        for &k in &movers {
            let i = &t.scene.instances[k];
            r.set_instance_previous_transform(k, i.rotation, i.position - SPEED);
        }
        settle(&mut r, t.camera, target);
        r
    };
    let views = views();

    // 1. Cost, first (the machine coolest).
    let big = target(&kernel, timed_size);
    let mut c1 = true;
    for (k, v) in views.iter().enumerate() {
        let mut r = make(v, timed_size, &big);
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
            r.set_motion_blur(None);
            let plain = timed(&mut r);
            r.set_motion_blur(Some(blur));
            let blurred = timed(&mut r);
            added.push(blurred - plain);
            rounds.push_str(&format!(" {plain:.3}/{blurred:.3}"));
        }
        let add = median(&mut added);
        let bar = [Some(1.7), Some(1.7), Some(0.8), None][k];
        let ok = bar.is_none_or(|b| add <= b);
        c1 &= ok;
        say(format!(
            "1. cost, {}: +{add:.3} ms{} (rounds, without/with ms:{rounds})",
            v.name,
            match bar {
                Some(b) => format!(" (at most {b}): {}", verdict(ok)),
                None => " (reported)".to_string(),
            }
        ));
    }

    // 2. to 4., each view.
    let small = target(&kernel, (w, h));
    let (mut c2, mut c3, mut c4) = (true, true, true);
    let mut unsettled = 0;
    for v in &views {
        let t = &v.scene;
        let movers = v.movers();
        let mut r = make(v, (w, h), &small);
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
        let still = supersampled(&mut r);
        r.set_motion_blur(Some(blur));
        let ours = supersampled(&mut r);
        r.set_motion_blur(None);
        // The shutter sampled outright.
        let mut sum = vec![[0.0f64; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        for k in 0..instants {
            let time = ((k as f32 + 0.5) / instants as f32 - 0.5) * shutter;
            let cell = (k * 7 + k / 16) % 16;
            r.set_pixel_offset(
                (cell % 4) as f32 * 0.25 - 0.375,
                (cell / 4) as f32 * 0.25 - 0.375,
            );
            for &m in &movers {
                let i = &t.scene.instances[m];
                r.set_instance_transform(m, i.rotation, i.position + SPEED * time as f64);
            }
            if !settle(&mut r, v.camera_at(time), &small) {
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
            .map(|p| p.map(|x| (x / instants as f64) as f32))
            .collect();
        let ours_seconds = started.elapsed().as_secs_f64();
        let motion = CyclesMotion {
            shutter,
            camera_before: v.camera_at(-1.0),
            movers: movers
                .iter()
                .map(|&m| {
                    let i = &t.scene.instances[m];
                    (m, i.rotation, i.position - SPEED)
                })
                .collect(),
        };
        let tag = v.name.chars().nth(1).unwrap_or('x');
        let reference = render_instanced_motion(
            &t.scene,
            t.origin,
            &t.lights,
            t.sky,
            &motion,
            t.camera,
            (w, h),
            samples,
            1,
            1,
            &out_dir.join(format!(
                "m21_gate1_cycles_{tag}{}",
                if dev { "_dev" } else { "" }
            )),
        )?;
        let mean = |px: &[[f32; 3]]| {
            px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>() / (3 * px.len()) as f64
        };
        let e_ours = relative_rmse(&ours, &truth);
        let e_still = relative_rmse(&still, &truth);
        let ok2 = e_ours <= 0.35 * e_still;
        let cy_ours = relative_rmse(&ours, &reference.pixels);
        let cy_truth = relative_rmse(&truth, &reference.pixels);
        let cy_still = relative_rmse(&still, &reference.pixels);
        let ok3 = cy_ours <= 1.12 * cy_truth;
        let light = mean(&ours) / mean(&truth) - 1.0;
        let ok4 = light.abs() <= 0.01;
        c2 &= ok2;
        c3 &= ok3;
        c4 &= ok4;
        say(format!(
            "2. {}: from the sampled shutter {:.2}%, the still image {:.2}% ({:.3}x, at most 0.35x): {}",
            v.name,
            100.0 * e_ours,
            100.0 * e_still,
            e_ours / e_still,
            verdict(ok2)
        ));
        say(format!(
            "3. {}: from Cycles ({:.0} s, {}) {:.2}%, the sampled shutter {:.2}% ({:.3}x, at most 1.12x), the still image {:.2}%; means against Cycles: ours {:+.2}%, sampled shutter {:+.2}%, still {:+.2}% (ours took {ours_seconds:.0} s): {}",
            v.name,
            reference.seconds,
            reference.device,
            100.0 * cy_ours,
            100.0 * cy_truth,
            cy_ours / cy_truth,
            100.0 * cy_still,
            100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0),
            100.0 * (mean(&truth) / mean(&reference.pixels) - 1.0),
            100.0 * (mean(&still) / mean(&reference.pixels) - 1.0),
            verdict(ok3)
        ));
        say(format!(
            "4. {}: the mean of ours {:+.3}% from the sampled shutter's (within 1%): {}",
            v.name,
            100.0 * light,
            verdict(ok4)
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
                "m21_gate1_{tag}_{name}{}.png",
                if dev { "_dev" } else { "" }
            )))
        };
        save("ours", &ours)?;
        save("still", &still)?;
        save("shutter", &truth)?;
        save("cycles", &reference.pixels)?;
    }
    say(format!(
        "views that did not settle within 20,000 frames: {unsettled}"
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {}, 4 {}: {}",
        verdict(c1),
        verdict(c2),
        verdict(c3),
        verdict(c4),
        verdict(c1 && c2 && c3 && c4)
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m21_gate1_dev.txt"
        } else {
            "m21_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
