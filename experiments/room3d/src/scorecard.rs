//! The basics scorecard's scenes and measurements (phase 1 of DESIGN §3.8),
//! shared by the `scorecard` binary and gates that measure the same way.
//! See `src/bin/scorecard.rs` for what is measured.

use std::fmt::Write as _;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::cycles::{relative_rmse, render_instanced, render_instanced_lights};
use crate::field::Field;
use crate::showcase::{Showcase, mover_poses};
use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedScene, Light, MaskConeKernel, RectLight, Renderer,
    RendererOptions, wgpu,
};

/// The speed half's size.
pub const W: u32 = 1280;
pub const H: u32 = 720;
/// The quality half's: Cycles' size, rendered 4×4 larger.
pub const QW: u32 = 640;
pub const QH: u32 = 360;
pub const SS: u32 = 4;
pub const MOTION_FRAMES: usize = 120;

/// The viewer's uniform sky.
pub fn viewer_sky() -> Vec3 {
    Vec3::new(0.6, 0.75, 1.0) * 0.3
}

#[derive(Clone, Copy)]
pub enum Motion {
    /// The eye turned about the vertical through the target by this many
    /// degrees.
    Orbit(f32),
    /// The eye walked this far (meters) along the view, horizontally, while
    /// the view turns this many degrees.
    Walk(f32, f32),
    /// The animation or the movers play; the camera holds.
    Animate,
}

pub enum Reference {
    /// Through Blender's glTF importer: the file, samples, suns (toward,
    /// irradiance), the viewer's ground (height under the origin, half
    /// width).
    Gltf {
        path: PathBuf,
        samples: u32,
        suns: Vec<(Vec3, Vec3)>,
        ground: Option<(f32, f32)>,
    },
    /// Instanced Cycles, one rect light, 4 bounces.
    Field { light: RectLight, radiance: f32 },
    /// M10 gate 25's reference: 12 bounces, 4,096 samples (with movers:
    /// the same, in a folder of its own).
    Room,
}

pub struct Case {
    pub name: &'static str,
    pub scene: InstancedScene,
    pub origin: DVec3,
    pub lights: Vec<Light>,
    pub sky: Option<Vec3>,
    pub camera: Camera,
    pub motion: Motion,
    /// The animations playing together, and their longest duration.
    pub animation: Option<(continuum_gltf::Animations, Vec<usize>, f32)>,
    pub deforming: Vec<usize>,
    /// The showcase's movers (instances), posed by `showcase::mover_poses`.
    pub movers: Vec<usize>,
    /// Bounces instead of the default (`CacheOptions::bounces`).
    pub bounces: Option<u32>,
    /// `CacheOptions::bounce_rest` instead of the default.
    pub bounce_rest: Option<bool>,
    pub reference: Reference,
}

impl Case {
    /// The camera `s` (0 to 1, eased) of the way through its move.
    pub fn camera_at(&self, s: f32) -> Camera {
        let c = self.camera;
        match self.motion {
            Motion::Orbit(deg) => {
                let r = Mat3::from_rotation_z((deg * s).to_radians());
                Camera {
                    eye: c.target + r * (c.eye - c.target),
                    ..c
                }
            }
            Motion::Walk(d, deg) => {
                let along = c.target - c.eye;
                let flat = Vec3::new(along.x, along.y, 0.0).normalize_or_zero();
                let eye = c.eye + flat * d * s;
                let r = Mat3::from_rotation_z((deg * s).to_radians());
                Camera {
                    eye,
                    target: eye + r * along,
                    ..c
                }
            }
            Motion::Animate => c,
        }
    }

    /// Poses the deforming geometries and the movers at `time` seconds.
    pub fn pose(&self, r: &mut Renderer, time: f32) {
        if let Some((animations, all, duration)) = &self.animation {
            for p in animations.deform(all, time % duration) {
                r.set_vertices(p.geometry, p.triangles, p.normals);
            }
        }
        for (&i, (rotation, position)) in self.movers.iter().zip(mover_poses(time * 60.0)) {
            r.set_transform(i, rotation, position.as_dvec3() + self.origin);
        }
    }

    /// The animation's time where the motion stops.
    pub fn stop_time(&self) -> Option<f32> {
        let t = MOTION_FRAMES as f32 / 60.0;
        match &self.animation {
            Some((_, _, d)) => Some(t % d),
            None => (!self.movers.is_empty()).then_some(t),
        }
    }

    pub fn renderer(&self, size: (u32, u32), mut cache_options: CacheOptions) -> Renderer {
        // Development: `SC_SKY_LAG=1` with `CacheOptions::sky_lag`.
        if std::env::var_os("SC_SKY_LAG").is_some() {
            cache_options.sky_lag = true;
        }
        if let Some(b) = self.bounces {
            cache_options.bounces = b;
        }
        if let Some(r) = self.bounce_rest {
            cache_options.bounce_rest = r;
        }
        // Development: `SC_BOUNCE_REST=1` for `CacheOptions::bounce_rest`.
        if std::env::var_os("SC_BOUNCE_REST").is_some() {
            cache_options.bounce_rest = true;
        }
        // Development: `SC_BOUNCE_TAIL` for `CacheOptions::bounce_tail`.
        if let Some(t) = std::env::var("SC_BOUNCE_TAIL")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            cache_options.bounce_tail = t;
        }
        // Development: `SC_BOUNCES` bounces instead of the default.
        if let Some(b) = std::env::var("SC_BOUNCES")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            cache_options.bounces = b;
        }
        let kernel = MaskConeKernel::new().expect("GPU");
        Renderer::new(
            kernel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            size,
            &self.scene,
            self.origin,
            &self.lights,
            RendererOptions {
                sky: self.sky,
                deforming: self.deforming.clone(),
                movable: self.movers.clone(),
                // Development: `SC_NOBOUNCE=1` without bounce light.
                bounce: std::env::var_os("SC_NOBOUNCE").is_none(),
                cache_options,
                ..Default::default()
            },
        )
    }
}

pub fn target(r: &Renderer, (w, h): (u32, u32)) -> wgpu::TextureView {
    r.kernel()
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

/// A glTF file, its sun and sky (or the viewer's ground, sun and sky for
/// an animated file), seen from `view` (eye and target from the scene's
/// center) or the viewer's default view.
pub fn gltf_case(
    name: &'static str,
    path: PathBuf,
    view: Option<(Vec3, Vec3, f32)>,
    motion: Motion,
    samples: u32,
) -> Case {
    let imported = continuum_gltf::load(&path).expect("the file (see DESIGN §3.8 for the assets)");
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    let animated = matches!(motion, Motion::Animate);
    let mut scene = imported.scene;
    let mut ground = None;
    let (toward, irradiance) = if animated {
        // The viewer's ground: 20 radii across, just below the scene.
        let h = (10.0 * radius) as f32;
        let c = [(-h, -h), (h, -h), (h, h), (-h, h)].map(|(x, y)| Vec3::new(x, y, 0.0));
        scene.geometries.push(Surface::from_triangles(vec![
            [c[0], c[1], c[2]],
            [c[0], c[2], c[3]],
        ]));
        scene.geometry_textures.resize(scene.geometries.len(), None);
        let z = lo.z - 1e-3 * radius;
        scene.instances.push(Instance {
            geometry: scene.geometries.len() - 1,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: DVec3::new(origin.x, origin.y, z),
            albedo: Vec3::splat(0.5),
        });
        ground = Some(((z - origin.z) as f32, h));
        let (az, el) = (200f64.to_radians(), 30f64.to_radians());
        (
            DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin()),
            Vec3::new(1.0, 0.92, 0.8) * std::f32::consts::PI,
        )
    } else {
        (DVec3::new(-0.6, -0.3, 0.75).normalize(), Vec3::splat(3.0))
    };
    let lights = vec![Light::sun(
        Vec3::ZERO,
        toward,
        radius * continuum_gpu::SUN_RADII,
        irradiance,
    )];
    let camera = match view {
        Some((eye, target, fov)) => Camera {
            eye,
            target,
            vertical_fov_degrees: fov,
        },
        None => {
            let (az, el) = (225f64.to_radians(), 25f64.to_radians());
            let dir = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
            Camera {
                eye: (dir * radius * 2.2).as_vec3(),
                target: Vec3::ZERO,
                vertical_fov_degrees: 40.0,
            }
        }
    };
    let animation = animated.then(|| {
        let all = imported.animations.together();
        let duration = all
            .iter()
            .map(|&k| imported.animations.list[k].duration)
            .fold(1e-3, f32::max);
        (imported.animations.clone(), all, duration)
    });
    let deforming = if animated {
        imported.animations.deforming()
    } else {
        Vec::new()
    };
    Case {
        name,
        scene,
        origin,
        lights,
        sky: Some(viewer_sky()),
        camera,
        motion,
        animation,
        deforming,
        movers: Vec::new(),
        bounces: None,
        bounce_rest: None,
        reference: Reference::Gltf {
            path,
            samples,
            suns: vec![(toward.as_vec3(), irradiance)],
            ground,
        },
    }
}

pub fn field_case() -> Case {
    let field = Field::new(1_000_000, DVec3::ZERO).with_sun_at(1000.0);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    Case {
        name: "field",
        lights: vec![Light::rect(light, Vec3::splat(field.radiance))],
        sky: Some(field.sky()),
        camera,
        motion: Motion::Walk(10.0, 20.0),
        animation: None,
        deforming: Vec::new(),
        movers: Vec::new(),
        bounces: None,
        bounce_rest: None,
        reference: Reference::Field {
            light,
            radiance: field.radiance,
        },
        origin,
        scene: field.scene,
    }
}

/// The showcase's static surfaces (M10 gate 25's room), or with its two
/// movers playing (the camera still).
pub fn room_case(with_movers: bool) -> Case {
    let s = Showcase::new();
    let n = s.first_mover;
    let scene = if with_movers {
        s.instanced()
    } else {
        InstancedScene {
            geometries: s.surfaces[..n].to_vec(),
            instances: (0..n)
                .map(|k| Instance {
                    geometry: k,
                    rotation: Mat3::IDENTITY,
                    scale: 1.0,
                    position: DVec3::ZERO,
                    albedo: s.albedo[k],
                })
                .collect(),
            ..Default::default()
        }
    };
    Case {
        name: if with_movers { "movers" } else { "room" },
        scene,
        origin: DVec3::ZERO,
        lights: vec![Light::white(s.light, s.radiance)],
        sky: None,
        camera: s.camera(0.0),
        motion: if with_movers {
            Motion::Animate
        } else {
            Motion::Walk(1.0, 25.0)
        },
        animation: None,
        deforming: Vec::new(),
        movers: if with_movers {
            vec![n, n + 1]
        } else {
            Vec::new()
        },
        bounces: None,
        bounce_rest: None,
        reference: Reference::Room,
    }
}

/// Stops the run after a frame of more than 5 s: one GPU submission that
/// long starves the WindowServer (with the display off, past its 40 s
/// watchdog, which logs the user out; 2026-11-22).
pub fn guard(start: Instant, what: &str) {
    let s = start.elapsed().as_secs_f64();
    if s > 5.0 {
        eprintln!("a frame took {s:.1} s ({what}): stopping the run");
        std::process::exit(2);
    }
}

pub fn median_p(v: &[f64], p: f64) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let at = ((s.len() as f64 * p) as usize).min(s.len() - 1);
    (s[s.len() / 2], s[at])
}

/// Renders still until settled (32 quiet frames) or `cap_s` seconds; the
/// frame and seconds where the quiet run began.
pub fn settle(
    r: &mut Renderer,
    camera: Camera,
    view: &wgpu::TextureView,
    cap_s: f64,
    mut on_frame: impl FnMut(&Renderer, usize, f64),
) -> Option<(usize, f64)> {
    let start = Instant::now();
    let mut stamps = Vec::new();
    for f in 1.. {
        let frame = Instant::now();
        r.render(camera, view);
        r.wait();
        guard(frame, "settling");
        let t = start.elapsed().as_secs_f64();
        stamps.push(t);
        on_frame(r, f, t);
        if f > 32 && r.is_settled(32) {
            return Some((f - 32, stamps[f - 33]));
        }
        if t > cap_s {
            return None;
        }
    }
    unreachable!()
}

pub struct Speed {
    pub load_min: f64,
    pub under_1: Option<(usize, f64)>,
    pub settled: Option<(usize, f64)>,
    pub frame: (f64, f64),
    pub motion: (f64, f64),
    pub motion_fallback: (f64, f64),
    pub stop_error: f64,
    pub resettled: Option<(usize, f64)>,
    pub gpu_mb: f64,
}

pub fn speed(case: &Case, run_start: Instant) -> Speed {
    speed_capped(case, run_start, 60.0)
}

/// [`speed`] with warm-up capped at `cap` seconds (and settling again after
/// the motion at half of it).
pub fn speed_capped(case: &Case, run_start: Instant, cap: f64) -> Speed {
    let load_min = run_start.elapsed().as_secs_f64() / 60.0;
    let defaults = CacheOptions::default();
    let mut r = case.renderer((W, H), defaults);
    let view = target(&r, (W, H));
    let pixels = (W * H) as f64;
    case.pose(&mut r, 0.0);
    let mut under_1 = None;
    let settled = settle(&mut r, case.camera, &view, cap, |r, f, t| {
        let share = r.stats().map_or(1.0, |s| s.fallback_pixels as f64 / pixels);
        if under_1.is_none() && f > 4 && share < 0.01 {
            under_1 = Some((f, t));
        }
    });
    let mut times = Vec::new();
    for _ in 0..60 {
        let t = Instant::now();
        r.render(case.camera, &view);
        r.wait();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let frame = median_p(&times, 0.9);
    let (mut times, mut shares) = (Vec::new(), Vec::new());
    for k in 1..=MOTION_FRAMES {
        let x = k as f32 / MOTION_FRAMES as f32;
        let camera = case.camera_at(x * x * (3.0 - 2.0 * x));
        let t = Instant::now();
        case.pose(&mut r, k as f32 / 60.0);
        r.render(camera, &view);
        r.wait();
        guard(t, "moving");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        shares.push(r.stats().map_or(1.0, |s| s.fallback_pixels as f64 / pixels));
        // Development: `SC_TRACE=1` prints the cache's work every 20 moving
        // frames.
        if std::env::var_os("SC_TRACE").is_some()
            && (k <= 6 || k % 20 == 0)
            && let Some(s) = r.stats()
        {
            println!(
                "  {} frame {k}: fallback {:.1}%, requests {}, baked {} pages, re-baked {} pages, candidates {}, invalidated {}, sky invalidated {}, sky listed {}, sky charged {}, sky re-baked {}, evicted {}, budget {}, missed {}, visible {}, deferred {}",
                case.name,
                100.0 * shares[k - 1],
                s.requests,
                s.pages_baked,
                s.rebaked_pages,
                s.candidates,
                s.invalidated,
                s.sky_invalidated,
                s.sky_listed,
                s.sky_charged,
                s.sky_rebaked,
                s.evicted,
                s.budget,
                s.missed,
                s.visible_pages,
                s.deferred
            );
        }
    }
    // Development: `SC_PROFILE=1` profiles ten more moving frames.
    if std::env::var_os("SC_PROFILE").is_some() {
        let mut bake = std::collections::BTreeMap::<&str, f64>::new();
        let mut passes = [0.0f64; 16];
        let (mut posing, mut rendering) = (0.0, 0.0);
        for k in 0..10 {
            let t = Instant::now();
            case.pose(&mut r, (MOTION_FRAMES + k) as f32 / 60.0);
            posing += t.elapsed().as_secs_f64() * 100.0;
            let t = Instant::now();
            r.render(case.camera_at(1.0), &view);
            r.wait();
            rendering += t.elapsed().as_secs_f64() * 100.0;
            case.pose(&mut r, (MOTION_FRAMES + k) as f32 / 60.0 + 0.5 / 60.0);
            let p = r.render_profiled(case.camera_at(1.0), &view);
            for (i, d) in p.passes.iter().enumerate() {
                passes[i] += d.as_secs_f64() * 100.0;
            }
            for (label, d) in p.bake_passes {
                if !label.is_empty() {
                    *bake.entry(label).or_insert(0.0) += d.as_secs_f64() * 100.0;
                }
            }
            if let Some(s) = r.stats() {
                println!(
                    "  {}: baked {} pages {} texels, re-baked {} pages {} texels, requests {}, budget {}, planning {:?}, waiting {:?}",
                    case.name,
                    s.pages_baked,
                    s.texels_baked,
                    s.rebaked_pages,
                    s.rebaked_texels,
                    s.requests,
                    s.budget,
                    s.planning,
                    s.waiting
                );
            }
        }
        let b: Vec<String> = bake.iter().map(|(k, t)| format!("{k} {t:.2}")).collect();
        let c: Vec<String> = passes
            .iter()
            .enumerate()
            .filter(|(_, t)| **t > 0.05)
            .map(|(i, t)| format!("{i}: {t:.2}"))
            .collect();
        println!(
            "  {}: posing {posing:.2} ms, render and wait {rendering:.2} ms a frame",
            case.name
        );
        println!("  {} bake passes (ms): {}", case.name, b.join(", "));
        println!("  {} camera passes (ms): {}", case.name, c.join(", "));
    }
    let motion = median_p(&times, 0.99);
    let motion_fallback = (
        100.0 * shares.iter().sum::<f64>() / shares.len() as f64,
        100.0 * shares.iter().copied().fold(0.0, f64::max),
    );
    let stopped = case.camera_at(1.0);
    let at_stop = r.read_colors();
    let resettled = settle(&mut r, stopped, &view, 0.5 * cap, |_, _, _| {});
    let settled_image = r.read_colors();
    let stop_error = relative_rmse(&at_stop, &settled_image);
    // Development: `SC_SAVE_STOP=1` saves both (out/scorecard).
    if std::env::var_os("SC_SAVE_STOP").is_some() {
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/scorecard");
        let m = settled_image
            .iter()
            .flatten()
            .map(|&x| x as f64)
            .sum::<f64>()
            / (3 * settled_image.len()) as f64;
        let e = (0.18 / m) as f32;
        save(
            &at_stop,
            (W, H),
            e,
            &out.join(format!("{}.stop.png", case.name)),
        );
        save(
            &settled_image,
            (W, H),
            e,
            &out.join(format!("{}.stop_settled.png", case.name)),
        );
    }
    Speed {
        load_min,
        under_1,
        settled,
        frame,
        motion,
        motion_fallback,
        stop_error,
        resettled,
        gpu_mb: r.gpu_bytes() as f64 / 1e6,
    }
}

pub struct Quality {
    pub rmse: f64,
    pub off: f64,
    pub settled: Option<usize>,
    pub mean_ours: f64,
    pub mean_cycles: f64,
}

/// The scene with its movers where the motion stopped.
pub fn posed(case: &Case) -> InstancedScene {
    let mut scene = case.scene.clone();
    if let Some(t) = case.stop_time() {
        for (&i, (rotation, position)) in case.movers.iter().zip(mover_poses(t * 60.0)) {
            scene.instances[i].rotation = rotation;
            scene.instances[i].position = position.as_dvec3() + case.origin;
        }
    }
    scene
}

/// The Cycles reference at 640×360, cached.
pub fn reference(case: &Case, out: &Path) -> std::io::Result<Vec<[f32; 3]>> {
    let camera = case.camera_at(1.0);
    match &case.reference {
        Reference::Room => Ok(render_instanced_lights(
            &posed(case),
            case.origin,
            &case.lights,
            Vec3::ZERO,
            camera,
            (QW, QH),
            4096,
            12,
            1,
            &if case.movers.is_empty() {
                out.parent().unwrap().join("m10_gate25_cycles")
            } else {
                out.join("cycles_movers")
            },
        )?
        .pixels),
        Reference::Field { light, radiance } => Ok(render_instanced(
            &case.scene,
            case.origin,
            *light,
            *radiance,
            case.sky.unwrap_or(Vec3::ZERO),
            camera,
            (QW, QH),
            1024,
            4,
            1,
            &out.join("cycles_field"),
        )?
        .pixels),
        Reference::Gltf {
            path,
            samples,
            suns,
            ground,
        } => {
            let v = |x: Vec3| format!("[{}, {}, {}]", x.x, x.y, x.z);
            let suns: Vec<String> = suns
                .iter()
                .map(|&(toward, irradiance)| {
                    format!(
                        "{{\"toward\": {}, \"irradiance\": {}, \"angle\": 0.5}}",
                        v(toward),
                        v(irradiance)
                    )
                })
                .collect();
            let sky = case.sky.unwrap_or(Vec3::ZERO);
            let mut job = format!(
                "\"gltf\": {:?}, \"origin\": [{}, {}, {}], \"camera\": {{\"eye\": {}, \"target\": {}, \"fov\": {}}}, \"size\": [{QW}, {QH}], \"samples\": {samples}, \"bounces\": 4, \"environment\": null, \"sky\": {}, \"suns\": [{}]",
                path.canonicalize()?.to_string_lossy(),
                case.origin.x,
                case.origin.y,
                case.origin.z,
                v(camera.eye),
                v(camera.target),
                camera.vertical_fov_degrees,
                v(sky),
                suns.join(", ")
            );
            if let Some(t) = case.stop_time() {
                write!(job, ", \"time\": {t}").unwrap();
            }
            if let Some((z, half)) = ground {
                write!(
                    job,
                    ", \"ground\": {{\"z\": {z}, \"half\": {half}, \"albedo\": 0.5}}"
                )
                .unwrap();
            }
            // Keyed on the job and the script: either changing renders anew.
            let script =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("cycles/render_gltf_reference.py");
            let mut hasher = DefaultHasher::new();
            job.hash(&mut hasher);
            std::fs::read(&script)?.hash(&mut hasher);
            let raw = out.join(format!("cycles_{}_{:016x}.f32", case.name, hasher.finish()));
            if !raw.exists() {
                let job_path = raw.with_extension("json");
                std::fs::write(
                    &job_path,
                    format!("{{{job}, \"out\": {:?}}}", raw.to_string_lossy()),
                )?;
                let t = Instant::now();
                let status =
                    std::process::Command::new("/Applications/Blender.app/Contents/MacOS/Blender")
                        .args(["-b", "--factory-startup", "-noaudio", "--python"])
                        .arg(&script)
                        .arg("--")
                        .arg(&job_path)
                        .stdout(std::process::Stdio::null())
                        .status()?;
                if !status.success() || !raw.exists() {
                    return Err(std::io::Error::other("Blender failed"));
                }
                println!(
                    "  Cycles reference for {}: {:.0} s",
                    case.name,
                    t.elapsed().as_secs_f64()
                );
            }
            let bytes = std::fs::read(&raw)?;
            Ok(bytes
                .as_chunks::<12>()
                .0
                .iter()
                .map(|c| {
                    std::array::from_fn(|k| {
                        f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap())
                    })
                })
                .collect())
        }
    }
}

pub fn save(image: &[[f32; 3]], (w, h): (u32, u32), exposure: f32, path: &Path) {
    let srgb = |x: f32| {
        let x = (x * exposure).clamp(0.0, 1.0);
        let s = if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };
    let bytes: Vec<u8> = image.iter().flat_map(|c| c.map(srgb)).collect();
    let file = std::fs::File::create(path).expect("the image");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut wr| wr.write_image_data(&bytes))
        .expect("the PNG");
}

pub fn quality(case: &Case, out: &Path) -> std::io::Result<Quality> {
    let size = (QW * SS, QH * SS);
    let mut r = case.renderer(
        size,
        CacheOptions {
            frame_target_ms: 0.0,
            bake_budget: 120_000,
            ..Default::default()
        },
    );
    let view = target(&r, size);
    if let Some(t) = case.stop_time() {
        case.pose(&mut r, t);
    }
    let camera = case.camera_at(1.0);
    let mut settled = None;
    for f in 1..=600 {
        let frame = Instant::now();
        r.render(camera, &view);
        r.wait();
        guard(frame, "quality");
        if f > 32 && r.is_settled(32) {
            settled = Some(f - 32);
            break;
        }
    }
    let colors = r.read_colors();
    drop(r);
    let ours: Vec<[f32; 3]> = (0..QW * QH)
        .map(|i| {
            let (x, y) = (i % QW, i / QW);
            let mut sum = [0.0f32; 3];
            for dy in 0..SS {
                for dx in 0..SS {
                    let c = colors[((y * SS + dy) * QW * SS + x * SS + dx) as usize];
                    (0..3).for_each(|k| sum[k] += c[k]);
                }
            }
            sum.map(|v| v / (SS * SS) as f32)
        })
        .collect();
    let reference = reference(case, out)?;
    let mean =
        |v: &[[f32; 3]]| v.iter().flatten().map(|&x| x as f64).sum::<f64>() / (3 * v.len()) as f64;
    let m = mean(&reference);
    let cap = (4.0 * m) as f32;
    let clamp = |v: &[[f32; 3]]| v.iter().map(|c| c.map(|x| x.min(cap))).collect::<Vec<_>>();
    let (oc, rc) = (clamp(&ours), clamp(&reference));
    let off = oc
        .iter()
        .zip(&rc)
        .filter(|(a, b)| (0..3).any(|k| ((a[k] - b[k]) as f64).abs() > 0.1 * m))
        .count();
    let exposure = (0.18 / m) as f32;
    save(
        &ours,
        (QW, QH),
        exposure,
        &out.join(format!("{}.png", case.name)),
    );
    save(
        &reference,
        (QW, QH),
        exposure,
        &out.join(format!("{}.cycles.png", case.name)),
    );
    // Ours minus Cycles: red brighter, blue darker, full scale at half the
    // mean.
    let diff: Vec<[f32; 3]> = oc
        .iter()
        .zip(&rc)
        .map(|(a, b)| {
            let d = ((a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2])) / 3.0 / (0.5 * m as f32);
            [d.max(0.0), 0.0, (-d).max(0.0)]
        })
        .collect();
    save(
        &diff,
        (QW, QH),
        1.0,
        &out.join(format!("{}.diff.png", case.name)),
    );
    Ok(Quality {
        rmse: relative_rmse(&oc, &rc),
        off: off as f64 / oc.len() as f64,
        settled,
        mean_ours: mean(&ours),
        mean_cycles: m,
    })
}

/// A scene's builder.
pub type Build = Box<dyn Fn() -> Case>;

/// The scorecard's scenes, by name, built on demand, from the test assets
/// under `assets` (`out/assets`).
pub fn cases(assets: &Path) -> Vec<(&'static str, Build)> {
    let sponza = assets.join("khronos/Models/Sponza/glTF/Sponza.gltf");
    let helmet = assets.join("khronos/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb");
    let (fox, brain) = (assets.join("glb/Fox.glb"), assets.join("glb/BrainStem.glb"));
    vec![
        (
            "sponza",
            Box::new(move || {
                gltf_case(
                    "sponza",
                    sponza.clone(),
                    Some((Vec3::new(5.0, 0.3, -3.4), Vec3::new(-10.0, 0.0, -2.0), 60.0)),
                    Motion::Walk(3.0, 30.0),
                    8192,
                )
            }),
        ),
        ("field", Box::new(field_case)),
        ("room", Box::new(|| room_case(false))),
        ("movers", Box::new(|| room_case(true))),
        (
            "helmet",
            Box::new(move || gltf_case("helmet", helmet.clone(), None, Motion::Orbit(40.0), 1024)),
        ),
        (
            "fox",
            Box::new(move || gltf_case("fox", fox.clone(), None, Motion::Animate, 1024)),
        ),
        (
            "brainstem",
            Box::new(move || gltf_case("brainstem", brain.clone(), None, Motion::Animate, 1024)),
        ),
    ]
}

/// One renderer of a pair being settled side by side: its own frames' times.
struct Side {
    r: Renderer,
    view: wgpu::TextureView,
    /// Its own frames' accumulated seconds after each frame.
    stamps: Vec<f64>,
    under_1: Option<(usize, f64)>,
    settled: Option<(usize, f64)>,
}

impl Side {
    fn new(case: &Case) -> Side {
        let mut r = case.renderer((W, H), CacheOptions::default());
        let view = target(&r, (W, H));
        case.pose(&mut r, 0.0);
        Side {
            r,
            view,
            stamps: Vec::new(),
            under_1: None,
            settled: None,
        }
    }

    /// One frame toward settling; true once settled.
    fn settle_step(&mut self, camera: Camera) -> bool {
        if self.settled.is_some() {
            return true;
        }
        let t = Instant::now();
        self.r.render(camera, &self.view);
        self.r.wait();
        guard(t, "settling");
        let own = self.stamps.last().copied().unwrap_or(0.0) + t.elapsed().as_secs_f64();
        self.stamps.push(own);
        let f = self.stamps.len();
        let share = self
            .r
            .stats()
            .map_or(1.0, |s| s.fallback_pixels as f64 / (W * H) as f64);
        if self.under_1.is_none() && f > 4 && share < 0.01 {
            self.under_1 = Some((f, own));
        }
        if f > 32 && self.r.is_settled(32) {
            self.settled = Some((f - 32, self.stamps[f - 33]));
        }
        self.settled.is_some()
    }

    fn frame(&mut self, case: &Case, camera: Camera, pose: Option<f32>) -> f64 {
        let t = Instant::now();
        if let Some(time) = pose {
            case.pose(&mut self.r, time);
        }
        self.r.render(camera, &self.view);
        self.r.wait();
        guard(t, "timed");
        t.elapsed().as_secs_f64() * 1e3
    }
}

/// Settles two renderers side by side, alternating frames until each is
/// settled or its own frames have taken `cap` seconds.
fn settle_pair(sides: &mut [Side; 2], camera: Camera, cap: f64) {
    loop {
        let mut busy = false;
        for s in sides.iter_mut() {
            if s.settled.is_none() && s.stamps.last().copied().unwrap_or(0.0) <= cap {
                s.settle_step(camera);
                busy = true;
            }
        }
        if !busy {
            break;
        }
    }
}

/// [`speed_capped`] for two configurations of a scene measured side by side
/// (M22 gate 7): both renderers alive, frames alternated, so the machine's
/// heat weighs on both alike. Warm-up and settling count each renderer's own
/// frames' time; the settled frames are six alternated blocks of ten; the
/// motion alternates frame by frame; the settling after it alternates as the
/// warm-up does (at half the cap).
pub fn speed_pair(a: &Case, b: &Case, run_start: Instant, cap: f64) -> [Speed; 2] {
    let load_min = run_start.elapsed().as_secs_f64() / 60.0;
    let cases = [a, b];
    let mut sides = [Side::new(a), Side::new(b)];
    settle_pair(&mut sides, a.camera, cap);
    let (under_1, settled) = (
        [sides[0].under_1, sides[1].under_1],
        [sides[0].settled, sides[1].settled],
    );
    let mut still = [Vec::new(), Vec::new()];
    for _ in 0..6 {
        for (k, s) in sides.iter_mut().enumerate() {
            for _ in 0..10 {
                still[k].push(s.frame(cases[k], cases[k].camera, None));
            }
        }
    }
    let mut moving = [Vec::new(), Vec::new()];
    let mut shares = [Vec::new(), Vec::new()];
    for f in 1..=MOTION_FRAMES {
        let x = f as f32 / MOTION_FRAMES as f32;
        for (k, s) in sides.iter_mut().enumerate() {
            let camera = cases[k].camera_at(x * x * (3.0 - 2.0 * x));
            moving[k].push(s.frame(cases[k], camera, Some(f as f32 / 60.0)));
            shares[k].push(
                s.r.stats()
                    .map_or(1.0, |st| st.fallback_pixels as f64 / (W * H) as f64),
            );
        }
    }
    let at_stop = [sides[0].r.read_colors(), sides[1].r.read_colors()];
    for s in sides.iter_mut() {
        s.stamps.clear();
        s.settled = None;
        s.under_1 = None;
    }
    settle_pair(&mut sides, a.camera_at(1.0), 0.5 * cap);
    let resettled = [sides[0].settled, sides[1].settled];
    std::array::from_fn(|k| {
        let settled_image = sides[k].r.read_colors();
        Speed {
            load_min,
            under_1: under_1[k],
            settled: settled[k],
            frame: median_p(&still[k], 0.9),
            motion: median_p(&moving[k], 0.99),
            motion_fallback: (
                100.0 * shares[k].iter().sum::<f64>() / shares[k].len() as f64,
                100.0 * shares[k].iter().copied().fold(0.0, f64::max),
            ),
            stop_error: relative_rmse(&at_stop[k], &settled_image),
            resettled: resettled[k],
            gpu_mb: sides[k].r.gpu_bytes() as f64 / 1e6,
        }
    })
}
