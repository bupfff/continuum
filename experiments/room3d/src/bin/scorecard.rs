//! The basics scorecard (phase 1 of DESIGN §3.8): a fixed set of scenes,
//! each measured the same way, run after each change that may move them.
//! Not a gate.
//!
//! Scenes: Sponza (the courtyard view, sun and sky), the million-boulder
//! field (its standing view, its sun at 1 km, its sky), the showcase's
//! closed room (one skylight panel, no sky), the same room with its two
//! movers (a bouncing ball, a spinning cube), DamagedHelmet (sun and sky),
//! and two animated characters (Fox, BrainStem: posed every frame over the
//! viewer's ground, under its sun and sky).
//!
//! **Speed** (first, for every scene, so timings are as cool as the
//! machine allows; each row states the minutes of load before it):
//! `Renderer` with its defaults at 1280×720, every frame waited for:
//! - warm-up: seconds (frames) from the first frame until under 1% of the
//!   pixels lack their page, and until settled (32 quiet frames; at most
//!   60 s);
//! - the settled frame: median and p90 of 60 frames;
//! - motion: 120 frames of a camera move (an orbit around an object, a
//!   walk and turn indoors and on the field) or of the animation or the
//!   movers playing (2 s at 60 fps): median and p99 frame, mean and worst share of pixels
//!   without their page; then held still: the image at the stop against
//!   the settled one (relative RMSE) and the seconds to settle again (at
//!   most 30 s);
//! - memory: `Renderer::gpu_bytes` after all that.
//!
//! **Quality:** the same scene at 2560×1440 with a fixed bake budget
//! (120,000 texels a frame), until settled or 600 frames, averaged 4×4
//! down to 640×360, against Cycles (Blender 5.2, cached under
//! `out/scorecard/`): relative RMSE with both clamped at 4× the reference's
//! mean (sun glints on smooth metal dominate otherwise), and the share of
//! pixels off by more than 10% of the mean in any channel. glTF files go
//! through Blender's own importer (`render_gltf_reference.py`); animated
//! characters and movers are compared at the pose where their motion
//! stopped.
//!
//! `SC_ONLY=sponza,field`: only these scenes. `SC_SPEED=0` / `SC_QUALITY=0`:
//! skip a half.
//!
//! `cargo run --release -p room3d --bin scorecard`

use std::fmt::Write as _;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedScene, Light, MaskConeKernel, RectLight, Renderer,
    RendererOptions, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced, render_instanced_lights};
use room3d::field::Field;
use room3d::showcase::{Showcase, mover_poses};

/// The speed half's size.
const W: u32 = 1280;
const H: u32 = 720;
/// The quality half's: Cycles' size, rendered 4×4 larger.
const QW: u32 = 640;
const QH: u32 = 360;
const SS: u32 = 4;
const MOTION_FRAMES: usize = 120;

/// The viewer's uniform sky.
fn viewer_sky() -> Vec3 {
    Vec3::new(0.6, 0.75, 1.0) * 0.3
}

#[derive(Clone, Copy)]
enum Motion {
    /// The eye turned about the vertical through the target by this many
    /// degrees.
    Orbit(f32),
    /// The eye walked this far (meters) along the view, horizontally, while
    /// the view turns this many degrees.
    Walk(f32, f32),
    /// The animation or the movers play; the camera holds.
    Animate,
}

enum Reference {
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

struct Case {
    name: &'static str,
    scene: InstancedScene,
    origin: DVec3,
    lights: Vec<Light>,
    sky: Option<Vec3>,
    camera: Camera,
    motion: Motion,
    /// The animations playing together, and their longest duration.
    animation: Option<(continuum_gltf::Animations, Vec<usize>, f32)>,
    deforming: Vec<usize>,
    /// The showcase's movers (instances), posed by `showcase::mover_poses`.
    movers: Vec<usize>,
    reference: Reference,
}

impl Case {
    /// The camera `s` (0 to 1, eased) of the way through its move.
    fn camera_at(&self, s: f32) -> Camera {
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
    fn pose(&self, r: &mut Renderer, time: f32) {
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
    fn stop_time(&self) -> Option<f32> {
        let t = MOTION_FRAMES as f32 / 60.0;
        match &self.animation {
            Some((_, _, d)) => Some(t % d),
            None => (!self.movers.is_empty()).then_some(t),
        }
    }

    fn renderer(&self, size: (u32, u32), cache_options: CacheOptions) -> Renderer {
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
                cache_options,
                ..Default::default()
            },
        )
    }
}

fn target(r: &Renderer, (w, h): (u32, u32)) -> wgpu::TextureView {
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
fn gltf_case(
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
        reference: Reference::Gltf {
            path,
            samples,
            suns: vec![(toward.as_vec3(), irradiance)],
            ground,
        },
    }
}

fn field_case() -> Case {
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
fn room_case(with_movers: bool) -> Case {
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
        reference: Reference::Room,
    }
}

fn median_p(v: &[f64], p: f64) -> (f64, f64) {
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
fn settle(
    r: &mut Renderer,
    camera: Camera,
    view: &wgpu::TextureView,
    cap_s: f64,
    mut on_frame: impl FnMut(&Renderer, usize, f64),
) -> Option<(usize, f64)> {
    let start = Instant::now();
    let mut stamps = Vec::new();
    for f in 1.. {
        r.render(camera, view);
        r.wait();
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

struct Speed {
    load_min: f64,
    under_1: Option<(usize, f64)>,
    settled: Option<(usize, f64)>,
    frame: (f64, f64),
    motion: (f64, f64),
    motion_fallback: (f64, f64),
    stop_error: f64,
    resettled: Option<(usize, f64)>,
    gpu_mb: f64,
}

fn speed(case: &Case, run_start: Instant) -> Speed {
    let load_min = run_start.elapsed().as_secs_f64() / 60.0;
    let defaults = CacheOptions::default();
    let mut r = case.renderer((W, H), defaults);
    let view = target(&r, (W, H));
    let pixels = (W * H) as f64;
    case.pose(&mut r, 0.0);
    let mut under_1 = None;
    let settled = settle(&mut r, case.camera, &view, 60.0, |r, f, t| {
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
        times.push(t.elapsed().as_secs_f64() * 1e3);
        shares.push(r.stats().map_or(1.0, |s| s.fallback_pixels as f64 / pixels));
    }
    let motion = median_p(&times, 0.99);
    let motion_fallback = (
        100.0 * shares.iter().sum::<f64>() / shares.len() as f64,
        100.0 * shares.iter().copied().fold(0.0, f64::max),
    );
    let stopped = case.camera_at(1.0);
    let at_stop = r.read_colors();
    let resettled = settle(&mut r, stopped, &view, 30.0, |_, _, _| {});
    let stop_error = relative_rmse(&at_stop, &r.read_colors());
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

struct Quality {
    rmse: f64,
    off: f64,
    settled: Option<usize>,
    mean_ours: f64,
    mean_cycles: f64,
}

/// The scene with its movers where the motion stopped.
fn posed(case: &Case) -> InstancedScene {
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
fn reference(case: &Case, out: &Path) -> std::io::Result<Vec<[f32; 3]>> {
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

fn save(image: &[[f32; 3]], (w, h): (u32, u32), exposure: f32, path: &Path) {
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

fn quality(case: &Case, out: &Path) -> std::io::Result<Quality> {
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
        r.render(camera, &view);
        r.wait();
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

fn main() -> std::io::Result<()> {
    let run_start = Instant::now();
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/scorecard");
    std::fs::create_dir_all(&out)?;
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/assets");
    let only: Option<Vec<String>> = std::env::var("SC_ONLY")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let wanted = |name: &str| only.as_ref().is_none_or(|o| o.iter().any(|n| n == name));
    type Build = Box<dyn Fn() -> Case>;
    let sponza = assets.join("khronos/Models/Sponza/glTF/Sponza.gltf");
    let helmet = assets.join("khronos/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb");
    let (fox, brain) = (assets.join("glb/Fox.glb"), assets.join("glb/BrainStem.glb"));
    let builds: Vec<(&str, Build)> = vec![
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
    ];
    let builds: Vec<_> = builds.into_iter().filter(|(n, _)| wanted(n)).collect();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let opt = |v: Option<(usize, f64)>| {
        v.map_or("not within the cap".to_string(), |(f, t)| {
            format!("{t:.2} s ({f})")
        })
    };
    if std::env::var("SC_SPEED").as_deref() != Ok("0") {
        say(format!(
            "## Speed ({W}×{H}, `Renderer` defaults)\n\n| Scene | Load before | Warm-up: under 1% / settled | Settled frame (median, p90) | Motion frame (median, p99) | Without page in motion (mean, worst) | Stop vs settled | Settles again | GPU memory |\n|---|---|---|---|---|---|---|---|---|"
        ));
        for (_, build) in &builds {
            let case = build();
            let s = speed(&case, run_start);
            say(format!(
                "| {} | {:.1} min | {} / {} | {:.2}, {:.2} ms | {:.2}, {:.2} ms | {:.2}%, {:.2}% | {:.2}% | {} | {:.0} MB |",
                case.name,
                s.load_min,
                opt(s.under_1),
                opt(s.settled),
                s.frame.0,
                s.frame.1,
                s.motion.0,
                s.motion.1,
                s.motion_fallback.0,
                s.motion_fallback.1,
                100.0 * s.stop_error,
                opt(s.resettled),
                s.gpu_mb
            ));
        }
    }
    if std::env::var("SC_QUALITY").as_deref() != Ok("0") {
        say(format!(
            "\n## Against Cycles ({QW}×{QH}, ours {SS}×{SS} supersampled at a fixed budget)\n\n| Scene | Clamped relative RMSE | Pixels off > 10% | Settled at | Mean: ours, Cycles |\n|---|---|---|---|---|"
        ));
        for (_, build) in &builds {
            let case = build();
            let q = quality(&case, &out)?;
            say(format!(
                "| {} | {:.2}% | {:.2}% | {} | {:.4}, {:.4} |",
                case.name,
                100.0 * q.rmse,
                100.0 * q.off,
                q.settled
                    .map_or("not in 600 frames".to_string(), |f| format!("frame {f}")),
                q.mean_ours,
                q.mean_cycles
            ));
        }
    }
    say(format!(
        "\nTotal {:.1} min.",
        run_start.elapsed().as_secs_f64() / 60.0
    ));
    std::fs::write(out.join("scorecard.md"), report)?;
    Ok(())
}
