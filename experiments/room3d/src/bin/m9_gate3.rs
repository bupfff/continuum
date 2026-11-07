//! M9 gate 3: moving objects under a sky, through the error ledger.
//!
//! The cache refuses movers with a sky: a sky cone sees the whole
//! hemisphere, so exact invalidation would re-bake nearly every page each
//! frame. This is the ledger's case: bounded deferral within ε.
//!
//! Method:
//! - **Light channels** stay exact (gate 1's stale pass and same-frame
//!   re-bake).
//! - **Charges.** Each frame something moves, every page is charged, for
//!   each moved object, the solid angle its old and its new bounding sphere
//!   subtend from the nearest point of the page's box (π if inside): a bound
//!   on how much the move can change the page's sky light per unit sky
//!   radiance. Charges accumulate in the page's ledger until it re-bakes.
//! - **ε.** A page whose ledger passes half of ε × max(its smallest sky
//!   value at the bake, 0.05π) is listed for the CPU to re-bake its sky
//!   channel; one that passes ε is marked invalid (shown missing, evicted),
//!   so no shown sky value is further than ε from a re-bake.
//! - **Pages on moved objects** keep their exact light channels; their sky
//!   shows the cache's fallback guess, flagged, until a sky re-bake after
//!   they stop.
//!
//! Scene: the million-boulder field with M5 gate 6's three colored lights
//! and the field's sky, gate 1's 65 movers and motion, 1280×720, hardware
//! camera rays, cache defaults, ε = 1%. Flight: M5 gate 8's path.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Soundness:** over the first 300 frames, after every 10th, for every
//!    page shading may read (not invalid, not flagged): its light channels
//!    equal a re-bake bit for bit, and its sky channel's largest change is at
//!    most its ledger (plus 1e-5) and at most its ε threshold. Zero
//!    violations in 30 checks.
//! 2. **Work:** pages whose sky re-baked for movers average at most 25% of
//!    the pages charged at all (which exact invalidation would re-bake).
//! 3. **Speed** (first, cool): three flights with movers alternating with
//!    three with them still, sky on in both: each moving flight's mean at
//!    most 1.25× the still flights' mean and at most 20 ms, 99th percentile
//!    at most 40 ms, fallback (with flagged sky) at most 10%.
//!
//! Control, before anything else: gate 1's sun-only warm frame at the
//! standing viewpoint (no sky, static) at most 3.3 ms, or the run is void.
//!
//! Changed before the first full-size run, after development runs on the
//! GPU test's 160×90 scene (`movers_under_a_sky_stay_within_the_ledger`,
//! disclosed in the results):
//! - **Charges telescope.** Summing every frame's old and new spheres
//!   charged a mover's whole solid angle twice per frame, so pages near a
//!   mover passed ε within a few frames. Since a page's bake, its sky can
//!   only have changed where a mover covered it at the bake or covers it
//!   now, so a page is charged, per mover moved since its bake, that
//!   mover's sphere's solid angle at the bake (once) plus its solid angle
//!   now (kept current). Still a bound; much tighter. Movers beyond the
//!   96th keep the per-frame sum.
//! - **Same-frame sky re-bakes.** A page read in the frame before whose
//!   ledger passes ε re-bakes its sky in the same frame (the stale pass
//!   appends it, with its sky samples, as gate 1 appends stale light);
//!   only pages not read are marked invalid. Listing at half of ε for the
//!   CPU is dropped for these pages: a development run at 640×360 showed
//!   67% fallback, pages near movers passing ε within one frame.
//! - A sky-only re-bake clears a sky guess and leaves any other mark (stale
//!   light) to the light channels' bakes; pages the stale pass appends for
//!   light re-bake light channels only.
//! - A moved object's page is listed once per sky bake and its sky
//!   re-baked by the CPU 30 frames later (the method said "after they
//!   stop").
//!
//! `cargo run --release -p room3d --bin m9_gate3`
//!
//! `M9_DEV=1` runs a small development version (640×360, short flights),
//! whose numbers are not the gate's.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, CacheStats, Camera, Instance, InstancedRenderer, InstancedScene, Light,
    MaskConeKernel, wgpu,
};
use flatland::img::Image;
use room3d::field::Field;

struct Size {
    w: u32,
    h: u32,
    /// Frames per flight, frames of the exactness run, flights of each kind.
    flight: usize,
    checked: usize,
    flights: usize,
}

const GATE: Size = Size {
    w: 1280,
    h: 720,
    flight: 600,
    checked: 300,
    flights: 3,
};

const DEV: Size = Size {
    w: 640,
    h: 360,
    flight: 200,
    checked: 60,
    flights: 1,
};

/// The movers' motion: each boulder circles a center 1.5 m from its start
/// once every 240 frames, turning 1/60 radian per frame; the box drives
/// along the flight at 0.7 m per frame.
struct Motion {
    /// Per mover: instance, start position, start rotation, phase (boulders)
    /// or `None` (the box).
    movers: Vec<(usize, DVec3, Mat3, Option<f64>)>,
    flat: DVec3,
}

impl Motion {
    fn place(&self, k: usize, frame: usize) -> (Mat3, DVec3) {
        let (_, start, rotation, phase) = self.movers[k];
        let f = frame as f64;
        match phase {
            Some(phi) => {
                let a = phi + std::f64::consts::TAU * f / 240.0;
                let center = start - DVec3::new(phi.cos(), phi.sin(), 0.0) * 1.5;
                (
                    Mat3::from_rotation_z(frame as f32 / 60.0) * rotation,
                    center + DVec3::new(a.cos(), a.sin(), 0.0) * 1.5,
                )
            }
            None => (rotation, start + self.flat * 0.7 * f),
        }
    }

    fn apply(&self, r: &mut InstancedRenderer, frame: usize) {
        for k in 0..self.movers.len() {
            let (rotation, position) = self.place(k, frame);
            r.set_instance_transform(self.movers[k].0, rotation, position);
        }
    }

    fn instances(&self) -> Vec<usize> {
        self.movers.iter().map(|m| m.0).collect()
    }
}

struct Warm {
    frames: Option<usize>,
    frame_time: Duration,
}

/// Gate 8's warm frame: render until 16 frames ask for nothing, then time 60.
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
    Warm {
        frames: warm_from,
        frame_time: start.elapsed() / 60,
    }
}

struct Flight {
    mean: Duration,
    p99: Duration,
    worst: Duration,
    fallback: f64,
    budget: f64,
    gpu_bake: f64,
    gpu_camera: f64,
    /// Over frames where something moved: mean of re-baked / visible pages,
    /// and means of re-baked pages, invalidated pages, missed pages and
    /// visible pages per frame.
    rebaked_share: f64,
    rebaked: f64,
    invalidated: f64,
    missed: f64,
    visible: f64,
    /// Over frames where something moved: mean of sky re-baked / charged
    /// pages; means of sky re-baked, charged and sky-invalidated pages.
    sky_share: f64,
    sky_rebaked: f64,
    sky_charged: f64,
    sky_invalidated: f64,
}

fn summarize(times: &[Duration], total: Duration, log: &[CacheStats], pixels: u32) -> Flight {
    let mut sorted = times.to_vec();
    sorted.sort();
    let n = log.len() as f64;
    let mean = |f: &dyn Fn(&CacheStats) -> f64| log.iter().map(f).sum::<f64>() / n;
    let moved: Vec<&CacheStats> = log.iter().filter(|s| s.moved > 0).collect();
    let m = moved.len().max(1) as f64;
    let moved_mean = |f: &dyn Fn(&CacheStats) -> f64| moved.iter().map(|s| f(s)).sum::<f64>() / m;
    Flight {
        mean: total / times.len() as u32,
        p99: sorted[(sorted.len() * 99) / 100 - 1],
        worst: *sorted.last().unwrap(),
        fallback: mean(&|s| s.fallback_pixels as f64 / pixels as f64),
        budget: mean(&|s| s.budget as f64),
        gpu_bake: mean(&|s| s.gpu_bake_ms as f64),
        gpu_camera: mean(&|s| s.gpu_render_ms as f64),
        rebaked_share: moved_mean(&|s| s.rebaked_pages as f64 / s.visible_pages.max(1) as f64),
        rebaked: moved_mean(&|s| s.rebaked_pages as f64),
        invalidated: moved_mean(&|s| s.invalidated as f64),
        missed: moved_mean(&|s| s.missed as f64),
        visible: moved_mean(&|s| s.visible_pages as f64),
        sky_share: moved_mean(&|s| s.sky_rebaked as f64 / s.sky_charged.max(1) as f64),
        sky_rebaked: moved_mean(&|s| s.sky_rebaked as f64),
        sky_charged: moved_mean(&|s| s.sky_charged as f64),
        sky_invalidated: moved_mean(&|s| s.sky_invalidated as f64),
    }
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M9_DEV").is_some();
    let size = if dev { DEV } else { GATE };
    if dev {
        println!(
            "DEVELOPMENT RUN ({}×{}): not the gate's numbers",
            size.w, size.h
        );
    }
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
    let lights = field.colored_lights(origin, 1000.0);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: size.w,
                height: size.h,
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

    // Gate 8's flight path.
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
    let flight_camera = |f: usize| {
        let t = f as f32 / (GATE.flight - 1) as f32;
        view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t)
    };

    // The movers: the 64 boulders nearest points spaced evenly from 15 m to
    // 450 m ahead of the camera's start (one per point), and a box 6 m to
    // the flight's left, 20 m ahead.
    let start = origin + ground.eye.as_dvec3();
    let flat_d = flat.as_dvec3();
    let left = DVec3::Z.cross(flat_d);
    let foot = DVec3::new(start.x, start.y, 0.0);
    let mut scene: InstancedScene = field.scene.clone();
    let mut taken = std::collections::HashSet::new();
    let mut movers: Vec<(usize, DVec3, Mat3, Option<f64>)> = Vec::new();
    for j in 0..64 {
        let point = foot + flat_d * (15.0 + 435.0 * j as f64 / 63.0);
        let k = (1..scene.instances.len())
            .filter(|k| !taken.contains(k))
            .min_by(|&a, &b| {
                let d = |k: usize| (scene.instances[k].position - point).truncate().length();
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        taken.insert(k);
        let i = &scene.instances[k];
        movers.push((
            k,
            i.position,
            i.rotation,
            Some(std::f64::consts::TAU * j as f64 / 64.0),
        ));
    }
    scene.geometries.push(Surface::cuboid(
        Vec3::new(-2.0, -1.0, 0.0),
        Vec3::new(2.0, 1.0, 1.6),
    ));
    let box_rotation = Mat3::from_rotation_z(flat.y.atan2(flat.x));
    let box_start = foot + flat_d * 20.0 + left * 6.0;
    scene.instances.push(Instance {
        geometry: scene.geometries.len() - 1,
        rotation: box_rotation,
        scale: 1.0,
        position: box_start,
        albedo: Vec3::new(0.65, 0.25, 0.2),
    });
    movers.push((scene.instances.len() - 1, box_start, box_rotation, None));
    let motion = Motion {
        movers,
        flat: flat_d,
    };
    let movable = motion.instances();
    let renderer = || {
        InstancedRenderer::with_movers(
            &kernel,
            format,
            size.w,
            size.h,
            1,
            &lights,
            &scene,
            origin,
            Some(CacheOptions::default()),
            &movable,
            &[],
        )
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    // Control: the sun-only warm frame at the standing viewpoint, static.
    let sky = field.sky();
    let control = {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            size.w,
            size.h,
            1,
            &[sun],
            &field.scene,
            origin,
            CacheOptions::default(),
        );
        warm(&mut r, ground, &target)
    };
    kernel.set_sky(sky);
    let fit = ms(control.frame_time) <= 3.3;
    say(format!(
        "control: sun-only warm frame {:.2} ms (at most 3.3 ms) after {:?} frames: {}",
        ms(control.frame_time),
        control.frames,
        if fit {
            "fit"
        } else {
            "NOT FIT: the run is void"
        }
    ));
    if !fit && !dev {
        return Ok(());
    }

    // 3. Speed, first: flights alternating, movers still and moving.
    let fly = |moving: bool| {
        let mut r = renderer();
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            if moving {
                motion.apply(&mut r, f);
            }
            r.frame(flight_camera(f * GATE.flight / size.flight), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let total = begin.elapsed();
        summarize(&times, total, r.cache_log(), size.w * size.h)
    };
    let mut still = Vec::new();
    let mut moving = Vec::new();
    for _ in 0..size.flights {
        still.push(fly(false));
        moving.push(fly(true));
    }
    let line = |label: &str, f: &Flight| {
        format!(
            "{label}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, budget {:.0}, GPU bake {:.2} ms, camera {:.2} ms; re-baked {:.1} of {:.0} visible pages per frame ({:.2}%), invalidated {:.1}, missed {:.1}; sky re-baked {:.1} of {:.0} charged ({:.2}%), sky-invalidated {:.1}",
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            100.0 * f.fallback,
            f.budget,
            f.gpu_bake,
            f.gpu_camera,
            f.rebaked,
            f.visible,
            100.0 * f.rebaked_share,
            f.invalidated,
            f.missed,
            f.sky_rebaked,
            f.sky_charged,
            100.0 * f.sky_share,
            f.sky_invalidated
        )
    };
    for (k, (a, b)) in still.iter().zip(&moving).enumerate() {
        say(line(&format!("flight {}, movers still", k + 1), a));
        say(line(&format!("flight {}, movers moving", k + 1), b));
    }
    let control_mean = still.iter().map(|f| ms(f.mean)).sum::<f64>() / still.len() as f64;
    let c3 = moving.iter().all(|f| {
        ms(f.mean) <= 20.0
            && ms(f.mean) <= 1.25 * control_mean
            && ms(f.p99) <= 40.0
            && f.fallback <= 0.10
    });
    let share = moving.iter().map(|f| f.sky_share).sum::<f64>() / moving.len() as f64;
    let c2 = share <= 0.25;
    say(format!(
        "3. speed: control mean {control_mean:.2} ms; moving means {} (each at most 20 ms and {:.2} ms; p99 at most 40 ms; fallback at most 10%): {}",
        moving
            .iter()
            .map(|f| format!("{:.2}", ms(f.mean)))
            .collect::<Vec<_>>()
            .join(", "),
        1.25 * control_mean,
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "2. work: sky re-baked pages {:.2}% of pages charged (at most 25%): {}",
        100.0 * share,
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 1. No stale light: verify after every 10th frame.
    let mut r = renderer();
    let mut worst = 0u64;
    let mut over = 0u32;
    let mut checks = 0;
    let mut checked_pages = Vec::new();
    for f in 0..size.checked {
        motion.apply(&mut r, f);
        r.frame(flight_camera(f * GATE.flight / size.flight), &target);
        if (f + 1) % 10 == 0 {
            if matches!(f + 1, 10 | 50 | 100 | 300) {
                // Reported: what the frame showed.
                let pixels = r.read_colors();
                let mut image = Image::new(size.w as usize, size.h as usize, [0; 3]);
                for (i, p) in pixels.iter().enumerate() {
                    let (x, y) = (i % size.w as usize, i / size.w as usize);
                    image.set(x, y, room3d::to_rgb(Vec3::from_array(*p)));
                }
                let tag = if dev { "_dev" } else { "" };
                image.save(&out_dir.join(format!("m9_gate3{tag}_frame{}.png", f + 1)))?;
            }
            let check = r.verify_sky();
            checks += 1;
            worst = worst.max(check.light_differing);
            over += check.violations + check.over_threshold;
            checked_pages.push(check.pages);
            say(format!(
                "frame {}: {} pages; light values differing {}; sky: {} changed, {} beyond the ledger, {} beyond ε (largest change {:.3e}, largest charge {:.3e})",
                f + 1,
                check.pages,
                check.light_differing,
                check.changed,
                check.violations,
                check.over_threshold,
                check.largest_change,
                check.largest_charge
            ));
        }
    }
    let c1 = worst == 0 && over == 0 && checks == size.checked / 10;
    say(format!(
        "1. soundness: {checks} checks, at most {worst} light values differing, {over} sky violations: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    let name = if dev {
        "m9_gate3_dev.txt"
    } else {
        "m9_gate3.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
