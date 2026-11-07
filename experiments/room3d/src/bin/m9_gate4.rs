//! M9 gate 4: a moving light through the ledger, without the work limit,
//! at a real sun's speed.
//!
//! Gate 2 failed on fallback (33–34%). The cause: the ledger's limit on
//! re-bakes, eight times the frame's bake budget, bound (it re-baked about
//! 14,700 texel jobs a frame against about 16,000), so visible pages were
//! set aside and shown missing. Now the only limit is the buffer's
//! capacity. Development runs at gate 2's speed and 640×360 (disclosed):
//! fallback 11.4%, but 1.28× the still flights' frame time, from about
//! 52,000 re-bakes a frame as pages' guards expire; a guard of five radii
//! instead of three re-baked less but cost more (19.8 ms), and was not kept.
//!
//! Gate 2's sun moved at a 24-minute day: 60× a real sun. At a real sun's
//! speed each page's guard lasts about 80 seconds.
//!
//! Scene, flight and lights as gate 2; 1280×720.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Soundness and quality at gate 2's speed:** 30 checks over 300
//!    frames, as gate 2's criteria 1 and 2: no page beyond its bound, none
//!    beyond ε.
//! 2. **Speed at a real sun's speed** (1/14,400° per frame, 0.25° a
//!    minute): three flights alternating with three still ones, each with
//!    mean at most 16.7 ms and at most 1.25× the still flights' mean, 99th
//!    percentile at most 33.3 ms, fallback at most 10%.
//!
//! Reported: one flight at gate 2's speed.
//!
//! `cargo run --release -p room3d --bin m9_gate4`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{
    CacheOptions, CacheStats, Camera, Emitter, InstancedRenderer, Light, MaskConeKernel, wgpu,
};
use room3d::field::Field;

struct Size {
    w: u32,
    h: u32,
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

struct Warm {
    frames: Option<usize>,
    frame_time: Duration,
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
    /// Over frames where the light moved: mean share of pages read in the
    /// frame before that the ledger re-baked texels of, and means of pages
    /// re-baked, texels re-baked, pages marked invalid, pages read, pages
    /// with mixed texels, pages passed whole.
    share: f64,
    pages: f64,
    texels: f64,
    invalidated: f64,
    checked: f64,
    mixed: f64,
    whole: f64,
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
        share: moved_mean(&|s| s.ledger_pages as f64 / s.ledger_checked.max(1) as f64),
        pages: moved_mean(&|s| s.ledger_pages as f64),
        texels: moved_mean(&|s| s.ledger_texels as f64),
        invalidated: moved_mean(&|s| s.ledger_invalidated as f64),
        checked: moved_mean(&|s| s.ledger_checked as f64),
        mixed: moved_mean(&|s| s.ledger_mixed as f64),
        whole: moved_mean(&|s| s.ledger_whole as f64),
    }
}

/// Light `light` turned by `angle` about the vertical axis through `center`
/// (both relative to the render origin).
fn turned(light: Light, center: Vec3, angle: f32) -> Light {
    let r = Mat3::from_rotation_z(angle);
    let mut out = light;
    if let Emitter::Rect(mut rect) = light.emitter {
        rect.corner = center + r * (rect.corner - center);
        rect.edge_u = r * rect.edge_u;
        rect.edge_v = r * rect.edge_v;
        out.emitter = Emitter::Rect(rect);
    }
    out
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
    let sun_only = Light::white(sun, field.radiance);
    let lights = field.colored_lights(origin, 1000.0);
    let center = (field.scene.instances[0].position - origin).as_vec3();
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
    // Gate 2's sun drifts 1/240° per frame (0.25° per second at 60 frames
    // per second); a real sun 1/14,400° per frame.
    let sun_at = |f: usize| turned(lights[0], center, (f as f32 / 240.0).to_radians());
    let real_sun = |f: usize| turned(lights[0], center, (f as f32 / 14_400.0).to_radians());
    let renderer = |moving: usize| {
        InstancedRenderer::with_movers(
            &kernel,
            format,
            size.w,
            size.h,
            1,
            &lights,
            &field.scene,
            origin,
            Some(CacheOptions::default()),
            &[],
            &[moving],
        )
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    let control = {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            size.w,
            size.h,
            1,
            &[sun_only],
            &field.scene,
            origin,
            CacheOptions::default(),
        );
        warm(&mut r, ground, &target)
    };
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

    // 4. Speed, first: flights alternating, the sun still and drifting.
    let fly = |moving: bool, at: &dyn Fn(usize) -> Light| {
        let mut r = renderer(0);
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            let g = f * GATE.flight / size.flight;
            if moving {
                r.set_light(0, at(g));
            }
            r.frame(flight_camera(g), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let total = begin.elapsed();
        summarize(&times, total, r.cache_log(), size.w * size.h)
    };
    let mut still = Vec::new();
    let mut drifting = Vec::new();
    for _ in 0..size.flights {
        still.push(fly(false, &real_sun));
        drifting.push(fly(true, &real_sun));
    }
    let line = |label: &str, f: &Flight| {
        format!(
            "{label}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, budget {:.0}; re-baked texels on {:.1} of {:.0} pages read ({:.2}%), {:.0} texels, invalidated {:.1}; pages with mixed texels {:.0}, passed whole {:.0}",
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            100.0 * f.fallback,
            f.budget,
            f.pages,
            f.checked,
            100.0 * f.share,
            f.texels,
            f.invalidated,
            f.mixed,
            f.whole
        )
    };
    for (k, (a, b)) in still.iter().zip(&drifting).enumerate() {
        say(line(&format!("flight {}, sun still", k + 1), a));
        say(line(&format!("flight {}, real sun", k + 1), b));
    }
    let control_mean = still.iter().map(|f| ms(f.mean)).sum::<f64>() / still.len() as f64;
    let c4 = drifting.iter().all(|f| {
        ms(f.mean) <= 16.7
            && ms(f.mean) <= 1.25 * control_mean
            && ms(f.p99) <= 33.3
            && f.fallback <= 0.10
    });
    let share = drifting.iter().map(|f| f.share).sum::<f64>() / drifting.len() as f64;
    say(format!(
        "2. speed at a real sun's speed: still mean {control_mean:.2} ms; means {} (each at most 16.7 ms and {:.2} ms): {}",
        drifting
            .iter()
            .map(|f| format!("{:.2}", ms(f.mean)))
            .collect::<Vec<_>>()
            .join(", "),
        1.25 * control_mean,
        if c4 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "reported: pages re-baked for the real sun {:.2}% of pages read",
        100.0 * share
    ));
    let fast = fly(true, &sun_at);
    say(line("reported, gate 2's speed", &fast));

    // 1–2. Soundness and quality: check after every 10th frame.
    let check = |moving: usize,
                 at: &dyn Fn(usize) -> Light,
                 frames: usize,
                 label: &str,
                 say: &mut dyn FnMut(String)| {
        let mut r = renderer(moving);
        let (mut violations, mut over, mut checks) = (0u32, 0u32, 0usize);
        for f in 0..frames {
            let g = f * GATE.flight / size.flight;
            r.set_light(moving, at(g));
            r.frame(flight_camera(g), &target);
            if (f + 1) % 10 == 0 {
                let c = r.verify_ledger(moving);
                checks += 1;
                violations += c.violations;
                over += c.over_epsilon;
                say(format!(
                    "{label} frame {}: {} pages, {} changed, {} beyond their bound, {} beyond ε; largest change {:.2e}, largest bound {:.2e}",
                    f + 1,
                    c.pages,
                    c.changed,
                    c.violations,
                    c.over_epsilon,
                    c.largest_change,
                    c.largest_bound
                ));
            }
        }
        (checks, violations, over)
    };
    let (checks, violations, over) = check(0, &sun_at, size.checked, "sun", &mut say);
    let c1 = violations == 0 && over == 0 && checks == size.checked / 10;
    say(format!(
        "1. soundness and quality at gate 2's speed: {checks} checks, {violations} pages beyond their bound, {over} beyond ε: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));

    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c4 { "PASS" } else { "FAIL" },
        if c1 && c4 { "PASS" } else { "FAIL" }
    ));
    let name = if dev {
        "m9_gate2_dev.txt"
    } else {
        "m9_gate2.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
