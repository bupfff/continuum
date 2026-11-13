//! M9 gate 6: a moving sun at real speed, its shadows' edges held within a
//! certified bound (Landmark 2).
//!
//! Gate 4 failed on speed (1.6× the still flights at a real sun's speed):
//! texels near shadow edges ("mixed") re-baked whenever the light moved at
//! all, because a moving light's ratio estimate wavers by a few thousandths
//! as the light cone's bins realign with the light, which no bound on the
//! light's motion covers.
//!
//! Now a moving rectangular light's cones keep their bins on a lattice fixed
//! in the frame of the page's version (the light's pose at its last whole
//! bake): the window is the lattice's 8×8 bins holding the light's
//! footprint, and each bin counts by the footprint's area in it. A bin's
//! coverage then no longer depends on where the light is, and the ratio
//! moves by at most the footprint's change: with every point of the light
//! moving at most δ (as seen from the page), at most 1.5 κ (P δ + π δ²) / Ω,
//! for the footprint's perimeter P and solid angle Ω and κ the spread of the
//! bins' weights. Mixed texels re-bake only once that bound passes ε (a
//! hashed share of it, a half to all), so at a real sun's speed about every
//! 6–12 frames instead of every frame; no snapping. Texels whose weights
//! spread too far (the light near their horizon) still re-bake whenever it
//! moves.
//!
//! Scene, flight and lights as gates 2 and 4 (the million-boulder field,
//! three coloured rect lights 1 km away, the warm "sun" moving); 1280×720.
//!
//! Pass criteria, fixed before the first run:
//! 1. **Speed at a real sun's speed** (1/14,400° per frame), run first: three
//!    flights alternating with three still ones, each moving flight's mean
//!    at most 16.7 ms and at most 1.25× the still flights' mean, its 99th
//!    percentile at most 33.3 ms, fallback at most 10% (gate 4's criterion
//!    2).
//! 2. **Soundness:** 30 checks over the flight, 10 each at a real sun's
//!    speed, 8× it, and gate 2's speed (60× it); each re-bakes every page the
//!    cache may show, in its own lattice: no page beyond its certified bound
//!    (with 1e-5 for rounding), none beyond ε (0.01).
//! 3. **Accuracy at rest:** after the real-sun flight, the sun held at its
//!    last pose and the view settled, the image within 1% (relative RMSE) of
//!    a renderer built with the sun fixed there (the standard light cones).
//!
//! Reported: one flight at gate 2's speed; mixed pages held within their
//! bound against re-baked, per frame.
//!
//! Options as gate 4's: the cache's defaults without the frame-period cap
//! (`frame_budget_ms`, added after gate 4 for M14's streaming), which on
//! this scene (frames about 15 ms) holds the bake at its floor.
//!
//! **Disclosed (before the run):** the mechanism was developed on the
//! library's tests (a drifting sun at five speeds, every check within its
//! bound) and three development runs (`M9_DEV=1`: 640×360, 200 frames, one
//! flight pair, three checks per speed):
//! - with the library's frame-period cap: still 9.08 ms, real sun 10.34 ms,
//!   fallback 11.83% (still 10.01%); hence gate 4's options;
//! - without it: real sun 15.52 ms against 13.24 still, fallback 21.54%, the
//!   moving flight's bake budget at its floor. Profiling (`m9_dev`) found
//!   pages set aside off screen re-baked whole when wanted again (about 60 a
//!   frame), texels near the light's horizon re-baking every frame on the
//!   lattice's larger window, and the lattice's growth check, evaluated for
//!   every page a moving frame, costing about 4 ms. Changed: set-aside pages
//!   checked texel by texel, texels whose weights spread more than κ = 8 on
//!   the light's own cone, the growth check only where mixed texels would
//!   hold, bounds from the page's and light's centers, a lattice margin of
//!   3%;
//! - then: still 13.40 ms, real sun 14.42 ms (1.08×), p99 28.44 ms,
//!   fallback 8.56%; 604 of 1,813 pages with mixed texels held a frame,
//!   1,562 texels re-baked; sound (largest change 8.8e-4 against bounds up
//!   to 9.7e-3 at a real sun's speed; at 8× and 60× every mixed texel
//!   re-baked each frame); at rest 0.098% from the fixed sun.
//!
//! `cargo run --release -p room3d --bin m9_gate6`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{
    CacheOptions, CacheStats, Camera, Emitter, InstancedRenderer, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

struct Size {
    w: u32,
    h: u32,
    flight: usize,
    /// Frames per soundness speed, checked every 10th.
    checked: usize,
    flights: usize,
}

const GATE: Size = Size {
    w: 1280,
    h: 720,
    flight: 600,
    checked: 100,
    flights: 3,
};

const DEV: Size = Size {
    w: 640,
    h: 360,
    flight: 200,
    checked: 30,
    flights: 1,
};

struct Warm {
    frames: Option<usize>,
    frame_time: Duration,
}

/// Renders until 16 frames in a row request and bake nothing (at most
/// 40,000), then times 60 warm frames.
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
    /// Over frames where the light moved: means of pages re-baked, texels
    /// re-baked, pages read, pages with mixed texels, of those held within
    /// their bound, and pages passed whole.
    pages: f64,
    texels: f64,
    checked: f64,
    mixed: f64,
    held: f64,
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
        pages: moved_mean(&|s| s.ledger_pages as f64),
        texels: moved_mean(&|s| s.ledger_texels as f64),
        checked: moved_mean(&|s| s.ledger_checked as f64),
        mixed: moved_mean(&|s| s.ledger_mixed as f64),
        held: moved_mean(&|s| s.ledger_held as f64),
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
    // A real sun turns 1/14,400° per frame (0.25° a minute at 60 frames a
    // second); gate 2's sun 1/240°.
    let at_speed = |per_frame_degrees: f32| {
        move |f: usize| {
            turned(
                lights[0],
                center,
                (f as f32 * per_frame_degrees).to_radians(),
            )
        }
    };
    let real_sun = at_speed(1.0 / 14_400.0);
    // Gate 4's options: the defaults of its day, without the frame-period
    // cap added since.
    let options = CacheOptions {
        frame_budget_ms: 0.0,
        ..Default::default()
    };
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
            Some(options),
            &[],
            &[moving],
        )
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    // The machine's fitness: gate 4's control, the sun alone warm (2.15 ms
    // in gate 4).
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
        "control: sun-only warm frame {:.2} ms (at most 3.3 ms; gate 4: 2.15 ms) after {:?} frames: {}",
        ms(control.frame_time),
        control.frames,
        if fit {
            "fit"
        } else {
            "NOT FIT: the run is void"
        }
    ));
    if !fit && !dev {
        std::fs::write(out_dir.join("m9_gate6.txt"), report)?;
        return Ok(());
    }

    // 1. Speed, first: flights alternating, the sun still and moving at a
    // real sun's speed.
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
    let mut moving = Vec::new();
    for _ in 0..size.flights {
        still.push(fly(false, &real_sun));
        moving.push(fly(true, &real_sun));
    }
    let line = |label: &str, f: &Flight| {
        format!(
            "{label}: mean {:.2} ms, p99 {:.2} ms, worst {:.2} ms, fallback {:.2}%, budget {:.0}; ledger re-baked texels on {:.1} of {:.0} pages read, {:.0} texels; pages with mixed texels {:.0}, held within their bound {:.0}; passed whole {:.0}",
            ms(f.mean),
            ms(f.p99),
            ms(f.worst),
            100.0 * f.fallback,
            f.budget,
            f.pages,
            f.checked,
            f.texels,
            f.mixed,
            f.held,
            f.whole
        )
    };
    for (k, (a, b)) in still.iter().zip(&moving).enumerate() {
        say(line(&format!("flight {}, sun still", k + 1), a));
        say(line(&format!("flight {}, real sun", k + 1), b));
    }
    let control_mean = still.iter().map(|f| ms(f.mean)).sum::<f64>() / still.len() as f64;
    let c1 = moving.iter().all(|f| {
        ms(f.mean) <= 16.7
            && ms(f.mean) <= 1.25 * control_mean
            && ms(f.p99) <= 33.3
            && f.fallback <= 0.10
    });
    say(format!(
        "1. speed at a real sun's speed: still mean {control_mean:.2} ms; moving means {} (each at most 16.7 ms and {:.2} ms), p99 {} (at most 33.3 ms), fallback {} (at most 10%): {}",
        moving
            .iter()
            .map(|f| format!("{:.2}", ms(f.mean)))
            .collect::<Vec<_>>()
            .join(", "),
        1.25 * control_mean,
        moving
            .iter()
            .map(|f| format!("{:.2}", ms(f.p99)))
            .collect::<Vec<_>>()
            .join(", "),
        moving
            .iter()
            .map(|f| format!("{:.2}%", 100.0 * f.fallback))
            .collect::<Vec<_>>()
            .join(", "),
        if c1 { "PASS" } else { "FAIL" }
    ));
    let fast = fly(true, &at_speed(1.0 / 240.0));
    say(line("reported, gate 2's speed", &fast));

    // 2. Soundness: check after every 10th frame, at three speeds.
    let check = |at: &dyn Fn(usize) -> Light, label: &str, say: &mut dyn FnMut(String)| {
        let mut r = renderer(0);
        let (mut violations, mut over, mut checks, mut pages) = (0u32, 0u32, 0usize, 0u64);
        let (mut largest_change, mut largest_bound) = (0.0f32, 0.0f32);
        for f in 0..size.checked {
            let g = f * GATE.flight / size.flight;
            r.set_light(0, at(g));
            r.frame(flight_camera(g), &target);
            if (f + 1) % 10 == 0 {
                let c = r.verify_ledger(0);
                checks += 1;
                violations += c.violations;
                over += c.over_epsilon;
                pages += c.pages as u64;
                largest_change = largest_change.max(c.largest_change);
                largest_bound = largest_bound.max(c.largest_bound);
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
        say(format!(
            "{label}: {checks} checks of {pages} pages in all, {violations} beyond their bound, {over} beyond ε; largest change {largest_change:.2e}, largest bound {largest_bound:.2e}"
        ));
        (checks, violations, over)
    };
    let mut sound = true;
    let mut total_checks = 0;
    for (label, per_frame) in [
        ("real sun", 1.0 / 14_400.0),
        ("8× a real sun", 8.0 / 14_400.0),
        ("gate 2's speed", 1.0 / 240.0),
    ] {
        let (checks, violations, over) = check(&at_speed(per_frame), label, &mut say);
        total_checks += checks;
        sound &= violations == 0 && over == 0;
    }
    let c2 = sound && total_checks == 3 * (size.checked / 10);
    say(format!(
        "2. soundness: {total_checks} checks at three speeds: {}",
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 3. Accuracy at rest: the real-sun flight, then the sun held and the
    // view settled, against the sun fixed there.
    let last = GATE.flight - 1;
    let held = {
        let mut r = renderer(0);
        for f in 0..size.flight {
            let g = f * GATE.flight / size.flight;
            r.set_light(0, real_sun(g));
            r.frame(flight_camera(g), &target);
        }
        let w = warm(&mut r, flight_camera(last), &target);
        say(format!(
            "3. moving renderer settled after {:?} frames",
            w.frames
        ));
        r.read_colors()
    };
    let fixed = {
        let mut fixed_lights = lights;
        fixed_lights[0] = real_sun(last);
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            size.w,
            size.h,
            1,
            &fixed_lights,
            &field.scene,
            origin,
            options,
        );
        let w = warm(&mut r, flight_camera(last), &target);
        say(format!(
            "3. fixed renderer settled after {:?} frames",
            w.frames
        ));
        r.read_colors()
    };
    let error = relative_rmse(&held, &fixed);
    let c3 = error <= 0.01;
    say(format!(
        "3. accuracy at rest: {:.3}% from the sun fixed there (at most 1%): {}",
        100.0 * error,
        if c3 { "PASS" } else { "FAIL" }
    ));

    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    let name = if dev {
        "m9_gate6_dev.txt"
    } else {
        "m9_gate6.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
