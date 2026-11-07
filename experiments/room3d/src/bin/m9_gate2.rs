//! M9 gate 2: a moving light through the error ledger (the paradigm's third
//! rule, on the GPU, in 3D).
//!
//! When a light moves, every cached texel's channel for it may change, so
//! exact invalidation re-bakes every page every frame. The ledger instead
//! certifies a bound on how far each page can be from a re-bake, and
//! re-bakes only pages whose bound exceeds ε.
//!
//! Method:
//! - Texels store the light's shadowed share of its unshadowed light
//!   (ratios, M5 gate 7); shading multiplies by the unshadowed light at the
//!   pixel, computed each frame for the light where it is now. Only the
//!   ratio can be stale.
//! - **A guard band.** For a light marked as moving, each texel also traces
//!   a cone through the light's window enlarged three times about its
//!   center. A page is *clear* if no texel's enlarged cone meets anything,
//!   *dark* if every texel's is fully covered. Each page keeps, per moving
//!   light, the frame it was baked (the light's pose then is kept in a
//!   ring), the smallest angle the light can shift within every texel's
//!   enlarged window (the guard), whether it is clear or dark, and the
//!   smallest unshadowed light at its texels.
//! - **The bound.** Each frame, per page read in the frame before: the
//!   light's shift δ as seen from the page (each corner's displacement over
//!   its distance to the page's box, conservatively). A clear or dark page
//!   whose δ stays within its guard cannot change: bound 0. Otherwise the
//!   window gained or lost at most the band of width δ around its edge, so
//!   the ratio moved by at most 2 (2 δ P + π δ²) / U, with P the window's
//!   perimeter in radians and U the page's smallest unshadowed light.
//! - Pages whose bound exceeds ε = 1% are re-baked for that light in the
//!   frame (whole pages, appended to the bake); what does not fit is marked
//!   invalid, so no pixel shows a ratio more than ε from a re-bake.
//!
//! Scene: the million-boulder field with M5 gate 6's three colored lights,
//! no sky, 1280×720, hardware camera rays, cache defaults, prefilter 1 bin.
//! The sun drifts at 0.25° per second (a 24-minute day, flatland gate 7's
//! "realistic" speed; 1/240° per frame at 60 frames per second), turning
//! about the vertical axis through the field's center. The other lights
//! stay. Flight: M5 gate 8's path, 600 frames, from an empty cache.
//!
//! Control, before anything else: the sun-only warm frame at the standing
//! viewpoint (static scene) at most 3.3 ms, or the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Soundness:** over the first 300 frames of the flight, after every
//!    10th frame, for every page shading may read, the largest change of a
//!    texel's sun ratio when re-baked from scratch is at most the page's
//!    certified bound plus 1e-5 (rounding). Zero violations in all 30
//!    checks.
//! 2. **Quality:** in the same checks, no texel's sun ratio is more than
//!    ε = 0.01 from its re-baked value.
//! 3. **Work:** in the timed flights, pages re-baked for the sun's motion
//!    average at most 25% of the pages read in the frame before (exact
//!    invalidation re-bakes all of them).
//! 4. **Speed** (measured first, cool): three flights with the sun drifting,
//!    alternating with three with it still. Each drifting flight has a mean
//!    frame time at most 16.7 ms and at most 1.25× the still flights' mean,
//!    a 99th percentile at most 33.3 ms, and fallback at most 10%.
//!
//! Reported, not gated: the share of pages clear, dark and mixed; bounds'
//! distribution against actual changes; the same with the lamp moving
//! (0.5 m per second in a circle) instead of the sun.
//!
//! Changed before the first full-size run, after a development test on
//! the small GPU test scene (`a_drifting_sun_stays_within_the_ledger`):
//! - **No dark state.** A page wholly covered in its guard (even by one
//!   mesh alone, with the light's own cone wholly covered) still let a
//!   re-bake 0.37° later see 1% of the light. Shadowed texels count as
//!   mixed and re-bake whenever the light moves.
//! - **Snapping.** A moving light's ratios within ε of 0 or 1 are stored as
//!   0 or 1: a nearly covered cone's estimate wavers by a few thousandths
//!   as its bins realign. A texel's cone that nothing covers stores 1, one
//!   wholly covered 0. Mixed texels' bounds add ε, so they re-bake whenever
//!   the light moves.
//! - **The horizon.** A texel can be clear only with the light's center at
//!   least twice its guard (and 0.1 radian) above its horizon: estimates of
//!   grazing light wavered by up to 6%.
//! - **Per-texel frames and a page's slack.** Each texel keeps the frame
//!   it was baked, so a page can re-bake some texels and keep the rest.
//!   Mixed texels re-bake alone, whenever the light moves, without a guard
//!   cone (they stay mixed). Each page keeps its slack, the smallest guard
//!   of its clear and fixed texels less the light's shift since the page's
//!   version; past a hashed share (a half to all) of it the page re-bakes
//!   whole, with fresh guards, so pages baked together do not all re-bake
//!   in one frame. (Per-texel shift checks cost about 5 ms a frame.)
//! - **Fixed texels** (facing away) only with the light's center at least
//!   twice its radius below their horizon; nearer, mixed. (A sliver of
//!   margin made a page's slack a sliver.)
//! - **Pages set aside.** A page not read in the frame before that needs a
//!   re-bake is kept, shown as missing until a pixel wants it, then
//!   re-baked the next frame: the ledger runs every frame there are moving
//!   lights. (Marking such pages invalid, evicted and requested again, cost
//!   whole pages of every light: fallback rose to about 30%.) The ledger's
//!   re-bakes are limited to eight times the frame's bake budget; past
//!   that, pages are set aside.
//!
//! `cargo run --release -p room3d --bin m9_gate2`
//!
//! `M9_DEV=1` runs a small development version (640×360, short flights),
//! whose numbers are not the gate's.

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
    // The sun drifts 1/240° per frame (0.25° per second at 60 frames per
    // second); the lamp (reported) circles 2 m at 0.5 m per second.
    let sun_at = |f: usize| turned(lights[0], center, (f as f32 / 240.0).to_radians());
    let lamp_at = |f: usize| {
        let mut l = lights[2];
        if let Emitter::Rect(mut rect) = l.emitter {
            let a = 0.25 * f as f32 / 60.0;
            rect.corner += Vec3::new(a.cos() - 1.0, a.sin(), 0.0) * 2.0;
            l.emitter = Emitter::Rect(rect);
        }
        l
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
    let fly = |moving: bool| {
        let mut r = renderer(0);
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            let g = f * GATE.flight / size.flight;
            if moving {
                r.set_light(0, sun_at(g));
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
        still.push(fly(false));
        drifting.push(fly(true));
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
        say(line(&format!("flight {}, sun drifting", k + 1), b));
    }
    let control_mean = still.iter().map(|f| ms(f.mean)).sum::<f64>() / still.len() as f64;
    let c4 = drifting.iter().all(|f| {
        ms(f.mean) <= 16.7
            && ms(f.mean) <= 1.25 * control_mean
            && ms(f.p99) <= 33.3
            && f.fallback <= 0.10
    });
    let share = drifting.iter().map(|f| f.share).sum::<f64>() / drifting.len() as f64;
    let c3 = share <= 0.25;
    say(format!(
        "4. speed: still mean {control_mean:.2} ms; drifting means {} (each at most 16.7 ms and {:.2} ms): {}",
        drifting
            .iter()
            .map(|f| format!("{:.2}", ms(f.mean)))
            .collect::<Vec<_>>()
            .join(", "),
        1.25 * control_mean,
        if c4 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "3. work: pages re-baked for the sun {:.2}% of pages read (at most 25%): {}",
        100.0 * share,
        if c3 { "PASS" } else { "FAIL" }
    ));

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
    let c1 = violations == 0 && checks == size.checked / 10;
    let c2 = over == 0 && checks == size.checked / 10;
    say(format!(
        "1. soundness: {checks} checks, {violations} pages beyond their bound: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "2. quality: {over} pages beyond ε: {}",
        if c2 { "PASS" } else { "FAIL" }
    ));

    // Reported: the lamp moving instead.
    let lamp_flight = {
        let mut r = renderer(2);
        let mut times = Vec::with_capacity(size.flight);
        let begin = Instant::now();
        for f in 0..size.flight {
            let call = Instant::now();
            let g = f * GATE.flight / size.flight;
            r.set_light(2, lamp_at(g));
            r.frame(flight_camera(g), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let total = begin.elapsed();
        summarize(&times, total, r.cache_log(), size.w * size.h)
    };
    say(line("reported, lamp circling", &lamp_flight));
    let (lc, lv, lo) = check(2, &lamp_at, size.checked / 3, "lamp", &mut say);
    say(format!(
        "reported, lamp: {lc} checks, {lv} pages beyond their bound, {lo} beyond ε"
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {}, 4 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c4 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 && c4 { "PASS" } else { "FAIL" }
    ));
    let name = if dev {
        "m9_gate2_dev.txt"
    } else {
        "m9_gate2.txt"
    };
    std::fs::write(out_dir.join(name), report)?;
    Ok(())
}
