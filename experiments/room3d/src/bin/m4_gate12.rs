//! M4 gate 12: pages requested three levels finer: less error while
//! flying (Landmark 2), quicker warm-up.
//!
//! The error while flying, measured against settled references at the same
//! cameras (development, `flight_error_dev`): on M9's flight over the
//! million-boulder field with sun and sky, the image averaged 24–26% from
//! settled (relative RMSE). Pixels showed coarser levels than they wanted
//! (0.2% found their wanted level): the ground's boulder shadows smeared
//! into blotches, boulders unshadowed. A pixel showing a page coarser than
//! it wants requests the page one level finer, so a region walks from its
//! coarsest page to its wanted one in ten rounds or more, each a bake and a
//! readback two frames late, baking every level in between.
//!
//! Now a pixel requests up to three levels finer (`CacheOptions::
//! refine_levels`, 3 by default): fewer rounds, fewer pages in between.
//!
//! Pass criteria, fixed before the run:
//! 1. **Speed in flight** (run first): three pairs of M9's flight (600
//!    frames, standing height to 150 m, 1280×720, sun and sky, default
//!    options), one level then three: the three levels' mean frame at most
//!    1.05× the one level's (over the three flights each).
//! 2. **Error while flying:** the same flight at a fixed bake budget (11,000
//!    units a frame, so the comparison does not depend on the machine's
//!    heat), every 60th frame's image against a renderer settled at that
//!    camera: the mean relative RMSE with three levels at most 0.9× that
//!    with one. Reported: the largest, and both with default options.
//! 3. **Warm-up and result:** at M5 gate 13's four viewpoints with the fixed
//!    budget, from empty: frames until warm (16 quiet frames) with three
//!    levels at most 1.0× those with one, and the warm images within 0.01%
//!    (relative RMSE) of each other.
//!
//! **Disclosed (before the run):** developed with `flight_error_dev` and
//! `sky_dev`. At the fixed budget the flight's mean error was 25.37%, 21.77%,
//! 20.77% and 20.76% with one to four levels (largest 27.34, 25.17, 26.09,
//! 28.85%); with default options 24.2–26.0% with one level and 19.2–23.3%
//! with three, moving with the machine's heat. With three levels the views
//! from 10 and 40 m up warmed in 477 and 482 frames against 628 and 639,
//! their warm images identical. The development mode (one flight pair)
//! failed criterion 1 (9.77 against 8.42 ms, the process's first flight
//! fast) and passed 2 (0.813×) and 3; three pairs (`M4_PAIRS=3`) gave 9.88
//! against 9.81 ms (1.007×). The gate runs three pairs as written.
//!
//! `M4_DEV=1`: a development check (100,000 boulders, a 200-frame flight,
//! two viewpoints); not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m4_gate12`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const FLIGHT: usize = 600;
const FIXED: u32 = 11_000;
const EVERY: usize = 60;

/// Frames until 16 in a row request and bake nothing (at most 60,000).
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Option<usize> {
    for _ in 0..60_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return Some(latest - 16);
        }
    }
    None
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M4_DEV").is_some();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say("DEVELOPMENT RUN: 100,000 boulders, a 200-frame flight, two viewpoints; not the gate's numbers".into());
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
    let field = Field::new(if dev { 100_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: W,
                height: H,
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
    let options = |levels: u32, fixed: bool| CacheOptions {
        refine_levels: levels,
        frame_target_ms: if fixed { 0.0 } else { 14.0 },
        bake_budget: if fixed { FIXED } else { 20_000 },
        ..Default::default()
    };
    let renderer = |o: CacheOptions| {
        InstancedRenderer::with_cache(
            &kernel,
            format,
            W,
            H,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            o,
        )
    };
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
        let t = f as f32 / (FLIGHT - 1) as f32;
        view(300.0 * t, 1.7 + 148.3 * t, 45.0 * t)
    };
    let frames = if dev { 200 } else { FLIGHT };

    // 1. Speed, first.
    let fly = |levels: u32| {
        let mut r = renderer(options(levels, false));
        let begin = Instant::now();
        let mut times = Vec::with_capacity(frames);
        for f in 0..frames {
            let call = Instant::now();
            r.frame(flight_camera(f * FLIGHT / frames), &target);
            times.push(call.elapsed());
        }
        r.finish();
        let mean = begin.elapsed() / frames as u32;
        times.sort();
        (mean, times[(times.len() * 99) / 100 - 1])
    };
    let mut means = [0.0f64; 2];
    let pairs = std::env::var("M4_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if dev { 1 } else { 3 });
    for _ in 0..pairs {
        for (k, levels) in [1u32, 3].into_iter().enumerate() {
            let (mean, p99) = fly(levels);
            means[k] += ms(mean) / pairs as f64;
            say(format!(
                "1. flight, {levels} level(s): mean {:.2} ms, p99 {:.2} ms",
                ms(mean),
                ms(p99)
            ));
        }
    }
    let c1 = means[1] <= 1.05 * means[0];
    say(format!(
        "1. speed in flight: three levels {:.2} ms against one {:.2} ms ({:.3}×; at most 1.05×): {}",
        means[1],
        means[0],
        means[1] / means[0],
        if c1 { "PASS" } else { "FAIL" }
    ));

    if std::env::var_os("M4_SPEED_ONLY").is_some() {
        return Ok(());
    }

    // 2. The error while flying, against settled references.
    let flight_error = |levels: u32, fixed: bool| {
        let mut r = renderer(options(levels, fixed));
        let mut errors = Vec::new();
        let every = EVERY * frames / FLIGHT;
        for f in 0..frames {
            let g = f * FLIGHT / frames;
            r.frame(flight_camera(g), &target);
            if (f + 1) % every == 0 {
                r.finish();
                let flying = r.read_colors();
                let mut reference = renderer(options(levels, fixed));
                settle(&mut reference, flight_camera(g), &target);
                reference.finish();
                errors.push(relative_rmse(&flying, &reference.read_colors()));
            }
        }
        let mean = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
        let largest = errors.iter().copied().fold(0.0, f64::max);
        (mean, largest)
    };
    let one = flight_error(1, true);
    let three = flight_error(3, true);
    let c2 = three.0 <= 0.9 * one.0;
    say(format!(
        "2. error while flying (fixed budget {FIXED}): three levels mean {:.2}% (largest {:.2}%) against one {:.2}% (largest {:.2}%) ({:.3}×; at most 0.9×): {}",
        100.0 * three.0,
        100.0 * three.1,
        100.0 * one.0,
        100.0 * one.1,
        three.0 / one.0,
        if c2 { "PASS" } else { "FAIL" }
    ));
    let one_d = flight_error(1, false);
    let three_d = flight_error(3, false);
    say(format!(
        "2. reported, default options: three levels mean {:.2}% (largest {:.2}%) against one {:.2}% (largest {:.2}%)",
        100.0 * three_d.0,
        100.0 * three_d.1,
        100.0 * one_d.0,
        100.0 * one_d.1
    ));

    // 3. Warm-up and result at four viewpoints.
    let views = [
        ("standing", view(0.0, 1.7, 0.0)),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ];
    let mut c3 = true;
    for &(name, camera) in views.iter().take(if dev { 2 } else { 4 }) {
        let mut a = renderer(options(1, true));
        let fa = settle(&mut a, camera, &target);
        a.finish();
        let mut b = renderer(options(3, true));
        let fb = settle(&mut b, camera, &target);
        b.finish();
        let same = relative_rmse(&b.read_colors(), &a.read_colors());
        let ok = matches!((fa, fb), (Some(x), Some(y)) if y <= x) && same <= 1e-4;
        c3 &= ok;
        say(format!(
            "3. {name}: warm in {} frames with three levels against {} with one; warm images {:.4}% apart: {}",
            fb.map_or("never".into(), |f| f.to_string()),
            fa.map_or("never".into(), |f| f.to_string()),
            100.0 * same,
            if ok { "PASS" } else { "FAIL" }
        ));
    }
    say(format!(
        "3. warm-up and result: {}",
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m4_gate12_dev.txt"
        } else {
            "m4_gate12.txt"
        }),
        report,
    )?;
    Ok(())
}
