//! M13 gate 7: coverage pages that keep up in flight (coverage only).
//!
//! M13 gate 6 passed with view-dependent coverage pages for far hierarchy
//! nodes (settled: 16.12 ms standing with 10,224 pages). In motion, a page
//! is valid while its contents shift by at most a quarter of a bin; pages
//! baked together expire together, and bakes cost about 0.087 ms a page at
//! 20 a frame (one workgroup lane per texel, 16 rays each). After the run
//! (development): a walk from a settled start needs 1.5 bakes a frame, but
//! a flight at 10 m/s 20 m up asks for about 2,000 a frame after 15 m and
//! frames reach 41 ms (`docs/results/m13-gate6.md`).
//!
//! Method, on gate 6's pass:
//! - Bakes with one invocation per ray (8,192 a page), texels' shares
//!   counted in workgroup memory, so a frame's few pages fill the GPU.
//! - A page's bake eye is led along the camera's motion (the last frame's
//!   eye step): by half the steps after which its contents would shift a
//!   quarter of a bin, so it is valid at once (a shift of an eighth of a
//!   bin, half the bound) and stays valid about twice as long along the
//!   path. The validity test is gate 6's, from the current eye: accuracy
//!   never depends on the lead.
//!
//! Changed before the first run, after development at full size (the
//! criteria as amended below):
//! - Pages read past 80% of their bound (a fifth of a bin) are asked again
//!   on a second list, baked with the budget the frame's requests leave, so
//!   pages baked together (a settled start) do not all expire together.
//! - The bakes' arguments pass runs as one workgroup of 256 (as a single
//!   invocation it took about 2 ms to drop thousands of requests).
//! - The budget: 56 pages a frame (development: about 0.028 ms a page on
//!   the flight's pages, so 1.6 ms; 48 lagged, 64 measured 1.66–1.85 ms).
//!
//! Scene: the million-boulder field without its ground, 1280×720, gate 6's
//! pages (512 texels, nodes of at least 8 instances, a pool of 32,768).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Flight, speed:** 20 m up, looking 10° down along the standing
//!    view's heading, at 10 m/s for 10 s (600 frames at 60 frames a
//!    second), from a settled start: the 95th percentile frame at most 1.2×
//!    the settled frame (median of 10, each waited for) at its place (as
//!    amended below).
//! 2. **Bake budget:** the flight's bakes a frame (a count) cost at most
//!    2 ms a frame, measured from a cold pool at the flight's start (as
//!    amended below).
//! 3. **Flight, accuracy:** at 10 frames evenly spread after the first
//!    second, coverage within 0.3× the error of that frame's one-ray image
//!    from its own 16×16 reference.
//! 4. **Fill:** gate 6's walk (1.7 m, 1.5 m/s along the standing view for
//!    10 s) from a cold pool at the same budget: the median frame at most
//!    1.25× the settled standing frame (gate 6: 34.0 ms against 16.1).
//!
//! Reported: bakes and requests a frame, pages read, the largest shift
//! read; frame times along both paths.
//!
//! Amended before any run, after development at full size (disclosed; the
//! bars unchanged), as M7 gate 4 amended its criterion 1:
//! - Criterion 1 compared each frame with the settled frame at the
//!   flight's start; but the view itself costs more further into the field
//!   (development: 20.4 ms settled at the start, 23.5 ms at the end of a
//!   3 s flight, frames in flight tracking the latter), which is not lag.
//!   Now: the 95th percentile of each frame's time over the settled frame
//!   at its place, settled frames (median of 10) taken every second along
//!   the path and interpolated, at most 1.2.
//! - Criterion 2's interleaved frames could not resolve 2 ms at the
//!   flight's cold start (frames of about 80 ms varying by several ms:
//!   development read −3.3 to +8.6 ms for 64 bakes). Now: the page passes'
//!   own time (the arguments and the bakes, each submitted and waited for
//!   alone), with the budget and without it, medians of 10 interleaved
//!   frames from a cold pool at the flight's start, differing by at most
//!   2 ms. (GPU timestamps around the passes read implausibly low on this
//!   device: 0.1 ms for 64 bakes.)
//!
//! `cargo run --release -p room3d --bin m13_gate7`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders; `M13_DEV=1M` at
//! full size: 3 s paths turned 20° from the gate's, three checks), whose
//! numbers are not the gate's; with it `M13_SECONDS=n` sets the paths'
//! length, `M13_END=1` times the settled frame at the flight's end and
//! `M13_COST=budget,pairs` the bakes alone (whole frames, interleaved).

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
/// Pages baked a frame while settling a view.
const SETTLE_BUDGET: u32 = 512;
/// Pages baked a frame on the paths (development: about 1.6 ms a frame).
const FLIGHT_BUDGET: u32 = 56;

fn target(kernel: &MaskConeKernel) -> wgpu::TextureView {
    kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
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

/// A per-pixel renderer through the pixel-cone entry point, with the
/// kernel's profile stage and prefilter as set.
fn renderer(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
) -> InstancedRenderer {
    kernel.set_pixel_cones(true);
    let r = InstancedRenderer::new_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        lights,
        &field.scene,
        origin,
    );
    kernel.set_pixel_cones(false);
    r
}

fn frame(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Vec<[f32; 3]> {
    r.frame(camera, target);
    r.finish();
    r.read_colors()
}

fn gray(px: &[[f32; 3]]) -> Vec<[f32; 3]> {
    px.iter().map(|p| [p[0]; 3]).collect()
}

/// The 16×16-ray reference and the one-ray image at `camera`.
fn reference(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    kernel.set_profile_stage(13);
    let mut r = renderer(kernel, field, origin, lights);
    let one = gray(&frame(&mut r, camera, target));
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    for k in 0..256 {
        let (dx, dy) = (
            ((k % 16) as f32 + 0.5) / 16.0 - 0.5,
            ((k / 16) as f32 + 0.5) / 16.0 - 0.5,
        );
        r.set_pixel_offset(dx, dy);
        for (s, p) in sum.iter_mut().zip(frame(&mut r, camera, target)) {
            for v in s.iter_mut() {
                *v += p[0] / 256.0;
            }
        }
    }
    (sum, one)
}

/// Frames at `camera` until a frame asks for no page (at most `max`):
/// the frames run and the pages baked.
fn settle(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    max: usize,
    dev: bool,
) -> (usize, u32) {
    let mut baked = 0;
    for k in 0..max {
        let start = Instant::now();
        r.frame(camera, target);
        r.wait();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let c = r.m13_page_counters();
        baked += c[2];
        if dev {
            println!(
                "  settling frame {k}: {ms:.2} ms, {} requests, {} baked, {} pages read; counters {c:?}",
                c[6], c[2], c[8]
            );
        }
        if c[6] == 0 {
            return (k + 1, baked);
        }
    }
    (max, baked)
}

/// The median of `times` (ms).
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// A path's frames: per frame its time (ms) and the pages' counters, and
/// the images at `checks`.
#[allow(clippy::type_complexity)]
fn fly(
    r: &mut InstancedRenderer,
    path: impl Fn(usize) -> Camera,
    frames: usize,
    checks: &[usize],
    target: &wgpu::TextureView,
) -> (
    Vec<f64>,
    Vec<[u32; 32]>,
    Vec<(usize, Camera, Vec<[f32; 3]>)>,
) {
    let (mut times, mut counters, mut saved) = (Vec::new(), Vec::new(), Vec::new());
    for f in 0..frames {
        let camera = path(f);
        let start = Instant::now();
        r.frame(camera, target);
        r.wait();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        counters.push(r.m13_page_counters());
        if checks.contains(&f) {
            saved.push((f, camera, gray(&frame(r, camera, target))));
        }
    }
    (times, counters, saved)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let view_target = target(&kernel);

    let dev = std::env::var_os("M13_DEV").is_some();
    let full = std::env::var("M13_DEV").is_ok_and(|v| v == "1M");
    let mut field = Field::new(if dev && !full { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    // The ground left out, for the references and the test alike.
    field.scene.instances.remove(0);
    let forward = (ground.target - ground.eye).normalize();
    let flat0 = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let turn: f32 = if dev { 20.0 } else { 0.0 };
    let (s, c) = turn.to_radians().sin_cos();
    let flat = Vec3::new(flat0.x * c - flat0.y * s, flat0.x * s + flat0.y * c, 0.0);
    let view = |height: f32, down_degrees: f32, ahead: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let (mut seconds, n_checks) = if dev { (3usize, 3usize) } else { (10, 10) };
    // `M13_SECONDS=n` in development: paths of n seconds.
    if let Some(n) = std::env::var("M13_SECONDS")
        .ok()
        .filter(|_| dev)
        .and_then(|v| v.parse().ok())
    {
        seconds = n;
    }
    let frames = 60 * seconds;
    let checks: Vec<usize> = (0..n_checks)
        .map(|k| 60 + k * (frames - 60) / n_checks)
        .collect();
    let flight = |f: usize| view(20.0, 10.0, 10.0 * f as f32 / 60.0);
    let walk = |f: usize| view(1.7, 0.0, 1.5 * f as f32 / 60.0);
    let new_renderer = || {
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(18);
        renderer(&kernel, &field, origin, &lights)
    };
    let settled_frame = |r: &mut InstancedRenderer, camera: Camera| {
        r.set_m13_page_budget(SETTLE_BUDGET);
        let (frames, baked) = settle(r, camera, &view_target, 200, false);
        let mut times: Vec<f64> = (0..10)
            .map(|_| {
                let start = Instant::now();
                r.frame(camera, &view_target);
                r.wait();
                ms(start.elapsed())
            })
            .collect();
        (median(&mut times), frames, baked)
    };
    let tag = if dev { "DEVELOPMENT " } else { "" };
    let mut pass = true;

    // 1. The flight: settled frames every second along it, then the flight
    // from a settled start.
    let mut r = new_renderer();
    let samples: Vec<f64> = (0..=seconds)
        .map(|k| settled_frame(&mut r, flight((60 * k).min(frames - 1))).0)
        .collect();
    drop(r);
    let settled_at = |f: usize| {
        let x = f as f64 / 60.0;
        let k = (x.floor() as usize).min(seconds - 1);
        let w = x - k as f64;
        samples[k] * (1.0 - w) + samples[k + 1] * w
    };
    let mut r = new_renderer();
    let (settled_flight, settle_frames, settle_baked) = settled_frame(&mut r, flight(0));
    r.set_m13_page_budget(FLIGHT_BUDGET);
    let (mut times, counters, saved) = fly(&mut r, flight, frames, &checks, &view_target);
    drop(r);
    let mut ratios: Vec<f64> = times
        .iter()
        .enumerate()
        .map(|(f, t)| t / settled_at(f))
        .collect();
    if dev {
        for (f, t) in times.iter().enumerate() {
            let q = t / settled_at(f);
            if q > 1.12 {
                println!(
                    "  slow frame {f}: {t:.2} ms ({q:.3}×), {} asked, {} baked",
                    counters[f][6], counters[f][2]
                );
            }
        }
    }
    ratios.sort_by(f64::total_cmp);
    let p95 = ratios[(ratios.len() * 95).div_ceil(100) - 1];
    let worst = times.iter().cloned().fold(0.0, f64::max);
    let c1 = p95 <= 1.2;
    pass &= c1;
    say(format!(
        "{tag}settled frames along the flight, every second: {}",
        samples
            .iter()
            .map(|t| format!("{t:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let stat = |cs: &[[u32; 32]], k: usize| {
        (
            cs.iter().map(|c| c[k] as f64).sum::<f64>() / cs.len() as f64,
            cs.iter().map(|c| c[k]).max().unwrap_or(0),
        )
    };
    let (baked_mean, baked_max) = stat(&counters, 2);
    let (asked_mean, asked_max) = stat(&counters, 6);
    let (read_mean, _) = stat(&counters, 8);
    let largest = counters.iter().map(|c| c[14]).max().unwrap_or(0);
    say(format!(
        "{tag}flight: settled start {settled_flight:.2} ms ({settle_frames} frames, {settle_baked} pages); 1. 95th percentile of frame over settled frame {p95:.3} (at most 1.2): {}; median frame {:.2} ms, worst {worst:.2} ms; bakes a frame {baked_mean:.1} (at most {baked_max}), requests {asked_mean:.1} (at most {asked_max}), pages read {read_mean:.0}; largest shift read {:.4} bin",
        verdict(c1),
        median(&mut times),
        largest as f64 / 1e4
    ));
    for (k, step) in [0usize, 60, 120, 180, 240, 300, 360, 420, 480, 540]
        .iter()
        .filter(|&&k| k < frames)
        .enumerate()
    {
        let _ = k;
        say(format!(
            "  flight frame {step}: {:.2} ms, {} asked, {} baked, {} pages read",
            times[*step], counters[*step][6], counters[*step][2], counters[*step][8]
        ));
    }

    // `M13_END=1` in development: the settled frame at the flight's end.
    // `M13_COST=budget,pairs` in development: the bakes' cost alone.
    if let Some((budget, pairs)) = std::env::var("M13_COST")
        .ok()
        .filter(|_| dev)
        .and_then(|v| {
            let p: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 2).then(|| (p[0], p[1] as usize))
        })
    {
        for start_at in [flight(0), walk(0)] {
            let mut r = new_renderer();
            let (mut without, mut with) = (Vec::new(), Vec::new());
            let (mut gpu0, mut gpu1) = (Vec::new(), Vec::new());
            for k in 0..2 * pairs {
                let b = if k % 2 == 0 { 0 } else { budget };
                r.set_m13_page_budget(b);
                let start = Instant::now();
                r.frame(start_at, &view_target);
                r.wait();
                let t = ms(start.elapsed());
                let g = r.m13_page_ms().unwrap_or(f64::NAN);
                if b == 0 {
                    without.push(t);
                    gpu0.push(g);
                } else {
                    with.push(t);
                    gpu1.push(g);
                }
            }
            let (a, b) = (median(&mut without), median(&mut with));
            println!(
                "cost of {budget} bakes: {a:.2} ms without, {b:.2} with: {:.2} ms ({:.3} a page); page passes on the GPU {:.3} ms without, {:.3} with (worst {:.3})",
                b - a,
                (b - a) / budget as f64,
                median(&mut gpu0),
                median(&mut gpu1),
                gpu1.iter().cloned().fold(0.0, f64::max)
            );
        }
        return Ok(());
    }
    if dev && std::env::var_os("M13_END").is_some() {
        let mut r = new_renderer();
        let (end, frames_end, baked_end) = settled_frame(&mut r, flight(frames - 1));
        println!(
            "settled at the flight's end: {end:.2} ms ({frames_end} frames, {baked_end} pages)"
        );
        return Ok(());
    }
    // 2. The bakes' cost: the page passes' own time (stage 18's last two
    // camera passes, each submitted and waited for alone), interleaved
    // frames without and with the budget, from a cold pool at the flight's
    // start.
    let mut r = new_renderer();
    let (mut without, mut with) = (Vec::new(), Vec::new());
    for k in 0..20 {
        let budget = if k % 2 == 0 { 0 } else { FLIGHT_BUDGET };
        r.set_m13_page_budget(budget);
        let p = r.frame_profiled(flight(0), &view_target);
        let t = ms(p.passes[6]) + ms(p.passes[7]);
        if budget == 0 {
            without.push(t)
        } else {
            with.push(t)
        }
    }
    drop(r);
    let cost = median(&mut with) - median(&mut without);
    let c2 = cost <= 2.0;
    pass &= c2;
    say(format!(
        "{tag}2. {FLIGHT_BUDGET} bakes a frame cost {cost:.2} ms (at most 2; the page passes {:.2} ms without, {:.2} with): {}",
        median(&mut without),
        median(&mut with),
        verdict(c2)
    ));

    // 4. Fill: gate 6's walk from a cold pool.
    let mut r = new_renderer();
    let (settled_standing, _, _) = settled_frame(&mut r, walk(0));
    drop(r);
    let mut r = new_renderer();
    r.set_m13_page_budget(FLIGHT_BUDGET);
    let (mut wtimes, wcounters, _) = fly(&mut r, walk, frames, &[], &view_target);
    drop(r);
    let wmedian = median(&mut wtimes);
    let c4 = wmedian <= 1.25 * settled_standing;
    pass &= c4;
    let (wbaked, _) = stat(&wcounters, 2);
    say(format!(
        "{tag}4. walk from a cold pool: median {wmedian:.2} ms (at most {:.2}, 1.25× the settled standing frame {settled_standing:.2}): {}; worst {:.2} ms; bakes a frame {wbaked:.1}; pages read at the end {}",
        1.25 * settled_standing,
        verdict(c4),
        wtimes.iter().cloned().fold(0.0, f64::max),
        wcounters.last().map(|c| c[8]).unwrap_or(0)
    ));

    // 3. The flight's accuracy at the checks.
    let mut c3 = true;
    for (f, camera, ours) in &saved {
        let (reference, one) = reference(&kernel, &field, origin, &lights, *camera, &view_target);
        let e_ours = relative_rmse(ours, &reference);
        let e_one = relative_rmse(&one, &reference);
        let ok = e_ours <= 0.3 * e_one;
        c3 &= ok;
        say(format!(
            "{tag}flight frame {f} ({:.2} s, {} pages read): coverage {:.2}% from its reference, one ray {:.2}% ({:.3}×, at most 0.3×): {}",
            *f as f64 / 60.0,
            counters[*f][8],
            100.0 * e_ours,
            100.0 * e_one,
            e_ours / e_one,
            verdict(ok)
        ));
    }
    pass &= c3;
    say(format!("{tag}3. flight accuracy: {}", verdict(c3)));
    kernel.set_profile_stage(0);
    if dev {
        return Ok(());
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate7.txt"), report)?;
    Ok(())
}
