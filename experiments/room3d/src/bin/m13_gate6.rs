//! M13 gate 6: hierarchy nodes holding coverage, as view-dependent coverage
//! pages (coverage only).
//!
//! The design is DESIGN §3.4, "The far field for sub-pixel coverage", part
//! D, on top of gate 5's pass (A, B, C, E). At the standing view's horizon
//! a pixel sees boulders from about 540 m to the field's edge, each about a
//! pixel across, and walks up to thousands of nodes reading their maps.
//!
//! Method:
//! - A node's page is its coverage seen from the bake eye at one texel per
//!   bin: 512 8-bit texels in a shape from 32×16 to 128×4 (or transposed);
//!   each texel the share of 4×4 hardware rays from the bake eye through its
//!   square, clipped to the node's box, that meet an instance in the node's
//!   item range. A header holds the bake eye, the page's plane (through the
//!   node's centre, facing the eye), its axes and texel sizes.
//! - During gate 5's tile walk, a node whose projected box fits a page shape
//!   at one texel per bin is an entry if its page is resident and valid;
//!   otherwise it is descended, and requested if it fits (deduplicated on
//!   the GPU). Pixels read a page per bin where the bin's ray crosses its
//!   plane, bilinearly, combined with other entries as independent
//!   occluders. Instances keep their shared maps; nodes holding movers never
//!   get pages.
//! - A page is valid while `2 f |ΔE⊥| L / D²` (the largest shift of its
//!   contents, in bins, since its bake) is at most 1/4 and its texels are
//!   0.7–1.4 bins wide; past that it is not read until re-baked. Pages bake
//!   within a time budget each frame, in a pool of 32,768.
//!
//! Changed before the first run, after development at full size
//! (`docs/results/m13-gate6.md`; criteria unchanged):
//! - Gate 5's pass lists subtree roots per tile and walks below them per
//!   pixel, so pages are found in both: the highest node on a path that fits
//!   a page shape and holds at least 8 instances is read from its page if
//!   resident and valid, otherwise asked for (marks on the GPU) and walked;
//!   nodes below it want no page. Reads are bilinear per open bin, with
//!   plane coordinates affine across the pixel.
//! - Bakes: after the pixels' pass, a single invocation takes at most the
//!   budget's requests and gives each a page (its own, or the next in the
//!   pool's ring, evicting the owner); eight workgroups a page, one texel a
//!   lane, rays from the node box's entry, opaque queries stepping past
//!   other instances (at most 16 a ray). Budget: 512 pages a frame while
//!   settling a view; 20 on the walk (development: 0.087 ms a page, 1.75 ms
//!   a frame). A view is settled once a frame asks for no page.
//!
//! Scene, views and reference are gates 4 and 5's (the million-boulder
//! field without its ground, standing at 1.7 m and 150 m up looking 45°
//! down, 1280×720, the reference 16×16 rays per pixel, one ray the central
//! one).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy, settled:** with every wanted page baked at the view,
//!    relative RMSE of per-pixel coverage from the reference at most 0.25×
//!    one ray's standing and 0.15× from 150 m (gate 5's bars).
//! 2. **Speed, settled:** at most 20 ms a frame in both views with the pages
//!    resident (median of 10 frames each waited for), timed first.
//! 3. **Memory:** the pages wanted in each view within the pool (32,768
//!    pages, at most 32 MB with headers).
//! 4. **Reuse in motion:** a 10 s walk at 1.5 m/s at standing height along
//!    the view, pages re-baked within a 2 ms budget a frame from a cold
//!    pool: at 10 frames evenly spread after the first second, coverage
//!    within 0.3× the error of that frame's one-ray image from its own 16×16
//!    reference.
//!
//! Reported: pages wanted, baked and read per view; entries per horizon
//! pixel; rays a frame for bakes; frames on the walk that read no stale page
//! and the largest bound read.
//!
//! `cargo run --release -p room3d --bin m13_gate6`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the passes alone;
//! `M13_DEV=1M` at full size), whose numbers are not the gate's; with it
//! `M13_ACC=1` measures accuracy on other views than the gate's, and
//! `M13_WALK=1` runs a 3 s walk along another view with three checks.

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
/// Pages baked a frame on the walk: within 2 ms on the M4 (development:
/// see `docs/results/m13-gate6.md`).
const WALK_BUDGET: u32 = 20;

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
    let acc = dev && std::env::var_os("M13_ACC").is_some();
    let walk_dev = dev && std::env::var_os("M13_WALK").is_some();
    let mut field = Field::new(if dev && !full { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    // The ground left out, for the reference and the test alike.
    field.scene.instances.remove(0);
    let forward = (ground.target - ground.eye).normalize();
    let flat0 = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let turned = |turn_degrees: f32| {
        let (s, c) = turn_degrees.to_radians().sin_cos();
        Vec3::new(flat0.x * c - flat0.y * s, flat0.x * s + flat0.y * c, 0.0)
    };
    let field_view = |height: f32, down_degrees: f32, turn_degrees: f32, ahead: f32| {
        let d = down_degrees.to_radians();
        let flat = turned(turn_degrees);
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = if acc {
        [
            (
                "standing (development view)",
                field_view(1.7, 0.0, 20.0, 0.0),
            ),
            (
                "120 m up (development view)",
                field_view(120.0, 40.0, -20.0, 0.0),
            ),
        ]
    } else {
        [
            ("standing", field_view(1.7, 0.0, 0.0, 0.0)),
            ("150 m up", field_view(150.0, 45.0, 0.0, 0.0)),
        ]
    };
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let v = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.set(i % W as usize, i / W as usize, [v; 3]);
        }
        image.save(&out_dir.join(format!("m13_gate6_{name}.png")))
    };
    // `M13_BAKECOST=1` in development: what WALK_BUDGET bakes add to a frame
    // at the standing view from a cold pool (medians of 10 frames without
    // and with them, interleaved).
    if dev && std::env::var_os("M13_BAKECOST").is_some() {
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(18);
        let camera = views[0].1;
        let mut r = renderer(&kernel, &field, origin, &lights);
        let (mut without, mut with) = (Vec::new(), Vec::new());
        for k in 0..20 {
            let budget = if k % 2 == 0 { 0 } else { WALK_BUDGET };
            r.set_m13_page_budget(budget);
            let start = Instant::now();
            r.frame(camera, &view_target);
            r.wait();
            let t = ms(start.elapsed());
            if budget == 0 {
                without.push(t)
            } else {
                with.push(t)
            }
        }
        without.sort_by(f64::total_cmp);
        with.sort_by(f64::total_cmp);
        println!(
            "bake cost: {:.2} ms a frame without bakes, {:.2} ms with {WALK_BUDGET}: {:.3} ms a page",
            without[5],
            with[5],
            (with[5] - without[5]) / WALK_BUDGET as f64
        );
        return Ok(());
    }
    let mut pass = true;
    let run_views = !walk_dev || acc;
    // 2. Speed (first) and 3. memory, with the pages settled at each view.
    let mut ours_images = Vec::new();
    let mut speed = Vec::new();
    for (name, camera) in views.into_iter().filter(|_| run_views) {
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(18);
        let mut r = renderer(&kernel, &field, origin, &lights);
        r.set_m13_page_budget(SETTLE_BUDGET);
        let (frames, baked) = settle(&mut r, camera, &view_target, 200, dev);
        let mut times: Vec<Duration> = (0..10)
            .map(|_| {
                let start = Instant::now();
                r.frame(camera, &view_target);
                r.wait();
                start.elapsed()
            })
            .collect();
        times.sort();
        let ours_ms = ms(times[5]);
        let out = frame(&mut r, camera, &view_target);
        let c = r.m13_page_counters();
        drop(r);
        let n = out.len() as f64;
        let roots = out.iter().map(|p| (p[2] as u32 / 2) as f64).sum::<f64>() / n;
        let max_roots = out.iter().map(|p| p[2] as u32 / 2).max().unwrap_or(0);
        let walked = out.iter().filter(|p| p[1] > 0.0).count().max(1) as f64;
        let visits = out.iter().map(|p| p[1] as f64).sum::<f64>() / walked;
        let max_visits = out.iter().map(|p| p[1]).fold(0.0f32, f32::max);
        let used = c[8];
        let wanted = used + c[6];
        let c2 = ours_ms <= 20.0;
        let c3 = wanted as usize <= 32_768;
        if !acc {
            pass &= c2 && c3;
        }
        say(format!(
            "{name}: settled in {frames} frames ({baked} pages baked, {} rays; {} nodes fit no page); 2. {ours_ms:.2} ms a frame (at most 20): {}; 3. pages wanted {wanted} (read {used}, asked {}; at most 32,768 in 16 MB of texels and 2.6 MB of headers): {}; page roots listed {} (summed over tiles); roots a tile {roots:.1} (at most {max_roots}); object nodes a walking pixel {visits:.1} (at most {max_visits:.0}); largest shift read {:.4} bin; mean coverage {:.4}",
            baked as u64 * 8192,
            c[11],
            verdict(c2),
            c[6],
            verdict(c3),
            c[12],
            c[14] as f64 / 1e4,
            out.iter().map(|p| p[0] as f64).sum::<f64>() / n
        ));
        if dev {
            save(&format!("{}_dev", name.replace(' ', "_")), &gray(&out))?;
        }
        ours_images.push(gray(&out));
        speed.push(ours_ms);
    }
    // 1. Accuracy, settled.
    if run_views && (!dev || acc) {
        for (k, (name, camera)) in views.into_iter().enumerate() {
            let (reference, one) =
                reference(&kernel, &field, origin, &lights, camera, &view_target);
            let ours = &ours_images[k];
            let e_ours = relative_rmse(ours, &reference);
            let e_one = relative_rmse(&one, &reference);
            let bar = if k == 0 { 0.25 } else { 0.15 };
            let c1 = e_ours <= bar * e_one;
            if !acc {
                pass &= c1;
            }
            let mean_cov =
                |px: &[[f32; 3]]| px.iter().map(|p| p[0] as f64).sum::<f64>() / px.len() as f64;
            say(format!(
                "{name}: 1. coverage {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most {bar}×): {}; mean coverage: reference {:.4}, ours {:.4}, one ray {:.4}",
                100.0 * e_ours,
                100.0 * e_one,
                e_ours / e_one,
                verdict(c1),
                mean_cov(&reference),
                mean_cov(ours),
                mean_cov(&one)
            ));
            if !dev {
                let slug = name.replace(' ', "_");
                save(&format!("{slug}_reference"), &reference)?;
                save(&format!("{slug}_ours"), ours)?;
            }
        }
    }
    // 4. Reuse in motion: a walk at 1.5 m/s at standing height along the
    // view, 60 frames a second, from a cold pool, at most WALK_BUDGET pages
    // baked a frame; checks at 10 frames evenly spread after the first
    // second (development: 3 s along the turned view, 3 checks).
    if !dev || walk_dev {
        let (seconds, checks, turn) = if dev { (3, 3, 20.0) } else { (10, 10, 0.0) };
        let frames = 60 * seconds;
        let at: Vec<usize> = (0..checks)
            .map(|k| 60 + k * (frames - 60) / checks)
            .collect();
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(18);
        let mut r = renderer(&kernel, &field, origin, &lights);
        r.set_m13_page_budget(WALK_BUDGET);
        let mut saved = Vec::new();
        let (mut baked, mut most_baked, mut largest, mut asked) = (0u64, 0u32, 0u32, 0u64);
        let mut frame_ms = Vec::new();
        for f in 0..frames {
            let camera = field_view(1.7, 0.0, turn, 1.5 * f as f32 / 60.0);
            let start = Instant::now();
            r.frame(camera, &view_target);
            r.wait();
            frame_ms.push(ms(start.elapsed()));
            let c = r.m13_page_counters();
            baked += c[2] as u64;
            most_baked = most_baked.max(c[2]);
            largest = largest.max(c[14]);
            asked += c[6] as u64;
            if at.contains(&f) {
                saved.push((f, camera, gray(&frame(&mut r, camera, &view_target)), c[8]));
            }
        }
        drop(r);
        frame_ms.sort_by(f64::total_cmp);
        say(format!(
            "walk: {frames} frames, {baked} pages baked ({:.1} a frame, at most {most_baked}; {:.0} rays a frame), {asked} asked in all; largest shift read {:.4} bin (bound 0.25); frame median {:.2} ms, worst {:.2} ms",
            baked as f64 / frames as f64,
            baked as f64 * 8192.0 / frames as f64,
            largest as f64 / 1e4,
            frame_ms[frames / 2],
            frame_ms[frames - 1]
        ));
        let mut c4 = true;
        for (f, camera, ours, used) in &saved {
            let (reference, one) =
                reference(&kernel, &field, origin, &lights, *camera, &view_target);
            let e_ours = relative_rmse(ours, &reference);
            let e_one = relative_rmse(&one, &reference);
            let ok = e_ours <= 0.3 * e_one;
            c4 &= ok;
            say(format!(
                "walk frame {f} ({:.2} s, {used} pages read): coverage {:.2}% from its reference, one ray {:.2}% ({:.3}×, at most 0.3×): {}",
                *f as f64 / 60.0,
                100.0 * e_ours,
                100.0 * e_one,
                e_ours / e_one,
                verdict(ok)
            ));
        }
        if !dev {
            pass &= c4;
        }
        say(format!("4. reuse in motion: {}", verdict(c4)));
    }
    kernel.set_profile_stage(0);
    if dev {
        return Ok(());
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate6.txt"), report)?;
    Ok(())
}
