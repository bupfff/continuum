//! M13 gate 8: the coverage pass in 10 ms (coverage only).
//!
//! After M13 gates 5–7 the settled pass takes 16.1 ms standing and 17.2
//! ms from 150 m up at full size, the pixels' pass 72–76% of it; from
//! 150 m the instances' set-up alone is 5.0 ms, computed again by every
//! subgroup that touches an instance (`docs/results/m13-10ms-profile.md`).
//! The design (DESIGN §3.4, "The 10 ms bar") moves per-instance and
//! per-page work into passes that run once a frame.
//!
//! Method, on gate 7's pass (stage 18):
//! - Instance records: one invocation per object writes, for each small
//!   instance in front of the eye and inside the view, its projected
//!   centre and radius, its maps' level, two directions' layers and
//!   weights, and the affine map from screen position to each direction's
//!   map coordinates; the walk's items read the record.
//! - Page records: one invocation per page in use decides its validity
//!   from the eye (and its refresh) and the homography from screen
//!   position to its texels; page reads use the record.
//! - The other passes trimmed where free.
//!
//! Changed before the first run, after development at full size (speed on
//! the gate's views, accuracy only on other views; criteria unchanged;
//! `docs/results/m13-gate8.md` has the steps and their numbers):
//! - The once-a-frame passes are one dispatch (pages, nodes, objects), the
//!   nodes' projections included: walks and the roots pass read a node's
//!   rectangle unless it straddles the eye's plane.
//! - Instance records name the nearest of 32×32 directions (the maps now
//!   have 32×32; as accurate in development as two of 16×16 blended), 32
//!   bytes each, from static per-object data (sphere, rotation quaternion,
//!   first layer); they also tell corner and bin rays whether a hit is
//!   large (no record: large), and those rays are plain opaque queries.
//! - Pages hold 2,048 texels (shapes 4×512 to 512×4), so wider horizon nodes
//!   fit; the pool is 16,384 pages (32 MB of texels).
//! - Each tile's leaf roots' instances are listed once in workgroup memory
//!   and read by its pixels; leaves never want pages.
//! - Bin rays where one of the pixel's own corners (not its 3×3 block's)
//!   hits a large object; corner rays only in tiles whose roots may hold a
//!   large instance (each node's largest instance at its nearest depth).
//! - Map reads skip bins outside the map's rectangle of non-zero texels.
//!
//! Scene, views and reference are gates 4–6's: the million-boulder field
//! without its ground, standing at 1.7 m and 150 m up looking 45° down,
//! 1280×720; per pixel the share of 16×16 rays through its square that
//! hit a boulder; one ray the central one. Pages as gate 7 (settled at 512
//! bakes a frame).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed:** settled, at most 10 ms a frame standing and from 150 m up
//!    (median of 10 frames each waited for), timed first.
//! 2. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.25× one ray's standing and 0.15× from 150 m (gate 6's
//!    bars; gate 6 measured 0.212× and 0.135×).
//!
//! Reported: the settled frame 20 m up looking 10° down (gate 7's flight
//! view); each pass's time alone in a development profile after the gate.
//!
//! `cargo run --release -p room3d --bin m13_gate8`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders; `M13_DEV=1M` at
//! full size), whose numbers are not the gate's; with it `M13_ACC=1`
//! measures accuracy on other views than the gate's.

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
    let mut field = Field::new(if dev && !full { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    // The ground left out, for the reference and the test alike.
    field.scene.instances.remove(0);
    let forward = (ground.target - ground.eye).normalize();
    let flat0 = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down_degrees: f32, turn_degrees: f32| {
        let d = down_degrees.to_radians();
        let (s, c) = turn_degrees.to_radians().sin_cos();
        let flat = Vec3::new(flat0.x * c - flat0.y * s, flat0.x * s + flat0.y * c, 0.0);
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    // `M13_ACC=1` in development: other views than the gate's.
    let views = if acc {
        [
            ("standing (development view)", view(1.7, 0.0, 20.0)),
            ("120 m up (development view)", view(120.0, 40.0, -20.0)),
        ]
    } else {
        [
            ("standing", view(1.7, 0.0, 0.0)),
            ("150 m up", view(150.0, 45.0, 0.0)),
        ]
    };
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let v = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.set(i % W as usize, i / W as usize, [v; 3]);
        }
        image.save(&out_dir.join(format!("m13_gate8_{name}.png")))
    };
    // Settled at `camera` (at most 200 frames of SETTLE_BUDGET bakes): the
    // median of 10 frames, the image and the pages' counters.
    let settled = |camera: Camera| {
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(18);
        let mut r = renderer(&kernel, &field, origin, &lights);
        r.set_m13_page_budget(SETTLE_BUDGET);
        let (frames, baked) = settle(&mut r, camera, &view_target, 200, false);
        let mut times: Vec<Duration> = (0..10)
            .map(|_| {
                let start = Instant::now();
                r.frame(camera, &view_target);
                r.wait();
                start.elapsed()
            })
            .collect();
        times.sort();
        let out = frame(&mut r, camera, &view_target);
        let c = r.m13_page_counters();
        (ms(times[5]), out, frames, baked, c)
    };
    let mut pass = true;
    // 1. Speed, first.
    let mut ours_images = Vec::new();
    for (k, (name, camera)) in views.into_iter().enumerate() {
        let (t, out, frames, baked, c) = settled(camera);
        let c1 = t <= 10.0;
        if !acc {
            pass &= c1;
        }
        let n = out.len() as f64;
        let _ = k;
        say(format!(
            "{name}: 1. settled frame {t:.2} ms (at most 10): {}; settled in {frames} frames ({baked} pages baked), pages read {}, roots a tile {:.1}; mean coverage {:.4}",
            verdict(c1),
            c[8],
            out.iter().map(|p| (p[2] as u32 / 2) as f64).sum::<f64>() / n,
            out.iter().map(|p| p[0] as f64).sum::<f64>() / n
        ));
        ours_images.push(gray(&out));
    }
    // Reported: gate 7's flight view, settled.
    if !acc {
        let (t, _, frames, baked, c) = settled(view(20.0, 10.0, 0.0));
        say(format!(
            "20 m up, 10° down (reported): settled frame {t:.2} ms; settled in {frames} frames ({baked} pages baked), pages read {}",
            c[8]
        ));
    }
    // 2. Accuracy.
    if !dev || acc {
        for (k, (name, camera)) in views.into_iter().enumerate() {
            let (reference, one) =
                reference(&kernel, &field, origin, &lights, camera, &view_target);
            let ours = &ours_images[k];
            let e_ours = relative_rmse(ours, &reference);
            let e_one = relative_rmse(&one, &reference);
            let bar = if k == 0 { 0.25 } else { 0.15 };
            let c2 = e_ours <= bar * e_one;
            if !acc {
                pass &= c2;
            }
            let mean =
                |px: &[[f32; 3]]| px.iter().map(|p| p[0] as f64).sum::<f64>() / px.len() as f64;
            say(format!(
                "{name}: 2. coverage {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most {bar}×): {}; mean coverage: reference {:.4}, ours {:.4}, one ray {:.4}",
                100.0 * e_ours,
                100.0 * e_one,
                e_ours / e_one,
                verdict(c2),
                mean(&reference),
                mean(ours),
                mean(&one)
            ));
            if !dev {
                let slug = name.replace(' ', "_");
                save(&format!("{slug}_reference"), &reference)?;
                save(&format!("{slug}_ours"), ours)?;
            }
        }
    }
    kernel.set_profile_stage(0);
    if dev {
        return Ok(());
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate8.txt"), report)?;
    Ok(())
}
