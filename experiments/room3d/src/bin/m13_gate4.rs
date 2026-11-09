//! M13 gate 4: sub-pixel geometry, near surfaces resolved by hardware rays
//! (coverage only).
//!
//! M13 gate 3 passed accuracy in both views (0.264× and 0.231× one ray's
//! error) with directional coverage maps and 4×4 sub-pixel bins, but took
//! 178–707 ms a frame. Development after it (`m13-gate3.md`) found walking
//! the triangles of instances larger than 8 pixels, per pixel, was 75–180
//! ms of about 200 at 10,000 boulders; the maps and the object walk about
//! 25 ms. Those large instances are near, resolved surfaces, where one ray
//! is right except at their edges.
//!
//! Method (profile stage 16):
//! 1. A hardware ray through each pixel's top-left corner (one per pixel,
//!    `hw_m13_corners`) records the object it hits and whether that object
//!    is large (more than 8 pixels across, or without a coverage map).
//! 2. Each pixel (`hw_m13_near`) whose four corners all hit the same large
//!    object is wholly covered by it. Every other pixel, and the last row
//!    and column, casts a hardware ray through each of its 4×4 bins'
//!    centres: a bin whose ray meets a large object is covered. Then gate
//!    3's walk (its development changes included: nodes over full bins
//!    skipped, far object nodes no taller than a bin splatted, map
//!    addressing hoisted) adds the small instances through their maps and
//!    the far object nodes, leaving large instances out.
//!
//! Changed before the first run, after a development run (`M13_DEV=1`,
//! disclosed): the first version walked large instances' triangles in every
//! pixel the corners did not resolve (standing 162 ms, 25.6% of pixels
//! resolved by corners; 150 m up 82 ms, 3.6%); the bins' rays replace those
//! walks. The criteria are unchanged.
//!
//! Scene, views and reference are gate 3's: the million-boulder field
//! without its ground (in the reference and the test alike), standing at
//! 1.7 m and 150 m up, 1280×720; per pixel, the share of 16×16 rays through
//! its square that hit a boulder. One ray: the central ray alone.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.5× one ray's (gate 3's bar).
//! 2. **Speed:** both passes at most 50 ms a frame at 1280×720 (median of 10
//!    frames each waited for). An intermediate bar: M13's goal stays a
//!    10 ms coverage pass, for which the far field's walk must still get
//!    cheaper.
//!
//! Reported: the share of pixels resolved by their corners; object and
//! triangle nodes per pixel.
//!
//! `cargo run --release -p room3d --bin m13_gate4`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the passes alone),
//! whose numbers are not the gate's.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;

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

    // `M13_DEV=1`: a development check (10,000 boulders, the pass alone, no
    // reference), whose numbers are not the gate's.
    let dev = std::env::var_os("M13_DEV").is_some();
    let mut field = Field::new(if dev { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    // The ground left out, for the reference and the test alike.
    field.scene.instances.remove(0);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let field_view = |height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", field_view(1.7, 0.0)),
        ("150 m up", field_view(150.0, 45.0)),
    ];
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let v = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.set(i % W as usize, i / W as usize, [v; 3]);
        }
        image.save(&out_dir.join(format!("m13_gate4_{name}.png")))
    };
    let mut pass = true;
    for (name, camera) in views {
        let slug = name.replace(' ', "_");
        // 2. Speed, first: the scalar coverage pass (the prefilter on, so
        // object nodes keep their coverage).
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(16);
        let mut r = renderer(&kernel, &field, origin, &lights);
        frame(&mut r, camera, &view_target);
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
        let ours = gray(&out);
        let mean = |c: usize| out.iter().map(|p| p[c] as f64).sum::<f64>() / out.len() as f64;
        let max = |c: usize| out.iter().map(|p| p[c]).fold(0.0f32, f32::max);
        let work = (mean(1), max(1), mean(2), max(2));
        let resolved = out.iter().filter(|p| p[1] == 0.0).count() as f64 / out.len() as f64;
        let (build, bytes) = kernel.m13_maps_info();
        drop(r);
        if dev {
            println!(
                "DEVELOPMENT {name}: {ours_ms:.2} ms, mean coverage {:.4}, {:.1}% resolved by corners, per pixel {:.1} object nodes, {:.1} triangle nodes (at most {:.0})",
                mean(0),
                100.0 * resolved,
                work.0,
                work.2,
                work.3
            );
            save(&format!("{slug}_dev"), &ours)?;
            continue;
        }
        // The reference: 16×16 rays per pixel; one ray: the central one.
        kernel.set_profile_stage(13);
        let mut r = renderer(&kernel, &field, origin, &lights);
        let one = gray(&frame(&mut r, camera, &view_target));
        let mut reference = vec![[0.0f32; 3]; (W * H) as usize];
        for k in 0..256 {
            let (dx, dy) = (
                ((k % 16) as f32 + 0.5) / 16.0 - 0.5,
                ((k / 16) as f32 + 0.5) / 16.0 - 0.5,
            );
            r.set_pixel_offset(dx, dy);
            for (s, p) in reference
                .iter_mut()
                .zip(frame(&mut r, camera, &view_target))
            {
                for v in s.iter_mut() {
                    *v += p[0] / 256.0;
                }
            }
        }
        drop(r);
        kernel.set_profile_stage(0);
        let e_ours = relative_rmse(&ours, &reference);
        let e_one = relative_rmse(&one, &reference);
        let c1 = e_ours <= 0.5 * e_one;
        let c2 = ours_ms <= 50.0;
        pass &= c1 && c2;
        let mean_cov =
            |px: &[[f32; 3]]| px.iter().map(|p| p[0] as f64).sum::<f64>() / px.len() as f64;
        say(format!(
            "{name}: 1. coverage {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most 0.5×): {}; 2. {:.2} ms a frame (at most 50): {}; mean coverage: reference {:.4}, ours {:.4}, one ray {:.4}",
            100.0 * e_ours,
            100.0 * e_one,
            e_ours / e_one,
            verdict(c1),
            ours_ms,
            verdict(c2),
            mean_cov(&reference),
            mean_cov(&ours),
            mean_cov(&one)
        ));
        say(format!(
            "  {:.1}% of pixels resolved by their corners; per pixel {:.1} object nodes (at most {:.0}), {:.1} triangle nodes (at most {:.0}); maps built in {:.2} s, {:.1} MB",
            100.0 * resolved,
            work.0,
            work.1,
            work.2,
            work.3,
            build.as_secs_f64(),
            bytes as f64 / 1e6
        ));
        save(&format!("{slug}_reference"), &reference)?;
        save(&format!("{slug}_ours"), &ours)?;
        save(&format!("{slug}_one_ray"), &one)?;
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate4.txt"), report)?;
    Ok(())
}
