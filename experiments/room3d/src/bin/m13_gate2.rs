//! M13 gate 2: sub-pixel geometry, a scalar per-node test (coverage only).
//!
//! M13 gate 1 failed: prefiltering each pixel's cone at the pixel halved
//! one ray's coverage error from 150 m (0.38×), but each node visit
//! rasterized the node into the cone's 8×8 bins, about 9 ns a visit across
//! the GPU: 1.2–2 s a frame. Its standing view also measured badly: the
//! ground hid buried boulder parts from the reference's rays but not from
//! the cone walk (`m13-gate1.md`).
//!
//! Method (profile stage 12 of the pixel-cone entry point): each pixel walks
//! the object hierarchy, then each object's triangle hierarchy, testing
//! boxes in screen space (the 8 corners projected):
//! - a box whose projection misses the pixel is skipped;
//! - a box within one pixel (its projection's longer side) adds the share of
//!   the pixel its projected box covers times its prefiltered opacity seen
//!   from the eye (object nodes 1 − exp(−front area / silhouette), triangle
//!   nodes front area / silhouette), with no bins;
//! - a larger box is opened; a triangle larger than a pixel adds the exact
//!   area of the pixel it covers (front faces only).
//!
//! Within an object, contributions add (its front faces do not overlap) up
//! to 1; objects combine as independent occluders. The walk stops once the
//! pixel is covered. Boxes crossing the eye's near plane are opened down to
//! triangles, and triangles crossing it are skipped.
//!
//! Changed before the first run, after development runs (`M13_DEV=1`, 10,000
//! boulders, disclosed): children are opened nearer first; points go to
//! camera space through one transform per object; and boxes are also culled
//! against the pixel frustum's four side planes, which hold on both sides
//! of the eye (without it, every pixel opened the boxes of boulders beside
//! the camera: standing took 880 ms with 2,622 triangle nodes per pixel;
//! with it, 190 ms with 50; 150 m up, 114 ms with 65 nodes per pixel). The
//! criteria are unchanged.
//!
//! Scene: the million-boulder field with its ground left out (for the
//! reference and the test alike: the measure is the boulders' coverage,
//! occlusion by resolved surfaces being later work), M7 gate 1's two
//! viewpoints (standing at 1.7 m; 150 m up, 45° down), 1280×720.
//!
//! Reference: per pixel, the share of 16×16 rays through its square (a grid
//! of sub-pixel offsets) that hit a boulder. One ray: the central ray alone.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.5× one ray's.
//! 2. **Speed:** the coverage pass at most 10 ms a frame at 1280×720 (median
//!    of 10 frames each waited for), leaving room for shading within a
//!    60 Hz frame.
//!
//! Reported: nodes and triangles per pixel.
//!
//! `cargo run --release -p room3d --bin m13_gate2`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the pass alone),
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
        image.save(&out_dir.join(format!("m13_gate2_{name}.png")))
    };
    let mut pass = true;
    for (name, camera) in views {
        let slug = name.replace(' ', "_");
        // 2. Speed, first: the scalar coverage pass (the prefilter on, so
        // object nodes keep their coverage).
        kernel.set_prefilter(8.0);
        kernel.set_profile_stage(12);
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
        drop(r);
        if dev {
            println!(
                "DEVELOPMENT {name}: {ours_ms:.2} ms, mean coverage {:.4}, per pixel {:.1} object nodes, {:.1} triangle nodes (at most {:.0})",
                mean(0),
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
                for c in 0..3 {
                    s[c] += p[0] / 256.0;
                }
            }
        }
        drop(r);
        kernel.set_profile_stage(0);
        let e_ours = relative_rmse(&ours, &reference);
        let e_one = relative_rmse(&one, &reference);
        let c1 = e_ours <= 0.5 * e_one;
        let c2 = ours_ms <= 10.0;
        pass &= c1 && c2;
        let mean_cov =
            |px: &[[f32; 3]]| px.iter().map(|p| p[0] as f64).sum::<f64>() / px.len() as f64;
        say(format!(
            "{name}: 1. coverage {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most 0.5×): {}; 2. {:.2} ms a frame (at most 10): {}; mean coverage: reference {:.4}, ours {:.4}, one ray {:.4}",
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
            "  per pixel {:.1} object nodes (at most {:.0}), {:.1} triangle nodes (at most {:.0})",
            work.0, work.1, work.2, work.3
        ));
        save(&format!("{slug}_reference"), &reference)?;
        save(&format!("{slug}_ours"), &ours)?;
        save(&format!("{slug}_one_ray"), &one)?;
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate2.txt"), report)?;
    Ok(())
}
