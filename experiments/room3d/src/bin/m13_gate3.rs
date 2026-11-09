//! M13 gate 3: sub-pixel geometry through directional coverage maps
//! (coverage only).
//!
//! M13 gate 2 failed: a scalar test per node was accurate off the horizon
//! (0.26–0.29× one ray's error) but lost correlation within a pixel at the
//! horizon (one row: 0.632 covered, ours 1.0), and took 257–281 ms a frame.
//! Development after it found the object walk alone takes 10.7–14.3 ms;
//! walking each small instance's triangle hierarchy per pixel was the cost
//! (`m13-gate2.md`).
//!
//! Method (profile stage 15, entry point `m13_coverage_maps`):
//! - **Sub-pixel bins:** each pixel keeps 4×4 bins of coverage; contributions
//!   combine per bin as independent occluders, and the pixel's coverage is
//!   the bins' mean.
//! - **The object walk** as gate 2's (screen-space boxes, culled by the
//!   pixel frustum, nearer child first); an object node within a pixel adds
//!   its prefiltered opacity spread over its projected box, per bin.
//! - **Small instances read a map:** each geometry has a directional
//!   coverage map, built once on the CPU: its opacity seen from 256
//!   directions (an octahedral grid of 16×16), each over its bounding
//!   sphere at 16×16 texels (4×4 samples a texel) with mip levels down to
//!   one texel. An instance at most 8 pixels across reads, per bin, the
//!   maps of the four directions nearest the view (blended bilinearly) at
//!   the level matching the bin's footprint.
//! - **Larger instances** walk their triangle hierarchy as in gate 2, per
//!   bin: nodes within a pixel splat by their projected box, triangles
//!   cover the bins whose centres they contain (front faces).
//!
//! Scene, views and reference are gate 2's: the million-boulder field
//! without its ground (in the reference and the test alike), standing at
//! 1.7 m and 150 m up, 1280×720; per pixel, the share of 16×16 rays through
//! its square that hit a boulder. One ray: the central ray alone.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.5× one ray's.
//! 2. **Speed:** the coverage pass at most 10 ms a frame at 1280×720 (median
//!    of 10 frames each waited for).
//!
//! Reported: the maps' build time and size; object and triangle nodes per
//! pixel.
//!
//! `cargo run --release -p room3d --bin m13_gate3`
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
        image.save(&out_dir.join(format!("m13_gate3_{name}.png")))
    };
    let mut pass = true;
    for (name, camera) in views {
        let slug = name.replace(' ', "_");
        // 2. Speed, first: the scalar coverage pass (the prefilter on, so
        // object nodes keep their coverage).
        kernel.set_prefilter(8.0);
        kernel.set_profile_stage(15);
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
        let (build, bytes) = kernel.m13_maps_info();
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
            "  per pixel {:.1} object nodes (at most {:.0}), {:.1} triangle nodes (at most {:.0}); maps built in {:.2} s, {:.1} MB",
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
    std::fs::write(out_dir.join("m13_gate3.txt"), report)?;
    Ok(())
}
