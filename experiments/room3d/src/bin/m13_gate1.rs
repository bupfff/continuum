//! M13 gate 1: sub-pixel geometry through prefiltered pixel cones
//! (coverage only).
//!
//! One ray per pixel sees one surface per pixel, so wherever a pixel holds
//! many (the boulder field from afar) the image is noise: 23.49% from the
//! area-integrated image from 150 m (M7 gate 1). M7 gate 1's pixel cones
//! resolved every pixel to an eighth of a pixel and cost hundreds of
//! hierarchy nodes per pixel (10–87 s a frame on the field). M13's plan
//! (DESIGN §3.4, §3.8): a pixel's cone stops at hierarchy nodes about its
//! own size and reads their prefiltered summaries, so cost follows the
//! screen, not the scene. This gate tests the geometric half alone: how
//! much of each pixel is covered by geometry. Appearance (color and
//! light) is the next gate's.
//!
//! Method: each pixel's cone (its square seen from the eye, 8×8 bins) walks
//! the object hierarchy and the triangle hierarchies as light cones do, but
//! with the prefilter threshold at the pixel (8 bins): object nodes and
//! triangle nodes that project within a pixel splat their prefiltered
//! coverage (front area over silhouette; objects as independent
//! occluders), larger ones are descended. Coverage combines across meshes as
//! independent occluders. The ground is left out: the measure is how much
//! of each pixel the boulders cover.
//!
//! Scene: the million-boulder field (M4), M7 gate 1's two viewpoints
//! (standing at 1.7 m; 150 m up, looking 45° down), 1280×720.
//!
//! Reference: per pixel, the share of 16×16 rays through its square (a grid
//! of sub-pixel offsets) that hit a boulder. One ray: the same with the
//! pixel's central ray alone (0 or 1).
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Accuracy:** relative RMSE of the boulders' per-pixel coverage from
//!    the reference at most 0.5× one ray's.
//! 2. **Speed:** the coverage pass at most 16.7 ms a frame at 1280×720
//!    (median of 10 frames each waited for, hardware or compute as the
//!    kernel chooses; the pass alone, no shading).
//! Reported: work per pixel (object nodes, meshes entered, triangle nodes),
//! and the same with M7 gate 1's threshold (one bin) for contrast.
//!
//! `cargo run --release -p room3d --bin m13_gate1`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, InstancedRenderer, KernelOptions, Light, MaskConeKernel, wgpu};
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
fn renderer(kernel: &MaskConeKernel, field: &Field, origin: DVec3, lights: &[Light]) -> InstancedRenderer {
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
    let counting = MaskConeKernel::new_with(KernelOptions {
        count_work: true,
        ..Default::default()
    })
    .expect("a GPU");
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let view_target = target(&kernel);
    let count_target = target(&counting);

    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
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
        image.save(&out_dir.join(format!("m13_gate1_{name}.png")))
    };
    let mut pass = true;
    for (name, camera) in views {
        let slug = name.replace(' ', "_");
        // 2. Speed, first: the coverage pass, prefiltered at the pixel.
        kernel.set_prefilter(8.0);
        kernel.set_profile_stage(10);
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
        let ours = gray(&frame(&mut r, camera, &view_target));
        drop(r);
        // The reference: 16×16 rays per pixel; one ray: the central one.
        kernel.set_profile_stage(11);
        let mut r = renderer(&kernel, &field, origin, &lights);
        let one = gray(&frame(&mut r, camera, &view_target));
        let mut reference = vec![[0.0f32; 3]; (W * H) as usize];
        for k in 0..256 {
            let (dx, dy) = (((k % 16) as f32 + 0.5) / 16.0 - 0.5, ((k / 16) as f32 + 0.5) / 16.0 - 0.5);
            r.set_pixel_offset(dx, dy);
            for (s, p) in reference.iter_mut().zip(frame(&mut r, camera, &view_target)) {
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
        let c2 = ours_ms <= 16.7;
        pass &= c1 && c2;
        let mean_cov = |px: &[[f32; 3]]| px.iter().map(|p| p[0] as f64).sum::<f64>() / px.len() as f64;
        say(format!(
            "{name}: 1. coverage {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most 0.5×): {}; 2. {:.2} ms a frame (at most 16.7): {}; mean coverage: reference {:.4}, ours {:.4}, one ray {:.4}",
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
        // Work per pixel, at the pixel's threshold and at one bin.
        let mut work = Vec::new();
        for bins in [8.0, 1.0] {
            counting.set_prefilter(bins);
            counting.set_profile_stage(10);
            let mut r = renderer(&counting, &field, origin, &lights);
            let start = Instant::now();
            let px = frame(&mut r, camera, &count_target);
            let took = start.elapsed();
            let mean = |c: usize| px.iter().map(|p| p[c] as f64).sum::<f64>() / px.len() as f64;
            let max = |c: usize| px.iter().map(|p| p[c]).fold(0.0f32, f32::max);
            work.push((bins, mean(1), mean(2), max(2), ms(took), relative_rmse(&gray(&px), &reference)));
        }
        counting.set_profile_stage(0);
        for (bins, objects, tri_nodes, tri_max, took, err) in work {
            say(format!(
                "  threshold {bins} bins: per pixel {objects:.1} object nodes, {tri_nodes:.1} triangle nodes (at most {tri_max:.0}); counting frame {took:.0} ms; coverage {:.2}% from the reference",
                100.0 * err
            ));
        }
        save(&format!("{slug}_reference"), &reference)?;
        save(&format!("{slug}_ours"), &ours)?;
        save(&format!("{slug}_one_ray"), &one)?;
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate1.txt"), report)?;
    Ok(())
}
