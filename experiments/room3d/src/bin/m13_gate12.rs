//! M13 gate 12 (stub): the sub-pixel path in the renderer, lit by its cache.
//!
//! Gates 5–11 ran sub-pixel coverage and colour as a diagnostic profile
//! stage with its own lighting model (one sun, a constant ambient); gate 11
//! met colour's bar there (0.335× and 0.386× one ray's error, 13.85 and
//! 14.69 ms settled; `docs/results/m13-gate11.md`). DESIGN §3.4, "Into the
//! renderer", brings it into the renderer as an option whose colour comes
//! from the renderer's own lighting (the cache: every light's area, the
//! sky, bounce light), not a second model.
//!
//! Method (to be developed; DESIGN §3.4, parts A–D):
//! - A. `RendererOptions` gains the sub-pixel camera path for instanced
//!   scenes in place of one ray a pixel; the lighting cache bakes as now.
//! - B. Large hits (corners, bin rays, samples of pixels the corners
//!   resolved) are shaded as the renderer's shade pass shades its hit.
//! - C. Small instances (map entries): direct light per light from the
//!   maps' normals and a visibility per instance and light; sky and bounce
//!   from the cache's coarsest level for the instance (or its node's).
//! - D. Coverage pages bake colour from the same shading.
//!
//! Scene: the million-boulder field with its ground, its lights and a sky,
//! 1280×720; gate 11's views (standing at 1.7 m, 150 m up looking 45° down).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Colour:** relative RMSE of the RGB image from a reference at most
//!    0.5× one ray's, in both views. The reference: per pixel the mean of
//!    16×16 samples, each one ray shaded by the renderer's own shading with
//!    the cache settled; one ray: the centre sample, shaded the same way.
//! 2. **Speed:** settled (the cache settled, every wanted page baked), at
//!    most 16.7 ms a frame in both views (median of 10 frames each waited
//!    for), timed first.
//!
//! Reported: the frame with one ray a pixel and the cache (the renderer's
//! present path) in the same views; the 20 m view settled.
//!
//! `cargo run --release -p room3d --bin m13_gate12`

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
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

/// A renderer with the lighting cache: the sub-pixel path (`sub`) or one
/// ray a pixel.
fn cached(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    sub: bool,
) -> InstancedRenderer {
    kernel.set_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_profile_stage(if sub { 20 } else { 0 });
    kernel.set_pixel_cones(sub);
    let r = InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        lights,
        &field.scene,
        origin,
        CacheOptions::default(),
    );
    kernel.set_pixel_cones(false);
    kernel.set_profile_stage(0);
    r
}

/// Frames until `quiet` frames in a row ask for and bake no page (at most
/// `max`), and (the sub-pixel path) its pages settle too: the frames run.
fn settle(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    quiet: usize,
    max: usize,
) -> usize {
    let mut calm = 0;
    for k in 1..=max {
        r.frame(camera, target);
        r.wait();
        let s = r.cache_stats();
        let pages = r.m13_page_counters()[6];
        if s.is_some_and(|s| s.requests == 0 && s.pages_baked == 0) && pages == 0 {
            calm += 1;
            if calm >= quiet {
                return k;
            }
        } else {
            calm = 0;
        }
    }
    max
}

fn frame(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Vec<[f32; 3]> {
    r.frame(camera, target);
    r.finish();
    r.read_colors()
}

fn main() {
    // `M13_BINDINGS=1` in development: each entry point's bindings, from
    // naga, in the model's module and the cache-lit one.
    if std::env::var_os("M13_BINDINGS").is_some() {
        let Some(kernel) = MaskConeKernel::new() else {
            println!("No GPU adapter available.");
            return;
        };
        for name in [
            "hw_m13_prep",
            "hw_m13_tile_roots",
            "hw_m13_corners_tiles",
            "hw_m13_bins",
            "hw_m13_bin_args",
            "hw_m13_bin_rays",
            "hw_m13_shade_args",
            "hw_m13_shade_rays",
            "hw_m13_tiles",
            "hw_m13_page_args",
            "hw_m13_page_bake",
        ] {
            println!(
                "{name}: model {:?}; cache {:?}",
                kernel.entry_bindings(name, false),
                kernel.entry_bindings(name, true)
            );
        }
        return;
    }
    // `M13_DEV=1` (10,000 boulders; `1M` at full size): the sub-pixel path
    // lit by the cache next to one ray a pixel, both settled, on other views
    // than the gate's; images saved, mean colours and the difference.
    if let Ok(v) = std::env::var("M13_DEV") {
        let Some(kernel) = MaskConeKernel::new() else {
            println!("No GPU adapter available.");
            return;
        };
        let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
        std::fs::create_dir_all(&out_dir).expect("out");
        let field = Field::new(if v == "1M" { 1_000_000 } else { 10_000 }, DVec3::ZERO);
        let origin = field.origin_near_camera();
        let (ground, light) = field.relative(origin);
        let lights = [Light::white(light, field.radiance)];
        kernel.set_sky(field.sky());
        let view_target = target(&kernel);
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
        let save = |name: &str, px: &[[f32; 3]], scale: f32| {
            let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                let c =
                    p.map(|v| ((v / scale).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
                image.set(i % W as usize, i / W as usize, c);
            }
            image
                .save(&out_dir.join(format!("m13_gate12_{name}.png")))
                .expect("save");
        };
        for (name, camera) in [
            ("standing_dev", view(1.7, 0.0, 20.0)),
            ("120m_dev", view(120.0, 40.0, -20.0)),
        ] {
            let mut one = cached(&kernel, &field, origin, &lights, false);
            let n1 = settle(&mut one, camera, &view_target, 8, 3000);
            let a = frame(&mut one, camera, &view_target);
            // `M13_ACC=1`: the reference, 16×16 samples a pixel shaded by the
            // renderer (its cache settled first), and the colour error.
            let reference = std::env::var_os("M13_ACC").is_some().then(|| {
                let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
                for k in 0..256 {
                    let (dx, dy) = (
                        ((k % 16) as f32 + 0.5) / 16.0 - 0.5,
                        ((k / 16) as f32 + 0.5) / 16.0 - 0.5,
                    );
                    one.set_pixel_offset(dx, dy);
                    for (s, p) in sum.iter_mut().zip(frame(&mut one, camera, &view_target)) {
                        for (v, c) in s.iter_mut().zip(p) {
                            *v += c / 256.0;
                        }
                    }
                }
                sum
            });
            drop(one);
            let mut sub = cached(&kernel, &field, origin, &lights, true);
            sub.set_m13_page_budget(512);
            let n2 = settle(&mut sub, camera, &view_target, 8, 3000);
            let mut times: Vec<f64> = (0..10)
                .map(|_| {
                    let t = Instant::now();
                    sub.frame(camera, &view_target);
                    sub.wait();
                    t.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            times.sort_by(f64::total_cmp);
            let b = frame(&mut sub, camera, &view_target);
            drop(sub);
            let mean = |px: &[[f32; 3]]| {
                let n = px.len() as f64;
                [0, 1, 2].map(|c| px.iter().map(|p| p[c] as f64).sum::<f64>() / n)
            };
            let mut l: Vec<f32> = a.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
            l.sort_by(f32::total_cmp);
            let scale = l[l.len() * 99 / 100].max(1e-6);
            save(&format!("{name}_one_ray"), &a, scale);
            save(&format!("{name}_sub"), &b, scale);
            if let Some(reference) = &reference {
                let e_ours = relative_rmse(&b, reference);
                let e_one = relative_rmse(&a, reference);
                println!(
                    "{name}: colour {:.2}% from the reference, one ray {:.2}% ({:.3}x); means reference {:?}",
                    100.0 * e_ours,
                    100.0 * e_one,
                    e_ours / e_one,
                    mean(reference)
                );
                save(&format!("{name}_reference"), reference, scale);
            }
            println!(
                "{name}: one ray settled in {n1} frames, sub-pixel in {n2} ({:.2} ms settled); means one ray {:?}, sub-pixel {:?}; relative difference {:.2}%",
                times[5],
                mean(&a),
                mean(&b),
                100.0 * relative_rmse(&b, &a)
            );
        }
        return;
    }
    println!(
        "M13 gate 12 is a stub: the method is not implemented yet (DESIGN §3.4, \"Into the renderer\")."
    );
}
