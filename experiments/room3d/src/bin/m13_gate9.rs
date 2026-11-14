//! M13 gate 9: colour in the coverage pass (appearance, first).
//!
//! M13 gate 8 brought the coverage pass to about 9 ms settled
//! (`docs/results/m13-gate8.md`), coverage only and without the field's
//! ground. The design (DESIGN §3.4, "Appearance") adds colour with one
//! shading model evaluated everywhere, the reference included, and the
//! ground, which makes large objects a background with a depth.
//!
//! The model: colour = albedo × (E_sun · max(0, n·l) · V + E_amb) / π, l
//! and E_sun from the light's rectangle (toward its centre; its radiance
//! times its solid angle), V a hard shadow (one hardware ray toward the
//! light's centre), E_amb a fifth of the sun's normal irradiance, shading
//! normals; rays that miss see a constant sky.
//!
//! Method, on gate 8's pass:
//! - Large objects (mapless or over 8 pixels) as a background: a pixel with
//!   large bins shades one large hit by the model and keeps its depth; walks
//!   cull nodes behind it; small entries in front composite over its bins;
//!   what is left of each bin shows the large colour or the sky.
//! - Per entry and bin the pixel's colour accumulates the bin's
//!   transmittance times the entry's coverage and colour, over 16.
//! - Instances: maps hold coverage and the texels' mean normal; records add
//!   the light's direction in the object's frame, the instance's sun
//!   visibility (one ray a frame) and its albedo.
//! - Pages: texels add the mean colour of their rays' hits, shaded by the
//!   model at bake (a shadow ray a hit).
//!
//! Changed before the first run, after development at full size (accuracy
//! on other views only; criteria unchanged; `docs/results/m13-gate9.md`):
//! - The background: objects far larger than most (the ground) keep bit 0
//!   of their hardware-structure mask; a pixel's background is what its
//!   centre's ray seeing only those meets (else the sky), its distance per
//!   bin by that hit's plane; where the pixel's corners' background rays
//!   disagree (the ground's edge, the horizon), each bin traces its own.
//! - Large objects that are not background: up to two a pixel, claimed by
//!   the bin rays (pixels whose own corners hit one); the sky can be claimed
//!   by bins that miss where the centre saw ground.
//! - Large and background colours: the corners' shaded hits of the same
//!   object, bilinearly at each bin's centre.
//! - Maps (RGBA8) hold coverage, mean normal and mean depth; a bin that
//!   shows the background skips the part of an instance behind it (the
//!   boulders' buried parts); page bakes skip hits behind the background.
//! - Records carry the light's irradiance at the instance over the eye's
//!   (the light is 3 km away; across the field it varies by 10–15%).
//! - Page texels as words (coverage, colour as square roots of the page
//!   scale's share); a pool of 8,192 pages of 2,048 texels.
//!
//! Scene: the million-boulder field with its ground, its light, 1280×720;
//! views standing at 1.7 m and 150 m up looking 45° down (gates 4–8's).
//! Reference: per pixel the mean of 16×16 samples (offsets on a regular
//! grid), each one ray shaded by the model. One ray: the centre sample.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Colour:** relative RMSE of the RGB image from the reference at most
//!    0.5× one ray's, standing and from 150 m.
//! 2. **Speed:** settled (every wanted page baked), at most 15 ms a frame
//!    in both views (median of 10 frames each waited for), timed first.
//!
//! Reported: the 20 m view (gate 7's flight view) settled; pixels with large
//! bins; mean colour against the reference's.
//!
//! `cargo run --release -p room3d --bin m13_gate9`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders; `M13_DEV=1M` at
//! full size), whose numbers are not the gate's; with it `M13_ACC=1`
//! measures colour on other views than the gate's, and `M13_REF=1` renders
//! only the reference and the one-ray image (saved as PNGs).

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
/// M13 gate 9's colour pass and its reference (profile stages).
const OURS: u32 = 20;
const REFERENCE: u32 = 19;

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

/// The reference (the mean of 16×16 shaded samples a pixel) and the one-ray
/// image (the centre sample) at `camera`.
fn reference(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    kernel.set_hardware_rays(true);
    kernel.set_profile_stage(REFERENCE);
    let mut r = renderer(kernel, field, origin, lights);
    let one = frame(&mut r, camera, target);
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    for k in 0..256 {
        let (dx, dy) = (
            ((k % 16) as f32 + 0.5) / 16.0 - 0.5,
            ((k / 16) as f32 + 0.5) / 16.0 - 0.5,
        );
        r.set_pixel_offset(dx, dy);
        for (s, p) in sum.iter_mut().zip(frame(&mut r, camera, target)) {
            for (v, c) in s.iter_mut().zip(p) {
                *v += c / 256.0;
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
    let ref_only = dev && std::env::var_os("M13_REF").is_some();
    // The field with its ground (`M13_NOGROUND=1` in development: without).
    let mut field = Field::new(if dev && !full { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    if dev && std::env::var_os("M13_NOGROUND").is_some() {
        field.scene.instances.remove(0);
    }
    let lights = [Light::white(light, field.radiance)];
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
    // Development (`M13_ACC`, `M13_REF`): other views than the gate's.
    let views = if acc || ref_only {
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
    // The images tone mapped to 8 bits by the reference's 99th percentile.
    let save = |name: &str, px: &[[f32; 3]], scale: f32| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let c = p.map(|v| ((v / scale).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
            image.set(i % W as usize, i / W as usize, c);
        }
        image.save(&out_dir.join(format!("m13_gate9_{name}.png")))
    };
    let scale_of = |px: &[[f32; 3]]| {
        let mut l: Vec<f32> = px.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
        l.sort_by(f32::total_cmp);
        l[l.len() * 99 / 100].max(1e-6)
    };
    if ref_only {
        for (name, camera) in views {
            let start = Instant::now();
            let (reference, one) =
                reference(&kernel, &field, origin, &lights, camera, &view_target);
            let slug = name.replace(' ', "_").replace(['(', ')'], "");
            let scale = scale_of(&reference);
            save(&format!("{slug}_reference"), &reference, scale)?;
            save(&format!("{slug}_one_ray"), &one, scale)?;
            println!(
                "{name}: reference in {:.1} s; one ray {:.2}% from it",
                start.elapsed().as_secs_f64(),
                100.0 * relative_rmse(&one, &reference)
            );
        }
        return Ok(());
    }
    // Settled at `camera` (at most 200 frames of SETTLE_BUDGET bakes): the
    // median of 10 frames, the image and the pages' counters.
    let settled = |camera: Camera| {
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(OURS);
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
    // `M13_PROFILE=1` in development: each pass's time alone at the views,
    // settled.
    if dev && std::env::var_os("M13_PROFILE").is_some() {
        let names = [
            "prep",
            "roots",
            "corners",
            "bins",
            "bin args",
            "bin rays",
            "pixels",
            "page args",
            "bakes",
        ];
        for (name, camera) in views {
            kernel.set_prefilter(8.0);
            kernel.set_hardware_rays(true);
            kernel.set_profile_stage(OURS);
            let mut r = renderer(&kernel, &field, origin, &lights);
            r.set_m13_page_budget(SETTLE_BUDGET);
            settle(&mut r, camera, &view_target, 200, false);
            let mut per: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
            for _ in 0..5 {
                let p = r.frame_profiled(camera, &view_target);
                for (k, v) in per.iter_mut().enumerate() {
                    v.push(ms(p.passes[k]));
                }
            }
            let parts: Vec<String> = per
                .iter_mut()
                .zip(names)
                .map(|(v, n)| {
                    v.sort_by(f64::total_cmp);
                    format!("{n} {:.2}", v[2])
                })
                .collect();
            println!("{name}: {}", parts.join(", "));
        }
        return Ok(());
    }
    let mut pass = true;
    // 2. Speed, first.
    let mut ours_images = Vec::new();
    for (name, camera) in views {
        let (t, out, frames, baked, c) = settled(camera);
        let c2 = t <= 15.0;
        if !acc {
            pass &= c2;
        }
        say(format!(
            "{name}: 2. settled frame {t:.2} ms (at most 15): {}; settled in {frames} frames ({baked} pages baked), pages read {}",
            verdict(c2),
            c[8]
        ));
        ours_images.push(out);
    }
    if !acc {
        let (t, _, frames, baked, c) = settled(view(20.0, 10.0, 0.0));
        say(format!(
            "20 m up, 10° down (reported): settled frame {t:.2} ms; settled in {frames} frames ({baked} pages baked), pages read {}",
            c[8]
        ));
    }
    // 1. Colour.
    if !dev || acc {
        for (k, (name, camera)) in views.into_iter().enumerate() {
            let (reference, one) =
                reference(&kernel, &field, origin, &lights, camera, &view_target);
            let ours = &ours_images[k];
            let e_ours = relative_rmse(ours, &reference);
            let e_one = relative_rmse(&one, &reference);
            let c1 = e_ours <= 0.5 * e_one;
            if !acc {
                pass &= c1;
            }
            let mean = |px: &[[f32; 3]]| {
                let n = px.len() as f64;
                [0, 1, 2].map(|c| px.iter().map(|p| p[c] as f64).sum::<f64>() / n)
            };
            let (mr, mo) = (mean(&reference), mean(ours));
            say(format!(
                "{name}: 1. colour {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most 0.5×): {}; mean colour: reference ({:.4}, {:.4}, {:.4}), ours ({:.4}, {:.4}, {:.4})",
                100.0 * e_ours,
                100.0 * e_one,
                e_ours / e_one,
                verdict(c1),
                mr[0],
                mr[1],
                mr[2],
                mo[0],
                mo[1],
                mo[2]
            ));
            let slug = name.replace(' ', "_").replace(['(', ')'], "");
            let scale = scale_of(&reference);
            save(&format!("{slug}_reference"), &reference, scale)?;
            save(&format!("{slug}_ours"), ours, scale)?;
            save(&format!("{slug}_one_ray"), &one, scale)?;
        }
    }
    kernel.set_profile_stage(0);
    if dev {
        return Ok(());
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate9.txt"), report)?;
    Ok(())
}
