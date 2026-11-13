//! M13 gate 5: the coverage pass with exact culling, the bins' rays only
//! near large objects, one walk per tile and lighter map reads (coverage
//! only).
//!
//! M13 gate 4 passed accuracy (0.218× one ray's coverage error standing,
//! 0.132× from 150 m) but took 122–135 ms against a 50 ms bar. Profiled at
//! full size after it (`docs/results/m13-far-field-profile.md`): from 150 m
//! (104 ms with the packet walk) the bins' rays take about 51 ms, the maps'
//! reads about 24 and a root-to-instance walk in every pixel about 27;
//! standing (122 ms) the sky's pixels walked boxes straddling the eye's
//! plane for nothing, and the horizon walks thousands of nodes. The design
//! (DESIGN §3.4, "The far field for sub-pixel coverage") splits the work in
//! five parts; this gate takes A, B, C and E, leaving the horizon's far
//! field (D, coverage pages for hierarchy nodes) to gate 6.
//!
//! Method (gate 4's corner rays first):
//! - A. A box is also rejected when the pixel's or tile's frustum lies
//!   wholly on the eye's side of one of its faces (exact).
//! - B. A pixel traces its 16 bin rays only if one of the 16 corners of its
//!   3×3 block of pixels hits a large object (more than 8 pixels across, or
//!   without a coverage map) and its own four corners do not all hit the
//!   same one.
//! - C. Each 8×8 tile walks the object hierarchy once against its frustum,
//!   nearest first, listing its entries (instances read from their maps,
//!   object nodes splatted as in gate 4) in workgroup memory in chunks;
//!   after each chunk every pixel adds the entries over its open bins, and
//!   the walk ends when every pixel's bins are full or the hierarchy is
//!   exhausted.
//! - E. The instances' maps as 8-bit texels in a texture array (a layer per
//!   direction), one hardware-filtered sample per bin and direction.
//!
//! Changed before the first run, after development at full size
//! (`docs/results/m13-gate5-development.md`; criteria unchanged):
//! - C as two passes: one invocation per tile walks to the roots of
//!   subtrees at most 8 pixels across (at most 63 a tile, the rest listed
//!   as they stand), and each pixel walks those subtrees nearest first, as a
//!   packet (gate 4's walk from lower roots), until its bins are full. A
//!   single invocation walking for the tile's 64 held the other 63 idle.
//! - B in its own pass, writing each pixel's covered bins.
//! - E reads two directions' maps (the nearest along the axis of the
//!   directions' grid whose fraction is nearer a half, blended) instead of
//!   four, with each bin's map coordinates affine in its screen position
//!   about the instance's centre ray; open bins kept as a bit mask.
//!
//! Scene, views and reference are gate 4's: the million-boulder field
//! without its ground (in the reference and the test alike), standing at
//! 1.7 m and 150 m up looking 45° down, 1280×720; per pixel, the share of
//! 16×16 rays through its square that hit a boulder. One ray: the central
//! ray alone.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** relative RMSE of per-pixel coverage from the reference
//!    at most 0.25× one ray's standing and 0.15× from 150 m (gate 4: 0.218×
//!    and 0.132×; a stated loss of about 15% for B and E).
//! 2. **Speed:** at most 25 ms a frame from 150 m and 50 ms standing at
//!    1280×720 (median of 10 frames each waited for), timed first. The
//!    standing bar leaves the horizon's far field to gate 6; M13's goal
//!    stays a 10 ms coverage pass.
//!
//! Reported: the share of pixels tracing bin rays and resolved by their
//! corners; subtree roots per tile and object nodes per pixel; each part's
//! time in a development profile after the gate.
//!
//! `cargo run --release -p room3d --bin m13_gate5`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders, the passes alone;
//! `M13_DEV=1M` at full size), whose numbers are not the gate's;
//! `M13_STAGE=16` with it times gate 4's path instead, as a control;
//! `M13_ACC=1` with it measures accuracy too, on other views than the gate's.

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
    // `M13_DEV=1M`: the development check at full size.
    let full = std::env::var("M13_DEV").is_ok_and(|v| v == "1M");
    // `M13_STAGE=16` in development: gate 4's path, as a timing control.
    let stage = std::env::var("M13_STAGE")
        .ok()
        .filter(|_| dev)
        .and_then(|v| v.parse().ok())
        .unwrap_or(17);
    let mut field = Field::new(if dev && !full { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    // The ground left out, for the reference and the test alike.
    field.scene.instances.remove(0);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let field_view = |height: f32, down_degrees: f32, turn_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        let (s, c) = turn_degrees.to_radians().sin_cos();
        let flat = Vec3::new(flat.x * c - flat.y * s, flat.x * s + flat.y * c, 0.0);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    // `M13_ACC=1` in development: accuracy too, on other views than the
    // gate's (turned 20° either way, 120 m up at 40°), so development does
    // not tune on the gate's views.
    let acc = dev && std::env::var_os("M13_ACC").is_some();
    let views = if acc {
        [
            ("standing (development view)", field_view(1.7, 0.0, 20.0)),
            (
                "120 m up (development view)",
                field_view(120.0, 40.0, -20.0),
            ),
        ]
    } else {
        [
            ("standing", field_view(1.7, 0.0, 0.0)),
            ("150 m up", field_view(150.0, 45.0, 0.0)),
        ]
    };
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let v = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.set(i % W as usize, i / W as usize, [v; 3]);
        }
        image.save(&out_dir.join(format!("m13_gate5_{name}.png")))
    };
    // 2. Speed, first: both views, after building each renderer.
    let mut ours_images = Vec::new();
    let mut speed = Vec::new();
    for (name, camera) in views {
        kernel.set_prefilter(8.0);
        kernel.set_hardware_rays(true);
        kernel.set_profile_stage(stage);
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
        drop(r);
        let n = out.len() as f64;
        let resolved = out.iter().filter(|p| p[1] == 0.0).count() as f64 / n;
        let traced = out.iter().filter(|p| p[2] as u32 % 2 == 1).count() as f64 / n;
        let roots = out.iter().map(|p| (p[2] as u32 / 2) as f64).sum::<f64>() / n;
        let max_roots = out.iter().map(|p| p[2] as u32 / 2).max().unwrap_or(0);
        let walked = out.iter().filter(|p| p[1] > 0.0).count().max(1) as f64;
        let visits = out.iter().map(|p| p[1] as f64).sum::<f64>() / walked;
        let max_visits = out.iter().map(|p| p[1]).fold(0.0f32, f32::max);
        say(format!(
            "{name}: {ours_ms:.2} ms a frame; {:.1}% of pixels resolved by their corners, {:.1}% tracing bin rays; subtree roots a tile {roots:.1} (at most {max_roots}); object nodes a walking pixel {visits:.1} (at most {max_visits:.0}); mean coverage {:.4}",
            100.0 * resolved,
            100.0 * traced,
            out.iter().map(|p| p[0] as f64).sum::<f64>() / n
        ));
        if dev {
            save(&format!("{}_dev", name.replace(' ', "_")), &gray(&out))?;
        }
        ours_images.push(gray(&out));
        speed.push(ours_ms);
    }
    if dev && !acc {
        return Ok(());
    }
    let mut pass = true;
    for (k, (name, camera)) in views.into_iter().enumerate() {
        let slug = name.replace(' ', "_");
        let ours = &ours_images[k];
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
        let e_ours = relative_rmse(ours, &reference);
        let e_one = relative_rmse(&one, &reference);
        let (bar, ms_bar) = if k == 0 { (0.25, 50.0) } else { (0.15, 25.0) };
        let c1 = e_ours <= bar * e_one;
        let c2 = speed[k] <= ms_bar;
        pass &= c1 && c2;
        let mean_cov =
            |px: &[[f32; 3]]| px.iter().map(|p| p[0] as f64).sum::<f64>() / px.len() as f64;
        say(format!(
            "{name}: 1. coverage {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most {bar}×): {}; 2. {:.2} ms a frame (at most {ms_bar}): {}; mean coverage: reference {:.4}, ours {:.4}, one ray {:.4}",
            100.0 * e_ours,
            100.0 * e_one,
            e_ours / e_one,
            verdict(c1),
            speed[k],
            verdict(c2),
            mean_cov(&reference),
            mean_cov(ours),
            mean_cov(&one)
        ));
        if dev {
            continue;
        }
        save(&format!("{slug}_reference"), &reference)?;
        save(&format!("{slug}_ours"), ours)?;
        save(&format!("{slug}_one_ray"), &one)?;
    }
    if dev {
        return Ok(());
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate5.txt"), report)?;
    Ok(())
}
