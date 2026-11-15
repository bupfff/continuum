//! M5 gate 14: wide light cones (no stall near a large light).
//!
//! Found after M13 gate 14 (DESIGN §8, "Wide light cones"). A rectangle
//! light's cone projects the light onto a plane parallel to it at the
//! receiver's distance h from that plane, with 8×8 bins uniform there. For a
//! light s across the window is about s/h wide. A 40 m panel 5 cm above the
//! field's ground, under the boulders, gave windows of ±2,000. Bins then no
//! longer prefilter anything, and a cone walked to triangles: about 13 ms a
//! cone, frames of 12–24 s on 10,000 boulders on both the one-ray and the
//! sub-pixel path. At full size a development run waited 12 hours.
//!
//! The fix (designed before the stub): a window wider than WIDE_EXTENT (8
//! units of h) keeps the square that wide nearest the light's axis (the
//! foot point clamped into the window), where a receiver's light is densest
//! (it falls as the cube of the distance in that plane). The kept window's
//! visible share of its own unshadowed light stands for the whole light's
//! (`wide_visible`; where the kept part is below the receiver's horizon,
//! unshadowed). A sphere light's window is capped the same way, all of it
//! the disk. Windows no wider are unchanged.
//!
//! Scene: the million-boulder field with the field's sun and a light panel:
//! 40 m square, 5 cm above the ground, facing up, under the boulders, 130–170
//! m ahead of the field's camera along its line, radiance (0.5, 0.45, 0.35).
//! No sky. 1280×720; prefilter 1 bin with object-level prefiltering,
//! hardware rays, cache defaults (M5 gate 6's settings). Views:
//! - near: 110 m ahead of the field's camera, 8 m up, looking 15° down, the
//!   panel and its boulders filling the view;
//! - 20 m up: at the field's camera, 20° down (the view that stalled).
//!
//! Pass criterion, fixed before the first run:
//! 1. **No stall** (timed first): the cache's path (one ray a pixel), from
//!    an empty cache, 600 frames at each view, each waited for. After the
//!    first 10 frames no frame over 100 ms, and the 99th percentile at most
//!    33.3 ms.
//! 2. **Accuracy:** direct light by per-pixel cones (no cache), 16×16
//!    samples a pixel at 320×180 box filtered, within 3% relative RMSE of
//!    instanced Cycles at 4,096 samples a pixel (direct light only), over
//!    surfaces, in both views. Pixels where a sample sees a light, and
//!    their eight neighbours, are scored apart (M5 gate 6's method and
//!    bar).
//! 3. **Ordinary lights untouched:** M5 gate 6's three coloured lights at
//!    the field's camera, per-pixel cones at 1280×720, give an image
//!    bit-identical to the same with no window clipped (WIDE_EXTENT 1e30, a
//!    second kernel).
//!
//! Amended before any run (development after the stub, disclosed):
//! - **The method grew three parts.** On 10,000 boulders the clip alone
//!   left frames of 0.1–0.8 s: points just outside the light, near its
//!   plane, see grazing slivers whose walks are just as long.
//!   - The clip applies only to windows wider than 30° in angle (corners
//!     more than 30° from the window's centre). Gate 6's scene changed in
//!     12 pixels where far receivers near a light's plane saw thin slivers.
//!   - Each light cone's walk is capped: 4,096 nodes and triangles, 512
//!     where its window is wider than WIDE_EXTENT in the plane. A long
//!     walk is one invocation's serial work, so one such cone sets a
//!     frame's length however few are baked. A sliver stopped at its cap
//!     counts as blocked, a clipped cone as open.
//!   - The time the caller waits for the GPU after a frame
//!     (`InstancedRenderer::wait`) counts toward the frame period the
//!     cache's bake budget adapts to, as a frame call's own wait does.
//!     Without it, this harness's frames stayed at the largest budget,
//!     since this device's pass timestamps mostly read zero.
//! - **Criterion 3's second kernel turns all of it off**
//!   (`set_wide_cones(false)`), not only the clip.
//! - **Development at full size on other views** (near turned 30°; 40 m up):
//!   - criterion 1: 21.2/17.2 ms worst, 18.8/15.6 ms 99th percentile;
//!   - criterion 3: no pixel differs;
//!   - criterion 2 (an earlier variant, the same caps but counted open):
//!     near 4.72% (failing), 40 m up 1.49%.
//!
//!   A probe of the panel alone showed boulders resting on it dark, which
//!   Cycles shows lit beside it. Their light is densest beneath them, where
//!   their own body blocks the kept window, while the rest of the panel
//!   lights them. Some far boulders came out lit where Cycles has them
//!   shadowed by the field. Accuracy for receivers centimetres from a large
//!   light is expected to fail here; rays are the planned fix (DESIGN §8).
//!
//! Disclosed before the stub: the 20 m view is where the stall was found
//! (development, M13 gate 14's binary with `M13_FLOOR=1`); on 10,000
//! boulders the fix took its frames from 12–24 s to 9–17 ms.
//!
//! Reported: the panel alone against Cycles (near); the cache's settled
//! image against per-pixel cones (1280×720); frames to settle; the frame
//! times' mean and maximum.
//!
//! `cargo run --release -p room3d --bin m5_gate14`
//!
//! `M5_DEV=1` runs criteria 1 and 3 on 10,000 boulders at the gate's views
//! (no Cycles). `M5_DEV=1M` runs all three at full size on other views
//! (near turned 30° left; 40 m up, 30° down, 80 m ahead).

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, RectLight, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const FRAMES: usize = 600;

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

fn kernel_with(guards: bool) -> Option<MaskConeKernel> {
    let kernel = MaskConeKernel::new()?;
    kernel.set_wide_cones(guards);
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    Some(kernel)
}

/// The camera's line on the ground (unit, horizontal) and a camera on it.
struct Views {
    ground: Camera,
    flat0: Vec3,
}

impl Views {
    fn new(ground: Camera) -> Self {
        let f = (ground.target - ground.eye).normalize();
        Views {
            ground,
            flat0: Vec3::new(f.x, f.y, 0.0).normalize(),
        }
    }

    fn flat(&self, turn_degrees: f32) -> Vec3 {
        let (s, c) = turn_degrees.to_radians().sin_cos();
        let f = self.flat0;
        Vec3::new(f.x * c - f.y * s, f.x * s + f.y * c, 0.0)
    }

    /// `ahead` m along the line, `height` up, turned `turn` (left) and
    /// looking `down` degrees down.
    fn at(&self, ahead: f32, height: f32, down: f32, turn: f32) -> Camera {
        let d = down.to_radians();
        let g = self.ground.eye;
        let eye = Vec3::new(g.x, g.y, 0.0) + self.flat0 * ahead + Vec3::Z * height;
        Camera {
            eye,
            target: eye + self.flat(turn) * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: self.ground.vertical_fov_degrees,
        }
    }
}

/// The panel: 40 m square, 5 cm up, facing up, 130–170 m ahead.
fn panel(views: &Views) -> Light {
    let f = views.flat0;
    let left = Vec3::Z.cross(f);
    let g = views.ground.eye;
    let middle = Vec3::new(g.x, g.y, 0.05) + f * 150.0;
    let (u, v) = (f * 40.0, left * 40.0);
    let (u, v) = if u.cross(v).z > 0.0 { (u, v) } else { (v, u) };
    Light::rect(
        RectLight {
            corner: middle - (u + v) * 0.5,
            edge_u: u,
            edge_v: v,
            one_sided: true,
        },
        Vec3::new(0.5, 0.45, 0.35),
    )
}

fn cached(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
) -> InstancedRenderer {
    InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        lights,
        &field.scene,
        origin,
        CacheOptions::default(),
    )
}

/// Frame times (ms) of `n` frames from an empty cache, each waited for.
fn frame_times(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    n: usize,
) -> Vec<f64> {
    let trace = std::env::var_os("M5_TRACE").is_some();
    (0..n)
        .map(|k| {
            let s = Instant::now();
            r.frame(camera, target);
            r.wait();
            let t = s.elapsed().as_secs_f64() * 1000.0;
            if trace && (k < 30 || k % 50 == 0 || t > 100.0) {
                let c = r.cache_stats();
                eprintln!(
                    "frame {k}: {t:.1} ms; {:?}",
                    c.map(|c| (
                        c.pages_baked,
                        c.texels_baked,
                        c.budget,
                        c.spent,
                        c.gpu_bake_ms,
                        c.gpu_render_ms,
                        c.planning.as_secs_f64() * 1e3,
                        c.waiting.as_secs_f64() * 1e3
                    ))
                );
            }
            t
        })
        .collect()
}

/// Frames until 8 in a row ask for and bake no page (at most `max`).
fn settle(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    max: usize,
) -> usize {
    let mut calm = 0;
    for k in 1..=max {
        r.frame(camera, target);
        r.wait();
        if r.cache_stats()
            .is_some_and(|s| s.requests == 0 && s.pages_baked == 0)
        {
            calm += 1;
            if calm >= 8 {
                return k;
            }
        } else {
            calm = 0;
        }
    }
    max
}

/// Per pixel at `ss × ss` samples (per-pixel cones), box filtered to
/// `w × h`, and which pixels (with their eight neighbours) have a sample
/// that sees a light (M5 gate 6's).
fn supersampled(
    kernel: &MaskConeKernel,
    camera: Camera,
    (w, h): (u32, u32),
    ss: u32,
    lights: &[Light],
    field: &Field,
    origin: DVec3,
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let (big, _, _) =
        kernel.render_instanced_lights(camera, w * ss, h * ss, lights, &field.scene, origin, 1);
    let mut px = vec![[0.0f32; 3]; (w * h) as usize];
    let mut sees = vec![false; (w * h) as usize];
    let n = (ss * ss) as f32;
    for y in 0..(h * ss) as usize {
        for x in 0..(w * ss) as usize {
            let p = big[y * (w * ss) as usize + x];
            let i = (y / ss as usize) * w as usize + x / ss as usize;
            if lights.iter().any(|l| l.radiance.to_array() == p) {
                sees[i] = true;
            }
            for c in 0..3 {
                px[i][c] += p[c] / n;
            }
        }
    }
    let mut apart = sees.clone();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !sees[(y * w as i32 + x) as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                        apart[(ny * w as i32 + nx) as usize] = true;
                    }
                }
            }
        }
    }
    (px, apart)
}

/// Relative RMSE over the pixels not set apart.
fn surfaces_rmse(a: &[[f32; 3]], b: &[[f32; 3]], apart: &[bool]) -> f64 {
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(apart)
            .filter(|(_, x)| !**x)
            .map(|(p, _)| *p)
            .collect()
    };
    relative_rmse(&keep(a), &keep(b))
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * p).round() as usize]
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = kernel_with(true) else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let dev = std::env::var("M5_DEV").ok();
    let field = Field::new(
        if dev.as_deref() == Some("1") {
            10_000
        } else {
            1_000_000
        },
        DVec3::ZERO,
    );
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let views = Views::new(ground);
    let lights = [Light::white(sun, field.radiance), panel(&views)];
    let view_target = target(&kernel);
    // Development (`M5_PROBE=1`): the panel alone at the first development
    // view, per-pixel cones (4×4 a pixel at 320×180) beside Cycles, saved.
    if std::env::var_os("M5_PROBE").is_some() {
        let camera = views.at(110.0, 8.0, 15.0, 30.0);
        let alone = [lights[1]];
        let small = (320, 180);
        let (ours, apart) = supersampled(&kernel, camera, small, 4, &alone, &field, origin);
        let c = render_instanced_lights(
            &field.scene,
            origin,
            &alone,
            Vec3::ZERO,
            camera,
            small,
            4096,
            0,
            1,
            &out_dir.join("m5_gate14_cycles_neardevelopment_panel"),
        )?;
        let keep = |img: &[[f32; 3]]| -> f64 {
            img.iter()
                .zip(&apart)
                .filter(|(_, a)| !**a)
                .map(|(p, _)| (p[0] + p[1] + p[2]) as f64)
                .sum::<f64>()
        };
        println!(
            "probe: surfaces' sum ours {:.1}, Cycles {:.1}; error {:.2}%",
            keep(&ours),
            keep(&c.pixels),
            100.0 * surfaces_rmse(&ours, &c.pixels, &apart)
        );
        for (name, px) in [("ours", &ours), ("cycles", &c.pixels)] {
            let mut image = flatland::img::Image::new(320, 180, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                let v =
                    p.map(|v| ((v / 0.1).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
                image.set(i % 320, i / 320, v);
            }
            image.save(&out_dir.join(format!("m5_gate14_probe_{name}.png")))?;
        }
        return Ok(());
    }
    let views_run = if dev.as_deref() == Some("1M") {
        [
            ("near (development)", views.at(110.0, 8.0, 15.0, 30.0)),
            ("40 m up (development)", views.at(80.0, 40.0, 30.0, 0.0)),
        ]
    } else {
        [
            ("near", views.at(110.0, 8.0, 15.0, 0.0)),
            ("20 m up", views.at(0.0, 20.0, 20.0, 0.0)),
        ]
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let mut pass = true;
    // 1. No stall, first.
    let mut cached_images = Vec::new();
    for (name, camera) in views_run {
        let mut r = cached(&kernel, &field, origin, &lights);
        let t = frame_times(&mut r, camera, &view_target, FRAMES);
        if std::env::var_os("M5_TRACE").is_some() {
            // Development: the slowest frames and where they fell.
            let mut worst: Vec<(usize, f64)> = t.iter().cloned().enumerate().collect();
            worst.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("{name}: slowest frames {:?}", &worst[..8.min(worst.len())]);
        }
        let late = &t[10..];
        let max = late.iter().cloned().fold(0.0, f64::max);
        let p99 = percentile(late, 0.99);
        let c1 = max <= 100.0 && p99 <= 33.3;
        pass &= c1;
        let n = settle(&mut r, camera, &view_target, 3000);
        r.finish();
        cached_images.push(r.read_colors());
        say(format!(
            "{name}: 1. frames from an empty cache, after the first 10: max {max:.1} ms (at most 100), 99th percentile {p99:.1} ms (at most 33.3): {}; mean {:.1} ms; first 10 {:?} ms; settled {n} frames later",
            verdict(c1),
            late.iter().sum::<f64>() / late.len() as f64,
            t[..10]
                .iter()
                .map(|v| (v * 10.0).round() / 10.0)
                .collect::<Vec<_>>()
        ));
    }
    // 3. Ordinary lights untouched (a second kernel, no window clipped).
    {
        let colored = field.colored_lights(origin, 1000.0);
        let (a, _, _) =
            kernel.render_instanced_lights(ground, W, H, &colored, &field.scene, origin, 1);
        let whole = kernel_with(false).expect("a second kernel");
        let (b, _, _) =
            whole.render_instanced_lights(ground, W, H, &colored, &field.scene, origin, 1);
        let differing = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        let c3 = differing == 0;
        pass &= c3;
        say(format!(
            "3. M5 gate 6's three lights at the field's camera, per-pixel cones: {differing} pixels differ from no window clipped (none allowed): {}",
            verdict(c3)
        ));
    }
    if dev.as_deref() == Some("1") {
        say("Development on 10,000 boulders: no Cycles (criterion 2 not run).".into());
        return Ok(());
    }
    // 2. Accuracy against Cycles; the cache against per-pixel cones.
    let small = (320, 180);
    for (k, (name, camera)) in views_run.into_iter().enumerate() {
        let slug: String = name.chars().filter(|c| c.is_alphanumeric()).collect();
        let (ours, apart) = supersampled(&kernel, camera, small, 16, &lights, &field, origin);
        let cycles = render_instanced_lights(
            &field.scene,
            origin,
            &lights,
            Vec3::ZERO,
            camera,
            small,
            4096,
            0,
            1,
            &out_dir.join(format!("m5_gate14_cycles_{slug}")),
        )?;
        let e = surfaces_rmse(&ours, &cycles.pixels, &apart);
        let c2 = e <= 0.03;
        pass &= c2;
        say(format!(
            "{name}: 2. per-pixel cones (16×16, 320×180) vs Cycles (4,096 spp, {:.0} s), surfaces: {:.2}% (at most 3%): {}; {} pixels apart",
            cycles.seconds,
            100.0 * e,
            verdict(c2),
            apart.iter().filter(|a| **a).count()
        ));
        if k == 0 {
            let alone = [lights[1]];
            let (o, a) = supersampled(&kernel, camera, small, 16, &alone, &field, origin);
            let c = render_instanced_lights(
                &field.scene,
                origin,
                &alone,
                Vec3::ZERO,
                camera,
                small,
                4096,
                0,
                1,
                &out_dir.join(format!("m5_gate14_cycles_{slug}_panel")),
            )?;
            say(format!(
                "{name} (reported): the panel alone vs Cycles: {:.2}%",
                100.0 * surfaces_rmse(&o, &c.pixels, &a)
            ));
        }
        let (direct, _, _) =
            kernel.render_instanced_lights(camera, W, H, &lights, &field.scene, origin, 1);
        say(format!(
            "{name} (reported): the cache's settled image vs per-pixel cones: {:.2}%",
            100.0 * relative_rmse(&cached_images[k], &direct)
        ));
    }
    say(format!("Verdict: {}", verdict(pass)));
    let file = if dev.is_some() {
        "m5_gate14_dev.txt"
    } else {
        "m5_gate14.txt"
    };
    std::fs::write(out_dir.join(file), report)?;
    Ok(())
}
