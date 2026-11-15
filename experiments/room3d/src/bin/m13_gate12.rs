//! M13 gate 12: the sub-pixel path in the renderer, lit by its cache.
//!
//! Gates 5–11 ran sub-pixel coverage and colour as a diagnostic profile
//! stage with its own lighting model (one sun, a constant ambient); gate 11
//! met colour's bar there (0.335× and 0.386× one ray's error, 13.85 and
//! 14.69 ms settled; `docs/results/m13-gate11.md`). DESIGN §3.4, "Into the
//! renderer", brings it into the renderer as an option whose colour comes
//! from the renderer's own lighting (the cache: every light's area, the
//! sky, bounce light), not a second model.
//!
//! Method as developed (DESIGN §3.4, parts A–D, changed before the first
//! run; `docs/results/m13-gate12-development.md`):
//! - A. The renderer takes the sub-pixel path with its cache, pixel cones
//!   and profile stage 20 (`InstancedRenderer`; a `RendererOptions` field
//!   comes after the gate). The passes that shade come from a variant of
//!   the hardware-ray module; binding lists come from naga's analysis.
//! - B. Large hits (corners, the centres of pixels whose background needs
//!   one, samples of pixels the corners resolved, bins of objects no corner
//!   shows) are recorded and lit by the cache's own shading in passes of
//!   their own (the device's 29 buffers a stage do not hold both); a
//!   corner keeps its cached shares (the light's, the sky's), and a bin of
//!   an object its corners show is lit from them bilinearly with its own
//!   point, normal and albedo.
//! - C. Small instances: direct light from the maps' normals and a sun
//!   visibility per instance (the cosine-weighted share of five points on
//!   its bounding sphere's sunlit side, cached per object); the sky's
//!   radiance over half the hemisphere (a sphere resting on a plane). Not
//!   from the cache yet.
//! - D. Coverage pages bake colour by the model (from their own eye), not
//!   the cache yet.
//! - The cache bakes with a fixed budget on this path (no frame target or
//!   period cap: its camera passes alone pass both, which shrank the bakes
//!   to their least); settled frames bake nothing. Settled: eight frames in
//!   a row with no request and no page baked, and no coverage page asked.
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
//! Disclosed before the run: developed at full size on the gate's scene,
//! on other views for both speed and accuracy (0.420× and 0.426×; 16.0–16.6
//! ms, criterion 2 within a few percent of its bar).
//!
//! `cargo run --release -p room3d --bin m13_gate12`
//!
//! `M13_DEV=1` (10,000 boulders; `1M` at full size) runs development checks
//! on other views (`M13_ACC=1` colour and its split by pixel class,
//! `M13_NOSKY=1`, `M13_VIEW=standing|120`, `M13_VERBOSE=1`); `M13_BINDINGS=1`
//! prints each entry point's bindings.

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
        // A fixed bake budget for the sub-pixel path (no frame target, no
        // period cap): its camera passes alone pass both (about 16 ms), which
        // would shrink the bakes to their least and take thousands of frames
        // to settle. Settled frames bake nothing either way.
        if sub {
            CacheOptions {
                frame_target_ms: 0.0,
                frame_budget_ms: 0.0,
                ..CacheOptions::default()
            }
        } else {
            CacheOptions::default()
        },
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
        if std::env::var_os("M13_VERBOSE").is_some() && k % 100 == 0 {
            println!(
                "    frame {k}: cache requests {}, pages baked {}, texels {}, sky samples {}, resident {}, evicted {}, fallback pixels {}; coverage pages asked {pages}",
                s.map_or(0, |s| s.requests),
                s.map_or(0, |s| s.pages_baked),
                s.map_or(0, |s| s.texels_baked),
                s.map_or(0, |s| s.sky_samples),
                s.map_or(0, |s| s.resident),
                s.map_or(0, |s| s.evicted),
                s.map_or(0, |s| s.fallback_pixels)
            );
        }
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
            "hw_m13_shade_hits_a",
        ] {
            let vc = kernel.entry_bindings(name, true);
            println!(
                "{name}: model {:?}; cache ({}) {:?}",
                if name.contains("hits") {
                    Vec::new()
                } else {
                    kernel.entry_bindings(name, false)
                },
                vc.len(),
                vc
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
        // `M13_NOSKY=1` in development: lights alone.
        if std::env::var_os("M13_NOSKY").is_none() {
            kernel.set_sky(field.sky());
        }
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
        let only = std::env::var("M13_VIEW").ok();
        // (The gate's own views only when `M13_VIEW` names them, after the
        // run: `gate_standing`, `gate_150`.)
        for (name, camera) in [
            ("standing_dev", view(1.7, 0.0, 20.0)),
            ("120m_dev", view(120.0, 40.0, -20.0)),
            ("gate_standing", view(1.7, 0.0, 0.0)),
            ("gate_150", view(150.0, 45.0, 0.0)),
        ] {
            let chosen = match &only {
                Some(v) => name.starts_with(v.as_str()),
                None => !name.starts_with("gate"),
            };
            if !chosen {
                continue;
            }
            // `M13_NOONE=1`: the sub-pixel path alone (its speed).
            let noone = std::env::var_os("M13_NOONE").is_some();
            let mut one = cached(&kernel, &field, origin, &lights, false);
            let n1 = if noone {
                0
            } else {
                settle(&mut one, camera, &view_target, 8, 3000)
            };
            let mut one_times: Vec<f64> = (0..10)
                .map(|_| {
                    let t = Instant::now();
                    one.frame(camera, &view_target);
                    one.wait();
                    t.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            one_times.sort_by(f64::total_cmp);
            println!(
                "{name}: one ray a pixel with the cache, settled: {:.2} ms",
                one_times[5]
            );
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
            // (The median of 41 frames: the median's index 20, read below as
            // `times[5]` was for 10; kept the same name.)
            let mut times: Vec<f64> = (0..41)
                .map(|_| {
                    let t = Instant::now();
                    sub.frame(camera, &view_target);
                    sub.wait();
                    t.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            times.sort_by(f64::total_cmp);
            times.drain(..15);
            let b = frame(&mut sub, camera, &view_target);
            let hits = sub.m13_hit_counts();
            let work = sub.m13_work_counters();
            println!(
                "{name}: records lit by the cache {} ({} after the corners), pixels tracing bin rays {}",
                hits[0],
                hits[0].saturating_sub(hits[1]),
                work[0]
            );
            // With `M13_ACC`: each pixel's class (large objects, pages,
            // horizon, shadow edge), for the error's split.
            let class = std::env::var_os("M13_ACC").is_some().then(|| {
                sub.set_m13_state_word(28, 1);
                let c = frame(&mut sub, camera, &view_target);
                sub.set_m13_state_word(28, 0);
                c
            });
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
            if let (Some(reference), Some(class)) = (&reference, &class) {
                let sq = |x: &[f32; 3], y: &[f32; 3]| -> f64 {
                    x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum()
                };
                let names = ["large", "pages", "horizon", "shadow edge"];
                let mut ours_by = [0.0f64; 16];
                let mut one_by = [0.0f64; 16];
                let mut count = [0usize; 16];
                for k in 0..reference.len() {
                    let c = class[k];
                    let z = c[2].round() as usize;
                    let id = (c[0] > 0.5) as usize
                        | ((c[1] > 0.5) as usize) << 1
                        | (z & 1) << 2
                        | ((z >> 1) & 1) << 3;
                    ours_by[id] += sq(&b[k], &reference[k]);
                    one_by[id] += sq(&a[k], &reference[k]);
                    count[id] += 1;
                }
                let total_one: f64 = one_by.iter().sum();
                for id in 0..16 {
                    if count[id] == 0 {
                        continue;
                    }
                    let parts: Vec<&str> = (0..4)
                        .filter(|q| id >> q & 1 == 1)
                        .map(|q| names[q])
                        .collect();
                    println!(
                        "  {:>26}: {:5.1}% of pixels; squared error ours {:.4}, one ray {:.4} (of one ray's total); ratio there {:.3}",
                        if parts.is_empty() {
                            "small only".to_string()
                        } else {
                            parts.join("+")
                        },
                        100.0 * count[id] as f64 / reference.len() as f64,
                        ours_by[id] / total_one,
                        one_by[id] / total_one,
                        (ours_by[id] / one_by[id].max(1e-30)).sqrt()
                    );
                }
            }
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
    gate();
}

/// The gate: speed first in both views (settled), the 20 m view reported,
/// then colour against the reference.
fn gate() {
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return;
    };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir).expect("out");
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    kernel.set_sky(field.sky());
    let view_target = target(&kernel);
    let forward = (ground.target - ground.eye).normalize();
    let flat0 = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down_degrees: f32| {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat0 * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(1.7, 0.0)),
        ("150 m up", view(150.0, 45.0)),
    ];
    let median_ms = |r: &mut InstancedRenderer, camera: Camera| {
        let mut t: Vec<f64> = (0..10)
            .map(|_| {
                let s = Instant::now();
                r.frame(camera, &view_target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        t.sort_by(f64::total_cmp);
        t[5]
    };
    let mut pass = true;
    // 2. Speed, first; the images kept for criterion 1.
    let mut ours = Vec::new();
    for (name, camera) in views {
        let mut sub = cached(&kernel, &field, origin, &lights, true);
        sub.set_m13_page_budget(512);
        let frames = settle(&mut sub, camera, &view_target, 8, 3000);
        let t = median_ms(&mut sub, camera);
        let c2 = t <= 16.7 && frames < 3000;
        pass &= c2;
        say(format!(
            "{name}: 2. settled frame {t:.2} ms (at most 16.7): {}; settled in {frames} frames",
            verdict(c2)
        ));
        ours.push(frame(&mut sub, camera, &view_target));
    }
    {
        let camera = view(20.0, 10.0);
        let mut sub = cached(&kernel, &field, origin, &lights, true);
        sub.set_m13_page_budget(512);
        let frames = settle(&mut sub, camera, &view_target, 8, 3000);
        let t = median_ms(&mut sub, camera);
        say(format!(
            "20 m up, 10° down (reported): settled frame {t:.2} ms; settled in {frames} frames"
        ));
    }
    // 1. Colour, and the renderer's own path's frame (reported).
    let save = |name: &str, px: &[[f32; 3]], scale: f32| {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            let c = p.map(|v| ((v / scale).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
            image.set(i % W as usize, i / W as usize, c);
        }
        image
            .save(&out_dir.join(format!("m13_gate12_{name}.png")))
            .expect("save");
    };
    for (k, (name, camera)) in views.into_iter().enumerate() {
        let mut one = cached(&kernel, &field, origin, &lights, false);
        let frames = settle(&mut one, camera, &view_target, 8, 3000);
        let t_one = median_ms(&mut one, camera);
        let single = frame(&mut one, camera, &view_target);
        let mut reference = vec![[0.0f32; 3]; (W * H) as usize];
        for s in 0..256 {
            let (dx, dy) = (
                ((s % 16) as f32 + 0.5) / 16.0 - 0.5,
                ((s / 16) as f32 + 0.5) / 16.0 - 0.5,
            );
            one.set_pixel_offset(dx, dy);
            for (r, p) in reference
                .iter_mut()
                .zip(frame(&mut one, camera, &view_target))
            {
                for (v, c) in r.iter_mut().zip(p) {
                    *v += c / 256.0;
                }
            }
        }
        drop(one);
        let e_ours = relative_rmse(&ours[k], &reference);
        let e_one = relative_rmse(&single, &reference);
        let c1 = e_ours <= 0.5 * e_one;
        pass &= c1;
        let mean = |px: &[[f32; 3]]| {
            let n = px.len() as f64;
            [0, 1, 2].map(|c| px.iter().map(|p| p[c] as f64).sum::<f64>() / n)
        };
        let (mr, mo) = (mean(&reference), mean(&ours[k]));
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
        say(format!(
            "{name}: the renderer's own path (one ray a pixel, the cache) settled in {frames} frames, {t_one:.2} ms (reported)"
        ));
        let slug = name.replace(' ', "_");
        let mut l: Vec<f32> = reference.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
        l.sort_by(f32::total_cmp);
        let scale = l[l.len() * 99 / 100].max(1e-6);
        save(&format!("{slug}_reference"), &reference, scale);
        save(&format!("{slug}_ours"), &ours[k], scale);
        save(&format!("{slug}_one_ray"), &single, scale);
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate12.txt"), report).expect("report");
}
