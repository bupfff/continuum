//! Development (M20): depth of field against Cycles' thin lens, on the
//! textured scene (sun and sky). Not a gate.
//!
//! `DOF_W`/`DOF_H` set the size (640×360), `DOF_SAMPLES` Cycles' samples
//! (1024), `DOF_VIEW` the view (`far`: focused on the brick block, the
//! ground in front blurred; `near`: focused at 5 m, the distance blurred),
//! `DOF_APERTURE` the aperture's radius (0.1), `DOF_SCALE` ours relative to
//! Cycles' (1; to check the two lenses agree), `DOF_FOLIAGE` adds foliage.
//! Ours: 4×4 sub-pixel offsets, each settled. `time`: the cost at 1280×720
//! instead.

use std::path::Path;
use std::time::Instant;

use continuum_gpu::{CacheOptions, Camera, DepthOfField, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_dof, render_instanced_lights};
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return;
        }
    }
    eprintln!("WARNING: not settled");
}

/// The same gather on the CPU with every pixel within reach a tap (area 1,
/// its disc's edge one pixel soft): what the GPU's rings approximate.
fn cpu_dof(
    colors: &[[f32; 3]],
    hits: &[(u32, u32, f32, u32)],
    camera: Camera,
    lens: &DepthOfField,
    (w, h): (u32, u32),
    (dx, dy): (f32, f32),
) -> Vec<[f32; 3]> {
    let (w, h) = (w as usize, h as usize);
    let tz = (camera.vertical_fov_degrees.to_radians() * 0.5).tan();
    let aspect = w as f32 / h as f32;
    let f = h as f32 / (2.0 * tz);
    let max_px = lens.max_blur * h as f32;
    let coc: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let sx = ((x as f32 + 0.5 + dx) / w as f32 * 2.0 - 1.0) * tz * aspect;
            let sy = (1.0 - (y as f32 + 0.5 + dy) / h as f32 * 2.0) * tz;
            let inv_z = if hits[i].0 == u32::MAX {
                0.0
            } else {
                (1.0 + sx * sx + sy * sy).sqrt() / hits[i].2.max(1e-6)
            };
            (f * lens.aperture_radius * (1.0 / lens.focus_distance - inv_z)).clamp(-max_px, max_px)
        })
        .collect();
    let reach = coc.iter().fold(0.0f32, |m, c| m.max(c.abs())).ceil() as i64 + 1;
    let r_pixel = 0.5641896f32;
    let mut out = vec![[0.0f32; 3]; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let i = y as usize * w + x as usize;
            let cp = coc[i];
            let tol = (0.15 * cp.abs()).max(1.0);
            let (mut front, mut own, mut back, mut fill) =
                ([0.0f64; 4], [0.0f64; 4], [0.0f64; 4], [0.0f64; 4]);
            let add = |acc: &mut [f64; 4], col: [f32; 3], wgt: f64| {
                for k in 0..3 {
                    acc[k] += col[k] as f64 * wgt;
                }
                acc[3] += wgt;
            };
            for j in -reach..=reach {
                for k in -reach..=reach {
                    let (qx, qy) = (
                        (x + k).clamp(0, w as i64 - 1),
                        (y + j).clamp(0, h as i64 - 1),
                    );
                    let q = qy as usize * w + qx as usize;
                    let d = ((j * j + k * k) as f32).sqrt();
                    let c = coc[q];
                    let r = c.abs().max(r_pixel);
                    let s = if d == 0.0 { 0.5 } else { 1.0 };
                    let cover = ((r - d) / s + 0.5).clamp(0.0, 1.0) as f64;
                    let wgt = cover / (std::f64::consts::PI * (r * r) as f64);
                    if c < cp - tol {
                        add(&mut front, colors[q], wgt);
                    } else if c > cp + tol {
                        add(&mut back, colors[q], wgt);
                        if d <= reach as f32 {
                            add(&mut fill, colors[q], 1.0 / (1.0 + (d * d) as f64));
                        }
                    } else {
                        add(&mut own, colors[q], wgt);
                    }
                }
            }
            let norm = |a: [f64; 4]| if a[3] > 1.0 { a.map(|v| v / a[3]) } else { a };
            let (front, own, back) = (norm(front), norm(own), norm(back));
            let rest: [f64; 3] = if fill[3] > 0.0 {
                [fill[0] / fill[3], fill[1] / fill[3], fill[2] / fill[3]]
            } else if own[3] > 0.0 {
                [own[0] / own[3], own[1] / own[3], own[2] / own[3]]
            } else {
                colors[i].map(|v| v as f64)
            };
            for k in 0..3 {
                let behind = back[k] + (1.0 - back[3]) * rest[k];
                out[i][k] =
                    (front[k] + (1.0 - front[3]) * (own[k] + (1.0 - own[3]) * behind)) as f32;
            }
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let timing = std::env::args().any(|a| a == "time");
    let (w, h): (u32, u32) = if timing {
        (1280, 720)
    } else {
        (env("DOF_W", 640), env("DOF_H", 360))
    };
    let samples: u32 = env("DOF_SAMPLES", 1024);
    let view: String = env("DOF_VIEW", "far".to_string());
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let mut t = Textured::new();
    if std::env::var_os("DOF_FOLIAGE").is_some() {
        t = t.with_foliage();
    }
    kernel.set_sky(t.sky);
    let lens = DepthOfField {
        focus_distance: if view == "near" { 5.0 } else { 9.1 },
        aperture_radius: env("DOF_APERTURE", 0.1),
        ..Default::default()
    };
    let ours_lens = DepthOfField {
        aperture_radius: lens.aperture_radius * env("DOF_SCALE", 1.0f32),
        ..lens
    };
    let make = || {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        );
        settle(&mut r, t.camera, &target);
        r
    };
    let mut r = make();
    if timing {
        let timed = |r: &mut InstancedRenderer| {
            let mut v: Vec<f64> = (0..40)
                .map(|_| {
                    let s = Instant::now();
                    r.frame(t.camera, &target);
                    r.wait();
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[20]
        };
        for _ in 0..3 {
            r.set_depth_of_field(None);
            let plain = timed(&mut r);
            r.set_depth_of_field(Some(ours_lens));
            let lensed = timed(&mut r);
            println!(
                "1280x720 {view}: {plain:.3} ms, with depth of field {lensed:.3} ms (+{:.3}); highlights {:?}",
                lensed - plain,
                r.depth_of_field_highlights()
            );
        }
        return Ok(());
    }
    // With `cpu`, each frame through the CPU's gather.
    let supersampled = |r: &mut InstancedRenderer, cpu: Option<DepthOfField>| {
        let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
        r.set_footprint(0.25);
        for k in 0..16 {
            let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
            r.set_pixel_offset(dx, dy);
            settle(r, t.camera, &target);
            r.finish();
            let mut colors = r.read_colors();
            if let Some(lens) = &cpu {
                colors = cpu_dof(&colors, &r.read_hits(), t.camera, lens, (w, h), (dx, dy));
            }
            for (s, p) in sum.iter_mut().zip(colors) {
                for c in 0..3 {
                    s[c] += p[c] / 16.0;
                }
            }
        }
        r.set_pixel_offset(0.0, 0.0);
        r.set_footprint(1.0);
        sum
    };
    // The lens sampled outright in our own renderer (`DOF_TRUTH` = n: n × n
    // stratified points on the aperture by the concentric map): for each,
    // the eye moved to it and the image plane sheared (the pixel offset) so
    // the plane in focus stays put, with sub-pixel offsets cycling through
    // the 4 × 4 pattern.
    let truth_n = env("DOF_TRUTH", 0u32);
    let truth_file = out_dir.join(format!(
        "dof_dev_truth_{view}_{w}x{h}_{}_{truth_n}{}.f32",
        ours_lens.aperture_radius,
        if std::env::var_os("DOF_FOLIAGE").is_some() {
            "_foliage"
        } else {
            ""
        }
    ));
    let cached: Option<Vec<[f32; 3]>> = std::fs::read(&truth_file).ok().map(|b| {
        b.as_chunks::<12>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))
            })
            .collect()
    });
    let truth = match truth_n {
        0 => None,
        _ if cached.is_some() => cached,
        n => {
            let forward = (t.camera.target - t.camera.eye).normalize();
            let right = forward.cross(continuum::glam::Vec3::Z).normalize();
            let up = right.cross(forward);
            let f = h as f32 / (2.0 * (t.camera.vertical_fov_degrees.to_radians() * 0.5).tan());
            let mut sum = vec![[0.0f64; 3]; (w * h) as usize];
            r.set_footprint(0.25);
            for k in 0..n * n {
                let (a, b) = (k / n, k % n);
                // A fixed scramble of the sub-pixel cell, apart from the lens's.
                let cell = (k * 7 + k / 16) % 16;
                let (dx, dy) = (
                    (cell % 4) as f32 * 0.25 - 0.375,
                    (cell / 4) as f32 * 0.25 - 0.375,
                );
                let (u, v) = (
                    2.0 * (a as f32 + 0.5) / n as f32 - 1.0,
                    2.0 * (b as f32 + 0.5) / n as f32 - 1.0,
                );
                let (rad, phi) = if u.abs() > v.abs() {
                    (u, std::f32::consts::FRAC_PI_4 * v / u)
                } else {
                    (
                        v,
                        std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * u / v,
                    )
                };
                let (ux, uy) = (
                    rad * phi.cos() * ours_lens.aperture_radius,
                    rad * phi.sin() * ours_lens.aperture_radius,
                );
                let shift = right * ux + up * uy;
                let camera = Camera {
                    eye: t.camera.eye + shift,
                    target: t.camera.target + shift,
                    ..t.camera
                };
                r.set_pixel_offset(
                    dx - ux * f / ours_lens.focus_distance,
                    dy + uy * f / ours_lens.focus_distance,
                );
                settle(&mut r, camera, &target);
                r.finish();
                for (s, p) in sum.iter_mut().zip(r.read_colors()) {
                    for c in 0..3 {
                        s[c] += p[c] as f64;
                    }
                }
            }
            r.set_pixel_offset(0.0, 0.0);
            r.set_footprint(1.0);
            let scale = 1.0 / (n * n) as f64;
            let image: Vec<[f32; 3]> = sum.iter().map(|p| p.map(|v| (v * scale) as f32)).collect();
            let bytes: Vec<u8> = image
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(&truth_file, bytes)?;
            Some(image)
        }
    };
    let pinhole = supersampled(&mut r, None);
    let exact = if std::env::var_os("DOF_EXACT").is_some() {
        Some(supersampled(&mut r, Some(ours_lens)))
    } else {
        None
    };
    r.set_depth_of_field(Some(ours_lens));
    let ours = supersampled(&mut r, None);
    let reference = render_instanced_dof(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        &lens,
        t.camera,
        (w, h),
        samples,
        1,
        1,
        &out_dir.join(format!("dof_dev_cycles_{view}")),
    )?;
    let cycles_pinhole = render_instanced_lights(
        &t.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (w, h),
        samples,
        1,
        1,
        &out_dir.join(format!("dof_dev_cycles_{view}")),
    )?;
    // The lens's effect: each renderer's image through it less its pinhole
    // image, ours against Cycles' (each renderer's own shading cancels).
    let effect = |a: &[[f32; 3]], b: &[[f32; 3]]| -> Vec<[f32; 3]> {
        a.iter()
            .zip(b)
            .map(|(p, q)| [p[0] - q[0], p[1] - q[1], p[2] - q[2]])
            .collect()
    };
    let ours_effect = effect(&ours, &pinhole);
    let cycles_effect = effect(&reference.pixels, &cycles_pinhole.pixels);
    let norm = |v: &[[f32; 3]]| {
        v.iter()
            .flatten()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let diff = effect(&ours_effect, &cycles_effect);
    println!(
        "  the lens's effect: ours {:.2}% of Cycles' size, off by {:.2}% of it; the pinholes {:.2}% apart",
        100.0 * norm(&ours_effect) / norm(&cycles_effect),
        100.0 * norm(&diff) / norm(&cycles_effect),
        100.0 * relative_rmse(&pinhole, &cycles_pinhole.pixels)
    );
    if let Some(truth) = &truth {
        let e = |a: &[[f32; 3]]| relative_rmse(a, truth);
        // Where ours (and the exact gather) differ from the sampled lens.
        let mut maps: Vec<(&str, &Vec<[f32; 3]>)> = vec![("ours", &ours)];
        if let Some(x) = &exact {
            maps.push(("exact", x));
        }
        for (name, img) in maps {
            let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
            for (i, (a, b)) in img.iter().zip(truth.iter()).enumerate() {
                let v = |k: usize| ((0.5 + 4.0 * (a[k] - b[k])).clamp(0.0, 1.0) * 255.0) as u8;
                image.set(i % w as usize, i / w as usize, [v(0), v(1), v(2)]);
            }
            image.save(&out_dir.join(format!("dof_dev_{view}_{name}_truth_diff.png")))?;
        }
        println!(
            "  against the lens sampled in our renderer: ours {:.2}%, pinhole {:.2}% ({:.3}x){}; the sampled lens against Cycles {:.2}%",
            100.0 * e(&ours),
            100.0 * e(&pinhole),
            e(&ours) / e(&pinhole),
            exact.as_ref().map_or(String::new(), |x| format!(
                ", the CPU's exact gather {:.2}% ({:.3}x)",
                100.0 * e(x),
                e(x) / e(&pinhole)
            )),
            100.0 * relative_rmse(truth, &reference.pixels)
        );
    }
    if let Some(exact) = &exact {
        let exact_effect = effect(exact, &pinhole);
        println!(
            "  the CPU's exact gather: off from Cycles' effect by {:.2}% of it; the GPU's off from it by {:.2}%",
            100.0 * norm(&effect(&exact_effect, &cycles_effect)) / norm(&cycles_effect),
            100.0 * norm(&effect(&ours_effect, &exact_effect)) / norm(&cycles_effect)
        );
    }
    if std::env::var_os("DOF_NOISE").is_some() {
        // Cycles' own noise in the effect: two pinhole renders with
        // different seeds differ by as much as two independent renders do.
        let other = render_instanced_lights(
            &t.scene,
            t.origin,
            &t.lights,
            t.sky,
            t.camera,
            (w, h),
            samples,
            1,
            2,
            &out_dir.join(format!("dof_dev_cycles_{view}")),
        )?;
        println!(
            "  Cycles' noise: two seeds' pinholes differ by {:.2}% of the effect",
            100.0 * norm(&effect(&other.pixels, &cycles_pinhole.pixels)) / norm(&cycles_effect)
        );
    }
    // Where the effects differ, as an image (grey 0.5 is none).
    let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
    for (i, d) in diff.iter().enumerate() {
        let v = |x: f32| ((0.5 + 2.0 * x).clamp(0.0, 1.0) * 255.0) as u8;
        image.set(i % w as usize, i / w as usize, [v(d[0]), v(d[1]), v(d[2])]);
    }
    image.save(&out_dir.join(format!("dof_dev_{view}_effect_diff.png")))?;
    let mean = |px: &[[f32; 3]]| {
        px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>() / (3 * px.len()) as f64
    };
    println!(
        "{view} {w}x{h} (focus {} m, aperture {} m, ours x{}), Cycles {samples} samples ({:.0} s, {}): ours {:.2}%, pinhole {:.2}% ({:.3}x); means: ours {:+.2}%, pinhole {:+.2}%",
        lens.focus_distance,
        lens.aperture_radius,
        env("DOF_SCALE", 1.0f32),
        reference.seconds,
        reference.device,
        100.0 * relative_rmse(&ours, &reference.pixels),
        100.0 * relative_rmse(&pinhole, &reference.pixels),
        relative_rmse(&ours, &reference.pixels) / relative_rmse(&pinhole, &reference.pixels),
        100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0),
        100.0 * (mean(&pinhole) / mean(&reference.pixels) - 1.0)
    );
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                room3d::to_rgb(continuum::glam::Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("dof_dev_{view}_{name}.png")))
    };
    save("ours", &ours)?;
    save("pinhole", &pinhole)?;
    save("cycles", &reference.pixels)?;
    Ok(())
}
