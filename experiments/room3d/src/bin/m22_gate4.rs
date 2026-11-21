//! M22 gate 4: bounce light on by default in `Renderer`.
//!
//! Method: `RendererOptions::bounce` (on by default) turns on the cache's
//! bounce light (`CacheOptions::indirect`), with the ledger for movers (M22
//! gates 1–2) and settling (gate 3), except with moving lights, the
//! sub-pixel path, or movers without hardware ray queries.
//! `Renderer::is_settled` now waits for bounce samples too. With it, the
//! time budget's GPU model steers only while the frame's timestamps arrive
//! in order (the bake's span before the camera passes'); on this device
//! they often do not, and a stray sample held the budget at its floor for
//! hundreds of frames (DESIGN §3.4), so the frame period steers instead.
//!
//! Views, `Renderer` with its defaults (anti-aliasing, tone map, bloom,
//! auto-exposure, the default time budget) at 1280×720, against the same
//! with `bounce: false`: (a) M6's textured scene with its sun and sky; (b)
//! M10 gate 22's night scene; (a) and (b) seen in development; (c), not
//! used in development: M10 gate 2's materials (`with_materials`) seen from
//! the far side (eye (6, 22, 3) toward (−2, 8, 0)).
//!
//! Pass criterion, fixed before the run (bars set with development known:
//! 1.075× and 1.119×; 0.39×; 2.5× and 1.84×):
//! 1. **Cost:** in each view, the median over five interleaved rounds (30
//!    warm frames each, each waited for; off, then on) of the ratio at most
//!    1.2×.
//! 2. **Accuracy** (view (b), M10 gate 21's Cycles reference: one bounce,
//!    4,096 samples per pixel): the settled linear image over gate 22's
//!    kept pixels at most 0.5× the error with bounce off.
//! 3. **Settling:** in each view, frames until `is_settled(32)` with bounce
//!    at most 3× without.
//! 4. **Control** (run after, its own binary): M14 gate 11's timed flights
//!    (36,864 rocks, sky and bounce) with this code still hold its first
//!    criterion (mean and 99th percentile at most 16.7 ms, worst at most
//!    33.3 ms).
//!
//! Rehearsal (`M22_REHEARSE=1`): views (a) and (b) as the gate measures
//! them, three rounds; view (c) only run for 30 frames at 320×180, its
//! numbers not printed.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Camera, MaskConeKernel, Renderer, RendererOptions, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

fn target(r: &Renderer, w: u32, h: u32) -> wgpu::TextureView {
    r.kernel()
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn settle(r: &mut Renderer, camera: Camera, view: &wgpu::TextureView) -> (usize, f64) {
    let start = Instant::now();
    let mut frames = 0;
    while frames < 20_000 && !(frames > 32 && r.is_settled(32)) {
        r.render(camera, view);
        frames += 1;
    }
    (frames, start.elapsed().as_secs_f64())
}

fn warm(r: &mut Renderer, camera: Camera, view: &wgpu::TextureView) -> f64 {
    let mut t: Vec<f64> = (0..30)
        .map(|_| {
            let s = Instant::now();
            r.render(camera, view);
            r.wait();
            s.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[15]
}

fn main() -> std::io::Result<()> {
    let rehearse = std::env::var_os("M22_REHEARSE").is_some();
    let (w, h) = (1280u32, 720u32);
    let rounds = if rehearse { 3 } else { 5 };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        std::io::stdout().flush().ok();
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    say(format!(
        "M22 gate 4{}: {w}x{h}",
        if rehearse { " (rehearsal)" } else { "" }
    ));
    let mut all = [true; 3];
    for name in ["(a) day", "(b) night", "(c) materials, far side"] {
        let (t, camera) = match name {
            "(b) night" => {
                let t = Textured::new().with_lamps();
                let c = t.camera;
                (t, c)
            }
            "(c) materials, far side" => (
                Textured::new().with_materials(),
                Camera {
                    eye: Vec3::new(6.0, 22.0, 3.0),
                    target: Vec3::new(-2.0, 8.0, 0.0),
                    vertical_fov_degrees: 45.0,
                },
            ),
            _ => {
                let t = Textured::new();
                let c = t.camera;
                (t, c)
            }
        };
        let make = |bounce: bool, (w, h): (u32, u32)| {
            let kernel = MaskConeKernel::new().expect("GPU");
            Renderer::new(
                kernel,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                (w, h),
                &t.scene,
                t.origin,
                &t.lights,
                RendererOptions {
                    sky: (t.sky != Vec3::ZERO).then_some(t.sky),
                    bounce,
                    ..Default::default()
                },
            )
        };
        if rehearse && name.starts_with("(c)") {
            for bounce in [true, false] {
                let mut r = make(bounce, (320, 180));
                let v = target(&r, 320, 180);
                for _ in 0..30 {
                    r.render(camera, &v);
                }
                r.wait();
            }
            say(format!(
                "{name}: ran 30 frames at 320x180 (numbers not shown)"
            ));
            continue;
        }
        let mut on = make(true, (w, h));
        let mut off = make(false, (w, h));
        let (v_on, v_off) = (target(&on, w, h), target(&off, w, h));
        let (f_on, s_on) = settle(&mut on, camera, &v_on);
        let (f_off, s_off) = settle(&mut off, camera, &v_off);
        let mut ratios = Vec::new();
        for _ in 0..rounds {
            let a = warm(&mut off, camera, &v_off);
            let b = warm(&mut on, camera, &v_on);
            ratios.push((b / a, a, b));
        }
        ratios.sort_by(|x, y| x.0.total_cmp(&y.0));
        let (ratio, a, b) = ratios[ratios.len() / 2];
        let c1 = ratio <= 1.2;
        say(format!(
            "1. {name}: warm frame {b:.2} ms with bounce, {a:.2} ms without: median ratio {ratio:.3}x over {rounds} rounds (at most 1.2x): {}",
            verdict(c1)
        ));
        let settle_ratio = f_on as f64 / f_off.max(1) as f64;
        let c3 = settle_ratio <= 3.0 && f_on < 20_000;
        say(format!(
            "3. {name}: settled in {f_on} frames ({s_on:.1} s) with bounce, {f_off} ({s_off:.1} s) without: {settle_ratio:.2}x (at most 3x): {}; GPU memory {:.0} MB and {:.0} MB",
            verdict(c3),
            on.gpu_bytes() as f64 / 1e6,
            off.gpu_bytes() as f64 / 1e6
        ));
        all[0] &= c1;
        all[2] &= c3;
        if name == "(b) night" {
            let reference = render_instanced_lights(
                &t.scene,
                t.origin,
                &t.lights,
                t.sky,
                t.camera,
                (w, h),
                4096,
                1,
                1,
                &out_dir.join("m10_gate21_cycles"),
            )?;
            let peak = t
                .lights
                .iter()
                .map(|l| l.radiance.max_element())
                .fold(0.0f32, f32::max);
            let sees: Vec<bool> = reference
                .pixels
                .iter()
                .map(|p| p.iter().any(|&c| c > 0.01 * peak))
                .collect();
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
            let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
                img.iter()
                    .zip(&apart)
                    .filter(|(_, a)| !**a)
                    .map(|(p, _)| *p)
                    .collect()
            };
            let kept = keep(&reference.pixels);
            let e_on = relative_rmse(&keep(&on.read_colors()), &kept);
            let e_off = relative_rmse(&keep(&off.read_colors()), &kept);
            let ratio = e_on / e_off.max(1e-30);
            let c2 = ratio <= 0.5;
            say(format!(
                "2. {name}: from Cycles over surfaces {:.2}% with bounce, {:.2}% without ({ratio:.3}x, at most 0.5x): {}",
                100.0 * e_on,
                100.0 * e_off,
                verdict(c2)
            ));
            all[1] &= c2;
        }
    }
    say(format!(
        "Verdict (criteria 1-3; the control runs on its own): 1 {}, 2 {}, 3 {}: {}",
        verdict(all[0]),
        verdict(all[1]),
        verdict(all[2]),
        verdict(all.iter().all(|&x| x))
    ));
    std::fs::write(
        out_dir.join(if rehearse {
            "m22_gate4_rehearsal.txt"
        } else {
            "m22_gate4.txt"
        }),
        report,
    )?;
    Ok(())
}
