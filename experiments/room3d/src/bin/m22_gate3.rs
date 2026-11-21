//! M22 gate 3: bounce light that settles.
//!
//! Bounce samples read their hits' direct light from the first page
//! resident at or coarser than a bin's footprint, and kept it: finer pages
//! arriving later (asked for by pixels, or by other samples) never reached
//! them, so a settled still scene's bounce light was not what a fresh bake
//! gives (DESIGN §3.4, M22: 137–229 pages over ε in development, changes up
//! to 77% of the largest bounce value).
//!
//! Method (`CacheOptions::bounce_settles`): bounce rays read their hits one
//! mip coarser than a bin's footprint (a quarter of the pages to want); a
//! bin that finds only a coarser page asks for its page, and the sample
//! re-bakes once it may have arrived (as with missing pages).
//!
//! Views, 1280×720, a fixed bake budget (`frame_target_ms` 0), bounce light
//! on: (a) M6's textured scene, sun only; (b) M10 gate 22's night scene
//! (`with_lamps`); (c) the textured scene with its sky; (a)–(c) seen in
//! development; (d), not used in development: the textured scene with its
//! sky from the far side (eye (6, 22, 3) toward (−2, 8, 0)). In each, a
//! renderer with `bounce_settles` (new) and one without (old), each settled
//! from empty (32 frames asking for, baking and re-baking nothing).
//! `verify_bounce` re-bakes every resident page's bounce samples in place
//! and compares (restoring the cache).
//!
//! Pass criterion, fixed before the run (bars set with development known:
//! new 0.11–0.56% of pages over ε, largest change 1.4–2.7% of the largest
//! value; old 2.4–3.5% and 45–68%; settling 1.47–1.96× the old's frames):
//! 1. **Settled:** in each view the new renderer's pages over ε (0.01 of
//!    the largest bounce value) at most 1% of its bounce pages, and its
//!    largest change at most 5% of the largest bounce value.
//! 2. **Settling:** in each view the new renderer settles in at most 3× the
//!    old's frames.
//! 3. **Accuracy** (view (b), M10 gate 21's Cycles reference: one bounce,
//!    4,096 samples per pixel): the settled image supersampled 4×4, over
//!    the pixels gate 22 kept (not the lamps' disks and their neighbours),
//!    within 5% relative RMSE of Cycles and at most 0.2 percentage points
//!    above the old renderer's error.
//!
//! Rehearsal (`M22_REHEARSE=1`): views (a)–(c) as the gate measures them;
//! view (d) only run for 30 frames at 320×180, its numbers not printed.
//! Development mode (`M22_DEV=1`): 640×360, views (a)–(c), criterion 3
//! skipped (the reference is 1280×720); view (d) as in the rehearsal.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const EPS: f32 = 0.01;

/// Renders until 32 frames in a row asked for, baked and re-baked nothing;
/// the frames it took.
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    let start = r.cache_log().len();
    for k in 1..=30_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0 && s.indirect_samples == 0)
        {
            return k;
        }
    }
    panic!("the view never settled");
}

/// The view supersampled 4×4 (gate 22's): each sub-pixel offset's frame
/// after it settles, averaged.
fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    (w, h): (u32, u32),
) -> Vec<[f32; 3]> {
    let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        settle(r, camera, target);
        r.finish();
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    sum
}

struct View {
    name: &'static str,
    t: Textured,
    camera: Camera,
}

fn target(kernel: &MaskConeKernel, w: u32, h: u32) -> wgpu::TextureView {
    kernel
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

fn renderer(
    kernel: &MaskConeKernel,
    v: &View,
    (w, h): (u32, u32),
    settles: bool,
) -> InstancedRenderer {
    kernel.set_sky(v.t.sky);
    InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        w,
        h,
        1,
        &v.t.lights,
        &v.t.scene,
        v.t.origin,
        CacheOptions {
            indirect: true,
            frame_target_ms: 0.0,
            bounce_settles: settles,
            ..Default::default()
        },
    )
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M22_DEV").is_some();
    let rehearse = std::env::var_os("M22_REHEARSE").is_some();
    let size = if dev { (640u32, 360u32) } else { (1280, 720) };
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
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    say(format!(
        "M22 gate 3{}: {}x{}, adapter {}",
        if dev {
            " (development mode)"
        } else if rehearse {
            " (rehearsal)"
        } else {
            ""
        },
        size.0,
        size.1,
        kernel.adapter_name
    ));
    let mut sun = Textured::new();
    sun.sky = Vec3::ZERO;
    let lamps = Textured::new().with_lamps();
    let sky = Textured::new();
    let far = Textured::new();
    let views = [
        View {
            name: "(a) sun",
            camera: sun.camera,
            t: sun,
        },
        View {
            name: "(b) lamps",
            camera: lamps.camera,
            t: lamps,
        },
        View {
            name: "(c) sky",
            camera: sky.camera,
            t: sky,
        },
        View {
            name: "(d) sky, far side",
            camera: Camera {
                eye: Vec3::new(6.0, 22.0, 3.0),
                target: Vec3::new(-2.0, 8.0, 0.0),
                vertical_fov_degrees: 45.0,
            },
            t: far,
        },
    ];
    let tgt = target(&kernel, size.0, size.1);
    let mut all = [true; 3];
    for (vi, v) in views.iter().enumerate() {
        if (dev || rehearse) && vi == 3 {
            // Unseen: run briefly, small, its numbers not printed.
            let small = target(&kernel, 320, 180);
            for settles in [true, false] {
                let mut r = renderer(&kernel, v, (320, 180), settles);
                for _ in 0..30 {
                    r.frame(v.camera, &small);
                }
                r.finish();
            }
            say(format!(
                "{}: ran 30 frames at 320x180 (numbers not shown)",
                v.name
            ));
            continue;
        }
        let mut new = renderer(&kernel, v, size, true);
        let start = Instant::now();
        let f_new = settle(&mut new, v.camera, &tgt);
        let s_new = start.elapsed().as_secs_f64();
        let c_new = new.verify_bounce(EPS);
        let mut old = renderer(&kernel, v, size, false);
        let start = Instant::now();
        let f_old = settle(&mut old, v.camera, &tgt);
        let s_old = start.elapsed().as_secs_f64();
        let c_old = old.verify_bounce(EPS);
        let share = c_new.over_epsilon as f64 / c_new.pages.max(1) as f64;
        let largest = c_new.largest_change / c_new.largest_value.max(1e-30);
        let c1 = share <= 0.01 && largest <= 0.05;
        say(format!(
            "1. {}: settled, pages over eps {} of {} ({:.2}%, at most 1%), largest change {:.2}% of the largest value (at most 5%): {}; old {} of {} ({:.2}%), {:.2}%",
            v.name,
            c_new.over_epsilon,
            c_new.pages,
            100.0 * share,
            100.0 * largest,
            verdict(c1),
            c_old.over_epsilon,
            c_old.pages,
            100.0 * c_old.over_epsilon as f64 / c_old.pages.max(1) as f64,
            100.0 * c_old.largest_change / c_old.largest_value.max(1e-30)
        ));
        let ratio = f_new as f64 / f_old.max(1) as f64;
        let c2 = ratio <= 3.0;
        say(format!(
            "2. {}: settled in {f_new} frames ({s_new:.1} s), old {f_old} ({s_old:.1} s): {ratio:.2}x (at most 3x): {}",
            v.name,
            verdict(c2)
        ));
        all[0] &= c1;
        all[1] &= c2;
        // 3. Accuracy against Cycles, view (b) at full size.
        if vi == 1 && !dev {
            let t = &v.t;
            let reference = render_instanced_lights(
                &t.scene,
                t.origin,
                &t.lights,
                t.sky,
                t.camera,
                size,
                4096,
                1,
                1,
                &out_dir.join("m10_gate21_cycles"),
            )?;
            let (w, h) = size;
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
            let e_new = relative_rmse(&keep(&supersampled(&mut new, v.camera, &tgt, size)), &kept);
            let e_old = relative_rmse(&keep(&supersampled(&mut old, v.camera, &tgt, size)), &kept);
            let c3 = e_new <= 0.05 && e_new <= e_old + 0.002;
            say(format!(
                "3. {}: from Cycles over surfaces {:.3}% (at most 5% and the old's + 0.2 points), old {:.3}%: {}",
                v.name,
                100.0 * e_new,
                100.0 * e_old,
                verdict(c3)
            ));
            all[2] &= c3;
        }
    }
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {}: {}",
        verdict(all[0]),
        verdict(all[1]),
        if dev { "skipped" } else { verdict(all[2]) },
        verdict(all.iter().all(|&x| x))
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m22_gate3_dev.txt"
        } else if rehearse {
            "m22_gate3_rehearsal.txt"
        } else {
            "m22_gate3.txt"
        }),
        report,
    )?;
    Ok(())
}
