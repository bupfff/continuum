//! Development (M22, bounce light by default): `Renderer` with its default
//! options (bounce light on) against the same with `bounce: false`, at
//! 1280×720 (`DD_W`/`DD_H`) with the default time budget. Not a gate.
//!
//! Scenes: `day` (M6's textured scene with its sun and sky) and `night` (M10
//! gate 22's, `with_lamps`). For each: frames and seconds until
//! `is_settled(32)` (at most 20,000 frames), then `DD_ROUNDS` (5) rounds of
//! 30 warm frames each (each waited for), off then on, medians; for the
//! night scene the settled linear image against M10 gate 21's Cycles
//! reference over gate 22's kept pixels.

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Camera, MaskConeKernel, Renderer, RendererOptions, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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
    let trace = std::env::var_os("DD_TRACE").is_some();
    let mut acc = [0u64; 7];
    while frames < 20_000 && !(frames > 32 && r.is_settled(32)) {
        r.render(camera, view);
        frames += 1;
        if trace && let Some(st) = r.stats() {
            for (a, v) in acc.iter_mut().zip([
                st.requests,
                st.pages_baked,
                st.indirect_samples,
                st.indirect_incomplete,
                st.budget,
                st.spent,
                st.evicted,
            ]) {
                *a += v as u64;
            }
            if std::env::var_os("DD_FRAMES").is_some() && frames % 25 == 0 && frames < 600 {
                println!(
                    "    frame {frames}: budget {} cap {} spent {} gpu bake {:.2} ms render {:.2} ms timed {} waiting {:.2} ms planning {:.2} ms",
                    st.budget,
                    st.budget_cap,
                    st.spent,
                    st.gpu_bake_ms,
                    st.gpu_render_ms,
                    st.timed,
                    st.waiting.as_secs_f64() * 1e3,
                    st.planning.as_secs_f64() * 1e3
                );
            }
            if frames % 100 == 0 {
                println!(
                    "  frames {:5}: requests {:7} baked {:6} samples {:7} incomplete {:6} budget {:6} spent {:6} evicted {:5}",
                    frames - 100,
                    acc[0],
                    acc[1],
                    acc[2],
                    acc[3],
                    acc[4] / 100,
                    acc[5] / 100,
                    acc[6]
                );
                acc = [0; 7];
            }
        }
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
    let (w, h): (u32, u32) = (env("DD_W", 1280), env("DD_H", 720));
    let rounds: usize = env("DD_ROUNDS", 5);
    let only: String = env("DD_SCENE", String::new());
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    for name in ["day", "night"] {
        if !only.is_empty() && only != name {
            continue;
        }
        let t = if name == "night" {
            Textured::new().with_lamps()
        } else {
            Textured::new()
        };
        let make = |bounce: bool| {
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
                    cache_options: continuum_gpu::CacheOptions {
                        bake_budget: env(
                            "DD_BUDGET",
                            continuum_gpu::CacheOptions::default().bake_budget,
                        ),
                        frame_budget_ms: env(
                            "DD_FRAME_BUDGET",
                            continuum_gpu::CacheOptions::default().frame_budget_ms,
                        ),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
        };
        let mut on = make(true);
        let mut off = make(false);
        let (v_on, v_off) = (target(&on, w, h), target(&off, w, h));
        let (f_on, s_on) = settle(&mut on, t.camera, &v_on);
        let (f_off, s_off) = settle(&mut off, t.camera, &v_off);
        println!(
            "{name}: settled in {f_on} frames ({s_on:.1} s) with bounce, {f_off} ({s_off:.1} s) without; GPU memory {:.0} MB and {:.0} MB",
            on.gpu_bytes() as f64 / 1e6,
            off.gpu_bytes() as f64 / 1e6
        );
        let mut ratios = Vec::new();
        for k in 0..rounds {
            let a = warm(&mut off, t.camera, &v_off);
            let b = warm(&mut on, t.camera, &v_on);
            ratios.push(b / a);
            println!(
                "{name} round {k}: off {a:.2} ms, on {b:.2} ms ({:.3}x)",
                b / a
            );
        }
        ratios.sort_by(f64::total_cmp);
        println!(
            "{name}: warm frame median ratio {:.3}x",
            ratios[ratios.len() / 2]
        );
        if name == "night" && (w, h) == (1280, 720) {
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
            println!(
                "night: from Cycles over surfaces {:.2}% with bounce, {:.2}% without",
                100.0 * e_on,
                100.0 * e_off
            );
        }
    }
    Ok(())
}
