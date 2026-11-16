//! Development (M22): bounce light with movers through the ledger, on the
//! textured scene (sun and sky, bounce light on) with six boulders moving.
//! Not a gate.
//!
//! While they move (`BM_FRAMES`, 120 frames at `BM_SPEED` m a frame along
//! +x, 0.02), every `BM_EVERY` (30) frames the image is compared with a
//! renderer settled afresh at the same poses; ours with the ledger at
//! `BM_EPS` (0.01) against a renderer whose ledger never re-bakes (ε 1e9).
//! Then they stop and ours settles: compared again. 640×360 (`BM_W`/`BM_H`).

use std::path::Path;
use std::time::Instant;

use continuum::glam::DVec3;
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    let start = r.cache_log().len();
    for n in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0 && s.indirect_samples == 0)
        {
            return n;
        }
    }
    eprintln!("WARNING: not settled");
    20_000
}

fn main() -> std::io::Result<()> {
    let (w, h): (u32, u32) = (env("BM_W", 640), env("BM_H", 360));
    let frames: usize = env("BM_FRAMES", 120);
    let every: usize = env("BM_EVERY", 30);
    let speed: f64 = env("BM_SPEED", 0.02);
    let eps: f32 = env("BM_EPS", 0.01);
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
    // `BM_SKY=0`: the sun alone (movers under a sky re-bake skies at length:
    // M9 gate 3).
    if env("BM_SKY", 1u32) == 0 {
        t.sky = continuum::glam::Vec3::ZERO;
    }
    kernel.set_sky(t.sky);
    let movers: Vec<usize> = (1..=6).collect();
    let indirect = env("BM_INDIRECT", 1u32) == 1;
    // A fixed bake budget: residency (which bounce samples read) then does
    // not depend on timing.
    let options = |eps: f32| CacheOptions {
        indirect,
        frame_target_ms: 0.0,
        bounce_epsilon: eps,
        ..Default::default()
    };
    let make = |eps: f32| {
        InstancedRenderer::with_movers(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            Some(options(eps)),
            &movers,
            &[],
        )
    };
    let place = |r: &mut InstancedRenderer, at: f64| {
        for &k in &movers {
            let i = &t.scene.instances[k];
            r.set_instance_transform(k, i.rotation, i.position + DVec3::new(at, 0.0, 0.0));
        }
    };
    // A renderer settled afresh with the boulders at `at`.
    let fresh = |at: f64| {
        let mut scene = t.scene.clone();
        for &k in &movers {
            scene.instances[k].position += DVec3::new(at, 0.0, 0.0);
        }
        // The same path (movers), built afresh at these poses.
        let mut r = InstancedRenderer::with_movers(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &scene,
            t.origin,
            Some(options(eps)),
            &movers,
            &[],
        );
        settle(&mut r, t.camera, &target);
        r.finish();
        r.read_colors()
    };
    let mut ours = make(eps);
    let mut stale = make(1e9);
    let n_ours = settle(&mut ours, t.camera, &target);
    let n_stale = settle(&mut stale, t.camera, &target);
    println!("settled at rest: ours {n_ours} frames, without re-bakes {n_stale}");
    {
        ours.finish();
        let reference = fresh(0.0);
        println!(
            "at rest before moving: ours {:.4}% from a fresh bake (movers built there)",
            100.0 * relative_rmse(&ours.read_colors(), &reference)
        );
    }
    let mut times = Vec::new();
    for f in 1..=frames {
        let at = speed * f as f64;
        place(&mut ours, at);
        place(&mut stale, at);
        let s = Instant::now();
        ours.frame(t.camera, &target);
        ours.wait();
        times.push(s.elapsed().as_secs_f64() * 1000.0);
        stale.frame(t.camera, &target);
        if f % every == 0 {
            ours.finish();
            stale.finish();
            let reference = fresh(at);
            if std::env::var_os("BM_SAVE").is_some() {
                let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
                    let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
                    for (i, (a, b)) in px.iter().zip(reference.iter()).enumerate() {
                        let v =
                            |k: usize| ((0.5 + 8.0 * (a[k] - b[k])).clamp(0.0, 1.0) * 255.0) as u8;
                        image.set(i % w as usize, i / w as usize, [v(0), v(1), v(2)]);
                    }
                    image.save(&out_dir.join(format!("bounce_movers_{f}_{name}_diff.png")))
                };
                save("ours", &ours.read_colors())?;
                save("stale", &stale.read_colors())?;
            }
            let log = ours.cache_log();
            let recent = &log[log.len().saturating_sub(every)..];
            let samples: u32 = recent.iter().map(|s| s.indirect_samples).sum();
            println!(
                "frame {f} (boulders +{at:.2} m): ours {:.3}% from a fresh bake, without re-bakes {:.3}%; bounce samples baked over the last {every} frames {samples}",
                100.0 * relative_rmse(&ours.read_colors(), &reference),
                100.0 * relative_rmse(&stale.read_colors(), &reference),
            );
        }
    }
    times.sort_by(f64::total_cmp);
    times.push(0.0);
    println!(
        "moving frames: median {:.2} ms, 90th percentile {:.2} ms",
        times[times.len() / 2],
        times[times.len() * 9 / 10]
    );
    // Stopped: ours settles (the drain), then against a fresh bake.
    let at = speed * frames as f64;
    let n = settle(&mut ours, t.camera, &target);
    ours.finish();
    let reference = fresh(at);
    println!(
        "stopped: ours settled in {n} frames, {:.4}% from a fresh bake",
        100.0 * relative_rmse(&ours.read_colors(), &reference)
    );
    Ok(())
}
