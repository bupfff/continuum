//! Development (M22, after gate 1): which pages are over ε while moving.
//! Not a gate. Gate 1's flight (a) (boulders 1–`CD_MOVERS` at `CD_SPEED` m
//! a frame along +x) at 640×360, ours (ε 0.01) and the ledger without
//! re-bakes; at every frame listed in `CD_CHECKS` (default 10,20,30,40,60)
//! both are checked with `verify_bounce` and the pages over ε classified
//! (read in the last frame, listed, baked while moving). With
//! `CONTINUUM_TRACE_BOUNCE=1` each page over ε is printed (stderr).

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::textured::Textured;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> bool {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0 && s.indirect_samples == 0)
        {
            return true;
        }
    }
    false
}

fn main() {
    let w: u32 = env("CD_W", 640);
    let h: u32 = env("CD_H", 360);
    let frames: usize = env("CD_FRAMES", 120);
    let speed: f64 = env("CD_SPEED", 0.02);
    let movers_n: usize = env("CD_MOVERS", 6);
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
    t.sky = Vec3::ZERO;
    kernel.set_sky(t.sky);
    let movers: Vec<usize> = (1..=movers_n).collect();
    let make = |indirect: bool, eps: f32| {
        InstancedRenderer::with_movers(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            Some(CacheOptions {
                indirect,
                bounce_epsilon: eps,
                bounce_share: env("CD_SHARE", 1.0),
                frame_target_ms: 0.0,
                ..Default::default()
            }),
            &movers,
            &[],
        )
    };
    let checks: Vec<usize> = std::env::var("CD_CHECKS")
        .unwrap_or_else(|_| "10,20,30,40,60".into())
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
    let frames = frames.min(*checks.iter().max().unwrap_or(&60));
    let names = ["ours", "no re-bakes"];
    let mut rs = [make(true, 0.01), make(true, 1e9)];
    for r in rs.iter_mut() {
        assert!(settle(r, t.camera, &target), "not settled");
    }
    for f in 1..=frames {
        let at = speed * f as f64;
        for (k, r) in rs.iter_mut().enumerate() {
            for &m in &movers {
                let i = &t.scene.instances[m];
                r.set_instance_transform(m, i.rotation, i.position + DVec3::new(at, 0.0, 0.0));
            }
            r.frame(t.camera, &target);
            r.wait();
            let st = r.cache_stats().unwrap();
            if checks.contains(&f) || std::env::var_os("CD_EVERY").is_some() {
                println!(
                    "frame {f} {}: bounce samples {}, pages baked {}, requests {}, resident {}, evicted {}, spent {} of {}, stale re-baked {}, invalidated {}, incomplete {}",
                    names[k],
                    st.indirect_samples,
                    st.pages_baked,
                    st.requests,
                    st.resident,
                    st.evicted,
                    st.spent,
                    st.budget,
                    st.rebaked_pages,
                    st.invalidated,
                    st.indirect_incomplete
                );
            }
        }
        if checks.contains(&f) {
            for (k, r) in rs.iter_mut().enumerate() {
                eprintln!("--- frame {f} {}", names[k]);
                let c = r.verify_bounce(0.01);
                println!(
                    "frame {f} {:12}: over eps {} of {} (visible {}, listed {}, baked moving {}), largest charge {:.2}, missed {}",
                    names[k],
                    c.over_epsilon,
                    c.pages,
                    c.visible_over,
                    c.listed_over,
                    c.moving_over,
                    c.largest_charge,
                    c.missed
                );
            }
        }
    }
}
