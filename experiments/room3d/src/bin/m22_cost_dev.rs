//! Development (M22, after gate 1): where the moving frame's cost goes with
//! bounce light and movers. Not a gate.
//!
//! Gate 1's flight (a) (boulders 1–6 at `CD_SPEED` m a frame, 0.02, for
//! `CD_FRAMES`, 120) at 640×360 (`CD_W`/`CD_H`), a fixed bake budget, sun
//! only; three renderers timed interleaved each frame: ours (ε 0.01), the
//! ledger without re-bakes (ε 1e9) and bounce light off. Prints each one's
//! median frame and its mean bounce samples, pages baked and stale pages
//! re-baked a frame, and GPU bake/render times where the device has them.
//! `CD_GATE1_ORDER=1` times them as gate 1 did: ours, bounce off, then the
//! renderer without re-bakes not waited for (its GPU work runs into the
//! next frame of ours).

use std::time::Instant;

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
    let first: usize = env("CD_FIRST", 1);
    let movers: Vec<usize> = (first..first + movers_n).collect();
    // Along +x, or −y with `CD_DIR=y`.
    let step = if std::env::var("CD_DIR").is_ok_and(|d| d == "y") {
        DVec3::new(0.0, -speed, 0.0)
    } else {
        DVec3::new(speed, 0.0, 0.0)
    };
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
    let names = ["ours", "no re-bakes", "bounce off"];
    let mut rs = [make(true, 0.01), make(true, 1e9), make(false, 0.01)];
    for r in rs.iter_mut() {
        assert!(settle(r, t.camera, &target), "not settled");
    }
    let gate1_order = std::env::var_os("CD_GATE1_ORDER").is_some();
    // `CD_PROFILE=1`: before each of ours' timed frames, one profiled frame
    // (its passes each submitted and waited for alone), summed by pass.
    let profile = std::env::var_os("CD_PROFILE").is_some();
    let mut passes: std::collections::BTreeMap<&str, f64> = Default::default();
    let order: [usize; 3] = if gate1_order { [0, 2, 1] } else { [0, 1, 2] };
    let mut times = [Vec::new(), Vec::new(), Vec::new()];
    let mut sums = [[0f64; 6]; 3];
    for f in 1..=frames {
        let at = f as f64;
        for k in order {
            let r = &mut rs[k];
            for &m in &movers {
                let i = &t.scene.instances[m];
                r.set_instance_transform(m, i.rotation, i.position + step * at);
            }
            if profile && k == 0 {
                let p = r.frame_profiled(t.camera, &target);
                for (label, d) in p.bake_passes {
                    if !label.is_empty() {
                        *passes.entry(label).or_insert(0.0) += d.as_secs_f64() * 1000.0;
                    }
                }
                for (i, d) in p.passes.iter().enumerate() {
                    *passes.entry(["camera 0", "camera 1", "camera 2", "camera 3", "camera 4", "camera 5", "camera 6", "camera 7", "camera 8", "camera 9", "camera 10", "camera 11", "camera 12", "camera 13", "camera 14", "camera 15"][i]).or_insert(0.0) += d.as_secs_f64() * 1000.0;
                }
                *passes.entry("bake (all)").or_insert(0.0) += p.bake.as_secs_f64() * 1000.0;
                *passes.entry("planning").or_insert(0.0) += p.planning.as_secs_f64() * 1000.0;
            }
            let s = Instant::now();
            r.frame(t.camera, &target);
            if !(gate1_order && k == 1) {
                r.wait();
            }
            times[k].push(s.elapsed().as_secs_f64() * 1000.0);
            if let Some(st) = r.cache_stats() {
                sums[k][0] += st.indirect_samples as f64;
                sums[k][1] += st.pages_baked as f64;
                sums[k][2] += st.rebaked_pages as f64;
                sums[k][3] += st.gpu_bake_ms as f64;
                sums[k][4] += st.gpu_render_ms as f64;
                sums[k][5] += st.indirect_incomplete as f64;
            }
        }
    }
    let n = frames as f64;
    for k in 0..3 {
        let mut v = times[k].clone();
        v.sort_by(f64::total_cmp);
        println!(
            "{:12} median {:6.2} ms (p10 {:6.2}, p90 {:6.2}); a frame: bounce samples {:7.1} ({:6.1} incomplete), pages baked {:6.1}, stale re-baked {:6.1}",
            names[k],
            v[v.len() / 2],
            v[v.len() / 10],
            v[9 * v.len() / 10],
            sums[k][0] / n,
            sums[k][5] / n,
            sums[k][1] / n,
            sums[k][2] / n
        );
    }
    for (label, ms) in &passes {
        if *ms > 0.0 {
            println!("profiled {label:24} {:7.3} ms a frame", ms / n);
        }
    }
}
