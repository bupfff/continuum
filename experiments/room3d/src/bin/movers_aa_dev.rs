//! Development: the showcase (instanced, its ball and cube moving) through
//! the cache with movers, anti-aliasing on or off (`MA_AA=0`), bounce light
//! on or off (`MA_BOUNCE=0`), at `MA_W`×`MA_H` (1280×720). Times frames
//! (each waited for) along the camera path, then profiles a few (each pass
//! submitted and waited for alone). Not a gate.

use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::showcase::{Showcase, mover_poses};

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let (w, h): (u32, u32) = (env("MA_W", 1280), env("MA_H", 720));
    let aa = env("MA_AA", 1) == 1;
    let bounce = env("MA_BOUNCE", 1) == 1;
    let moving = env("MA_MOVE", 1) == 1;
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
    let show = Showcase::new();
    let scene = show.instanced();
    let movers: Vec<usize> = (show.first_mover..show.first_mover + 2).collect();
    let lights = [Light::rect(show.light, Vec3::splat(show.radiance))];
    let mut r = InstancedRenderer::with_movers(
        &kernel,
        format,
        w,
        h,
        1,
        &lights,
        &scene,
        continuum::glam::DVec3::ZERO,
        Some(CacheOptions {
            indirect: bounce,
            anti_aliasing: aa,
            ..Default::default()
        }),
        &movers,
        &[],
    );
    let mut times = Vec::new();
    let place = |r: &mut InstancedRenderer, f: f32| {
        for (k, (rot, pos)) in mover_poses(f).into_iter().enumerate() {
            r.set_instance_transform(movers[k], rot, pos.as_dvec3());
        }
    };
    for f in 0..300 {
        if moving {
            place(&mut r, f as f32);
        }
        let s = Instant::now();
        let cam = if std::env::var_os("MA_FIXED").is_some() {
            show.camera(0.0)
        } else {
            show.camera(f as f32)
        };
        r.frame(cam, &target);
        r.wait();
        times.push(s.elapsed().as_secs_f64() * 1000.0);
        if f % 60 == 30
            && let Some(st) = r.cache_stats()
        {
            println!(
                "frame {f}: rebaked pages {}, texels {}, candidates {}, invalidated {}, missed {}, visible {}, pages baked {}",
                st.rebaked_pages,
                st.rebaked_texels,
                st.candidates,
                st.invalidated,
                st.missed,
                st.visible_pages,
                st.pages_baked
            );
        }
    }
    let mut v = times[60..].to_vec();
    v.sort_by(f64::total_cmp);
    println!(
        "aa {aa}, bounce {bounce}, moving {moving}: median {:.2} ms, p90 {:.2} ms",
        v[v.len() / 2],
        v[v.len() * 9 / 10]
    );
    let mut sums = [0.0f64; 16];
    let mut bake = std::collections::BTreeMap::<&str, f64>::new();
    for f in 300..320 {
        if moving {
            place(&mut r, f as f32);
        }
        let p = r.frame_profiled(show.camera(f as f32), &target);
        for (i, d) in p.passes.iter().enumerate() {
            sums[i] += d.as_secs_f64() * 1000.0 / 20.0;
        }
        for (label, d) in p.bake_passes {
            if !label.is_empty() {
                *bake.entry(label).or_insert(0.0) += d.as_secs_f64() * 1000.0 / 20.0;
            }
        }
    }
    let passes: Vec<String> = sums
        .iter()
        .enumerate()
        .filter(|(_, t)| **t > 0.05)
        .map(|(i, t)| format!("{i}: {t:.2}"))
        .collect();
    println!("camera passes (ms): {}", passes.join(", "));
    let b: Vec<String> = bake.iter().map(|(k, t)| format!("{k} {t:.2}")).collect();
    println!("bake passes (ms): {}", b.join(", "));
}
