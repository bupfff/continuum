//! Development (files people use): loads a glTF file and builds `Renderer`
//! with its defaults at 1280×720, timing each step and listing the largest
//! GPU buffers; then renders `GD_FRAMES` (120) frames, each waited for, from
//! `GD_EYE` toward `GD_TARGET` (`x,y,z` from the scene's center; else the
//! file's camera or the viewer's), saving the last to `GD_OUT` (a PNG,
//! linear colors times `GD_EXPOSURE`, clamped). `GD_CUTOUTS=1` only cuts the
//! alpha cutouts and counts triangles. Not a gate.
//!
//! `cargo run --release -p room3d --bin gltf_dev -- scene.gltf`

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Light, MaskConeKernel, Renderer, RendererOptions, wgpu};

fn main() {
    let path = std::env::args().nth(1).expect("a glTF file");
    let t = Instant::now();
    let mut imported = continuum_gltf::load(Path::new(&path)).expect("the file");
    if std::env::var_os("GD_NOCUT").is_some() {
        imported.scene.geometry_alpha_cutoff.clear();
    }
    println!(
        "load {:.2} s: {:?}",
        t.elapsed().as_secs_f64(),
        imported.stats
    );
    let tri: usize = imported
        .scene
        .geometries
        .iter()
        .map(|g| g.triangles.len())
        .sum();
    let charts: usize = imported
        .scene
        .geometries
        .iter()
        .map(|g| g.charts.len())
        .sum();
    let big = imported
        .scene
        .geometries
        .iter()
        .map(|g| g.triangles.len())
        .max()
        .unwrap_or(0);
    println!(
        "geometries: {} with {tri} triangles (largest {big}), {charts} charts",
        imported.scene.geometries.len()
    );
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    let mut lights = imported.lights(origin, 0.05);
    if lights.is_empty() {
        let toward = continuum::glam::DVec3::new(-0.6, -0.3, 0.75);
        lights.push(Light::sun(
            Vec3::ZERO,
            toward,
            radius * 50.0,
            Vec3::splat(3.0),
        ));
    }
    if std::env::var_os("GD_CUTOUTS").is_some() {
        let t = Instant::now();
        let cut = imported.scene.with_cutouts();
        println!("cutouts {:.2} s", t.elapsed().as_secs_f64());
        let mut rows: Vec<(usize, usize, usize)> = (0..cut.geometries.len())
            .map(|g| {
                (
                    g,
                    imported.scene.geometries[g].triangles.len(),
                    cut.geometries[g].triangles.len(),
                )
            })
            .filter(|r| r.1 != r.2)
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        for (g, a, b) in rows.iter().take(12) {
            println!(
                "  geometry {g}: {a} -> {b} triangles; texture {:?}, cutoff {:?}",
                imported.scene.geometry_textures.get(*g).copied().flatten(),
                imported
                    .scene
                    .geometry_alpha_cutoff
                    .get(*g)
                    .copied()
                    .flatten()
            );
        }
        let total: usize = cut.geometries.iter().map(|g| g.triangles.len()).sum();
        let charts: usize = cut.geometries.iter().map(|g| g.charts.len()).sum();
        println!(
            "  total {total} triangles, {charts} charts after cutouts ({} geometries changed)",
            rows.len()
        );
        for (g, _, _) in rows.iter().take(3) {
            let s = &cut.geometries[*g];
            println!("  geometry {g}: {} charts", s.charts.len());
        }
        return;
    }
    let t = Instant::now();
    let kernel = MaskConeKernel::new().expect("GPU");
    println!("kernel {:.2} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let mut r = Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (1280, 720),
        &imported.scene,
        origin,
        &lights,
        RendererOptions {
            sky: (std::env::var("GD_SKY").as_deref() != Ok("0"))
                .then_some(Vec3::new(0.6, 0.75, 1.0) * 0.3),
            bounce: std::env::var("GD_BOUNCE").as_deref() != Ok("0"),
            anti_aliasing: std::env::var("GD_AA").as_deref() != Ok("0"),
            cache_options: continuum_gpu::CacheOptions {
                glossy_rays: std::env::var("GD_GLOSSY").as_deref() != Ok("0"),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    r.wait();
    println!(
        "renderer {:.2} s, {:.0} MB",
        t.elapsed().as_secs_f64(),
        r.gpu_bytes() as f64 / 1e6
    );
    let (scene, cache) = r.memory_report();
    let top = |v: &[(u32, u64)]| {
        v.iter()
            .take(6)
            .map(|(k, b)| format!("{k}: {:.0} MB", *b as f64 / 1e6))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("scene buffers: {}", top(&scene));
    println!("cache buffers: {}", top(&cache));
    let vec = |name: &str| {
        std::env::var(name).ok().map(|v| {
            let c: Vec<f32> = v
                .split(',')
                .map(|x| x.trim().parse().expect("x,y,z"))
                .collect();
            Vec3::new(c[0], c[1], c[2])
        })
    };
    let (eye, target, fov) = match (vec("GD_EYE"), vec("GD_TARGET"), imported.camera) {
        (Some(e), Some(t), _) => (e, t, 60.0),
        (_, _, Some(c)) => (
            (c.eye - origin).as_vec3(),
            (c.target - origin).as_vec3(),
            c.vertical_fov_degrees,
        ),
        _ => {
            let (az, el) = (225f64.to_radians(), 25f64.to_radians());
            let dir =
                continuum::glam::DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
            ((dir * radius * 2.2).as_vec3(), Vec3::ZERO, 40.0)
        }
    };
    println!("bounds {lo:.2?} to {hi:.2?}; eye {eye:.2}, target {target:.2}");
    let camera = continuum_gpu::Camera {
        eye,
        target,
        vertical_fov_degrees: fov,
    };
    let view = r
        .kernel()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1280,
                height: 720,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let frames: usize = std::env::var("GD_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let mut times = Vec::new();
    let start = Instant::now();
    let mut settled = None;
    for f in 0..frames {
        let t = Instant::now();
        r.render(camera, &view);
        r.wait();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        if std::env::var_os("GD_TRACE").is_some()
            && f % 25 == 0
            && let Some(s) = r.stats()
        {
            println!(
                "frame {f}: {:.1} ms (planning {:?}, evicting {:?}, waiting {:?}); fallback {}, requests {}, baked {} pages {} texels, budget {}, resident {}, evicted {}, bake {:.1} ms, render {:.1} ms, bounce samples {} ({} incomplete)",
                times[f],
                s.planning,
                s.evicting,
                s.waiting,
                s.fallback_pixels,
                s.requests,
                s.pages_baked,
                s.texels_baked,
                s.budget,
                s.resident,
                s.evicted,
                s.gpu_bake_ms,
                s.gpu_render_ms,
                s.indirect_samples,
                s.indirect_incomplete
            );
        }
        if settled.is_none() && f > 32 && r.is_settled(32) {
            settled = Some((f, start.elapsed().as_secs_f64()));
        }
    }
    if frames > 0 {
        let mut late = times[frames / 2..].to_vec();
        late.sort_by(f64::total_cmp);
        println!(
            "frames: first {:.1} ms, later half median {:.2} ms, p90 {:.2} ms; settled at {settled:?} (frame, s)",
            times[0],
            late[late.len() / 2],
            late[late.len() * 9 / 10]
        );
    }
    if std::env::var_os("GD_PROFILE").is_some() {
        let mut sums = [0.0f64; 16];
        let mut bake = std::collections::BTreeMap::<&str, f64>::new();
        for _ in 0..10 {
            let p = r.render_profiled(camera, &view);
            for (i, d) in p.passes.iter().enumerate() {
                sums[i] += d.as_secs_f64() * 100.0;
            }
            for (label, d) in p.bake_passes {
                if !label.is_empty() {
                    *bake.entry(label).or_insert(0.0) += d.as_secs_f64() * 100.0;
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
    if let Some(out) = std::env::var_os("GD_OUT") {
        let exposure: f32 = std::env::var("GD_EXPOSURE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let colors = r.read_colors();
        let srgb = |x: f32| {
            let x = (x * exposure).clamp(0.0, 1.0);
            let s = if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u8
        };
        let bytes: Vec<u8> = colors.iter().flat_map(|c| c.map(srgb)).collect();
        let file = std::fs::File::create(&out).expect("the image");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut w| w.write_image_data(&bytes))
            .expect("the PNG");
    }
}
