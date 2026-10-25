//! Diagnostic (not a gate): how many sky samples a view cache really needs.
//! Warms the million-boulder field's cache with sun and sky (a sky sample at
//! every texel) at gate 8's viewpoints, reads every resident page's sky
//! channel back, and finds for each page the coarsest stride (8, 4, 2 or 1)
//! whose bilinear interpolation, as the fill pass does it, stays within a
//! tolerance of every texel. That is an oracle: it bounds what any criterion
//! choosing strides per page could save. Tolerances are fractions of the
//! mean sky texel over all pages.
//! `cargo run --release -p room3d --bin sky_density_oracle -- [viewpoint ...]`
//! (viewpoints 0–3: standing, 10 m, 40 m, 150 m up; all by default).

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, CachePage, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::field::Field;

/// Linear interpolation span of texel `x` at `stride`, as the shader's
/// `sky_span`: (first sample, second sample, weight of the second).
fn span(x: u32, stride: u32, inside: u32) -> (u32, u32, f32) {
    let x0 = (x / stride) * stride;
    let x1 = x0 + stride;
    if x1 >= inside || x1 >= 9 {
        return (x0, x0, 0.0);
    }
    (x0, x1, (x - x0) as f32 / stride as f32)
}

/// The largest error over a page's inside texels when its sky is
/// interpolated from samples at `stride`.
fn page_error(p: &CachePage, stride: u32) -> f32 {
    let at = |x: u32, y: u32| p.texels[(y * 9 + x) as usize];
    let mut worst = 0.0f32;
    for y in 0..p.inside[1] {
        for x in 0..p.inside[0] {
            let (x0, x1, fx) = span(x, stride, p.inside[0]);
            let (y0, y1, fy) = span(y, stride, p.inside[1]);
            let a = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let b = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            let v = a + (b - a) * fy;
            worst = worst.max((v - at(x, y)).abs());
        }
    }
    worst
}

/// Sky samples a page needs at `stride` (listed texels: inside and one past).
fn samples(p: &CachePage, stride: u32) -> u64 {
    let listed = p.inside.map(|n| (n + 1).min(9));
    (listed[0].div_ceil(stride) * listed[1].div_ceil(stride)) as u64
}

fn main() {
    let chosen: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    // Development switches: SKY_PREFILTER (bins, default 8), SELF (own-object
    // mask spacing, default the cache's).
    kernel.set_sky_prefilter(
        std::env::var("SKY_PREFILTER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8.0),
    );
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    let lights = [Light::white(light, field.radiance)];
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |height: f32, down: f32| {
        let d = down.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(1.7, 0.0)),
        ("10 m up", view(10.0, 15.0)),
        ("40 m up", view(40.0, 30.0)),
        ("150 m up", view(150.0, 45.0)),
    ];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
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
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    println!("| Viewpoint | Pages | Samples, every texel | Tolerance | Samples, oracle | Saving | Pages at stride 8 / 4 / 2 / 1 |");
    println!("|---|---|---|---|---|---|---|");
    for (k, (name, camera)) in views.iter().enumerate() {
        if !chosen.is_empty() && !chosen.contains(&k) {
            continue;
        }
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            1280,
            720,
            1,
            &lights,
            &field.scene,
            origin,
            CacheOptions {
                self_spacing: std::env::var("SELF")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(CacheOptions::default().self_spacing),
                ..Default::default()
            },
        );
        let mut frames = 0;
        loop {
            r.frame(*camera, &target);
            frames += 1;
            let latest = r.cache_stats().unwrap().frame as usize;
            if (latest >= 16
                && r.cache_log()[latest - 16..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0))
                || frames > 40_000
            {
                break;
            }
        }
        r.finish();
        let pages = r.read_cache_pages(1);
        drop(r);
        let (sum, count) = pages.iter().fold((0.0f64, 0u64), |(s, n), p| {
            let mut s = s;
            for y in 0..p.inside[1] {
                for x in 0..p.inside[0] {
                    s += p.texels[(y * 9 + x) as usize] as f64;
                }
            }
            (s, n + (p.inside[0] * p.inside[1]) as u64)
        });
        let mean = (sum / count.max(1) as f64) as f32;
        let full: u64 = pages.iter().map(|p| samples(p, 1)).sum();
        if std::env::var("DUMP").is_ok() {
            // Error quantiles at stride 2, and a few pages needing stride 1.
            let mut e: Vec<f32> = pages.iter().map(|p| page_error(p, 2) / mean).collect();
            e.sort_by(f32::total_cmp);
            let q = |f: f64| e[((e.len() - 1) as f64 * f) as usize];
            println!(
                "mean {mean:.3}; stride-2 error / mean: median {:.4}, 75% {:.4}, 90% {:.4}, 99% {:.4}",
                q(0.5),
                q(0.75),
                q(0.9),
                q(0.99)
            );
            let mut mips = std::collections::BTreeMap::new();
            for p in &pages {
                *mips.entry(p.mip).or_insert(0) += 1;
            }
            println!("pages by mip: {mips:?}");
            for p in pages
                .iter()
                .filter(|p| p.inside == [9, 9] && page_error(p, 2) > 0.05 * mean)
                .step_by(997)
                .take(4)
            {
                println!(
                    "object {} chart {} mip {} texel {:.4} m:",
                    p.object, p.chart, p.mip, p.texel
                );
                for y in 0..9 {
                    let row: Vec<String> = (0..9)
                        .map(|x| format!("{:.3}", p.texels[y * 9 + x]))
                        .collect();
                    println!("  {}", row.join(" "));
                }
            }
        }
        for tolerance in [0.005f32, 0.01, 0.02] {
            let mut needed = 0u64;
            let mut at = [0usize; 4];
            for p in &pages {
                let mut chosen = 1;
                for (i, stride) in [8u32, 4, 2].into_iter().enumerate() {
                    if page_error(p, stride) <= tolerance * mean {
                        chosen = stride;
                        at[i] += 1;
                        break;
                    }
                }
                if chosen == 1 {
                    at[3] += 1;
                }
                needed += samples(p, chosen);
            }
            println!(
                "| {name} (warm after {frames} frames) | {} | {full} | {:.1}% | {needed} | {:.1}× | {} / {} / {} / {} |",
                pages.len(),
                100.0 * tolerance,
                full as f64 / needed as f64,
                at[0],
                at[1],
                at[2],
                at[3]
            );
        }
    }
}
