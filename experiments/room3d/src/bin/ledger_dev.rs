//! Development: what the moving light's mixed texels hold (candidate 4,
//! M9 gate 4's follow-up). Warms M9 gate 4's scene at the standing
//! viewpoint at 640×360, then reads the moving light's channel back and
//! counts texels stored as 0, as 1, and in between; then moves the light at
//! a real sun's speed for 60 frames and reports the ledger's re-bakes. Not
//! a gate.

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{CacheOptions, Camera, Emitter, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::field::Field;

const W: u32 = 640;
const H: u32 = 360;

fn turned(light: Light, center: Vec3, angle: f32) -> Light {
    let r = Mat3::from_rotation_z(angle);
    let mut out = light;
    if let Emitter::Rect(mut rect) = light.emitter {
        rect.corner = center + r * (rect.corner - center);
        rect.edge_u = r * rect.edge_u;
        rect.edge_v = r * rect.edge_v;
        out.emitter = Emitter::Rect(rect);
    }
    out
}

fn main() {
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_sky(Vec3::ZERO);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, _) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0);
    let center = (field.scene.instances[0].position - origin).as_vec3();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let mut r = InstancedRenderer::with_movers(
        &kernel, format, W, H, 1, &lights, &field.scene, origin, Some(CacheOptions::default()), &[], &[0],
    );
    let camera: Camera = ground;
    for _ in 0..20_000 {
        r.frame(camera, &target);
        let log = r.cache_log();
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize).min(log.len());
        if latest >= 32 && log[latest - 32..latest].iter().all(|s| s.requests == 0 && s.pages_baked == 0) {
            break;
        }
    }
    r.finish();
    let pages = r.read_cache_pages(0);
    let (mut zero, mut one, mut mid, mut near0, mut near1) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for p in &pages {
        for y in 0..p.inside[1] as usize {
            for x in 0..p.inside[0] as usize {
                let v = p.texels[y * 9 + x];
                if v == 0.0 {
                    zero += 1;
                } else if v == 1.0 {
                    one += 1;
                } else {
                    mid += 1;
                    if v < 0.05 { near0 += 1; }
                    if v > 0.95 { near1 += 1; }
                }
            }
        }
    }
    let all = (zero + one + mid) as f64;
    println!(
        "{} pages, {} texels: 0 {:.1}%, 1 {:.1}%, between {:.1}% (below 0.05: {:.1}%, above 0.95: {:.1}%)",
        pages.len(), all, 100.0 * zero as f64 / all, 100.0 * one as f64 / all, 100.0 * mid as f64 / all,
        100.0 * near0 as f64 / all, 100.0 * near1 as f64 / all
    );
    let start = r.cache_stats().map_or(0, |s| s.frame as usize);
    for f in 1..=60 {
        r.set_light(0, turned(lights[0], center, (f as f32 / 14_400.0).to_radians()));
        r.frame(camera, &target);
    }
    r.finish();
    let log = r.cache_log();
    let moving = &log[start..];
    let n = moving.len() as f64;
    let avg = |f: fn(&continuum_gpu::CacheStats) -> u32| moving.iter().map(|s| f(s) as f64).sum::<f64>() / n;
    println!(
        "moving 60 frames: ledger texels {:.0}, pages checked {:.0}, mixed {:.0}, whole {:.0}, fallback {:.0} px",
        avg(|s| s.ledger_texels), avg(|s| s.ledger_checked), avg(|s| s.ledger_mixed), avg(|s| s.ledger_whole), avg(|s| s.fallback_pixels)
    );
}
