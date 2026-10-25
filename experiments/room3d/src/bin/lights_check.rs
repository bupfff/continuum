//! Check (not a gate): several colored lights on the million-boulder field.
//! 1. One white light through the new light list: the warm cached sun-only
//!    image against per-pixel exact cones, at gate 8's standing viewpoint
//!    (gate 8: 14.49%).
//! 2. Lighting is linear: per pixel, two colored lights render as the sum of
//!    each alone.
//! 3. Re-coloring: a warm cache with two lights, re-colored with
//!    `set_light_radiance`, bakes nothing and matches a fresh cache with the
//!    new colors bit for bit.

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, RectLight, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;

/// Frames until 16 quiet frames.
fn warm(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    let mut frames = 0;
    loop {
        r.frame(camera, target);
        frames += 1;
        let latest = r.cache_stats().unwrap().frame as usize;
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return frames;
        }
    }
}

fn main() {
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, sun) = field.relative(origin);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: W,
                height: H,
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

    // 1. One white light.
    let mut r = InstancedRenderer::with_cache(
        &kernel,
        format,
        W,
        H,
        1,
        sun,
        &field.scene,
        origin,
        field.radiance,
        CacheOptions::default(),
    );
    let frames = warm(&mut r, camera, &target);
    let cached = r.read_colors();
    drop(r);
    kernel.set_prefilter(0.0);
    let exact = kernel
        .render_instanced(camera, W, H, sun, &field.scene, origin, field.radiance, 1)
        .0;
    kernel.set_prefilter(1.0);
    println!(
        "1. one white light, warm after {frames} frames: cached vs per-pixel exact {:.2}% (gate 8: 14.49%)",
        100.0 * relative_rmse(&cached, &exact)
    );

    // A second light: a small warm panel low in the west, 200 m from the
    // camera, facing it.
    let lamp = {
        let center = camera.eye + Vec3::new(-150.0, 120.0, 25.0);
        let toward = (camera.eye - center).normalize();
        let u = Vec3::Z.cross(toward).normalize() * 4.0;
        let v = toward.cross(u).normalize() * 4.0;
        RectLight {
            corner: center - (u + v) * 0.5,
            edge_u: u,
            edge_v: v,
            one_sided: true,
        }
    };
    // Facing the camera: edge_u × edge_v must point toward it.
    let lamp = if lamp.edge_u.cross(lamp.edge_v).dot(camera.eye - lamp.corner) < 0.0 {
        RectLight {
            edge_u: lamp.edge_v,
            edge_v: lamp.edge_u,
            ..lamp
        }
    } else {
        lamp
    };
    let a = Light {
        rect: sun,
        radiance: Vec3::new(1.0, 0.85, 0.7) * field.radiance,
    };
    let b = Light {
        rect: lamp,
        radiance: Vec3::new(0.3, 0.5, 1.0) * 2000.0,
    };

    // 2. Linearity per pixel (compute rays, 640×360).
    kernel.set_hardware_rays(false);
    let render = |lights: &[Light]| {
        kernel
            .render_instanced_lights(camera, 640, 360, lights, &field.scene, origin, 1)
            .0
    };
    let both = render(&[a, b]);
    let only_a = render(&[a]);
    let only_b = render(&[b]);
    let sum: Vec<[f32; 3]> = only_a
        .iter()
        .zip(&only_b)
        .map(|(x, y)| [x[0] + y[0], x[1] + y[1], x[2] + y[2]])
        .collect();
    let lit_b = only_b.iter().filter(|p| p[2] > 0.0).count();
    println!(
        "2. two lights per pixel vs the sum of each alone: {:.6}% apart ({lit_b} pixels lit by the second light)",
        100.0 * relative_rmse(&both, &sum)
    );
    kernel.set_hardware_rays(true);

    // 3. Re-coloring a warm cache.
    let a2 = Light {
        radiance: Vec3::new(0.6, 0.7, 1.0) * field.radiance,
        ..a
    };
    let b2 = Light {
        radiance: Vec3::new(1.0, 0.4, 0.1) * 4000.0,
        ..b
    };
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        W,
        H,
        1,
        &[a, b],
        &field.scene,
        origin,
        CacheOptions::default(),
    );
    let frames = warm(&mut r, camera, &target);
    r.set_light_radiance(0, a2.radiance);
    r.set_light_radiance(1, b2.radiance);
    let before = r.cache_log().len();
    for _ in 0..4 {
        r.frame(camera, &target);
    }
    r.finish();
    let baked: u32 = r.cache_log()[before..].iter().map(|s| s.pages_baked).sum();
    let recolored = r.read_colors();
    drop(r);
    let mut fresh = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        W,
        H,
        1,
        &[a2, b2],
        &field.scene,
        origin,
        CacheOptions::default(),
    );
    warm(&mut fresh, camera, &target);
    let fresh_image = fresh.read_colors();
    let same = recolored == fresh_image;
    println!(
        "3. two lights warm after {frames} frames; re-colored: {baked} pages baked, vs a fresh cache: bit-identical {same} ({:.6}% apart)",
        100.0 * relative_rmse(&recolored, &fresh_image)
    );
}
