//! Sky through hardware rays against cones, both against Cycles
//! (development, not a gate): the million-boulder field with its sun at
//! 1 km, sky only and sun and sky, the cache settled at 4×4 sub-pixel
//! offsets (Cycles averages each pixel's area), 640×360, against instanced
//! Cycles at 1,024 samples per pixel (cached on disk).
//!
//! `cargo run --release -p room3d --bin sky_rays_dev -- [view 0-3]
//! [boulders=N]`

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced};
use room3d::field::Field;

const W: u32 = 640;
const H: u32 = 360;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    for k in 1..=60_000usize {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return k;
        }
    }
    60_000
}

fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<[f32; 3]>, usize) {
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    let mut frames = 0;
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        frames += settle(r, camera, target);
        r.finish();
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    (sum, frames)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let view_index: usize = args
        .iter()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(0);
    let boulders: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("boulders="))
        .map_or(1_000_000, |v| v.parse().unwrap());
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let field = Field::new(boulders, DVec3::ZERO).with_sun_at(1000.0);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let sky = field.sky();
    kernel.set_sky(sky);
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
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |ahead: f32, height: f32, down: f32| {
        let d = down.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", ground),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ];
    let (name, camera) = views[view_index.min(3)];
    for (label, radiance) in [("sky only", 0.0f32), ("sun and sky", field.radiance)] {
        let start = Instant::now();
        let reference = render_instanced(
            &field.scene,
            origin,
            light,
            radiance,
            sky,
            camera,
            (W, H),
            1024,
            0,
            1,
            &out_dir.join("sky_rays_dev_cycles"),
        )?;
        let cycles_s = start.elapsed().as_secs_f64();
        let mut row = format!("{name}, {label} (Cycles {cycles_s:.0} s):");
        for rays in [false, true] {
            let mut r = InstancedRenderer::with_cache(
                &kernel,
                format,
                W,
                H,
                1,
                light,
                &field.scene,
                origin,
                radiance,
                CacheOptions {
                    frame_target_ms: 0.0,
                    bake_budget: 11_000,
                    sky_rays: rays,
                    ..Default::default()
                },
            );
            let (image, frames) = supersampled(&mut r, camera, &target);
            let e = relative_rmse(&image, &reference.pixels);
            row += &format!(
                " {} {:.2}% ({frames} frames)",
                if rays { "rays" } else { "cones" },
                100.0 * e
            );
        }
        println!("{row}");
    }
    Ok(())
}
