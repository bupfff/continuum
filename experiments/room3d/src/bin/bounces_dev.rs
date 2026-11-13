//! How much light past the first bounce carries (development, not a gate):
//! Cycles at 1 and at 4 bounces on M6's textured scene by day and at night
//! (M10 gate 22's lamps), 640×360, 4,096 samples per pixel (cached).
//!
//! With `cache`: also the showcase room through the cache with one bounce,
//! settled at 4×4 sub-pixel offsets, against Cycles at 1 and 4 bounces.
//!
//! `cargo run --release -p room3d --bin bounces_dev -- [cache]`

use std::path::Path;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use continuum_gpu::{Instance, InstancedScene, Light};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::showcase::Showcase;
use room3d::textured::Textured;

/// The showcase's static surfaces as an instanced scene (one instance each,
/// in place), its light, and its first camera.
fn showcase() -> (InstancedScene, Vec<Light>, continuum_gpu::Camera) {
    let s = Showcase::new();
    let n = s.first_mover;
    let scene = InstancedScene {
        geometries: s.surfaces[..n].to_vec(),
        instances: (0..n)
            .map(|k| Instance {
                geometry: k,
                rotation: Mat3::IDENTITY,
                scale: 1.0,
                position: DVec3::ZERO,
                albedo: s.albedo[k],
            })
            .collect(),
        textures: Vec::new(),
        geometry_textures: Vec::new(),
        geometry_materials: Vec::new(),
        geometry_emission: Vec::new(),
        geometry_emissive_textures: Vec::new(),
        geometry_blend: Vec::new(),
        geometry_normal_maps: Vec::new(),
        geometry_metallic_roughness: Vec::new(),
        geometry_alpha_cutoff: Vec::new(),
        geometry_colors: Vec::new(),
        texture_wrap: Vec::new(),
        geometry_levels: Vec::new(),
        level_source: None,
    };
    (
        scene,
        vec![Light::white(s.light, s.radiance)],
        s.camera(0.0),
    )
}

/// The room through the cache with one bounce, settled at 4×4 sub-pixel
/// offsets (640×360).
fn cached_room(
    scene: &InstancedScene,
    lights: &[Light],
    camera: continuum_gpu::Camera,
) -> Vec<[f32; 3]> {
    let (w, h) = (640u32, 360u32);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_hardware_rays(true);
    kernel.set_sky(Vec3::ZERO);
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
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        w,
        h,
        1,
        lights,
        scene,
        DVec3::ZERO,
        CacheOptions {
            indirect: true,
            frame_target_ms: 0.0,
            bounces: std::env::args()
                .find_map(|a| a.strip_prefix("bounces=").map(|v| v.parse().unwrap()))
                .unwrap_or(1),
            ..Default::default()
        },
    );
    let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
    let begin = std::time::Instant::now();
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        let start = r.cache_log().len();
        for _ in 0..50_000 {
            r.frame(camera, &target);
            let log = r.cache_log();
            if log.len() >= start + 32
                && log[log.len() - 32..]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0)
            {
                break;
            }
        }
        r.finish();
        if k == 0 {
            let log = r.cache_log();
            println!(
                "  first offset settled from empty in {} frames ({:.1} s): {} pages, {} bounce samples baked",
                log.len() - 32,
                begin.elapsed().as_secs_f64(),
                log.iter().map(|s| s.pages_baked as u64).sum::<u64>(),
                log.iter().map(|s| s.indirect_samples as u64).sum::<u64>()
            );
        }
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    sum
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let day = Textured::new();
    let night = Textured::new().with_lamps();
    let (room, room_lights, room_camera) = showcase();
    for (name, scene, origin, lights, sky, camera) in [
        (
            "day",
            &day.scene,
            day.origin,
            day.lights.clone(),
            day.sky,
            day.camera,
        ),
        (
            "night",
            &night.scene,
            night.origin,
            night.lights.clone(),
            night.sky,
            night.camera,
        ),
        (
            "showcase room",
            &room,
            DVec3::ZERO,
            room_lights.clone(),
            Vec3::ZERO,
            room_camera,
        ),
    ] {
        let render = |bounces: u32| {
            render_instanced_lights(
                scene,
                origin,
                &lights,
                sky,
                camera,
                (640, 360),
                4096,
                bounces,
                1,
                &out_dir.join("bounces_dev_cycles"),
            )
        };
        let zero = render(0)?;
        let one = render(1)?;
        let four = render(4)?;
        let twelve = render(12)?;
        let mean = |r: &room3d::cycles::Reference| {
            r.pixels.iter().flatten().map(|&v| v as f64).sum::<f64>() / (3 * r.pixels.len()) as f64
        };
        if name == "showcase room" && std::env::args().any(|a| a == "cache") {
            let ours = cached_room(scene, &lights, camera);
            let m = ours.iter().flatten().map(|&v| v as f64).sum::<f64>() / (3 * ours.len()) as f64;
            println!(
                "showcase room through the cache: mean radiance {m:.4}; against Cycles 1 bounce {:.2}%, 4 bounces {:.2}%, 12 bounces {:.2}% (Cycles' 12: mean {:.4})",
                100.0 * relative_rmse(&ours, &one.pixels),
                100.0 * relative_rmse(&ours, &four.pixels),
                100.0 * relative_rmse(&ours, &twelve.pixels),
                twelve
                    .pixels
                    .iter()
                    .flatten()
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / (3 * twelve.pixels.len()) as f64
            );
        }
        println!(
            "{name}: mean radiance 0 bounces {:.4}, 1 bounce {:.4}, 4 bounces {:.4}; 1 bounce against 4: {:.2}% (relative RMSE), 0 against 4: {:.2}%",
            mean(&zero),
            mean(&one),
            mean(&four),
            100.0 * relative_rmse(&one.pixels, &four.pixels),
            100.0 * relative_rmse(&zero.pixels, &four.pixels)
        );
    }
    Ok(())
}
