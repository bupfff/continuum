//! Coarser levels' shading against the finest (development, not a gate):
//! M14 gate 1's rock field, a renderer with every finest level resident
//! against one streaming camera levels within a budget, each settled at the
//! three flights' last views and averaged over 4×4 sub-pixel offsets; the
//! colour difference (relative RMSE) and the mean brightness ratio, with
//! the levels' own normals and with averaged ones (`Detail::
//! finest_normals`).
//!
//! `cargo run --release -p room3d --bin normals_cmp_dev -- [side=N]`

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, StreamOptions, wgpu};
use room3d::cycles::relative_rmse;
use room3d::rocks::{Detail, Rocks};

const W: u32 = 1280;
const H: u32 = 720;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    for f in 1..=4000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        let cache_quiet = latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0);
        let log = r.stream_log();
        let stream_quiet = log.len() < 16
            || log[log.len() - 16..]
                .iter()
                .all(|s| s.loads == 0 && s.evictions == 0);
        if cache_quiet && stream_quiet && f >= 16 {
            r.finish();
            return f;
        }
    }
    4000
}

fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> Vec<[f32; 3]> {
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        r.frame(camera, target);
        r.finish();
        for (c, s) in color.iter_mut().zip(r.read_colors()) {
            for i in 0..3 {
                c[i] += s[i] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    color
}

fn mean(image: &[[f32; 3]]) -> f64 {
    image.iter().flatten().map(|&v| v as f64).sum::<f64>() / (3 * image.len()) as f64
}

fn main() {
    let side: usize = std::env::args()
        .find_map(|a| a.strip_prefix("side=").map(|v| v.parse().unwrap()))
        .unwrap_or(16);
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
    for averaged in [false, true] {
        let rocks = Rocks::with_detail(
            side,
            6.0,
            Detail {
                finest_normals: averaged,
                ..Default::default()
            },
        );
        let half = rocks.half as f32;
        let make = |options: StreamOptions| {
            InstancedRenderer::with_streaming(
                &kernel,
                format,
                W,
                H,
                1,
                &[rocks.sun],
                &rocks.scene,
                DVec3::ZERO,
                Some(CacheOptions::default()),
                options,
            )
        };
        let mut streamed = make(StreamOptions {
            budget_triangles: 5 << 19,
            ..Default::default()
        });
        let mut resident = make(StreamOptions {
            resident: true,
            ..Default::default()
        });
        let mut row = format!("averaged normals {averaged}:");
        for k in 0..3 {
            let last = room3d::rocks::flight(k, 1.0, half, 50.0);
            settle(&mut resident, last, &target);
            let a = supersampled(&mut resident, last, &target);
            settle(&mut streamed, last, &target);
            let b = supersampled(&mut streamed, last, &target);
            row += &format!(
                " view {k}: {:.3}% apart, streamed/resident mean {:.4};",
                100.0 * relative_rmse(&b, &a),
                mean(&b) / mean(&a)
            );
        }
        println!("{row}");
    }
}
