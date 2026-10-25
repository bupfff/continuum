//! Development check of sky light through the view cache (not a gate): warms
//! a small field's cache with sky only and compares with per-pixel cones.
//! `cargo run --release -p room3d --bin cache_sky_check -- [boulders]`

use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let count = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20_000usize);
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(count, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    let (w, h) = (640u32, 360u32);
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
    for (name, radiance) in [("sky", 0.0), ("both", field.radiance)] {
        let mut r = InstancedRenderer::with_cache(
            &kernel,
            format,
            w,
            h,
            1,
            light,
            &field.scene,
            origin,
            radiance,
            CacheOptions::default(),
        );
        let mut frames = 0;
        loop {
            r.frame(camera, &target);
            frames += 1;
            let latest = r.cache_stats().unwrap().frame as usize;
            if (latest >= 16
                && r.cache_log()[latest - 16..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0))
                || frames > 6000
            {
                break;
            }
        }
        r.finish();
        let cached = r.read_colors();
        let budget = r.cache_log().last().unwrap().budget;
        drop(r);
        let (pp, t, _) =
            kernel.render_instanced(camera, w, h, light, &field.scene, origin, radiance, 1);
        println!(
            "{name}: warm after {frames} frames (budget {budget}), cached vs per-pixel {:.2}%, per-pixel {:.1} s",
            100.0 * relative_rmse(&cached, &pp),
            t.as_secs_f64()
        );
        for (tag, img) in [("cached", &cached), ("perpixel", &pp)] {
            let mut image = Image::new(w as usize, h as usize, [0; 3]);
            for (i, p) in img.iter().enumerate() {
                image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
            }
            image.save(&out_dir.join(format!("cache_sky_{name}_{tag}.png")))?;
        }
    }
    Ok(())
}
