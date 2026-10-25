//! Check (not a gate): packet traversal must not change any result. Renders
//! the million-boulder field per pixel with sun and sky (320×180), and warms
//! the lighting cache at the standing viewpoint (1280×720), with invocations
//! walking alone and in packets, and compares bit for bit. Also reports how
//! far the sky's horizon cull moves the per-pixel image.

use continuum::glam::DVec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, KernelOptions, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

fn kernel(without_packets: bool) -> MaskConeKernel {
    let k = MaskConeKernel::new_with(KernelOptions {
        without_packets,
        ..Default::default()
    })
    .expect("GPU");
    k.set_prefilter(1.0);
    k.set_sky_prefilter(8.0);
    k
}

fn main() {
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    let per_pixel = |k: &MaskConeKernel| {
        k.set_hardware_rays(false);
        k.set_sky(field.sky());
        let (px, t, _) = k.render_instanced(
            camera,
            320,
            180,
            light,
            &field.scene,
            origin,
            field.radiance,
            1,
        );
        (px, t)
    };
    let cached = |k: &MaskConeKernel| {
        k.set_hardware_rays(true);
        k.set_sky(field.sky());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = k
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
        let mut r = InstancedRenderer::with_cache(
            k,
            format,
            1280,
            720,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            CacheOptions {
                frame_target_ms: 0.0,
                ..Default::default()
            },
        );
        // A fixed budget, so both runs bake the same pages in the same frames.
        for _ in 0..80 {
            r.frame(camera, &target);
        }
        r.finish();
        r.read_colors()
    };
    let solo = kernel(true);
    let packets = kernel(false);
    let (a, ta) = per_pixel(&solo);
    let (b, tb) = per_pixel(&packets);
    let same = a.iter().zip(&b).all(|(x, y)| x == y);
    println!(
        "per pixel, sun and sky: solo {:.2} s, packets {:.2} s, bit-identical: {same}",
        ta.as_secs_f64(),
        tb.as_secs_f64()
    );
    solo.set_sky_horizon_cull(false);
    let (c, tc) = per_pixel(&solo);
    solo.set_sky_horizon_cull(true);
    println!(
        "horizon cull moves the per-pixel image by {:.3}% (relative RMSE); without it, solo {:.2} s",
        100.0 * relative_rmse(&a, &c),
        tc.as_secs_f64()
    );
    let ca = cached(&solo);
    let cb = cached(&packets);
    let same = ca.iter().zip(&cb).all(|(x, y)| x == y);
    println!("cache after 80 frames, solo vs packets: bit-identical: {same}");
}
