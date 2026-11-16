//! Development (M19): the display's upscalers against a native render, and
//! bloom's and the upscalers' cost. Not a gate.
//!
//! The textured scene with foliage at 640×360 drawn to a 1280×720 float
//! target (render scale 2) by each upscaler, against the scene rendered at
//! 1280×720 (each settled; no bloom, no auto-exposure, exposure 1).

use std::time::Instant;

use continuum_gpu::{
    Bloom, CacheOptions, Camera, InstancedRenderer, MaskConeKernel, Upscale, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return;
        }
    }
}

fn read(kernel: &MaskConeKernel, texture: &wgpu::Texture) -> Vec<[f32; 3]> {
    let device = kernel.device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (W * H * 16) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 16),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    kernel.queue().submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    buffer
        .slice(..)
        .get_mapped_range()
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| {
            let f = |k: usize| f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
            [f(0), f(4), f(8)]
        })
        .collect()
}

fn main() {
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let format = wgpu::TextureFormat::Rgba32Float;
    let texture = kernel.device().create_texture(&wgpu::TextureDescriptor {
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target = texture.create_view(&Default::default());
    let t = Textured::new().with_foliage();
    kernel.set_sky(t.sky);
    let make = |scale: u32| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            scale,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        );
        settle(&mut r, t.camera, &target);
        r
    };
    let timed = |r: &mut InstancedRenderer| {
        let mut v: Vec<f64> = (0..40)
            .map(|_| {
                let s = Instant::now();
                r.frame(t.camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v[20]
    };
    let mut native = make(1);
    let t_native = timed(&mut native);
    native.frame(t.camera, &target);
    native.wait();
    let reference = read(&kernel, &texture);
    let mut half = make(2);
    for up in [Upscale::Nearest, Upscale::Bilinear, Upscale::CatmullRom] {
        half.set_upscale(up);
        let ms = timed(&mut half);
        half.frame(t.camera, &target);
        half.wait();
        let img = read(&kernel, &texture);
        println!(
            "{up:?}: {:.2}% from the native render; frame {ms:.3} ms ({:.3}x native {t_native:.3})",
            100.0 * relative_rmse(&img, &reference),
            ms / t_native
        );
    }
    native.set_bloom(Some(Bloom::default()));
    let t_bloom = timed(&mut native);
    native.set_bloom(None);
    let t_plain = timed(&mut native);
    println!(
        "bloom: {t_bloom:.3} ms against {t_plain:.3} ms (+{:.3})",
        t_bloom - t_plain
    );
}
