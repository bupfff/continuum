//! Development tool (not a gate): dumps the field's standing view, sun only: the warm cached image and the
//! per-pixel exact image, as raw f32, for comparisons across builds.
use continuum::glam::DVec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;

fn main() {
    let out = std::env::args().nth(1).expect("output prefix");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
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
    let mut r = InstancedRenderer::with_cache(
        &kernel,
        format,
        1280,
        720,
        1,
        light,
        &field.scene,
        origin,
        field.radiance,
        CacheOptions::default(),
    );
    loop {
        r.frame(camera, &target);
        let latest = r.cache_stats().unwrap().frame as usize;
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            break;
        }
    }
    r.finish();
    let save = |name: &str, px: &[[f32; 3]]| {
        let bytes: Vec<u8> = px.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{out}_{name}.bin"), bytes).unwrap();
    };
    save("cached", &r.read_colors());
    drop(r);
    kernel.set_prefilter(0.0);
    let exact = kernel
        .render_instanced(
            camera,
            1280,
            720,
            light,
            &field.scene,
            origin,
            field.radiance,
            1,
        )
        .0;
    save("exact", &exact);
}
