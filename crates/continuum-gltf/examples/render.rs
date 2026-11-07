//! Renders a glTF file to a PNG: lit by the file's own lights (or else a sun
//! 30° up), seen from its camera (or one framed around the scene), through
//! Continuum's lighting cache until the view's lighting is complete.
//!
//! `cargo run --release -p continuum-gltf --example render -- scene.glb [out.png] [W H] [--sky]`

use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, Light, MaskConeKernel, Renderer, RendererOptions, wgpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let input = args
        .first()
        .ok_or("usage: render scene.glb [out.png] [W H] [--sky]")?;
    let output = args
        .get(1)
        .filter(|a| !a.starts_with("--") && a.parse::<u32>().is_err())
        .cloned()
        .unwrap_or_else(|| "render.png".to_string());
    let sizes: Vec<u32> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let (width, height) = (
        *sizes.first().unwrap_or(&1280),
        *sizes.get(1).unwrap_or(&720),
    );
    let sky = args.iter().any(|a| a == "--sky");

    let imported = continuum_gltf::load(Path::new(input))?;
    println!("{input}: {:?}", imported.stats);
    // Render relative to the scene's center.
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    let camera = match imported.camera {
        Some(c) => Camera {
            eye: (c.eye - origin).as_vec3(),
            target: (c.target - origin).as_vec3(),
            vertical_fov_degrees: c.vertical_fov_degrees,
        },
        None => {
            let (az, el) = (225f64.to_radians(), 25f64.to_radians());
            let toward = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
            Camera {
                eye: (toward * radius * 2.2).as_vec3(),
                target: Vec3::ZERO,
                vertical_fov_degrees: 40.0,
            }
        }
    };
    // The file's lights (point and spot lights 5 cm in radius), or a sun
    // giving irradiance π (white at albedo 1) on surfaces facing it.
    let mut lights = imported.lights(origin, 0.05);
    lights.truncate(continuum_gpu::MAX_LIGHTS);
    if lights.is_empty() {
        let (az, el) = (200f64.to_radians(), 30f64.to_radians());
        let toward = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
        lights.push(Light::sun(
            Vec3::ZERO,
            toward,
            radius * 50.0,
            Vec3::new(1.0, 0.92, 0.8) * std::f32::consts::PI,
        ));
    }

    let kernel = MaskConeKernel::new().ok_or("no GPU adapter")?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = Renderer::new(
        kernel,
        format,
        (width, height),
        &imported.scene,
        origin,
        &lights,
        RendererOptions {
            sky: sky.then_some(Vec3::new(0.6, 0.75, 1.0) * 0.3),
            ..Default::default()
        },
    );
    let target = renderer
        .kernel()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width,
                height,
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
    // Frames until the cache has baked everything the view needs.
    let mut frames = 0;
    while frames < 50_000 && !renderer.is_settled(16) {
        renderer.render(camera, &target);
        frames += 1;
    }
    println!("lighting complete after {frames} frames");

    // Linear colors to 8-bit sRGB.
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    let pixels: Vec<u8> = renderer
        .read_colors()
        .iter()
        .flat_map(|p| p.map(encode))
        .collect();
    let file = std::fs::File::create(&output)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    println!("wrote {output}");
    Ok(())
}
