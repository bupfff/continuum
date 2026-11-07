//! M7 development: one ray per pixel and pixel cones against a 16×16
//! supersampled reference, per-pixel lighting, at a small size. Not a gate.
//!
//! `cargo run --release -p room3d --bin m7_dev -- [textured|field|field150] [W H] [--cached]`

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, Light, MaskConeKernel, Renderer, RendererOptions, wgpu};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::textured::Textured;
use room3d::to_rgb;

#[allow(clippy::too_many_arguments)]
fn render(
    scene: &continuum_gpu::InstancedScene,
    origin: DVec3,
    lights: &[Light],
    sky: Option<Vec3>,
    camera: Camera,
    (w, h): (u32, u32),
    pixel_cones: bool,
    cached: bool,
) -> (Vec<[f32; 3]>, f64) {
    let kernel = MaskConeKernel::new().expect("GPU");
    // Development: cut light cones short (1: none) to time the rest.
    if let Some(stage) = std::env::var("M7_STAGE").ok().and_then(|s| s.parse().ok()) {
        kernel.set_profile_stage(stage);
    }
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(
        kernel,
        format,
        (w, h),
        scene,
        origin,
        lights,
        RendererOptions {
            cache: cached,
            sky,
            pixel_cones,
            ..Default::default()
        },
    );
    let target = r
        .kernel()
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
    if cached {
        let mut frames = 0;
        while frames < 20_000 && !r.is_settled(16) {
            r.render(camera, &target);
            frames += 1;
        }
        eprintln!("  settled after {frames} frames");
    }
    r.render(camera, &target);
    r.wait();
    let start = Instant::now();
    r.render(camera, &target);
    r.wait();
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (r.read_colors(), ms)
}

fn downsample(big: &[[f32; 3]], (w, h): (u32, u32), ss: u32) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; (w * h) as usize];
    let n = (ss * ss) as f32;
    for y in 0..(h * ss) as usize {
        for x in 0..(w * ss) as usize {
            let p = big[y * (w * ss) as usize + x];
            let i = (y / ss as usize) * w as usize + x / ss as usize;
            for c in 0..3 {
                out[i][c] += p[c] / n;
            }
        }
    }
    out
}

fn save(path: &Path, pixels: &[[f32; 3]], (w, h): (u32, u32)) {
    let mut image = Image::new(w as usize, h as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.first().map(String::as_str).unwrap_or("textured");
    let sizes: Vec<u32> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let size = (
        *sizes.first().unwrap_or(&320),
        *sizes.get(1).unwrap_or(&180),
    );
    let cached = args.iter().any(|a| a == "--cached");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out).unwrap();
    let (scene, origin, lights, sky, camera) = match which {
        "textured" => {
            let t = Textured::new();
            (t.scene, t.origin, t.lights, Some(t.sky), t.camera)
        }
        _ => {
            let field = Field::new(1_000_000, DVec3::ZERO);
            let origin = field.origin_near_camera();
            let (ground, light) = field.relative(origin);
            let forward = (ground.target - ground.eye).normalize();
            let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
            let (height, down) = if which == "field150" {
                (150.0f32, 45f32)
            } else {
                (1.7, 0.0)
            };
            let d = down.to_radians();
            let eye = Vec3::new(ground.eye.x, ground.eye.y, height);
            let camera = Camera {
                eye,
                target: eye + flat * d.cos() - Vec3::Z * d.sin(),
                vertical_fov_degrees: ground.vertical_fov_degrees,
            };
            let lights = vec![Light::white(light, field.radiance)];
            (field.scene, origin, lights, None, camera)
        }
    };
    if std::env::var("M7_STAGE").as_deref() == Ok("9") {
        // Work per pixel of the pixel cones' walk.
        let (work, ms) = render(&scene, origin, &lights, sky, camera, size, true, false);
        let n = work.len() as f64;
        let mean = |c: usize| work.iter().map(|p| p[c] as f64).sum::<f64>() / n;
        let max = |c: usize| work.iter().map(|p| p[c]).fold(0.0f32, f32::max);
        println!(
            "{which}: {ms:.0} ms; per pixel: object nodes {:.0} (max {}), meshes {:.1} (max {}), triangle-hierarchy nodes {:.0} (max {})",
            mean(0),
            max(0),
            mean(1),
            max(1),
            mean(2),
            max(2)
        );
        return;
    }
    let ss = 16;
    let t = Instant::now();
    let (big, _) = render(
        &scene,
        origin,
        &lights,
        sky,
        camera,
        (size.0 * ss, size.1 * ss),
        false,
        false,
    );
    let reference = downsample(&big, size, ss);
    eprintln!("reference in {:.1} s", t.elapsed().as_secs_f64());
    let (one, one_ms) = render(&scene, origin, &lights, sky, camera, size, false, cached);
    let (cones, cones_ms) = render(&scene, origin, &lights, sky, camera, size, true, cached);
    println!(
        "{which} {}×{}{}: one ray {:.2}% ({one_ms:.1} ms), pixel cones {:.2}% ({cones_ms:.1} ms) from the 16×16 reference",
        size.0,
        size.1,
        if cached { " cached" } else { "" },
        100.0 * relative_rmse(&one, &reference),
        100.0 * relative_rmse(&cones, &reference),
    );
    save(
        &out.join(format!("m7_dev_{which}_ref.png")),
        &reference,
        size,
    );
    save(&out.join(format!("m7_dev_{which}_one.png")), &one, size);
    save(&out.join(format!("m7_dev_{which}_cones.png")), &cones, size);
}
