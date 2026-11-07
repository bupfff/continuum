//! M7 gate 3: anti-aliasing at silhouettes only.
//!
//! Pixel cones (M7 gate 1) resolved coverage in every pixel and cost 48× a
//! frame. Most pixels lie inside one surface, where one ray and the box
//! texture filter (gate 2) already approximate the area integral; the
//! aliasing is at edges.
//!
//! Method: in the cached shading pass, a pixel whose four neighbours'
//! visibility differs from its own (another object, or a depth 5% away)
//! is an edge. An edge pixel averages four rays through the centers of its
//! 2×2 quarters (compute rays through the scene's hierarchy), each shaded
//! from the cache; other pixels keep their one ray.
//!
//! Scenes, 1280×720, cache settled (32 quiet frames after the call):
//! M6's textured scene, and the million-boulder field from 150 m up (M7
//! gate 1's view), sun only.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Quality:** in both views, relative RMSE from the area-integrated
//!    cached image (16×16 sub-pixel offsets, each settled, textures for
//!    1/16 of a pixel) at most 0.75× that of one ray per pixel.
//! 2. **Speed:** the warm frame at most 1.5× one ray's in both views
//!    (median of 30 frames each waited for).
//!
//! Reported: the share of pixels found to be edges.
//!
//! `cargo run --release -p room3d --bin m7_gate3`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 frames rendered since the call requested and baked
/// nothing.
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..60_000 {
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
    panic!("the view never settled");
}

struct View<'a> {
    name: &'static str,
    scene: &'a InstancedScene,
    origin: DVec3,
    lights: Vec<Light>,
    sky: Vec3,
    camera: Camera,
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
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
    let t = Textured::new();
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let d = 45f32.to_radians();
    let eye = Vec3::new(ground.eye.x, ground.eye.y, 150.0);
    let high = Camera {
        eye,
        target: eye + flat * d.cos() - Vec3::Z * d.sin(),
        vertical_fov_degrees: ground.vertical_fov_degrees,
    };
    let views = [
        View {
            name: "textured",
            scene: &t.scene,
            origin: t.origin,
            lights: t.lights.clone(),
            sky: t.sky,
            camera: t.camera,
        },
        View {
            name: "field, 150 m up",
            scene: &field.scene,
            origin,
            lights: vec![Light::white(sun, field.radiance)],
            sky: Vec3::ZERO,
            camera: high,
        },
    ];
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let (mut c1, mut c2) = (true, true);
    for v in &views {
        kernel.set_sky(v.sky);
        let renderer = |edge_rays: bool| {
            InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                W,
                H,
                1,
                &v.lights,
                v.scene,
                v.origin,
                CacheOptions {
                    edge_rays,
                    ..Default::default()
                },
            )
        };
        let warm = |r: &mut InstancedRenderer| {
            let mut times: Vec<Duration> = (0..30)
                .map(|_| {
                    let start = Instant::now();
                    r.frame(v.camera, &target);
                    r.wait();
                    start.elapsed()
                })
                .collect();
            times.sort();
            times[15]
        };
        // 2. Speed, first.
        let mut one = renderer(false);
        settle(&mut one, v.camera, &target);
        let one_time = warm(&mut one);
        let mut edge = renderer(true);
        settle(&mut edge, v.camera, &target);
        let edge_time = warm(&mut edge);
        edge.finish();
        let edges = edge.cache_stats().map_or(0, |s| s.edge_pixels);
        let speed = ms(edge_time) <= 1.5 * ms(one_time);
        c2 &= speed;
        // 1. Quality against the area-integrated cached image.
        one.finish();
        let one_image = one.read_colors();
        let edge_image = edge.read_colors();
        let mut reference = vec![[0.0f64; 3]; (W * H) as usize];
        one.set_footprint(1.0 / 16.0);
        for k in 0..256 {
            let offset = |i: u32| (i as f32 + 0.5) / 16.0 - 0.5;
            one.set_pixel_offset(offset(k % 16), offset(k / 16));
            settle(&mut one, v.camera, &target);
            one.finish();
            for (s, p) in reference.iter_mut().zip(one.read_colors()) {
                for c in 0..3 {
                    s[c] += p[c] as f64 / 256.0;
                }
            }
        }
        let reference: Vec<[f32; 3]> = reference.iter().map(|s| s.map(|x| x as f32)).collect();
        let (e_one, e_edge) = (
            relative_rmse(&one_image, &reference),
            relative_rmse(&edge_image, &reference),
        );
        let quality = e_edge <= 0.75 * e_one;
        c1 &= quality;
        say(format!(
            "{}: quality {:.2}% with edge rays, {:.2}% one ray ({:.2}×, at most 0.75×): {}; speed {:.2} ms against {:.2} ms ({:.2}×, at most 1.5×): {}; edge pixels {:.1}%",
            v.name,
            100.0 * e_edge,
            100.0 * e_one,
            e_edge / e_one,
            if quality { "PASS" } else { "FAIL" },
            ms(edge_time),
            ms(one_time),
            ms(edge_time) / ms(one_time),
            if speed { "PASS" } else { "FAIL" },
            100.0 * edges as f64 / (W * H) as f64
        ));
    }
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m7_gate3.txt"), report)?;
    Ok(())
}
