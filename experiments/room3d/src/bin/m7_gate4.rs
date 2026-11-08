//! M7 gate 4: silhouette rays through hardware ray queries.
//!
//! M7 gate 3: four rays through an edge pixel's 2×2 quarters halve the
//! boulder field's error from the area integral from 150 m up (0.49×), but
//! as compute rays through a million objects they cost 13× the frame there,
//! and 1.66× on the textured scene with only 1% of pixels edges.
//!
//! Method (`CacheOptions::edge_rays` with hardware rays):
//! - A pass after the camera's hardware visibility finds edge pixels (as
//!   gate 3: a neighbour sees another object or a depth 5% away), traces
//!   their four quarter-rays with ray queries (front faces only, as the
//!   camera's), and lists each edge pixel with its four hits.
//! - A separate shading pass shades only the listed pixels: each hit from
//!   the cache, averaged. The main shading pass no longer runs the edge
//!   test or the four-sample loop.
//! - Without hardware rays, gate 3's compute path is unchanged.
//!
//! Scenes and views as gate 3 (M6's textured scene; the field from 150 m
//! up, sun only), 1280×720, cache settled (32 quiet frames after the call).
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Quality as the compute path's:** relative RMSE from the
//!    area-integrated cached image (16×16 sub-pixel offsets, each settled,
//!    textures for 1/16 of a pixel, as gate 3) with hardware silhouette
//!    rays at most 1.05× that of gate 3's compute path (the same scene with
//!    the kernel's hardware rays off), measured in this run.
//! 2. **Speed** (measured first): the warm frame with hardware silhouette
//!    rays at most 1.5× one ray's (median of 30 frames each waited for).
//!
//! Reported, not gated: the share of edge pixels; the warm frame of gate 3's
//! compute path; the image's distance from the compute path's; its error
//! with one ray.
//!
//! Changed after this binary's first commit, before any full-size run
//! (development runs, `edges_dev`, 640×360; the field with 100,000
//! boulders): criterion 1 was "within 0.1% of the compute path's image",
//! but hardware and software camera rays alone differ by 0.84% on the field
//! with one ray per pixel, so it measured the camera rays, not the edge
//! rays (hardware edges were 0.73% from the compute path). Development
//! numbers: textured 1.21× one ray's frame (compute path 2.20×), 0.058%
//! from the compute path; field 2.55× (compute path 16.0×). Reusing the
//! pixel's color for quarters on its own object saved little (2.49×) and
//! changed the image by 20%; not kept.
//!
//! `cargo run --release -p room3d --bin m7_gate4`

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
        kernel.set_hardware_rays(true);
        let mut one = renderer(false);
        settle(&mut one, v.camera, &target);
        let one_time = warm(&mut one);
        let mut edge = renderer(true);
        assert!(edge.hardware_rays(), "the gate needs hardware rays");
        settle(&mut edge, v.camera, &target);
        let edge_time = warm(&mut edge);
        edge.finish();
        let edges = edge.cache_stats().map_or(0, |s| s.edge_pixels);
        let speed = ms(edge_time) <= 1.5 * ms(one_time);
        c2 &= speed;
        // Gate 3's compute path: the kernel's hardware rays off.
        kernel.set_hardware_rays(false);
        let mut compute = renderer(true);
        settle(&mut compute, v.camera, &target);
        let compute_time = warm(&mut compute);
        compute.finish();
        let compute_image = compute.read_colors();
        drop(compute);
        kernel.set_hardware_rays(true);
        // 1. The same image as the compute path.
        one.finish();
        let one_image = one.read_colors();
        let edge_image = edge.read_colors();
        let same = relative_rmse(&edge_image, &compute_image);
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
        let (e_one, e_edge, e_compute) = (
            relative_rmse(&one_image, &reference),
            relative_rmse(&edge_image, &reference),
            relative_rmse(&compute_image, &reference),
        );
        let quality = e_edge <= 1.05 * e_compute;
        c1 &= quality;
        say(format!(
            "{}: 1. from the area integral {:.2}% with hardware silhouette rays, {:.2}% compute path ({:.3}×, at most 1.05×): {}; 2. speed {:.2} ms against {:.2} ms one ray ({:.2}×, at most 1.5×): {}; reported: compute path {:.2} ms ({:.2}×), edge pixels {:.1}%, {:.3}% from the compute path's image, one ray {:.2}% from the area integral ({:.2}× with silhouette rays)",
            v.name,
            100.0 * e_edge,
            100.0 * e_compute,
            e_edge / e_compute,
            if quality { "PASS" } else { "FAIL" },
            ms(edge_time),
            ms(one_time),
            ms(edge_time) / ms(one_time),
            if speed { "PASS" } else { "FAIL" },
            ms(compute_time),
            ms(compute_time) / ms(one_time),
            100.0 * edges as f64 / (W * H) as f64,
            100.0 * same,
            100.0 * e_one,
            e_edge / e_one,
        ));
    }
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m7_gate4.txt"), report)?;
    Ok(())
}
