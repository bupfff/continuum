//! Development (animated characters under a sky): a glTF file's skinned and
//! morphing meshes posed every frame through `Renderer`, over a grey ground,
//! under the viewer's sun with the sky (`AD_SKY=0`: the sun alone), at
//! 1280×720 seen from the viewer's default view. Times `AD_FRAMES` (300)
//! frames, prints the cache's work every 30, then profiles ten frames' bake
//! passes. `AD_LAG=1`: `CacheOptions::sky_lag`. Not a gate.
//!
//! `cargo run --release -p room3d --bin anim_dev -- Fox.glb`

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Instance, Light, MaskConeKernel, Renderer, RendererOptions, wgpu};

fn main() {
    let path = std::env::args().nth(1).expect("a glTF file");
    let sky = std::env::var("AD_SKY").as_deref() != Ok("0");
    let frames: usize = std::env::var("AD_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let imported = continuum_gltf::load(Path::new(&path)).expect("the file");
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    let animations = imported.animations;
    let deforming = animations.deforming();
    let mut scene = imported.scene;
    // The viewer's ground: 20 radii across, just below the scene.
    let h = (10.0 * radius) as f32;
    let c = [(-h, -h), (h, -h), (h, h), (-h, h)].map(|(x, y)| Vec3::new(x, y, 0.0));
    scene.geometries.push(Surface::from_triangles(vec![
        [c[0], c[1], c[2]],
        [c[0], c[2], c[3]],
    ]));
    scene.geometry_textures.resize(scene.geometries.len(), None);
    scene.instances.push(Instance {
        geometry: scene.geometries.len() - 1,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position: DVec3::new(origin.x, origin.y, lo.z - 1e-3 * radius),
        albedo: Vec3::splat(0.5),
    });
    let (az, el) = (200f64.to_radians(), 30f64.to_radians());
    let toward = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
    let lights = [Light::sun(
        Vec3::ZERO,
        toward,
        radius * 50.0,
        Vec3::new(1.0, 0.92, 0.8) * std::f32::consts::PI,
    )];
    let (az, el) = (225f64.to_radians(), 25f64.to_radians());
    let dir = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
    let camera = Camera {
        eye: (dir * radius * 2.2).as_vec3(),
        target: Vec3::ZERO,
        vertical_fov_degrees: 40.0,
    };
    let (w, h): (u32, u32) = (
        std::env::var("AD_W")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1280),
        std::env::var("AD_H")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(720),
    );
    // `AD_BGRA=1`: a window's format.
    let fmt = if std::env::var_os("AD_BGRA").is_some() {
        wgpu::TextureFormat::Bgra8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8UnormSrgb
    };
    let kernel = MaskConeKernel::new().expect("GPU");
    if std::env::var("AD_HW").as_deref() == Ok("0") {
        kernel.set_hardware_rays(false);
    }
    let mut r = Renderer::new(
        kernel,
        fmt,
        if std::env::var_os("AD_GROW").is_some() {
            (w / 2, h / 2)
        } else {
            (w, h)
        },
        &scene,
        origin,
        &lights,
        RendererOptions {
            sky: sky.then_some(Vec3::new(0.6, 0.75, 1.0) * 0.3),
            deforming: deforming.clone(),
            bounce: std::env::var("AD_BOUNCE").as_deref() != Ok("0"),
            anti_aliasing: std::env::var("AD_AA").as_deref() != Ok("0"),
            scale: std::env::var("AD_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            cache_options: continuum_gpu::CacheOptions {
                sky_lag: std::env::var("AD_LAG").as_deref() == Ok("1"),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let view = r
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
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let all = animations.together();
    let duration = all
        .iter()
        .map(|&k| animations.list[k].duration)
        .fold(1e-3, f32::max);
    let pose = |r: &mut Renderer, f: usize| {
        let step: f32 = std::env::var("AD_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        for p in animations.deform(&all, (f as f32 * step / 60.0) % duration) {
            if std::env::var_os("AD_DEBUG").is_some() && f.is_multiple_of(20) {
                let b = p.triangles.iter().flatten().fold(
                    (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                    |(lo, hi), &v| (lo.min(v), hi.max(v)),
                );
                eprintln!("pose {f}: bounds {b:?}");
            }
            r.set_vertices(p.geometry, p.triangles, p.normals);
        }
    };
    println!("{} deforming geometries, sky {sky}", deforming.len());
    let mut times = Vec::new();
    for f in 0..frames {
        pose(&mut r, f);
        let t = Instant::now();
        // `AD_RESIZE=1`: resized (to the same size) after frame 10, as a
        // window is.
        if f == 10 && std::env::var_os("AD_RESIZE").is_some() {
            r.resize(w, h);
        }
        // `AD_GROW=1`: built at half the size and resized to it after
        // frame 10, as a window on a Retina screen is.
        if f == 10 && std::env::var_os("AD_GROW").is_some() {
            r.resize(w, h);
        }
        // `AD_VIEWER=1`: the viewer's settings every frame.
        if std::env::var_os("AD_VIEWER").is_some() {
            r.set_exposure(1.0);
            r.set_tone_map(continuum_gpu::ToneMap::PbrNeutral);
            r.set_bloom(Some(continuum_gpu::Bloom::default()));
        }
        r.render(camera, &view);
        // `AD_NOWAIT=1`: frames pipelined, as a window renders them;
        // `AD_PACE=<ms>` between them.
        if std::env::var_os("AD_NOWAIT").is_none() || f + 1 == frames {
            r.wait();
        }
        if let Some(ms) = std::env::var("AD_PACE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            std::thread::sleep(std::time::Duration::from_millis(ms));
        }
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        if f % 30 == 29
            && let Some(s) = r.stats()
        {
            println!(
                "frame {f}: lag {}, {:.1} ms; baked {} pages {} texels, re-baked {} pages {} texels, sky re-baked {} charged {}, ledger {} pages, requests {}, budget {}",
                r.deforming_lag(),
                times[f],
                s.pages_baked,
                s.texels_baked,
                s.rebaked_pages,
                s.rebaked_texels,
                s.sky_rebaked,
                s.sky_charged,
                s.ledger_pages,
                s.requests,
                s.budget
            );
        }
    }
    if let Some(out) = std::env::var_os("AD_OUT") {
        let colors = r.read_colors();
        let srgb = |x: f32| {
            let x = x.clamp(0.0, 1.0);
            let s = if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u8
        };
        let bytes: Vec<u8> = colors.iter().flat_map(|c| c.map(srgb)).collect();
        let file = std::fs::File::create(out).expect("the image");
        // At `AD_SCALE` k the frame is w / k wide.
        let k = ((w * h) as f64 / colors.len() as f64).sqrt().round() as u32;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w / k, h / k);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut w| w.write_image_data(&bytes))
            .expect("the PNG");
    }
    let mut late = times[frames / 2..].to_vec();
    late.sort_by(f64::total_cmp);
    println!(
        "frames: later half median {:.2} ms, p90 {:.2} ms",
        late[late.len() / 2],
        late[late.len() * 9 / 10]
    );
    let mut bake = std::collections::BTreeMap::<&str, f64>::new();
    let mut sums = [0.0f64; 16];
    for f in frames..frames + 10 {
        pose(&mut r, f);
        let p = r.render_profiled(camera, &view);
        for (label, d) in p.bake_passes {
            if !label.is_empty() {
                *bake.entry(label).or_insert(0.0) += d.as_secs_f64() * 100.0;
            }
        }
        for (i, d) in p.passes.iter().enumerate() {
            sums[i] += d.as_secs_f64() * 100.0;
        }
    }
    let b: Vec<String> = bake.iter().map(|(k, t)| format!("{k} {t:.2}")).collect();
    println!("bake passes (ms): {}", b.join(", "));
    let c: Vec<String> = sums
        .iter()
        .enumerate()
        .filter(|(_, t)| **t > 0.05)
        .map(|(i, t)| format!("{i}: {t:.2}"))
        .collect();
    println!("camera passes (ms): {}", c.join(", "));
}
