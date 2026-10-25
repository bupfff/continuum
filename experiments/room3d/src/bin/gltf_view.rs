//! Development tool (not a gate): imports a glTF file, prints what it found,
//! and renders it lit by a sun and a sky, per pixel and through the view
//! cache, writing both images to `out/`. The camera is the file's first, or
//! one framed from the scene's bounds.
//! `cargo run --release -p room3d --bin gltf_view -- <file.gltf|glb> [W H]`

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, RectLight, wgpu,
};
use flatland::img::Image;
use room3d::cycles::relative_rmse;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(args.get(1).expect("a glTF file"));
    let w: u32 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(960);
    let h: u32 = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(540);
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let start = Instant::now();
    let imported = continuum_gltf::load(path).map_err(std::io::Error::other)?;
    println!(
        "imported in {:.2} s: {:?}",
        start.elapsed().as_secs_f64(),
        imported.stats
    );
    // Own-object sky masks at the cache's default spacing: grids per
    // geometry, and whether any chart is too large for them.
    let spacing = CacheOptions::default().self_spacing;
    for (g, s) in imported.scene.geometries.iter().enumerate() {
        let cells: u64 = s
            .charts
            .iter()
            .map(|c| ((c.x / spacing).ceil().max(1.0) * (c.y / spacing).ceil().max(1.0)) as u64)
            .sum();
        let largest = s
            .charts
            .iter()
            .map(|c| c.max_element())
            .fold(0.0f32, f32::max);
        println!(
            "  geometry {g}: {} triangles, {} charts, largest {largest:.2}, self texels {cells}{}",
            s.triangles.len(),
            s.charts.len(),
            if largest / spacing > 1024.0 {
                " (too large: walks itself)"
            } else {
                ""
            }
        );
    }
    let (lo, hi) = imported.bounds;
    let center = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    println!("bounds {lo:?} .. {hi:?}");
    let origin = center;
    let camera = match imported.camera {
        Some(c) => Camera {
            eye: (c.eye - origin).as_vec3(),
            target: (c.target - origin).as_vec3(),
            vertical_fov_degrees: c.vertical_fov_degrees,
        },
        None => {
            let (az, el) = (225f64.to_radians(), 25f64.to_radians());
            let dir = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
            Camera {
                eye: (dir * radius * 2.2).as_vec3(),
                target: Vec3::ZERO,
                vertical_fov_degrees: 40.0,
            }
        }
    };
    // A sun 30° up from azimuth 200°, 0.5° across, far beyond the scene.
    let (az, el) = (200f64.to_radians(), 30f64.to_radians());
    let toward = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
    let distance = radius * 50.0;
    let size = 2.0 * distance * 0.25f64.to_radians().tan();
    let wdir = (-toward).as_vec3();
    let a = Vec3::Z.cross(wdir).normalize() * size as f32;
    let b = wdir.cross(a).normalize() * size as f32;
    let middle = toward * distance;
    let sun = RectLight {
        corner: (middle - (a + b).as_dvec3() * 0.5).as_vec3(),
        edge_u: a,
        edge_v: b,
        one_sided: true,
    };
    // Irradiance π on a surface facing the sun: white at albedo 1.
    let solid = (size / distance) * (size / distance);
    let radiance = (std::f64::consts::PI / solid) as f32;
    let lights = [Light {
        rect: sun,
        radiance: Vec3::new(1.0, 0.92, 0.8) * radiance,
    }];
    let sky = Vec3::new(0.6, 0.75, 1.0) * 0.3;

    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_sky(sky);
    let start = Instant::now();
    let (per_pixel, t, _) =
        kernel.render_instanced_lights(camera, w, h, &lights, &imported.scene, origin, 1);
    println!(
        "per pixel: {:.2} s (GPU {:.0} ms)",
        start.elapsed().as_secs_f64(),
        t.as_secs_f64() * 1e3
    );
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
    let start = Instant::now();
    let mut r = InstancedRenderer::with_cache_lights(
        &kernel,
        format,
        w,
        h,
        1,
        &lights,
        &imported.scene,
        origin,
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
            || frames > 20_000
        {
            break;
        }
    }
    r.finish();
    let cached = r.read_colors();
    let t0 = Instant::now();
    for _ in 0..30 {
        r.frame(camera, &target);
    }
    r.finish();
    println!(
        "cache: warm after {frames} frames ({:.1} s), warm frame {:.2} ms, cached vs per-pixel {:.2}%",
        start.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() * 1e3 / 30.0,
        100.0 * relative_rmse(&cached, &per_pixel)
    );
    let name = path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .replace(' ', "_");
    for (tag, img) in [("perpixel", &per_pixel), ("cached", &cached)] {
        let mut image = Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in img.iter().enumerate() {
            image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
        }
        image.save(&out_dir.join(format!("gltf_{name}_{tag}.png")))?;
    }
    Ok(())
}
