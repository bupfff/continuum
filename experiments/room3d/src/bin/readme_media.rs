//! Frames for the README's animations (not a gate): real frames as the
//! renderer draws them, one per time step, read back and saved as PNGs in
//! `out/media/<clip>/`. `scripts/make_gif.py` turns them into GIFs.
//!
//! - `lamps`: M10 gate 21's night scene (five lamps, no sky, bounce light
//!   off: moving objects take neither yet), a boulder moving among them
//!   while the brightest lamp cycles through colors. Moves re-bake exactly
//!   in the frame (M9 gate 1); re-coloring bakes nothing.
//! - `field`: the million-boulder field under sun and sky, flying 120 m
//!   forward and rising to 20 m (the start view warmed first, then one
//!   frame per step, every second one kept).
//!
//! `cargo run --release -p room3d --bin readme_media -- lamps|field`

use std::path::{Path, PathBuf};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::field::Field;
use room3d::textured::Textured;

const W: u32 = 960;
const H: u32 = 540;

fn target(kernel: &MaskConeKernel) -> wgpu::TextureView {
    kernel
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Renders until 32 frames requested and baked nothing.
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..50_000 {
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
    panic!("never settled");
}

fn save(r: &mut InstancedRenderer, dir: &Path, k: usize) -> std::io::Result<()> {
    r.finish();
    let px = r.read_colors();
    let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in px.iter().enumerate() {
        image.set(
            i % W as usize,
            i / W as usize,
            room3d::to_rgb(Vec3::from_array(*p)),
        );
    }
    image.save(&dir.join(format!("{k:04}.png")))
}

/// A hue on the color wheel at `power` (the sum of channels).
fn hue(h: f32, power: f32) -> Vec3 {
    let c = |o: f32| 0.5 + 0.5 * (std::f32::consts::TAU * (h + o)).cos();
    let v = Vec3::new(c(0.0), c(-1.0 / 3.0), c(-2.0 / 3.0)) + Vec3::splat(0.15);
    v * (power / (v.x + v.y + v.z))
}

fn lamps(kernel: &MaskConeKernel, dir: &Path) -> std::io::Result<()> {
    kernel.set_sky(Vec3::ZERO);
    let t = Textured::new().with_lamps();
    // The boulder that moves: the one nearest the warm lamp.
    let lamp = Vec3::new(-1.6, 7.2, 0.0).as_dvec3();
    let mover = (1..t.scene.instances.len())
        .filter(|&i| t.scene.instances[i].geometry == 1)
        .min_by(|&a, &b| {
            let d = |i: usize| (t.scene.instances[i].position - lamp).length();
            d(a).total_cmp(&d(b))
        })
        .expect("a boulder");
    let rest = t.scene.instances[mover];
    let view = target(kernel);
    let mut r = InstancedRenderer::with_movers(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        &t.lights,
        &t.scene,
        t.origin,
        Some(CacheOptions::default()),
        &[mover],
        &[],
    );
    settle(&mut r, t.camera, &view);
    let frames = 150;
    let power = {
        let w = Textured::lamp_warm();
        w.x + w.y + w.z
    };
    for k in 0..frames {
        let s = k as f32 / frames as f32;
        let a = std::f32::consts::TAU * s;
        let position = DVec3::new(-1.0 + 1.8 * a.sin() as f64, 6.6, rest.position.z)
            + DVec3::new(0.0, 0.7 * (2.0 * a).sin() as f64, 0.0);
        r.set_instance_transform(mover, Mat3::from_rotation_z(a) * rest.rotation, position);
        r.set_light_radiance(3, hue(0.08 + s, power));
        r.frame(t.camera, &view);
        save(&mut r, dir, k)?;
    }
    Ok(())
}

fn field(kernel: &MaskConeKernel, dir: &Path) -> std::io::Result<()> {
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let lights = [Light::white(sun, field.radiance)];
    kernel.set_sky(field.sky());
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let camera = |f: usize, n: usize| {
        let t = f as f32 / (n - 1) as f32;
        let (ahead, height, down) = (120.0 * t, 1.7 + 18.3 * t, (14.0 * t).to_radians());
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * down.cos() - Vec3::Z * down.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let view = target(kernel);
    let mut r = InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        &lights,
        &field.scene,
        origin,
        CacheOptions::default(),
    );
    let n = 300;
    settle(&mut r, camera(0, n), &view);
    for f in 0..n {
        r.frame(camera(f, n), &view);
        if f % 2 == 0 {
            save(&mut r, dir, f / 2)?;
        }
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let clip = std::env::args().nth(1).unwrap_or_else(|| "lamps".into());
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("out/media")
        .join(&clip);
    std::fs::create_dir_all(&dir)?;
    let kernel = MaskConeKernel::new().expect("a GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    match clip.as_str() {
        "lamps" => lamps(&kernel, &dir)?,
        "field" => field(&kernel, &dir)?,
        other => panic!("unknown clip {other}: lamps or field"),
    }
    println!("frames in {}", dir.display());
    Ok(())
}
