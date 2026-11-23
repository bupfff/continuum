//! Frames for the README's newer animations (not a gate): `Renderer` with
//! its defaults as the viewer sets them, the displayed image (tone map,
//! bloom, auto-exposure) read back and saved as PNGs in
//! `out/media/<clip>/`; `scripts/make_gif.py` turns them into GIFs. Each
//! clip warms its first view (until settled, or `RC_WARM` frames), then
//! saves a frame after every `RC_STEP` rendered (2: the cache keeps up as
//! it does in the viewer at about twice the GIF's rate).
//!
//! - `sponza`: Sponza's courtyard under the viewer's outdoor HDRI (its sun
//!   and three more lights taken from the image), walking 6 m while
//!   turning 10°.
//! - `field`: the million-boulder field under its sun and sky (bounce light
//!   off, as the viewer has it), rising from eye height to 60 m over 150 m
//!   forward while tilting down. Not in the README: two frames a step leave
//!   new coarse ground pages darker until their sky bakes (a dark band in
//!   the middle distance, 2026-11-23).
//! - `helmet`: FlightHelmet in the viewer's studio HDRI on its ground, the
//!   camera orbiting 120°.
//!
//! `cargo run --release -p room3d --bin readme_clips -- sponza|field|helmet`

use std::path::{Path, PathBuf};

use continuum::glam::{DVec3, Mat3, Quat, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedScene, Light, MaskConeKernel, Renderer,
    RendererOptions, wgpu,
};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;

fn num(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The display texture, readable.
fn display(r: &Renderer) -> wgpu::Texture {
    r.kernel().device().create_texture(&wgpu::TextureDescriptor {
        label: Some("display"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Saves the displayed image as `dir/kkkk.png`.
fn save(r: &Renderer, texture: &wgpu::Texture, dir: &Path, k: usize) -> std::io::Result<()> {
    let device = r.kernel().device();
    let row = (W * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame"),
        size: (row * H) as u64,
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
                bytes_per_row: Some(row),
                rows_per_image: Some(H),
            },
        },
        texture.size(),
    );
    r.kernel().queue().submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU");
    let data = slice.get_mapped_range().expect("mapped");
    let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
    for y in 0..H as usize {
        for x in 0..W as usize {
            let i = y * row as usize + 4 * x;
            image.set(x, y, [data[i], data[i + 1], data[i + 2]]);
        }
    }
    image.save(&dir.join(format!("{k:04}.png")))
}

/// Warms `start`, then renders `frames` saved frames along `path` (0 to 1).
fn record(
    mut r: Renderer,
    dir: &Path,
    frames: usize,
    path: impl Fn(f32) -> Camera,
) -> std::io::Result<()> {
    let texture = display(&r);
    let view = texture.create_view(&Default::default());
    let warm = num("RC_WARM", 2400);
    let step = num("RC_STEP", 2).max(1);
    for f in 0..warm {
        r.render(path(0.0), &view);
        r.wait();
        if f > 32 && r.is_settled(32) {
            println!("warm after {f} frames");
            break;
        }
    }
    for k in 0..frames {
        let cam = path(k as f32 / (frames - 1).max(1) as f32);
        for _ in 0..step {
            r.render(cam, &view);
        }
        r.wait();
        save(&r, &texture, dir, k)?;
    }
    Ok(())
}

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("out/assets")
}

fn hdri(name: &str) -> continuum_gpu::Environment {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../apps/viewer/assets/hdri")
        .join(name);
    continuum_gltf::load_environment(&path).expect("the viewer's HDRI")
}

fn sponza(kernel: MaskConeKernel, dir: &Path) -> std::io::Result<()> {
    let imported = continuum_gltf::load(&assets().join("khronos/Models/Sponza/glTF/Sponza.gltf"))
        .expect("Sponza (DESIGN §3.8 lists the assets)");
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let r = Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (W, H),
        &imported.scene,
        origin,
        &[],
        RendererOptions {
            environment: Some(hdri("kloofendal_48d_partly_cloudy_puresky_1k.hdr")),
            cache_options: CacheOptions {
                sky_lag: true,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (eye0, target0) = (Vec3::new(5.0, 0.3, -3.4), Vec3::new(-10.0, 0.0, -2.0));
    let along = target0 - eye0;
    let flat = Vec3::new(along.x, along.y, 0.0).normalize();
    record(r, dir, num("RC_FRAMES", 160), |s| {
        let eye = eye0 + flat * 6.0 * s;
        Camera {
            eye,
            target: eye + Mat3::from_rotation_z((10.0 * s).to_radians()) * along,
            vertical_fov_degrees: 60.0,
        }
    })
}

fn field(kernel: MaskConeKernel, dir: &Path) -> std::io::Result<()> {
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let r = Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (W, H),
        &field.scene,
        origin,
        &[Light::white(sun, field.radiance)],
        RendererOptions {
            sky: Some(field.sky()),
            bounce: false,
            cache_options: CacheOptions {
                sky_lag: true,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let right = flat.cross(Vec3::Z).normalize();
    record(r, dir, num("RC_FRAMES", 200), |s| {
        let e = s * s * (3.0 - 2.0 * s);
        let eye = ground.eye + flat * 150.0 * e + Vec3::Z * 58.0 * e;
        let d = Quat::from_axis_angle(right, -(28.0 * e).to_radians()) * flat;
        Camera {
            eye,
            target: eye + d,
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    })
}

fn helmet(kernel: MaskConeKernel, dir: &Path) -> std::io::Result<()> {
    let imported = continuum_gltf::load(&assets().join("khronos/Models/FlightHelmet/glTF/FlightHelmet.gltf"))
        .expect("FlightHelmet");
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    let mut scene: InstancedScene = imported.scene;
    // The viewer's ground: 20 radii across, just below the object.
    let h = (10.0 * radius) as f32;
    let c = [(-h, -h), (h, -h), (h, h), (-h, h)].map(|(x, y)| Vec3::new(x, y, 0.0));
    scene
        .geometries
        .push(Surface::from_triangles(vec![[c[0], c[1], c[2]], [c[0], c[2], c[3]]]));
    scene.geometry_textures.resize(scene.geometries.len(), None);
    scene.instances.push(Instance {
        geometry: scene.geometries.len() - 1,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position: DVec3::new(origin.x, origin.y, lo.z - 1e-3 * radius),
        albedo: Vec3::splat(0.5),
    });
    let r = Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (W, H),
        &scene,
        origin,
        &[],
        RendererOptions {
            environment: Some(hdri("studio_small_09_1k.hdr")),
            ..Default::default()
        },
    );
    let dist = (radius * 2.0) as f32;
    let target = Vec3::new(0.0, 0.0, -0.1 * radius as f32);
    record(r, dir, num("RC_FRAMES", 160), |s| {
        let (az, el) = ((215.0 + 120.0 * s).to_radians(), 18f32.to_radians());
        Camera {
            eye: target + Vec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin()) * dist,
            target,
            vertical_fov_degrees: 40.0,
        }
    })
}

fn main() -> std::io::Result<()> {
    let clip = std::env::args().nth(1).unwrap_or_else(|| "sponza".into());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("out/media")
        .join(&clip);
    std::fs::create_dir_all(&dir)?;
    let kernel = MaskConeKernel::new().expect("a GPU");
    match clip.as_str() {
        "sponza" => sponza(kernel, &dir)?,
        "field" => field(kernel, &dir)?,
        "helmet" => helmet(kernel, &dir)?,
        other => panic!("unknown clip {other}: sponza, field or helmet"),
    }
    println!("frames in {}", dir.display());
    Ok(())
}
