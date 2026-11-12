//! GPU tests of the library on whatever adapter wgpu picks (CI chooses one
//! with `WGPU_BACKEND` and `WGPU_ADAPTER_NAME`: DX12 on WARP, Vulkan on
//! lavapipe). They check the kernel against the CPU reference, and the
//! renderer against an image rendered on Apple Silicon (Metal), against its
//! own cache, against a glTF round trip, and with hardware rays against
//! compute rays.
//!
//! Small on purpose: software adapters run them in seconds.
//!
//! `CONTINUUM_BLESS=1 cargo test --release -p continuum-gltf --test gpu`
//! rewrites the golden image (only on the Mac, and only on purpose).

use std::path::PathBuf;

use continuum::CoverageMask;
use continuum::cone3::{PyramidCone, box_mesh};
use continuum::glam::{DVec3, Mat3, Vec2, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedScene, Light, MaskConeKernel, RectLight, Renderer,
    RendererOptions, Spot, Texture, wgpu,
};

const WIDTH: u32 = 160;
const HEIGHT: u32 = 90;

fn kernel() -> MaskConeKernel {
    let kernel = MaskConeKernel::new().expect("a GPU adapter (set WGPU_BACKEND to choose one)");
    eprintln!("adapter: {}", kernel.adapter_name);
    kernel
}

/// A checkered 8×8 texture.
fn checker() -> Texture {
    let rgba = (0..64)
        .flat_map(|k| {
            if (k % 8 + k / 8) % 2 == 0 {
                [230, 220, 200, 255]
            } else {
                [90, 110, 140, 255]
            }
        })
        .collect();
    Texture {
        width: 8,
        height: 8,
        rgba,
    }
}

/// A textured ground, boxes and spheres placed `offset` from the world's
/// origin (rendered about `offset`, so large offsets test f64 placement).
fn scene(offset: DVec3) -> InstancedScene {
    let h = 3.0;
    let ground = [
        Vec3::new(-h, -h, 0.0),
        Vec3::new(h, -h, 0.0),
        Vec3::new(h, h, 0.0),
        Vec3::new(-h, h, 0.0),
    ];
    let uv = [
        Vec2::new(0.0, 0.0),
        Vec2::new(3.0, 0.0),
        Vec2::new(3.0, 3.0),
        Vec2::new(0.0, 3.0),
    ];
    let ground = Surface::from_mesh(
        vec![
            [ground[0], ground[1], ground[2]],
            [ground[0], ground[2], ground[3]],
        ],
        None,
        Some(vec![[uv[0], uv[1], uv[2]], [uv[0], uv[2], uv[3]]]),
    );
    let cube = Surface::cuboid(Vec3::splat(-0.25), Vec3::splat(0.25));
    let sphere = Surface::parametric(24, 12, |u, v| {
        let (theta, phi) = (u * std::f32::consts::TAU, (v - 0.5) * std::f32::consts::PI);
        Vec3::new(theta.cos() * phi.cos(), theta.sin() * phi.cos(), phi.sin()) * 0.3
    });
    let mut instances = vec![Instance {
        geometry: 0,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position: offset,
        albedo: Vec3::splat(0.8),
    }];
    for i in 0..4 {
        for j in 0..4 {
            let k = i * 4 + j;
            let at = DVec3::new(-1.8 + 1.2 * i as f64, -1.8 + 1.2 * j as f64, 0.0);
            let cube = k % 2 == 0;
            instances.push(Instance {
                geometry: if cube { 1 } else { 2 },
                rotation: Mat3::from_rotation_z(0.4 * k as f32) * Mat3::from_rotation_x(0.2),
                scale: 0.8 + 0.1 * (k % 3) as f32,
                position: offset + at + DVec3::Z * if cube { 0.2 } else { 0.25 },
                albedo: Vec3::new(0.4 + 0.1 * (k % 4) as f32, 0.5, 0.7 - 0.1 * (k % 3) as f32),
            });
        }
    }
    InstancedScene {
        geometries: vec![ground, cube, sphere],
        instances,
        textures: vec![checker()],
        geometry_textures: vec![Some(0), None, None],
        geometry_materials: Vec::new(),
        geometry_emission: Vec::new(),
        geometry_emissive_textures: Vec::new(),
        geometry_blend: Vec::new(),
        geometry_normal_maps: Vec::new(),
        geometry_metallic_roughness: Vec::new(),
        geometry_alpha_cutoff: Vec::new(),
        geometry_colors: Vec::new(),
        texture_wrap: Vec::new(),
        geometry_levels: Vec::new(),
        level_source: None,
    }
}

/// A sun, a point light and a spot.
fn lights() -> Vec<Light> {
    vec![
        Light::sun(
            Vec3::ZERO,
            DVec3::new(0.5, -0.4, 0.75),
            200.0,
            Vec3::new(2.4, 2.2, 1.9),
        ),
        Light::point(Vec3::new(-1.2, 0.6, 1.2), 0.1, Vec3::new(1.0, 0.6, 0.3)),
        Light::spot(
            Vec3::new(1.2, -1.2, 2.0),
            0.05,
            Vec3::new(1.0, 1.5, 4.5),
            Spot {
                axis: Vec3::new(-0.3, 0.3, -1.0),
                inner: 0.3,
                outer: 0.5,
            },
        ),
    ]
}

const SKY: Vec3 = Vec3::new(0.15, 0.2, 0.3);

fn camera() -> Camera {
    Camera {
        eye: Vec3::new(0.5, -5.0, 2.6),
        target: Vec3::new(0.0, 0.0, 0.2),
        vertical_fov_degrees: 45.0,
    }
}

fn target(renderer: &Renderer) -> wgpu::TextureView {
    renderer
        .kernel()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
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

fn renderer(
    kernel: MaskConeKernel,
    scene: &InstancedScene,
    origin: DVec3,
    options: RendererOptions,
) -> Renderer {
    Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (WIDTH, HEIGHT),
        scene,
        origin,
        &lights(),
        options,
    )
}

/// One frame with a mask cone per pixel per light.
fn per_pixel(kernel: MaskConeKernel, scene: &InstancedScene, origin: DVec3, sky: bool) -> Image {
    let mut r = renderer(
        kernel,
        scene,
        origin,
        RendererOptions {
            cache: false,
            sky: sky.then_some(SKY),
            ..Default::default()
        },
    );
    let view = target(&r);
    r.render(camera(), &view);
    r.read_colors()
}

type Image = Vec<[f32; 3]>;

/// Writes `image` as an sRGB PNG to `$CONTINUUM_TEST_OUT/name.png`, if set
/// (for looking at failures).
fn dump(name: &str, image: &Image) {
    let Some(dir) = std::env::var_os("CONTINUUM_TEST_OUT") else {
        return;
    };
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    let pixels: Vec<u8> = image.iter().flat_map(|p| p.map(encode)).collect();
    let path = PathBuf::from(dir).join(format!("{name}.png"));
    let file = std::fs::File::create(&path).unwrap();
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), WIDTH, HEIGHT);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
}

/// How far `a` is from `b`.
#[derive(Debug)]
struct Difference {
    /// Root mean square difference over the mean of `b` (all channels).
    relative_rmse: f64,
    /// Pixels differing by more than 2% of `b`'s mean in some channel.
    far_pixels: f64,
}

fn difference(a: &Image, b: &Image) -> Difference {
    assert_eq!(a.len(), b.len());
    let n = (3 * b.len()) as f64;
    let mean = b.iter().flatten().map(|&x| x as f64).sum::<f64>() / n;
    let mut err = 0.0;
    let mut far = 0;
    for (x, y) in a.iter().zip(b) {
        let mut worst = 0.0f64;
        for c in 0..3 {
            let d = (x[c] - y[c]) as f64;
            err += d * d;
            worst = worst.max(d.abs());
        }
        far += (worst > 0.02 * mean) as usize;
    }
    Difference {
        relative_rmse: (err / n).sqrt() / mean,
        far_pixels: far as f64 / b.len() as f64,
    }
}

fn assert_finite_and_lit(image: &Image) {
    assert!(
        image.iter().flatten().all(|c| c.is_finite() && *c >= 0.0),
        "non-finite or negative colors"
    );
    let mean = image.iter().flatten().sum::<f32>() / (3 * image.len()) as f32;
    assert!(mean > 0.05, "image is dark: mean {mean}");
}

/// The kernel's mask cones agree with the CPU reference (`continuum::cone3`)
/// to 1e-4 of the light's weight at every receiver, as in M2 gate 1.
#[test]
fn cones_match_the_cpu() {
    let meshes: Vec<Vec<[Vec3; 3]>> = vec![
        box_mesh(Vec3::new(0.30, 0.10, 0.55), Vec3::new(0.70, 0.35, 0.58)),
        box_mesh(Vec3::new(0.62, 0.40, 0.30), Vec3::new(0.85, 0.60, 0.33)),
        box_mesh(Vec3::new(0.15, 0.45, 0.20), Vec3::new(0.25, 0.55, 0.30)),
        box_mesh(Vec3::new(0.40, 0.65, 0.45), Vec3::new(0.48, 0.90, 0.62)),
    ];
    let light = RectLight {
        corner: Vec3::new(0.4, 0.4, 1.0),
        edge_u: Vec3::new(0.2, 0.0, 0.0),
        edge_v: Vec3::new(0.0, 0.2, 0.0),
        one_sided: false,
    };
    let size = 48;
    let receivers: Vec<(Vec3, Vec3)> = (0..size * size)
        .map(|k| {
            let (i, j) = (k % size, k / size);
            let p = Vec3::new(
                (i as f32 + 0.5) / size as f32,
                (j as f32 + 0.5) / size as f32,
                0.0,
            );
            (p, Vec3::Z)
        })
        .collect();
    let (gpu, _) = kernel().irradiance(&receivers, light, &meshes, 1);
    let mut worst = 0.0f32;
    for (&(p, _), g) in receivers.iter().zip(&gpu) {
        let cone =
            PyramidCone::toward_rect_light(p, light.corner, light.edge_u, light.edge_v, Vec3::Z)
                .expect("receivers are below the light");
        let mut mask = CoverageMask::EMPTY;
        for mesh in &meshes {
            let tris: Vec<_> = mesh.iter().filter(|t| cone.may_touch(t)).collect();
            cone.add_mesh(&mut mask, tris);
        }
        let cpu = cone.visible(&mask);
        worst = worst.max((cpu - g).abs() / cone.total_weight());
    }
    eprintln!("worst difference: {worst:.2e} of the light's weight");
    assert!(worst <= 1e-4, "GPU differs from the CPU by {worst:.2e}");
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/per_pixel.f32")
}

/// The per-pixel render (three lights and a sky, 10 km from the world's
/// origin) matches the image rendered on Apple Silicon: across GPUs and
/// shader compilers, only rounding and a few edge pixels may differ.
#[test]
fn per_pixel_matches_the_golden_image() {
    let offset = DVec3::new(10_000.0, -7_000.0, 0.0);
    let image = per_pixel(kernel(), &scene(offset), offset, true);
    dump("per_pixel", &image);
    assert_finite_and_lit(&image);
    let path = golden_path();
    if std::env::var_os("CONTINUUM_BLESS").is_some() {
        let bytes: Vec<u8> = image
            .iter()
            .flatten()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        eprintln!("wrote {}", path.display());
        return;
    }
    let bytes = std::fs::read(&path).expect("the golden image (CONTINUUM_BLESS=1 writes it)");
    let golden: Image = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>()
        .as_chunks::<3>()
        .0
        .to_vec();
    let d = difference(&image, &golden);
    eprintln!("against the golden image: {d:?}");
    assert!(
        d.relative_rmse <= 0.02 && d.far_pixels <= 0.005,
        "differs from the golden image: {d:?}"
    );
}

/// Through the view cache, once the view has settled, the image is close to
/// per-pixel cones. The cache interpolates texels two pixels wide, so shadow
/// edges are softer: 10.0% relative RMSE on Apple M4 at this small size,
/// where the black background also lowers the mean. A regression check, not
/// a quality claim.
#[test]
fn cache_settles_near_per_pixel() {
    let s = scene(DVec3::ZERO);
    let reference = per_pixel(kernel(), &s, DVec3::ZERO, false);
    let mut r = renderer(
        kernel(),
        &s,
        DVec3::ZERO,
        RendererOptions {
            cache_options: CacheOptions {
                // Bake a fixed amount each frame, however slow the adapter.
                frame_target_ms: 0.0,
                // A small pool: software adapters have little memory to spare.
                pages: 1 << 15,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let view = target(&r);
    let mut frames = 0;
    while frames < 2_000 && !r.is_settled(8) {
        r.render(camera(), &view);
        frames += 1;
    }
    assert!(r.is_settled(8), "not settled after {frames} frames");
    let image = r.read_colors();
    dump("cache_per_pixel", &reference);
    dump("cache_settled", &image);
    assert_finite_and_lit(&image);
    let d = difference(&image, &reference);
    eprintln!("settled after {frames} frames; against per-pixel cones: {d:?}");
    assert!(d.relative_rmse <= 0.13, "cache far from per-pixel: {d:?}");

    // Re-coloring a light re-weights the cache and bakes nothing.
    r.set_light_radiance(1, Vec3::new(0.2, 0.9, 0.4) * 40.0);
    for _ in 0..4 {
        r.render(camera(), &view);
    }
    r.wait();
    assert!(r.is_settled(2), "re-coloring a light made the cache bake");
}

/// Writing the scene as `.glb` and loading it back renders the same image,
/// up to f32 rounding of the transforms (M6 gate 1).
#[test]
fn gltf_round_trip_renders_the_same() {
    let s = scene(DVec3::ZERO);
    let path = std::env::temp_dir().join(format!("continuum-test-{}.glb", std::process::id()));
    continuum_gltf::save(&s, &path).unwrap();
    let imported = continuum_gltf::load(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(imported.scene.instances.len(), s.instances.len());
    let original = per_pixel(kernel(), &s, DVec3::ZERO, false);
    let loaded = per_pixel(kernel(), &imported.scene, DVec3::ZERO, false);
    let d = difference(&loaded, &original);
    eprintln!("round trip against the original: {d:?}");
    assert!(
        d.relative_rmse <= 0.01 && d.far_pixels <= 0.002,
        "round trip differs: {d:?}"
    );
}

/// Metallic and roughness survive a glTF round trip, per instance; a
/// Lambertian geometry stays Lambertian.
#[test]
fn gltf_round_trip_keeps_materials() {
    let mut s = scene(DVec3::ZERO);
    s.geometry_materials = vec![None; s.geometries.len()];
    s.geometry_materials[1] = Some(continuum_gpu::Material {
        metallic: 1.0,
        roughness: 0.3,
    });
    s.geometry_emission = vec![Vec3::ZERO; s.geometries.len()];
    s.geometry_emission[2] = Vec3::new(4.0, 2.0, 0.5);
    s.geometry_normal_maps = vec![Some(0), None, None];
    s.geometry_materials[0] = Some(continuum_gpu::Material {
        metallic: 0.0,
        roughness: 0.7,
    });
    s.geometry_metallic_roughness = vec![Some(0), None, None];
    s.geometry_alpha_cutoff = vec![Some(0.25), None, None];
    s.geometry_emission[0] = Vec3::new(0.5, 0.25, 0.125);
    s.geometry_emissive_textures = vec![Some(0), None, None];
    s.geometry_blend = vec![None, Some(0.4), None];
    s.texture_wrap = vec![[continuum_gpu::Wrap::Clamp, continuum_gpu::Wrap::Mirror]];
    let path = std::env::temp_dir().join(format!("continuum-mat-{}.glb", std::process::id()));
    continuum_gltf::save(&s, &path).unwrap();
    let imported = continuum_gltf::load(&path).unwrap().scene;
    std::fs::remove_file(&path).ok();
    // Instances come back in another order: matched by position.
    let of = |scene: &InstancedScene, g: usize| scene.geometry_materials.get(g).copied().flatten();
    let glow = |scene: &InstancedScene, g: usize| {
        scene
            .geometry_emission
            .get(g)
            .copied()
            .unwrap_or(Vec3::ZERO)
    };
    for a in &s.instances {
        let b = imported
            .instances
            .iter()
            .min_by(|x, y| {
                let d = |i: &Instance| (i.position - a.position).length();
                d(x).total_cmp(&d(y))
            })
            .unwrap();
        assert_eq!(of(&s, a.geometry), of(&imported, b.geometry));
        let (x, y) = (glow(&s, a.geometry), glow(&imported, b.geometry));
        assert!((x - y).abs().max_element() < 1e-6, "{x} against {y}");
        let bumps = |scene: &InstancedScene, g: usize| {
            scene
                .geometry_normal_maps
                .get(g)
                .copied()
                .flatten()
                .is_some()
        };
        assert_eq!(bumps(&s, a.geometry), bumps(&imported, b.geometry));
        let cut = |scene: &InstancedScene, g: usize| {
            scene.geometry_alpha_cutoff.get(g).copied().flatten()
        };
        assert_eq!(cut(&s, a.geometry), cut(&imported, b.geometry));
        let mr = |scene: &InstancedScene, g: usize| {
            scene
                .geometry_metallic_roughness
                .get(g)
                .copied()
                .flatten()
                .is_some()
        };
        assert_eq!(mr(&s, a.geometry), mr(&imported, b.geometry));
        let emissive = |scene: &InstancedScene, g: usize| {
            scene
                .geometry_emissive_textures
                .get(g)
                .copied()
                .flatten()
                .is_some()
        };
        assert_eq!(emissive(&s, a.geometry), emissive(&imported, b.geometry));
        let blend =
            |scene: &InstancedScene, g: usize| scene.geometry_blend.get(g).copied().flatten();
        assert_eq!(blend(&s, a.geometry), blend(&imported, b.geometry));
        let wrap = |scene: &InstancedScene, g: usize| {
            scene.geometry_textures.get(g).copied().flatten().map(|t| {
                scene
                    .texture_wrap
                    .get(t as usize)
                    .copied()
                    .unwrap_or_default()
            })
        };
        assert_eq!(wrap(&s, a.geometry), wrap(&imported, b.geometry));
    }
}

/// The cache's settled image of `scene`.
fn settled(scene: &InstancedScene) -> Image {
    let mut r = renderer(
        kernel(),
        scene,
        DVec3::ZERO,
        RendererOptions {
            cache_options: CacheOptions {
                frame_target_ms: 0.0,
                pages: 1 << 15,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let view = target(&r);
    let mut frames = 0;
    while frames < 2_000 && !r.is_settled(32) {
        r.render(camera(), &view);
        frames += 1;
    }
    r.read_colors()
}

/// A metallic-roughness texture multiplies its material's factors: a white
/// one changes nothing; a darker green (rougher scale) changes the image.
#[test]
fn metallic_roughness_texture_multiplies_the_factors() {
    let mut s = scene(DVec3::ZERO);
    s.geometry_materials = vec![
        Some(continuum_gpu::Material {
            metallic: 0.0,
            roughness: 0.5,
        }),
        None,
        None,
    ];
    let plain = settled(&s);
    let with = |green: u8| {
        let mut t = s.clone();
        t.textures.push(Texture {
            width: 1,
            height: 1,
            rgba: vec![255, green, 255, 255],
        });
        t.geometry_metallic_roughness = vec![Some(t.textures.len() as u32 - 1), None, None];
        settled(&t)
    };
    let white = difference(&with(255), &plain);
    eprintln!("white texture against none: {white:?}");
    assert!(
        white.relative_rmse < 1e-5,
        "a white texture changed the image: {white:?}"
    );
    let smoother = difference(&with(40), &plain);
    eprintln!("smoother against none: {smoother:?}");
    assert!(
        smoother.relative_rmse > 1e-3,
        "the texture had no effect: {smoother:?}"
    );
}

/// Exposure scales what is drawn, before the tone curve: on a float target
/// (linear light), the target holds the colors times the factor, while
/// `read_colors` stays unscaled.
#[test]
fn exposure_scales_the_drawn_colors() {
    let s = scene(DVec3::ZERO);
    let format = wgpu::TextureFormat::Rgba32Float;
    let mut r = Renderer::new(
        kernel(),
        format,
        (WIDTH, HEIGHT),
        &s,
        DVec3::ZERO,
        &lights(),
        // A pool small enough for software adapters (as `settled`).
        RendererOptions {
            cache_options: CacheOptions {
                frame_target_ms: 0.0,
                pages: 1 << 15,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let device = r.kernel().device().clone();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("float target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let mut frames = 0;
    while frames < 2_000 && !r.is_settled(32) {
        r.render(camera(), &view);
        frames += 1;
    }
    let drawn = |r: &mut Renderer, exposure: f32| -> Vec<f32> {
        r.set_exposure(exposure);
        r.render(camera(), &view);
        let queue = r.kernel().queue().clone();
        let bytes = (WIDTH * HEIGHT * 16) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
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
                    bytes_per_row: Some(WIDTH * 16),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    let one = drawn(&mut r, 1.0);
    let colors = r.read_colors();
    let two = drawn(&mut r, 2.0);
    let unscaled = r.read_colors();
    let mut worst = (0.0f32, 0.0f32, 0.0f32);
    for (k, c) in colors.iter().enumerate() {
        for ch in 0..3 {
            let (a, b) = (one[4 * k + ch], two[4 * k + ch]);
            worst.0 = worst.0.max((a - c[ch]).abs());
            worst.1 = worst.1.max((b - 2.0 * a).abs());
            worst.2 = worst.2.max((unscaled[k][ch] - c[ch]).abs());
        }
    }
    eprintln!(
        "exposure: worst |drawn − colors| {}, |2× − twice| {}, read_colors change {}",
        worst.0, worst.1, worst.2
    );
    assert!(
        worst.0 < 1e-6 && worst.1 < 1e-6 && worst.2 < 1e-6,
        "{worst:?}"
    );
    assert!(
        colors.iter().flatten().any(|&c| c > 0.05),
        "the image is dark"
    );
}

/// sRGB byte `b` decoded to linear, as the kernel decodes vertex colors.
fn srgb(b: u8) -> f32 {
    let s = b as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Vertex colors multiply the base color (M10 gate 17): one color on every
/// corner of a geometry renders as the instances' albedo times it, with a
/// texture (the ground) and without (the cubes), bounce light included.
#[test]
fn vertex_colors_multiply_the_base_color() {
    let s = scene(DVec3::ZERO);
    let colors = [
        Vec3::new(srgb(250), srgb(180), srgb(90)),
        Vec3::new(srgb(60), srgb(200), srgb(255)),
    ];
    let mut colored = s.clone();
    colored.geometry_colors = vec![
        Some(vec![[colors[0]; 3]; s.geometries[0].triangles.len()]),
        Some(vec![[colors[1]; 3]; s.geometries[1].triangles.len()]),
        None,
    ];
    let mut tinted = s.clone();
    for i in &mut tinted.instances {
        if i.geometry < 2 {
            i.albedo *= colors[i.geometry];
        }
    }
    let (a, b) = (settled(&colored), settled(&tinted));
    let d = difference(&a, &b);
    eprintln!("vertex colors against tinted albedos: {d:?}");
    assert!(d.relative_rmse < 1e-4, "{d:?}");
    let plain = difference(&settled(&s), &b);
    assert!(
        plain.relative_rmse > 0.05,
        "the colors had no effect: {plain:?}"
    );
}

/// Vertex colors survive a glTF round trip (`COLOR_0`).
#[test]
fn gltf_round_trip_keeps_vertex_colors() {
    let mut s = scene(DVec3::ZERO);
    let sphere = &s.geometries[2];
    let colors: Vec<[Vec3; 3]> = sphere
        .triangles
        .iter()
        .map(|t| t.map(|p| (p / 0.6 + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE)))
        .collect();
    s.geometry_colors = vec![None, None, Some(colors)];
    let path = std::env::temp_dir().join(format!("continuum-colors-{}.glb", std::process::id()));
    continuum_gltf::save(&s, &path).unwrap();
    let imported = continuum_gltf::load(&path).unwrap().scene;
    std::fs::remove_file(&path).ok();
    let mut found = 0;
    for (g, geometry) in imported.geometries.iter().enumerate() {
        let Some(colors) = imported.geometry_colors.get(g).cloned().flatten() else {
            continue;
        };
        found += 1;
        assert_eq!(colors.len(), geometry.triangles.len());
        // glTF is Y-up: the saved sphere comes back in Continuum's axes, so
        // each corner's color is still the function of its position.
        for (t, c) in geometry.triangles.iter().zip(&colors) {
            for (p, c) in t.iter().zip(c) {
                let want = (*p / 0.6 + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE);
                assert!((*c - want).abs().max_element() < 1e-5, "{c} against {want}");
            }
        }
    }
    assert_eq!(found, 1, "one geometry with colors");
}

/// Emission adds to what a surface reflects, exactly, in the cached path.
#[test]
fn emission_adds_to_the_cached_image() {
    let s = scene(DVec3::ZERO);
    let mut glowing = s.clone();
    let glow = Vec3::new(0.5, 0.25, 0.0);
    glowing.geometry_emission = vec![Vec3::ZERO; s.geometries.len()];
    glowing.geometry_emission[0] = glow;
    let (plain, lit) = (settled(&s), settled(&glowing));
    let (mut on, mut off) = (0, 0);
    for (a, b) in plain.iter().zip(&lit) {
        let d = Vec3::from_array(*b) - Vec3::from_array(*a);
        if (d - glow).abs().max_element() < 1e-4 {
            on += 1;
        } else if d.abs().max_element() < 1e-4 {
            off += 1;
        }
    }
    let n = plain.len();
    eprintln!("emission: {on} pixels +glow, {off} unchanged, of {n}");
    assert!(on > n / 5, "too few pixels show the emission");
    assert!(
        on + off >= n * 99 / 100,
        "pixels changed by other than the emission"
    );
}

/// Camera rays through hardware ray queries and through the compute
/// hierarchy give the same image (where the adapter has ray queries).
#[test]
fn hardware_and_compute_rays_agree() {
    let k = kernel();
    if !k.has_hardware_rays() {
        eprintln!("no hardware ray queries on this adapter: skipped");
        return;
    }
    let s = scene(DVec3::ZERO);
    let hardware = per_pixel(k, &s, DVec3::ZERO, true);
    let k = kernel();
    k.set_hardware_rays(false);
    let compute = per_pixel(k, &s, DVec3::ZERO, true);
    let d = difference(&compute, &hardware);
    eprintln!("compute against hardware rays: {d:?}");
    assert!(
        d.relative_rmse <= 0.01 && d.far_pixels <= 0.002,
        "ray paths differ: {d:?}"
    );
}

/// Where mover `k` of `movers_at` is after `frame` frames of circling.
fn mover_place(s: &InstancedScene, instance: usize, frame: u32) -> (Mat3, DVec3) {
    let i = &s.instances[instance];
    let a = 0.15 * frame as f64 + instance as f64;
    let offset = DVec3::new(
        a.cos() - (instance as f64).cos(),
        a.sin() - (instance as f64).sin(),
        0.0,
    ) * 0.35;
    (
        Mat3::from_rotation_z(0.1 * frame as f32) * i.rotation,
        i.position + offset,
    )
}

/// Moving objects in the view cache (M9 gate 1): no page the cache may show
/// ever holds stale light, and once settled it shows what a cache built for
/// the final placements shows.
#[test]
fn movers_leave_no_stale_light() {
    let s = scene(DVec3::ZERO);
    let movable: Vec<usize> = vec![1, 2, 6, 11];
    let options = RendererOptions {
        cache_options: CacheOptions {
            frame_target_ms: 0.0,
            pages: 1 << 15,
            ..Default::default()
        },
        movable: movable.clone(),
        ..Default::default()
    };
    let mut r = renderer(kernel(), &s, DVec3::ZERO, options.clone());
    let view = target(&r);
    for _ in 0..30 {
        r.render(camera(), &view);
    }
    let mut rebaked = 0;
    for frame in 1..=40 {
        for &m in &movable {
            let (rotation, position) = mover_place(&s, m, frame);
            r.set_transform(m, rotation, position);
        }
        r.render(camera(), &view);
        if let Some(stats) = r.stats() {
            rebaked += stats.rebaked_pages;
        }
        if frame % 8 == 0 {
            let check = r.verify_cache();
            eprintln!("frame {frame}: {check:?}");
            assert!(check.pages > 0, "no pages to check");
            assert_eq!(check.differing, 0, "stale light in the cache: {check:?}");
        }
    }
    assert!(rebaked > 0, "nothing was re-baked");
    // Requests sample one pixel per 4×4 tile, a different one each frame
    // over 16 frames: 32 quiet frames see every pixel twice.
    let mut frames = 0;
    while frames < 2_000 && !r.is_settled(32) {
        r.render(camera(), &view);
        frames += 1;
    }
    assert!(r.is_settled(32), "not settled after {frames} frames");
    let moved = r.read_colors();

    // A cache built with the movers where they ended.
    let mut placed = s.clone();
    for &m in &movable {
        let (rotation, position) = mover_place(&s, m, 40);
        placed.instances[m].rotation = rotation;
        placed.instances[m].position = position;
    }
    let mut fresh = renderer(kernel(), &placed, DVec3::ZERO, options);
    let mut frames = 0;
    while frames < 2_000 && !fresh.is_settled(32) {
        fresh.render(camera(), &view_for(&fresh));
        frames += 1;
    }
    let expected = fresh.read_colors();
    dump("movers_moved", &moved);
    dump("movers_fresh", &expected);
    let d = difference(&moved, &expected);
    eprintln!("moved against fresh: {d:?}");
    assert!(
        d.relative_rmse == 0.0,
        "the moved cache differs from a fresh one: {d:?}"
    );
}

/// Moving objects under a sky (M9 gate 3): light channels stay exact, and
/// no page the cache may show has a sky further from a re-bake than its
/// ledger, nor than its ε threshold.
#[test]
fn movers_under_a_sky_stay_within_the_ledger() {
    let s = scene(DVec3::ZERO);
    let movable: Vec<usize> = vec![1, 2, 6, 11];
    let options = RendererOptions {
        cache_options: CacheOptions {
            frame_target_ms: 0.0,
            pages: 1 << 15,
            // Loose, so charges build up without invalidating: the bound is
            // what is tested.
            ledger_epsilon: 0.3,
            ..Default::default()
        },
        sky: Some(SKY),
        movable: movable.clone(),
        ..Default::default()
    };
    let mut r = renderer(kernel(), &s, DVec3::ZERO, options);
    let view = target(&r);
    for _ in 0..30 {
        r.render(camera(), &view);
    }
    let (mut charged, mut rebaked) = (0, 0);
    for frame in 1..=40 {
        for &m in &movable {
            let (rotation, position) = mover_place(&s, m, frame);
            r.set_transform(m, rotation, position);
        }
        r.render(camera(), &view);
        if let Some(stats) = r.stats() {
            charged += stats.sky_charged;
            rebaked += stats.sky_rebaked;
        }
        if frame % 8 == 0 {
            let check = r.verify_sky();
            eprintln!("frame {frame}: {check:?}");
            assert!(check.pages > 0, "no pages to check");
            assert_eq!(check.light_differing, 0, "stale light: {check:?}");
            assert_eq!(check.violations, 0, "sky beyond its ledger: {check:?}");
            assert_eq!(check.over_threshold, 0, "sky beyond ε: {check:?}");
        }
    }
    eprintln!("sky: {rebaked} pages re-baked, {charged} charged");
    assert!(charged > 0, "nothing was charged");
    // Once the movers stop, their pages' skies re-bake and it settles.
    let mut frames = 0;
    while frames < 2_000 && !r.is_settled(32) {
        r.render(camera(), &view);
        frames += 1;
    }
    assert!(r.is_settled(32), "not settled after {frames} frames");
    let check = r.verify_sky();
    eprintln!("settled: {check:?}");
    assert_eq!(check.violations + check.over_threshold, 0, "{check:?}");
    assert_eq!(
        r.stats().map_or(1, |s| s.fallback_pixels),
        0,
        "sky guesses left"
    );
}

fn view_for(r: &Renderer) -> wgpu::TextureView {
    target(r)
}

/// A moving light through the error ledger (M9 gate 2): as the sun drifts,
/// no page the cache may show is further from a re-bake than the bound the
/// ledger certified, nor than ε.
#[test]
fn a_drifting_sun_stays_within_the_ledger() {
    let s = scene(DVec3::ZERO);
    let sun = |frame: u32| {
        let a = 0.002 * frame as f64;
        let toward = DVec3::new(
            0.5 * a.cos() + 0.4 * a.sin(),
            -0.4 * a.cos() + 0.5 * a.sin(),
            0.75,
        );
        Light::sun(Vec3::ZERO, toward, 200.0, Vec3::new(2.4, 2.2, 1.9))
    };
    let mut r = renderer(
        kernel(),
        &s,
        DVec3::ZERO,
        RendererOptions {
            cache_options: CacheOptions {
                frame_target_ms: 0.0,
                pages: 1 << 15,
                ..Default::default()
            },
            moving_lights: vec![0],
            ..Default::default()
        },
    );
    let view = target(&r);
    for _ in 0..30 {
        r.render(camera(), &view);
    }
    let mut rebaked = 0;
    for frame in 1..=60 {
        r.set_light(0, sun(frame));
        r.render(camera(), &view);
        if let Some(stats) = r.stats() {
            rebaked += stats.ledger_texels;
            eprintln!(
                "stats frame {}: moved {}, checked {}, whole {}, mixed {}, pages {}, texels {}, invalid {}",
                stats.frame,
                stats.moved,
                stats.ledger_checked,
                stats.ledger_whole,
                stats.ledger_mixed,
                stats.ledger_pages,
                stats.ledger_texels,
                stats.ledger_invalidated
            );
        }
        if frame % 10 == 0 {
            let check = r.verify_ledger(0);
            eprintln!("frame {frame}: {check:?}");
            assert!(check.pages > 0, "no pages to check");
            assert_eq!(check.violations, 0, "a change beyond its bound: {check:?}");
            assert_eq!(check.over_epsilon, 0, "a change beyond ε: {check:?}");
        }
    }
    eprintln!("texels re-baked by the ledger: {rebaked}");
}

/// Bounce light in the view cache (M10 gate 1): it settles, adds light (most
/// where direct light is weakest), and re-coloring a light still bakes
/// nothing.
#[test]
fn bounce_light_settles_and_recolors_free() {
    let s = scene(DVec3::ZERO);
    let settle = |indirect: bool| {
        let mut r = renderer(
            kernel(),
            &s,
            DVec3::ZERO,
            RendererOptions {
                sky: Some(SKY),
                cache_options: CacheOptions {
                    frame_target_ms: 0.0,
                    pages: 1 << 15,
                    indirect,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let view = target(&r);
        let mut frames = 0;
        while frames < 4_000 && !r.is_settled(32) {
            r.render(camera(), &view);
            frames += 1;
        }
        assert!(r.is_settled(32), "not settled after {frames} frames");
        (r, view, frames)
    };
    let (mut plain, _, plain_frames) = settle(false);
    let (mut bounce, view, bounce_frames) = settle(true);
    let a = plain.read_colors();
    let b = bounce.read_colors();
    dump("bounce_off", &a);
    dump("bounce_on", &b);
    assert_finite_and_lit(&b);
    let mean = |img: &Image| img.iter().map(|p| p[0] + p[1] + p[2]).sum::<f32>() / img.len() as f32;
    let darker = b
        .iter()
        .zip(&a)
        .filter(|(x, y)| x[0] + x[1] + x[2] < y[0] + y[1] + y[2] - 1e-4)
        .count();
    eprintln!(
        "settled in {plain_frames} and {bounce_frames} frames; mean {} without bounce light, {} with; {darker} pixels darker",
        mean(&a),
        mean(&b)
    );
    assert!(mean(&b) > mean(&a) * 1.01, "bounce light added nothing");
    assert!(
        darker < b.len() / 100,
        "bounce light darkened {darker} pixels"
    );
    bounce.set_light_radiance(0, Vec3::new(1.0, 0.8, 0.6) * 3.0);
    for _ in 0..30 {
        bounce.render(camera(), &view);
    }
    bounce.wait();
    assert!(
        bounce.is_settled(20),
        "re-coloring a light made the cache bake"
    );
}

/// With the sky later (M5 gate 9), pages bake their lights first and their
/// sky alone afterwards. Settled, the cache shows the same image as baking
/// each page whole; with bounce light the option is ignored.
#[test]
fn sky_later_settles_to_the_same_image() {
    let s = scene(DVec3::ZERO);
    let settle = |sky_later: bool, indirect: bool| {
        let mut r = renderer(
            kernel(),
            &s,
            DVec3::ZERO,
            RendererOptions {
                sky: Some(SKY),
                cache_options: CacheOptions {
                    frame_target_ms: 0.0,
                    bake_budget: 4_000,
                    pages: 1 << 15,
                    indirect,
                    sky_later,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let view = target(&r);
        let mut frames = 0;
        while frames < 4_000 && !r.is_settled(32) {
            r.render(camera(), &view);
            frames += 1;
        }
        assert!(r.is_settled(32), "not settled after {frames} frames");
        (r.read_colors(), frames)
    };
    for indirect in [false, true] {
        let (whole, whole_frames) = settle(false, indirect);
        let (later, later_frames) = settle(true, indirect);
        let d = difference(&later, &whole);
        eprintln!("bounce {indirect}: settled in {whole_frames} and {later_frames} frames; {d:?}");
        assert_finite_and_lit(&later);
        // Bounce light's samples depend on what is resident when they are
        // baked, which varies run to run (about 2e-4 here).
        let bound = if indirect { 2e-3 } else { 1e-4 };
        assert!(d.relative_rmse < bound, "bounce {indirect}: {d:?}");
    }
}
