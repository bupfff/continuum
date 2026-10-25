//! M6 gate 1: glTF import with automatic charts and base color textures.
//!
//! What was built: `continuum-gltf` loads glTF 2.0 (`.gltf`, `.glb`) as an
//! instanced scene (geometry per primitive, shared by primitives reading the
//! same accessors; instances from nodes, non-uniform transforms baked; Y-up
//! to Z-up; base color factor and texture) and writes scenes back
//! (`save`). `Surface::from_mesh` gives any triangle mesh automatic charts
//! (grown within 45° of a seed's normal, projected without flips); the view
//! cache's chart cells hold any number of triangles. Base color textures
//! are resampled into 1024² layers with mip levels and sampled per pixel at
//! the pixel's footprint; the cache stores light, so texture detail stays at
//! pixel resolution. Cycles references take textures (OBJ texture
//! coordinates, PNG images).
//!
//! Scene: `room3d::textured` (a tiled ground, 30 boulders with lichen, a
//! brick block; three generated 1024² textures; a sun 1 km away and a sky),
//! written to `.glb` and imported back. Prefilter 1 bin, sky prefilter ×8,
//! cache defaults, hardware rays.
//!
//! Pass criterion, fixed before the first run:
//! 1. Round trip: the imported scene per pixel (640×360, 1 sample per pixel,
//!    compute rays) within 0.01% relative RMSE of the original scene.
//! 2. Accuracy with textures: the imported scene per pixel, 16×16 samples
//!    per pixel at 320×180, within 3% relative RMSE of instanced Cycles at
//!    4,096 samples per pixel with the same textures.
//! 3. Automatic charts serve the cache: at 1280×720, the warm cached image of
//!    the imported scene (automatic charts) against its per-pixel image (1
//!    sample per pixel) at most 1 point above the original scene's (its
//!    parametric and box charts).
//! 4. Speed: the imported scene's warm frame at 1280×720 at most 16.7 ms.
//!
//! Reported, not gated: export and import times and sizes, and for glTF
//! files given on the command line (not in the repository): import time,
//! counts, cached against per-pixel, warm frame and frames until warm.
//! `cargo run --release -p room3d --bin m6_gate1 -- [asset.glb ...]`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum_gpu::{
    CacheOptions, Camera, Emitter, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

struct Warm {
    frames: Option<usize>,
    frame_time: Duration,
    image: Vec<[f32; 3]>,
}

fn warm(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Warm {
    let mut frames = 0usize;
    let mut warm_from = None;
    while frames < 40_000 {
        r.frame(camera, target);
        frames += 1;
        let latest = r.cache_stats().unwrap().frame as usize;
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            warm_from = Some(latest - 16);
            break;
        }
    }
    r.finish();
    let start = Instant::now();
    for _ in 0..60 {
        r.frame(camera, target);
    }
    r.finish();
    let frame_time = start.elapsed() / 60;
    Warm {
        frames: warm_from,
        frame_time,
        image: r.read_colors(),
    }
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let t = Textured::new();
    kernel.set_sky(t.sky);

    let glb = out_dir.join("m6_gate1_textured.glb");
    let start = Instant::now();
    continuum_gltf::save(&t.scene, &glb)?;
    let export = start.elapsed();
    let start = Instant::now();
    let imported = continuum_gltf::load(&glb).map_err(std::io::Error::other)?;
    let import = start.elapsed();
    let size = std::fs::metadata(&glb)?.len();
    println!(
        "export {:.0} ms, import {:.0} ms, {size} bytes: {:?}",
        ms(export),
        ms(import),
        imported.stats
    );

    // 1. Round trip, per pixel.
    kernel.set_hardware_rays(false);
    let per_pixel = |scene: &InstancedScene, camera: Camera, (w, h): (u32, u32), ss: u32| {
        let (big, _, _) =
            kernel.render_instanced_lights(camera, w * ss, h * ss, &t.lights, scene, t.origin, 1);
        let mut px = vec![[0.0f32; 3]; (w * h) as usize];
        for y in 0..(h * ss) as usize {
            for x in 0..(w * ss) as usize {
                let p = big[y * (w * ss) as usize + x];
                let o = &mut px[(y / ss as usize) * w as usize + x / ss as usize];
                for c in 0..3 {
                    o[c] += p[c] / (ss * ss) as f32;
                }
            }
        }
        px
    };
    let round_trip = relative_rmse(
        &per_pixel(&imported.scene, t.camera, (640, 360), 1),
        &per_pixel(&t.scene, t.camera, (640, 360), 1),
    );
    println!("1. round trip, per pixel: {:.5}%", 100.0 * round_trip);

    // 2. Against Cycles with textures.
    let ours = per_pixel(&imported.scene, t.camera, (320, 180), 16);
    let reference = render_instanced_lights(
        &imported.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (320, 180),
        4096,
        0,
        1,
        &out_dir.join("m6_gate1_cycles"),
    )?;
    let accuracy = relative_rmse(&ours, &reference.pixels);
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    let bias = mean(&ours) / mean(&reference.pixels) - 1.0;
    println!(
        "2. vs Cycles with textures: {:.2}% (ours {:+.2}% brighter, Cycles {:.1} s)",
        100.0 * accuracy,
        100.0 * bias,
        reference.seconds
    );
    kernel.set_hardware_rays(true);

    // 3–4. The cache, original and imported.
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
    let cache_error = |scene: &InstancedScene, camera: Camera, lights: &[Light]| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            lights,
            scene,
            t.origin,
            CacheOptions::default(),
        );
        let warm = warm(&mut r, camera, &target);
        drop(r);
        let exact = kernel
            .render_instanced_lights(camera, W, H, lights, scene, t.origin, 1)
            .0;
        (relative_rmse(&warm.image, &exact), warm)
    };
    let (original_error, original_warm) = cache_error(&t.scene, t.camera, &t.lights);
    let (imported_error, imported_warm) = cache_error(&imported.scene, t.camera, &t.lights);
    println!(
        "3. cached vs per-pixel: original {:.2}% (warm after {:?} frames), imported {:.2}% (after {:?})",
        100.0 * original_error,
        original_warm.frames,
        100.0 * imported_error,
        imported_warm.frames
    );
    println!(
        "4. imported warm frame {:.2} ms (original {:.2} ms)",
        ms(imported_warm.frame_time),
        ms(original_warm.frame_time)
    );

    // Reported: real assets, where they exist here.
    // Paths given on the command line (not in the repository).
    let assets: Vec<String> = std::env::args().skip(1).collect();
    let mut asset_rows = Vec::new();
    for path in &assets {
        let path = Path::new(path);
        let name = path
            .file_stem()
            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
        let start = Instant::now();
        let a = continuum_gltf::load(path).map_err(std::io::Error::other)?;
        let import = start.elapsed();
        let (lo, hi) = a.bounds;
        let center = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(1e-3);
        let dir = continuum::glam::DVec3::new(-0.64, -0.64, 0.42);
        let camera = Camera {
            eye: (dir * radius * 2.2).as_vec3(),
            target: continuum::glam::Vec3::ZERO,
            vertical_fov_degrees: 40.0,
        };
        // The textured scene's lights, scaled to the asset and centered on it.
        let scale = (radius / 10.0) as f32;
        let lights: Vec<Light> = t
            .lights
            .iter()
            .map(|l| {
                let mut l = *l;
                if let Emitter::Rect(r) = &mut l.emitter {
                    r.corner *= scale;
                    r.edge_u *= scale;
                    r.edge_v *= scale;
                }
                l
            })
            .collect();
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            &a.scene,
            center,
            CacheOptions::default(),
        );
        let w = warm(&mut r, camera, &target);
        drop(r);
        let exact = kernel
            .render_instanced_lights(camera, W, H, &lights, &a.scene, center, 1)
            .0;
        let e = relative_rmse(&w.image, &exact);
        println!(
            "{name}: import {:.0} ms, {:?}; cached vs per-pixel {:.2}%, warm {:.2} ms after {:?} frames",
            ms(import),
            a.stats,
            100.0 * e,
            ms(w.frame_time),
            w.frames
        );
        asset_rows.push((name, import, a.stats, e, w));
    }

    let c1 = round_trip <= 0.0001;
    let c2 = accuracy <= 0.03;
    let c3 = imported_warm.frames.is_some() && imported_error <= original_error + 0.01;
    let c4 = imported_warm.frames.is_some() && ms(imported_warm.frame_time) <= 16.7;
    let pass = c1 && c2 && c3 && c4;
    let f = |x: &Option<usize>| x.map_or("not warm".to_string(), |v| v.to_string());
    let mut out = String::new();
    writeln!(out, "# M6 gate 1 results\n").unwrap();
    writeln!(
        out,
        "{}, the textured scene (`room3d::textured`) through `.glb`: export {:.0} ms, import {:.0} ms, {size} bytes; {} geometries, {} instances, {} charts after import.\n",
        kernel.adapter_name,
        ms(export),
        ms(import),
        imported.scene.geometries.len(),
        imported.stats.instances,
        imported.stats.charts
    )
    .unwrap();
    writeln!(
        out,
        "1. Round trip, per pixel (640×360): **{:.5}%**.\n2. Imported, 16×16 per pixel (320×180), vs Cycles with textures (4,096 spp, {:.1} s): **{:.2}%** (ours {:+.2}% brighter).\n3. Cached vs per-pixel at {W}×{H}: imported **{:.2}%** (warm after {} frames), original {:.2}% (after {}).\n4. Imported warm frame: **{:.2} ms** (original {:.2} ms).\n",
        100.0 * round_trip,
        reference.seconds,
        100.0 * accuracy,
        100.0 * bias,
        100.0 * imported_error,
        f(&imported_warm.frames),
        100.0 * original_error,
        f(&original_warm.frames),
        ms(imported_warm.frame_time),
        ms(original_warm.frame_time)
    )
    .unwrap();
    if !asset_rows.is_empty() {
        writeln!(
            out,
            "Reported, real assets on this machine:\n\n| Asset | Import | Triangles | Charts | Textures | Baked transforms | Cached vs per-pixel | Warm frame | Frames until warm |\n|---|---|---|---|---|---|---|---|---|"
        )
        .unwrap();
        for (name, import, stats, e, w) in &asset_rows {
            writeln!(
                out,
                "| {name} | {:.0} ms | {} | {} | {} | {} | {:.2}% | {:.2} ms | {} |",
                ms(*import),
                stats.triangles,
                stats.charts,
                stats.textures,
                stats.baked,
                100.0 * e,
                ms(w.frame_time),
                f(&w.frames)
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "\n## Verdict\n\n1. Round trip within 0.01%: {}\n2. Within 3% of Cycles with textures: {}\n3. Automatic charts within 1 point of the original's: {}\n4. Warm frame at most 16.7 ms: {}\n\n→ {}",
        if c1 { "pass" } else { "fail" },
        if c2 { "pass" } else { "fail" },
        if c3 { "pass" } else { "fail" },
        if c4 { "pass" } else { "fail" },
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m6_gate1_results.md"), &out)?;
    Ok(())
}
