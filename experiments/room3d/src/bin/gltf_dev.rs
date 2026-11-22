//! Development (files people use): loads a glTF file and builds `Renderer`
//! with its defaults at `GD_W`×`GD_H` (1280×720; times `GD_SS` per axis,
//! averaged down for the images), timing each step and listing the largest
//! GPU buffers; then renders `GD_FRAMES` (120) frames, each waited for, from
//! `GD_EYE` toward `GD_TARGET` (`x,y,z` from the scene's center; else the
//! file's camera or the viewer's), saving the last to `GD_OUT` (a PNG,
//! linear colors times `GD_EXPOSURE`, clamped). `GD_CUTOUTS=1` only cuts the
//! alpha cutouts and counts triangles. `GD_CYCLES=<samples>` also renders
//! the file in Cycles through Blender's own importer (`GD_BOUNCES`, 4),
//! saves it beside `GD_OUT` (`.cycles.png`) and prints the relative RMSE.
//! `GD_RAW=1`: the linear images too (`.f32`, `.cycles.f32`).
//! Not a gate.
//!
//! `cargo run --release -p room3d --bin gltf_dev -- scene.gltf`

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Light, MaskConeKernel, Renderer, RendererOptions, wgpu};

fn main() {
    let path = std::env::args().nth(1).expect("a glTF file");
    let t = Instant::now();
    let mut imported = continuum_gltf::load(Path::new(&path)).expect("the file");
    if std::env::var_os("GD_NOCUT").is_some() {
        imported.scene.geometry_alpha_cutoff.clear();
    }
    // `GD_NONORMALS=1`: without normal maps (a diagnostic).
    if std::env::var_os("GD_NONORMALS").is_some() {
        imported.scene.geometry_normal_maps.clear();
    }
    // `GD_ONLY=a,b,…`: only the instances of these geometries (a diagnostic).
    if let Ok(v) = std::env::var("GD_ONLY") {
        let keep: Vec<usize> = v
            .split(',')
            .map(|x| x.trim().parse().expect("indices"))
            .collect();
        imported
            .scene
            .instances
            .retain(|i| keep.contains(&i.geometry));
    }
    println!(
        "load {:.2} s: {:?}",
        t.elapsed().as_secs_f64(),
        imported.stats
    );
    let tri: usize = imported
        .scene
        .geometries
        .iter()
        .map(|g| g.triangles.len())
        .sum();
    let charts: usize = imported
        .scene
        .geometries
        .iter()
        .map(|g| g.charts.len())
        .sum();
    let big = imported
        .scene
        .geometries
        .iter()
        .map(|g| g.triangles.len())
        .max()
        .unwrap_or(0);
    println!(
        "geometries: {} with {tri} triangles (largest {big}), {charts} charts",
        imported.scene.geometries.len()
    );
    let (lo, hi) = imported.bounds;
    let origin = (lo + hi) * 0.5;
    let radius = ((hi - lo).length() * 0.5).max(1e-3);
    let mut lights = imported.lights(origin, 0.05);
    // `GD_ENV`: an environment image (an HDRI) lights the scene instead of
    // the sun and sky, times `GD_ENV_STRENGTH` (1).
    let environment = std::env::var_os("GD_ENV").map(|p| {
        let mut env = continuum_gltf::load_environment(Path::new(&p)).expect("the environment");
        let k: f32 = std::env::var("GD_ENV_STRENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        env.rgb
            .iter_mut()
            .for_each(|c| c.iter_mut().for_each(|x| *x *= k));
        println!("environment lights: {:?}", env.clone().take_lights(4));
        env
    });
    // Suns added here (toward the sun, irradiance), for Cycles.
    let mut suns: Vec<(Vec3, Vec3)> = Vec::new();
    if lights.is_empty() && environment.is_none() && std::env::var("GD_SUN").as_deref() != Ok("0") {
        let toward = continuum::glam::DVec3::new(-0.6, -0.3, 0.75);
        lights.push(Light::sun(
            Vec3::ZERO,
            toward,
            radius * continuum_gpu::SUN_RADII,
            Vec3::splat(3.0),
        ));
        suns.push((toward.normalize().as_vec3(), Vec3::splat(3.0)));
    }
    if std::env::var_os("GD_CUTOUTS").is_some() {
        let t = Instant::now();
        let cut = imported.scene.with_cutouts();
        println!("cutouts {:.2} s", t.elapsed().as_secs_f64());
        let mut rows: Vec<(usize, usize, usize)> = (0..cut.geometries.len())
            .map(|g| {
                (
                    g,
                    imported.scene.geometries[g].triangles.len(),
                    cut.geometries[g].triangles.len(),
                )
            })
            .filter(|r| r.1 != r.2)
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        for (g, a, b) in rows.iter().take(12) {
            println!(
                "  geometry {g}: {a} -> {b} triangles; texture {:?}, cutoff {:?}",
                imported.scene.geometry_textures.get(*g).copied().flatten(),
                imported
                    .scene
                    .geometry_alpha_cutoff
                    .get(*g)
                    .copied()
                    .flatten()
            );
        }
        let total: usize = cut.geometries.iter().map(|g| g.triangles.len()).sum();
        let charts: usize = cut.geometries.iter().map(|g| g.charts.len()).sum();
        println!(
            "  total {total} triangles, {charts} charts after cutouts ({} geometries changed)",
            rows.len()
        );
        for (g, _, _) in rows.iter().take(3) {
            let s = &cut.geometries[*g];
            println!("  geometry {g}: {} charts", s.charts.len());
        }
        return;
    }
    let t = Instant::now();
    let kernel = MaskConeKernel::new().expect("GPU");
    // `GD_SKY_MAX=1`: sky cones combine meshes' coverage by the maximum
    // (a diagnostic).
    if std::env::var("GD_HW").as_deref() == Ok("0") {
        kernel.set_hardware_rays(false);
    }
    if std::env::var("GD_OBJECT_PREFILTER").as_deref() == Ok("0") {
        kernel.set_object_prefilter(false);
    }
    kernel.set_sky_combination_max(std::env::var("GD_SKY_MAX").as_deref() == Ok("1"));
    println!("kernel {:.2} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    // `GD_BUDGET`: a fixed bake budget (texels a frame) instead of the time
    // budget, for comparing renders.
    let budget: u32 = std::env::var("GD_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let num = |name: &str, default: u32| -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (w, h, ss) = (num("GD_W", 1280), num("GD_H", 720), num("GD_SS", 1).max(1));
    let mut r = Renderer::new(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (w * ss, h * ss),
        &imported.scene,
        origin,
        &lights,
        RendererOptions {
            sky: (environment.is_none() && std::env::var("GD_SKY").as_deref() != Ok("0"))
                .then_some(Vec3::new(0.6, 0.75, 1.0) * 0.3),
            environment,
            lights_from_environment: std::env::var("GD_ENV_LIGHTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            bounce: std::env::var("GD_BOUNCE").as_deref() != Ok("0"),
            anti_aliasing: std::env::var("GD_AA").as_deref() != Ok("0"),
            cache_options: continuum_gpu::CacheOptions {
                glossy_rays: std::env::var("GD_GLOSSY").as_deref() != Ok("0"),
                // `GD_CACHE_BOUNCES`: the cache's bounces (`GD_BOUNCES` is
                // Cycles').
                bounces: num("GD_CACHE_BOUNCES", 1),
                bounce_rest: std::env::var_os("GD_BOUNCE_REST").is_some(),
                hardware_shading: std::env::var("GD_HW_SHADE").as_deref() != Ok("0"),
                frame_target_ms: if budget > 0 { 0.0 } else { 14.0 },
                bake_budget: if budget > 0 {
                    budget
                } else {
                    continuum_gpu::CacheOptions::default().bake_budget
                },
                ..Default::default()
            },
            ..Default::default()
        },
    );
    r.wait();
    println!(
        "renderer {:.2} s, {:.0} MB",
        t.elapsed().as_secs_f64(),
        r.gpu_bytes() as f64 / 1e6
    );
    let (scene, cache) = r.memory_report();
    let top = |v: &[(u32, u64)]| {
        v.iter()
            .take(6)
            .map(|(k, b)| format!("{k}: {:.0} MB", *b as f64 / 1e6))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("scene buffers: {}", top(&scene));
    println!("cache buffers: {}", top(&cache));
    let vec = |name: &str| {
        std::env::var(name).ok().map(|v| {
            let c: Vec<f32> = v
                .split(',')
                .map(|x| x.trim().parse().expect("x,y,z"))
                .collect();
            Vec3::new(c[0], c[1], c[2])
        })
    };
    let (eye, target, fov) = match (vec("GD_EYE"), vec("GD_TARGET"), imported.camera) {
        (Some(e), Some(t), _) => (e, t, 60.0),
        (_, _, Some(c)) => (
            (c.eye - origin).as_vec3(),
            (c.target - origin).as_vec3(),
            c.vertical_fov_degrees,
        ),
        _ => {
            let (az, el) = (225f64.to_radians(), 25f64.to_radians());
            let dir =
                continuum::glam::DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
            ((dir * radius * 2.2).as_vec3(), Vec3::ZERO, 40.0)
        }
    };
    println!("bounds {lo:.2?} to {hi:.2?}; eye {eye:.2}, target {target:.2}");
    let camera = continuum_gpu::Camera {
        eye,
        target,
        vertical_fov_degrees: fov,
    };
    let view = r
        .kernel()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w * ss,
                height: h * ss,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let frames: usize = std::env::var("GD_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let mut times = Vec::new();
    let start = Instant::now();
    let mut settled = None;
    for f in 0..frames {
        let t = Instant::now();
        r.render(camera, &view);
        r.wait();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        if std::env::var_os("GD_TRACE").is_some()
            && f % 25 == 0
            && let Some(s) = r.stats()
        {
            println!(
                "frame {f}: {:.1} ms (planning {:?}, evicting {:?}, waiting {:?}); fallback {}, requests {}, baked {} pages {} texels, budget {}, resident {}, evicted {}, bake {:.1} ms, render {:.1} ms, bounce samples {} ({} incomplete)",
                times[f],
                s.planning,
                s.evicting,
                s.waiting,
                s.fallback_pixels,
                s.requests,
                s.pages_baked,
                s.texels_baked,
                s.budget,
                s.resident,
                s.evicted,
                s.gpu_bake_ms,
                s.gpu_render_ms,
                s.indirect_samples,
                s.indirect_incomplete
            );
        }
        if settled.is_none() && f > 32 && r.is_settled(32) {
            settled = Some((f, start.elapsed().as_secs_f64()));
        }
    }
    if frames > 0 {
        let mut late = times[frames / 2..].to_vec();
        late.sort_by(f64::total_cmp);
        println!(
            "frames: first {:.1} ms, later half median {:.2} ms, p90 {:.2} ms; settled at {settled:?} (frame, s)",
            times[0],
            late[late.len() / 2],
            late[late.len() * 9 / 10]
        );
    }
    if std::env::var_os("GD_PROFILE").is_some() {
        let mut sums = [0.0f64; 16];
        let mut bake = std::collections::BTreeMap::<&str, f64>::new();
        for _ in 0..10 {
            let p = r.render_profiled(camera, &view);
            for (i, d) in p.passes.iter().enumerate() {
                sums[i] += d.as_secs_f64() * 100.0;
            }
            for (label, d) in p.bake_passes {
                if !label.is_empty() {
                    *bake.entry(label).or_insert(0.0) += d.as_secs_f64() * 100.0;
                }
            }
        }
        let passes: Vec<String> = sums
            .iter()
            .enumerate()
            .filter(|(_, t)| **t > 0.05)
            .map(|(i, t)| format!("{i}: {t:.2}"))
            .collect();
        println!("camera passes (ms): {}", passes.join(", "));
        let b: Vec<String> = bake.iter().map(|(k, t)| format!("{k} {t:.2}")).collect();
        println!("bake passes (ms): {}", b.join(", "));
    }
    let exposure: f32 = std::env::var("GD_EXPOSURE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let colors = r.read_colors();
    // Averaged down `ss` × `ss` (Cycles averages each pixel's area).
    let ours: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut sum = [0.0f32; 3];
            for dy in 0..ss {
                for dx in 0..ss {
                    let c = colors[((y * ss + dy) * w * ss + x * ss + dx) as usize];
                    (0..3).for_each(|k| sum[k] += c[k]);
                }
            }
            sum.map(|v| v / (ss * ss) as f32)
        })
        .collect();
    let save = |image: &[[f32; 3]], path: &std::path::Path| {
        let srgb = |x: f32| {
            let x = (x * exposure).clamp(0.0, 1.0);
            let s = if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u8
        };
        let bytes: Vec<u8> = image.iter().flat_map(|c| c.map(srgb)).collect();
        let file = std::fs::File::create(path).expect("the image");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut wr| wr.write_image_data(&bytes))
            .expect("the PNG");
    };
    let out = std::env::var_os("GD_OUT").map(std::path::PathBuf::from);
    // `GD_RAW=1`: the linear images too (`.f32` beside the PNGs: RGB f32
    // rows from the top).
    let dump_raw = |image: &[[f32; 3]], path: &std::path::Path| {
        if std::env::var_os("GD_RAW").is_some() {
            let bytes: Vec<u8> = image
                .iter()
                .flatten()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            std::fs::write(path, bytes).expect("the raw image");
        }
    };
    if let Some(out) = &out {
        save(&ours, out);
        dump_raw(&ours, &out.with_extension("f32"));
    }
    if let Ok(samples) = std::env::var("GD_CYCLES") {
        let work = std::env::temp_dir().join(format!("gltf_dev_cycles_{}", std::process::id()));
        std::fs::create_dir_all(&work).expect("a work folder");
        let v = |x: Vec3| format!("[{}, {}, {}]", x.x, x.y, x.z);
        let suns: Vec<String> = suns
            .iter()
            .map(|&(toward, irradiance)| {
                format!(
                    "{{\"toward\": {}, \"irradiance\": {}, \"angle\": 0.5}}",
                    v(toward),
                    v(irradiance)
                )
            })
            .collect();
        let env_path = std::env::var("GD_ENV").ok();
        let strength: f32 = std::env::var("GD_ENV_STRENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let raw = work.join("reference.f32");
        let job = format!(
            "{{\"gltf\": {:?}, \"origin\": [{}, {}, {}], \"camera\": {{\"eye\": {}, \"target\": {}, \"fov\": {}}}, \"size\": [{w}, {h}], \"samples\": {samples}, \"bounces\": {}, \"environment\": {}, \"strength\": {strength}, \"sky\": {}, \"suns\": [{}], \"distribution\": {:?}, \"out\": {:?}}}",
            path,
            origin.x,
            origin.y,
            origin.z,
            v(eye),
            v(target),
            fov,
            num("GD_BOUNCES", 4),
            env_path.map_or("null".to_string(), |p| format!("{p:?}")),
            if std::env::var("GD_SKY").as_deref() == Ok("0") || std::env::var_os("GD_ENV").is_some()
            {
                v(Vec3::ZERO)
            } else {
                v(Vec3::new(0.6, 0.75, 1.0) * 0.3)
            },
            suns.join(", "),
            std::env::var("GD_DISTRIBUTION").unwrap_or_default(),
            raw.to_string_lossy()
        );
        let job_path = work.join("job.json");
        std::fs::write(&job_path, job).expect("the job");
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("cycles/render_gltf_reference.py");
        let t = Instant::now();
        let status = std::process::Command::new("/Applications/Blender.app/Contents/MacOS/Blender")
            .args(["-b", "--factory-startup", "-noaudio", "--python"])
            .arg(&script)
            .arg("--")
            .arg(&job_path)
            .stdout(std::process::Stdio::null())
            .status()
            .expect("Blender");
        assert!(status.success(), "Blender failed");
        let bytes = std::fs::read(&raw).expect("the reference");
        let reference: Vec<[f32; 3]> = bytes
            .as_chunks::<12>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))
            })
            .collect();
        println!(
            "cycles {:.0} s: relative RMSE {:.4}; mean ours {:.4}, cycles {:.4}",
            t.elapsed().as_secs_f64(),
            room3d::cycles::relative_rmse(&ours, &reference),
            ours.iter()
                .map(|c| (c[0] + c[1] + c[2]) as f64)
                .sum::<f64>()
                / (3 * ours.len()) as f64,
            reference
                .iter()
                .map(|c| (c[0] + c[1] + c[2]) as f64)
                .sum::<f64>()
                / (3 * reference.len()) as f64
        );
        // Highlights clamped at 4x the reference's mean (sun glints on
        // smooth metal dominate otherwise), and the share of pixels off by
        // more than 10% of the mean.
        let mean = reference.iter().flatten().map(|&x| x as f64).sum::<f64>()
            / (3 * reference.len()) as f64;
        let cap = (4.0 * mean) as f32;
        let clamp = |v: &[[f32; 3]]| v.iter().map(|c| c.map(|x| x.min(cap))).collect::<Vec<_>>();
        let (oc, rc) = (clamp(&ours), clamp(&reference));
        let off = oc
            .iter()
            .zip(&rc)
            .filter(|(a, b)| (0..3).any(|k| ((a[k] - b[k]) as f64).abs() > 0.1 * mean))
            .count();
        println!(
            "clamped at 4x the mean: relative RMSE {:.4}; pixels off by > 10% of the mean: {:.2}%",
            room3d::cycles::relative_rmse(&oc, &rc),
            100.0 * off as f64 / oc.len() as f64
        );
        if let Some(out) = &out {
            save(&reference, &out.with_extension("cycles.png"));
            dump_raw(&reference, &out.with_extension("cycles.f32"));
            // The difference, ours minus Cycles: red brighter, blue darker,
            // full scale at half the mean.
            let diff: Vec<[f32; 3]> = oc
                .iter()
                .zip(&rc)
                .map(|(a, b)| {
                    let d = ((a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2]))
                        / 3.0
                        / (0.5 * mean as f32)
                        / exposure;
                    [d.max(0.0), 0.0, (-d).max(0.0)]
                })
                .collect();
            save(&diff, &out.with_extension("diff.png"));
        }
        let _ = std::fs::remove_dir_all(&work);
    }
}
