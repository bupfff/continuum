//! M12 gate 1: deforming geometry.
//!
//! The first of the gaps the user ordered on 2026-11-15: geometry whose
//! vertices change every frame (skinned characters, morph targets, animated
//! scale, anything an application deforms). Design: DESIGN §3.4,
//! "Deforming geometry (M12 design)". In brief:
//! - `RendererOptions::deforming` lists geometries whose vertices change;
//!   their instances move as movers do. `Renderer::set_vertices` gives a
//!   pose (triangles and normals; topology, charts and texture coordinates
//!   stay the rest pose's).
//! - Per frame, for each deformed geometry: its triangles and normals
//!   uploaded; its triangle hierarchy refit on the CPU (the rest pose's
//!   tree, boxes and coverage samples recomputed); its hardware structure
//!   rebuilt; its new object-space box gives its instances' world boxes,
//!   which update the movers' subtree and structure.
//! - The cache: its instances count as moved (M9 G1's stale pass); its own
//!   pages are stale and re-bake in the same frame when read.
//!
//! Scene: the million-boulder field under M5 gate 6's three coloured
//! lights (`Field::colored_lights(origin, 1000.0)`), no sky, 1280×720,
//! hardware camera rays, prefilter 1 bin, cache defaults. A blob: a sphere
//! of 128 × 64 segments (16,384 triangles) 2 m across, its centre 1.2 m up,
//! 10 m ahead of the standing camera along its line, its radius
//! r(θ, φ, k) = 1 + 0.15 sin(3θ + 0.05k) sin(2φ) + 0.1 sin(5φ − 0.07k) at
//! frame k (normals from the surface's derivatives). The camera stands at
//! the field's camera, looking at the blob. Motion advances per frame.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Exact:** from an empty cache, over the first 300 frames with the
//!    blob deforming, after every 10th frame, every texel of every
//!    resident page that shading may read equals, bit for bit, its value
//!    re-baked from scratch with the scene as it then is
//!    (`verify_cache`). Zero differing texels in all 30 checks.
//! 2. **Accuracy:** at frame 150's pose, per-pixel cones (no cache), 16×16
//!    samples a pixel at 320×180 box filtered, within 3% relative RMSE of
//!    instanced Cycles (4,096 samples a pixel, direct light only) over
//!    surfaces. Pixels where a sample sees a light, and their eight
//!    neighbours, are scored apart (M5 gate 6's method).
//! 3. **Cost** (timed first): three runs of 600 frames from an empty
//!    cache with the blob deforming, alternating with three in which it
//!    stays at frame 0's pose. Each deforming run's mean frame at most
//!    1.25× the still runs' mean and at most 16.7 ms, its 99th percentile
//!    at most 33.3 ms, and fallback at most 10%.
//!
//! Reported: pages and texels re-baked per frame for the blob; the refit's
//! CPU time; per-pixel cones at frame 150 against a renderer built with
//! that pose as static geometry.
//!
//! As built (before any run): criterion 2's per-pixel cones are a renderer
//! without the cache built with the rest pose and given frame 150's pose
//! (`set_vertices`, so the refit is what is measured), averaged over 16×16
//! pixel offsets. Frame times are the renderer's frame call and the wait
//! for the GPU; the application's pose (here a parametric surface built
//! each frame on the CPU) is outside them, and the refit inside.
//!
//! Development before the run (disclosed):
//! - **A geometry's new structure is built in a submission of its own.**
//!   Built in the frame's encoder before the top-level build that places
//!   it, it was never hit (the blob vanished). A fresh structure per pose
//!   keeps frames in flight safe.
//! - **Deforming objects' cache pages take one mip coarser**
//!   (`DEFORMING_MIPS`), a quarter of the texels.
//! - **On 10,000 boulders with 100-frame runs:**
//!   - criterion 1: 10 checks, nothing differing;
//!   - criterion 2: 1.08% from Cycles; the refit is 0.000% from a hierarchy
//!     built for the pose;
//!   - criterion 3 fails: deforming 1.79–1.95× the still frames.
//!
//!   By frame 95 about 4,200 texel–light cones re-bake per frame, mostly
//!   receivers in the blob's shadows (cones passing near the blob walk its
//!   16,384 triangles), at about 5 µs each.
//!
//! `cargo run --release -p room3d --bin m12_gate1`
//!
//! `M12_DEV=1` runs on 10,000 boulders with 100-frame runs, one of each,
//! and 10 exactness checks. Development modes:
//! - `M12_ACC=1`: criterion 2 alone;
//! - `M12_TRACE=1`: frame statistics;
//! - `M12_PROBE=1`: images of the blob through each path.

use std::f32::consts::{PI, TAU};
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;

/// The blob at frame `k`, outward.
fn blob(k: usize) -> Surface {
    let k = k as f32;
    let s = Surface::parametric(128, 64, |u, v| {
        let (theta, phi) = (TAU * u, PI * v);
        let r = 1.0
            + 0.15 * (3.0 * theta + 0.05 * k).sin() * (2.0 * phi).sin()
            + 0.1 * (5.0 * phi - 0.07 * k).sin();
        Vec3::new(
            phi.sin() * theta.cos() * r,
            phi.sin() * theta.sin() * r,
            -phi.cos() * r,
        )
    });
    // Outward, judged at a triangle away from the poles.
    let t = s.triangles[s.triangles.len() / 2];
    if (t[1] - t[0]).cross(t[2] - t[0]).dot(t[0]) < 0.0 {
        s.flipped()
    } else {
        s
    }
}

fn target(kernel: &MaskConeKernel, w: u32, h: u32) -> wgpu::TextureView {
    kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
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

struct Setup {
    scene: InstancedScene,
    origin: DVec3,
    lights: Vec<Light>,
    camera: Camera,
    geometry: usize,
    instance: usize,
}

fn setup(boulders: usize) -> Setup {
    let field = Field::new(boulders, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, _) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0).to_vec();
    let f = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(f.x, f.y, 0.0).normalize();
    let centre = Vec3::new(ground.eye.x, ground.eye.y, 1.2) + flat * 10.0;
    let mut scene = field.scene.clone();
    let geometry = scene.geometries.len();
    scene.geometries.push(blob(0));
    let instance = scene.instances.len();
    scene.instances.push(Instance {
        geometry,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position: origin + centre.as_dvec3(),
        albedo: Vec3::new(0.6, 0.55, 0.5),
    });
    Setup {
        scene,
        origin,
        lights,
        camera: Camera {
            eye: ground.eye,
            target: centre,
            vertical_fov_degrees: ground.vertical_fov_degrees,
        },
        geometry,
        instance,
    }
}

fn renderer(
    kernel: &MaskConeKernel,
    s: &Setup,
    (w, h): (u32, u32),
    cache: Option<CacheOptions>,
) -> InstancedRenderer {
    let mut r = InstancedRenderer::with_movers(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        w,
        h,
        1,
        &s.lights,
        &s.scene,
        s.origin,
        cache,
        &[s.instance],
        &[],
    );
    r.set_deforming(&s.scene, &[s.geometry]);
    r
}

fn pose(r: &mut InstancedRenderer, s: &Setup, k: usize) {
    let b = blob(k);
    r.set_vertices(s.geometry, b.triangles, b.normals);
}

struct Run {
    mean: f64,
    p99: f64,
    fallback: f64,
}

/// `n` frames from an empty cache, the blob deforming (or still at frame
/// 0's pose): frame times (the frame call and the wait) and fallback.
fn run(kernel: &MaskConeKernel, s: &Setup, n: usize, deforming: bool) -> Run {
    let view = target(kernel, W, H);
    let mut r = renderer(kernel, s, (W, H), Some(CacheOptions::default()));
    let mut times = Vec::with_capacity(n);
    let mut fallback = 0.0;
    for k in 0..n {
        if deforming {
            pose(&mut r, s, k);
        }
        let t = Instant::now();
        r.frame(s.camera, &view);
        let call = t.elapsed().as_secs_f64() * 1000.0;
        r.wait();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        if let Some(c) = r.cache_stats() {
            fallback += c.fallback_pixels as f64 / (W * H) as f64;
            if std::env::var_os("M12_TRACE").is_some() && k % 10 == 5 {
                eprintln!(
                    "{} frame {k}: {:.1} ms (call {call:.1}); baked {} pages {} texels, rebaked {} pages {} texel-lights of {} candidates, moved {}, budget {}, gpu bake {:.1} render {:.1}",
                    if deforming { "deforming" } else { "still" },
                    times[k],
                    c.pages_baked,
                    c.texels_baked,
                    c.rebaked_pages,
                    c.rebaked_texels,
                    c.candidates,
                    c.moved,
                    c.budget,
                    c.gpu_bake_ms,
                    c.gpu_render_ms
                );
            }
        }
    }
    let mean = times.iter().sum::<f64>() / n as f64;
    times.sort_by(f64::total_cmp);
    Run {
        mean,
        p99: times[(n - 1) * 99 / 100],
        fallback: fallback / n as f64,
    }
}

/// Per pixel at 16×16 offsets (`r` at 320×180), and which pixels (with
/// their eight neighbours) have a sample that sees a light.
fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    view: &wgpu::TextureView,
    lights: &[Light],
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let (w, h) = (320usize, 180usize);
    let mut sum = vec![[0.0f32; 3]; w * h];
    let mut sees = vec![false; w * h];
    for s in 0..256 {
        let (dx, dy) = (
            ((s % 16) as f32 + 0.5) / 16.0 - 0.5,
            ((s / 16) as f32 + 0.5) / 16.0 - 0.5,
        );
        r.set_pixel_offset(dx, dy);
        r.frame(camera, view);
        r.finish();
        for (i, p) in r.read_colors().into_iter().enumerate() {
            if lights.iter().any(|l| l.radiance.to_array() == p) {
                sees[i] = true;
            }
            for c in 0..3 {
                sum[i][c] += p[c] / 256.0;
            }
        }
    }
    let mut apart = sees.clone();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !sees[y as usize * w + x as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                        apart[ny as usize * w + nx as usize] = true;
                    }
                }
            }
        }
    }
    (sum, apart)
}

fn surfaces_rmse(a: &[[f32; 3]], b: &[[f32; 3]], apart: &[bool]) -> f64 {
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(apart)
            .filter(|(_, x)| !**x)
            .map(|(p, _)| *p)
            .collect()
    };
    relative_rmse(&keep(a), &keep(b))
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    let dev = std::env::var_os("M12_DEV").is_some();
    let (boulders, frames, rounds, checked) = if dev {
        (10_000, 100, 1, 100)
    } else {
        (1_000_000, 600, 3, 300)
    };
    let s = setup(boulders);
    // Development (`M12_PROBE=1`): the blob through the renderer without
    // the cache at its rest pose and at frame 150's (posed), and with the
    // cache posed and settled; saved.
    if std::env::var_os("M12_PROBE").is_some() {
        let small = (320u32, 180u32);
        let view = target(&kernel, small.0, small.1);
        let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
            let mut image = flatland::img::Image::new(320, 180, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                let v = p.map(|v| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
                image.set(i % 320, i / 320, v);
            }
            image.save(&out_dir.join(format!("m12_probe_{name}.png")))
        };
        let mut a = renderer(&kernel, &s, small, None);
        a.frame(s.camera, &view);
        a.finish();
        save("rest", &a.read_colors())?;
        let mut z = renderer(&kernel, &s, small, None);
        pose(&mut z, &s, 0);
        z.frame(s.camera, &view);
        z.finish();
        save("rest_reposed", &z.read_colors())?;
        let mut b = renderer(&kernel, &s, small, None);
        pose(&mut b, &s, 150);
        b.frame(s.camera, &view);
        b.finish();
        save("posed", &b.read_colors())?;
        b.frame(s.camera, &view);
        b.finish();
        save("posed_next", &b.read_colors())?;
        let mut c = renderer(&kernel, &s, small, Some(CacheOptions::default()));
        pose(&mut c, &s, 150);
        for _ in 0..200 {
            c.frame(s.camera, &view);
            c.wait();
        }
        c.finish();
        save("cached", &c.read_colors())?;
        return Ok(());
    }
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let mut pass = true;
    // 3. Cost, first: deforming and still runs, alternating.
    let mut deforming = Vec::new();
    let mut still = Vec::new();
    let only_accuracy = std::env::var_os("M12_ACC").is_some();
    for _ in 0..if only_accuracy { 0 } else { rounds } {
        deforming.push(run(&kernel, &s, frames, true));
        still.push(run(&kernel, &s, frames, false));
    }
    let still_mean = still.iter().map(|r| r.mean).sum::<f64>() / rounds.max(1) as f64;
    for (k, d) in deforming.iter().enumerate() {
        let c3 =
            d.mean <= 1.25 * still_mean && d.mean <= 16.7 && d.p99 <= 33.3 && d.fallback <= 0.1;
        pass &= c3;
        say(format!(
            "3. deforming run {}: mean {:.2} ms ({:.3}× the still runs' {:.2} ms; at most 1.25× and 16.7 ms), 99th percentile {:.2} ms (at most 33.3), fallback {:.2}% (at most 10%): {}",
            k + 1,
            d.mean,
            d.mean / still_mean,
            still_mean,
            d.p99,
            100.0 * d.fallback,
            verdict(c3)
        ));
    }
    for (k, r) in still.iter().enumerate() {
        say(format!(
            "   still run {}: mean {:.2} ms, 99th percentile {:.2} ms, fallback {:.2}%",
            k + 1,
            r.mean,
            r.p99,
            100.0 * r.fallback
        ));
    }
    // 1. Exact.
    if !only_accuracy {
        let view = target(&kernel, W, H);
        let mut r = renderer(&kernel, &s, (W, H), Some(CacheOptions::default()));
        let mut worst = 0u64;
        let mut largest = 0.0f32;
        let mut pages = Vec::new();
        for k in 0..checked {
            pose(&mut r, &s, k);
            r.frame(s.camera, &view);
            r.wait();
            if (k + 1) % 10 == 0 {
                let c = r.verify_cache();
                worst = worst.max(c.differing);
                largest = largest.max(c.largest);
                pages.push(c.pages);
            }
        }
        let c1 = worst == 0;
        pass &= c1;
        say(format!(
            "1. exactness: {} checks, at most {worst} texel values differing (none allowed), largest change {largest:e}: {}; pages checked {:?}",
            pages.len(),
            verdict(c1),
            pages
        ));
    }
    // 2. Accuracy at frame 150's pose, through the refit.
    {
        let small = (320, 180);
        let view = target(&kernel, small.0, small.1);
        let mut r = renderer(&kernel, &s, small, None);
        pose(&mut r, &s, 150);
        let (ours, apart) = supersampled(&mut r, s.camera, &view, &s.lights);
        let mut posed = s.scene.clone();
        posed.geometries[s.geometry] = blob(150);
        let cycles = render_instanced_lights(
            &posed,
            s.origin,
            &s.lights,
            Vec3::ZERO,
            s.camera,
            small,
            4096,
            0,
            1,
            &out_dir.join(if dev {
                "m12_gate1_cycles_dev"
            } else {
                "m12_gate1_cycles"
            }),
        )?;
        let e = surfaces_rmse(&ours, &cycles.pixels, &apart);
        let c2 = e <= 0.03;
        pass &= c2;
        say(format!(
            "2. frame 150's pose, per-pixel cones (16×16, 320×180) vs Cycles (4,096 spp, {:.0} s), surfaces: {:.2}% (at most 3%): {}; {} pixels apart",
            cycles.seconds,
            100.0 * e,
            verdict(c2),
            apart.iter().filter(|a| **a).count()
        ));
        // Reported: against a renderer built with that pose as static
        // geometry.
        let mut fresh = InstancedRenderer::new_lights(
            &kernel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            small.0,
            small.1,
            1,
            &s.lights,
            &posed,
            s.origin,
        );
        let (built, _) = supersampled(&mut fresh, s.camera, &view, &s.lights);
        if dev {
            for (name, px) in [
                ("refit", &ours),
                ("built", &built),
                ("cycles", &cycles.pixels),
            ] {
                let mut image = flatland::img::Image::new(320, 180, [0; 3]);
                for (i, p) in px.iter().enumerate() {
                    let v = p
                        .map(|v| ((v / 1.0).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
                    image.set(i % 320, i / 320, v);
                }
                image.save(&out_dir.join(format!("m12_gate1_dev_{name}.png")))?;
            }
        }
        say(format!(
            "   (reported) the refit against a hierarchy built for the pose: {:.3}% over surfaces",
            100.0 * surfaces_rmse(&ours, &built, &apart)
        ));
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(
        out_dir.join(if dev {
            "m12_gate1_dev.txt"
        } else {
            "m12_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
