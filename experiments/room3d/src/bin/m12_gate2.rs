//! M12 gate 2: deforming geometry within the frame's budget.
//!
//! M12 gate 1 kept the cache exact for a deforming blob (no stale light in
//! 30 checks, 1.08% from Cycles) but frames cost 9.6–10.1× still ones:
//! exactness re-bakes, every frame, every texel in the blob's shadows
//! (`docs/results/m12-gate1.md`). Decided by the user (2026-11-15):
//! lighting near a deforming object may lag its pose within the frame's
//! budget, the shape current every frame, the lag reported.
//!
//! Method (`CacheOptions::deforming_budget`, 512 texel–light cones a
//! frame by default; 0 keeps gate 1's exactness):
//! - **In turn.** Each light's lighting near a deforming instance catches
//!   up with its pose in turn, the lights spread over a period of frames.
//!   The cache is told, for those lights only, of the move since that
//!   light last caught up: per light the box then and now, and a mask its
//!   own pages' re-bake follows. The other lights' light stays as it was.
//! - **The period** follows the re-bakes' measured cost (texel–light
//!   cones per light caught up, from the stats two frames on) so that a
//!   frame's re-bakes fit the budget; at most 16 frames
//!   (`DEFORMING_MAX_LAG`).
//! - **Catch-up.** When the object stops, the lights still lagging catch up
//!   in turn, after which the cache is exact again.
//! - **Reported:** `InstancedRenderer::deforming_lag`, the most frames any
//!   light lags.
//!
//! Scene: gate 1's (the million-boulder field, M5 gate 6's three coloured
//! lights, no sky, 1280×720, hardware camera rays, prefilter 1 bin, cache
//! defaults; the blob of 16,384 triangles 10 m ahead, waving each frame).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost** (timed first): three runs of 600 frames from an empty cache
//!    with the blob deforming, alternating with three with it still. Each
//!    deforming run's mean frame at most 1.25× the still runs' mean and at
//!    most 16.7 ms, its 99th percentile at most 33.3 ms or 1.25× the still
//!    runs' (whichever is larger), fallback at most 10%, and no light
//!    lagging more than 16 frames.
//! 2. **Caught up:** after 300 frames deforming, the blob held at frame
//!    300's pose for 32 frames: `verify_cache` finds no texel value
//!    differing (the lag leaves no stale light once the object stops).
//! 3. **Error while deforming:** at frames 100, 200 and 300 of a run with
//!    the blob deforming, the lagging cache's image against the exact
//!    cache's (deforming budget 0, the same poses from an empty cache),
//!    relative RMSE over the pixels of the blob's surroundings (its
//!    bounding sphere's screen box, widened by 1.5 radii): at most
//!    5% (provisional, from development).
//! 4. **Exact still exact:** with the deforming budget 0, 10 checks of
//!    `verify_cache` over 100 frames find no texel value differing.
//!
//! Reported: the period and lag over the runs; the whole image's error.
//!
//! **Not run** (2026-11-15): development at full size failed criterion 1
//! (deforming 1.6–1.95× still, 99th percentile 71–84 ms). The lag works:
//! the cache is exact once the blob stops, exact mode stays exact, and the
//! error while deforming is 0.15–4.7% around the blob. But one light's
//! catch-up spans tens of thousands of texels: fine ground pages near the
//! camera in the lamp's shadow region. Unexplained spikes remain
//! (`docs/results/m12-gate2-development.md`).
//!
//! `cargo run --release -p room3d --bin m12_gate2`
//!
//! `M12_DEV=1` runs on 10,000 boulders (100-frame runs, one each);
//! `M12_DEV=1M` at full size with the blob 14 m ahead; `M12_COST=1` one
//! deforming run alone (`M12_TRACE=1`: its frame statistics).

use std::f32::consts::{PI, TAU};
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::relative_rmse;
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

fn setup(boulders: usize, ahead: f32) -> Setup {
    let field = Field::new(boulders, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, _) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0).to_vec();
    let f = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(f.x, f.y, 0.0).normalize();
    let centre = Vec3::new(ground.eye.x, ground.eye.y, 1.2) + flat * ahead;
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
    lag: u32,
}

/// `n` frames from an empty cache, the blob deforming (or still at frame
/// 0's pose): frame times (the frame call and the wait) and fallback.
fn run(kernel: &MaskConeKernel, s: &Setup, n: usize, deforming: bool) -> Run {
    let view = target(kernel, W, H);
    let mut r = renderer(kernel, s, (W, H), Some(CacheOptions::default()));
    let mut times = Vec::with_capacity(n);
    let mut fallback = 0.0;
    let mut lag = 0;
    for k in 0..n {
        if deforming {
            pose(&mut r, s, k);
        }
        let t = Instant::now();
        r.frame(s.camera, &view);
        let call = t.elapsed().as_secs_f64() * 1000.0;
        r.wait();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        lag = lag.max(r.deforming_lag());
        if let Some(c) = r.cache_stats() {
            fallback += c.fallback_pixels as f64 / (W * H) as f64;
            if std::env::var_os("M12_TRACE").is_some() && k % 10 == 5 {
                eprintln!(
                    "{} frame {k}: {:.1} ms (call {call:.1}); lag {}; baked {} pages {} texels, rebaked {} pages {} texel-lights of {} candidates, moved {}, budget {}, gpu bake {:.1} render {:.1}",
                    if deforming { "deforming" } else { "still" },
                    times[k],
                    r.deforming_lag(),
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
        lag,
    }
}

/// The pixels of the blob's surroundings: its bounding sphere's (radius
/// 1.25 m) screen box widened by 1.5 radii.
fn surroundings(s: &Setup) -> Vec<bool> {
    let c = s.camera;
    let centre = s.camera.target;
    let forward = (c.target - c.eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let up = right.cross(forward);
    let tz = (c.vertical_fov_degrees.to_radians() * 0.5).tan();
    let aspect = W as f32 / H as f32;
    let d = (centre - c.eye).dot(forward);
    let r = 2.5 * 1.25;
    let px = |v: f32| v / d / (tz * aspect) * 0.5 * W as f32;
    let py = |v: f32| v / d / tz * 0.5 * H as f32;
    let (cx, cy) = (
        (1.0 + (centre - c.eye).dot(right) / d / (tz * aspect)) * 0.5 * W as f32,
        (1.0 - (centre - c.eye).dot(up) / d / tz) * 0.5 * H as f32,
    );
    let (hx, hy) = (px(r), py(r));
    (0..W * H)
        .map(|i| {
            let (x, y) = ((i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
            (x - cx).abs() <= hx && (y - cy).abs() <= hy
        })
        .collect()
}

fn pick(px: &[[f32; 3]], mask: &[bool]) -> Vec<[f32; 3]> {
    px.iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(p, _)| *p)
        .collect()
}

/// The bar of criterion 3 (provisional: development on another view read
/// 0.15–2.86%, and 2.6–4.7% on the gate's view on 10,000 boulders).
const BAR: f64 = 0.05;

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
    let dev = std::env::var("M12_DEV").ok();
    let (boulders, frames, rounds, ahead) = match dev.as_deref() {
        Some("1") => (10_000, 100, 1, 10.0),
        Some(_) => (1_000_000, 300, 1, 14.0),
        None => (1_000_000, 600, 3, 10.0),
    };
    let s = setup(boulders, ahead);
    // Development (`M12_COST=1`): one deforming run, traced, alone.
    if std::env::var_os("M12_COST").is_some() {
        let r = run(&kernel, &s, frames, true);
        println!(
            "deforming: mean {:.2} ms, p99 {:.2}, lag {}",
            r.mean, r.p99, r.lag
        );
        return Ok(());
    }
    // Development (`M12_ALIGN=1`): one deforming run, each frame's time
    // beside its own statistics (they arrive two frames on), the slowest
    // printed.
    if std::env::var_os("M12_ALIGN").is_some() {
        let view = target(&kernel, W, H);
        let mut r = renderer(&kernel, &s, (W, H), Some(CacheOptions::default()));
        let mut times = Vec::new();
        let mut stats = std::collections::BTreeMap::new();
        for k in 0..frames {
            pose(&mut r, &s, k);
            let t = Instant::now();
            r.frame(s.camera, &view);
            r.wait();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let c = r.cache_stats().expect("stats");
            stats.insert(c.frame, c);
            times.push((k, ms));
            if k % 25 == 0 {
                println!(
                    "call {k}: slices {}, lag {}",
                    r.deforming_slices(),
                    r.deforming_lag()
                );
            }
        }
        // The cache frame a call renders: the first statistics' frame plus
        // the call's index (statistics lag by a fixed number of frames).
        let first = stats.keys().next().copied().unwrap_or(0);
        let lag_calls = times.len() as u32 - stats.len() as u32;
        println!("statistics from frame {first}, {} calls without", lag_calls);
        let mut slow = times.clone();
        slow.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(k, ms) in slow.iter().take(12) {
            let f = first + k as u32;
            let c = stats.get(&f);
            println!(
                "call {k} (cache frame {f}): {ms:.1} ms; baked pages {:?}, texels {:?}, rebaked pages {:?}, texel-lights {:?}, candidates {:?}, budget {:?}",
                c.map(|c| c.pages_baked),
                c.map(|c| c.texels_baked),
                c.map(|c| c.rebaked_pages),
                c.map(|c| c.rebaked_texels),
                c.map(|c| c.candidates),
                c.map(|c| c.budget),
            );
        }
        let mean = times.iter().map(|t| t.1).sum::<f64>() / times.len() as f64;
        println!("mean {mean:.2} ms");
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
    // 1. Cost, first.
    let mut deforming = Vec::new();
    let mut still = Vec::new();
    for _ in 0..rounds {
        deforming.push(run(&kernel, &s, frames, true));
        still.push(run(&kernel, &s, frames, false));
    }
    let still_mean = still.iter().map(|r| r.mean).sum::<f64>() / rounds as f64;
    let still_p99 = still.iter().map(|r| r.p99).sum::<f64>() / rounds as f64;
    for (k, d) in deforming.iter().enumerate() {
        let c1 = d.mean <= 1.25 * still_mean
            && d.mean <= 16.7
            && d.p99 <= (1.25 * still_p99).max(33.3)
            && d.fallback <= 0.1
            && d.lag <= 16;
        pass &= c1;
        say(format!(
            "1. deforming run {}: mean {:.2} ms ({:.3}× the still runs' {:.2} ms; at most 1.25× and 16.7 ms), 99th percentile {:.2} ms (at most {:.2}), fallback {:.2}% (at most 10%), most lag {} frames (at most 16): {}",
            k + 1,
            d.mean,
            d.mean / still_mean,
            still_mean,
            d.p99,
            (1.25 * still_p99).max(33.3),
            100.0 * d.fallback,
            d.lag,
            verdict(c1)
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
    let view = target(&kernel, W, H);
    // 3. Error while deforming, then 2. caught up (the lagging renderer
    // held still after).
    {
        let mask = surroundings(&s);
        let mut lagging = renderer(&kernel, &s, (W, H), Some(CacheOptions::default()));
        let mut exact = renderer(
            &kernel,
            &s,
            (W, H),
            Some(CacheOptions {
                deforming_budget: 0,
                ..CacheOptions::default()
            }),
        );
        let checks = if dev.as_deref() == Some("1") {
            [50usize, 75, 100]
        } else {
            [100, 200, 300]
        };
        let last = checks[2];
        let mut worst: f64 = 0.0;
        for k in 1..=last {
            pose(&mut lagging, &s, k);
            pose(&mut exact, &s, k);
            lagging.frame(s.camera, &view);
            lagging.wait();
            exact.frame(s.camera, &view);
            exact.wait();
            if checks.contains(&k) {
                lagging.finish();
                exact.finish();
                let (a, b) = (lagging.read_colors(), exact.read_colors());
                let near = relative_rmse(&pick(&a, &mask), &pick(&b, &mask));
                worst = worst.max(near);
                say(format!(
                    "   frame {k}: the blob's surroundings ({} px) {:.3}% from exact, the whole image {:.3}%; lag {} frames",
                    mask.iter().filter(|m| **m).count(),
                    100.0 * near,
                    100.0 * relative_rmse(&a, &b),
                    lagging.deforming_lag()
                ));
            }
        }
        let c3 = worst <= BAR;
        pass &= c3;
        say(format!(
            "3. error while deforming: at most {:.3}% from exact over the surroundings (at most {:.3}%): {}",
            100.0 * worst,
            100.0 * BAR,
            verdict(c3)
        ));
        drop(exact);
        for _ in 0..32 {
            lagging.frame(s.camera, &view);
            lagging.wait();
        }
        let c = lagging.verify_cache();
        let c2 = c.differing == 0;
        pass &= c2;
        say(format!(
            "2. caught up: 32 frames held at frame {last}'s pose, lag {}, {} of {} texel values differing (none allowed), largest change {:e}: {}",
            lagging.deforming_lag(),
            c.differing,
            c.values,
            c.largest,
            verdict(c2)
        ));
    }
    // 4. Exact still exact.
    {
        let mut r = renderer(
            &kernel,
            &s,
            (W, H),
            Some(CacheOptions {
                deforming_budget: 0,
                ..CacheOptions::default()
            }),
        );
        let mut differing = 0;
        for k in 0..100 {
            pose(&mut r, &s, k);
            r.frame(s.camera, &view);
            r.wait();
            if (k + 1) % 10 == 0 {
                differing = differing.max(r.verify_cache().differing);
            }
        }
        let c4 = differing == 0;
        pass &= c4;
        say(format!(
            "4. exact still exact: 10 checks, at most {differing} texel values differing (none allowed): {}",
            verdict(c4)
        ));
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(
        out_dir.join(match dev.as_deref() {
            Some(_) => "m12_gate2_dev.txt",
            None => "m12_gate2.txt",
        }),
        report,
    )?;
    Ok(())
}
