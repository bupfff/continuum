//! M13 gate 13: the cache-lit sub-pixel path at the cost of gate 11's.
//!
//! Gate 12 brought the sub-pixel path under the renderer's lighting cache
//! (every light's area, the sky, bounce light): colour passed (0.435× and
//! 0.414× one ray's error against the renderer's own supersampled shading)
//! and speed failed (17.80 and 18.77 ms against 16.7) on a machine running
//! 8–16% slow; gate 11's path (its own lighting model, no cache) timed after
//! the run in the same state read 15.82 and 16.96 ms, so the cache's
//! lighting cost about 2 ms (1.11–1.13×; `docs/results/m13-gate12.md`).
//! Most of it is lighting the large hits' records with the cache's full
//! shading, about a million a frame (a corner each, then bins of objects no
//! corner shows and the samples of pixels the corners resolved).
//!
//! This gate asks for the renderer's lighting at little more than the
//! model's cost, and measures that against gate 11's path in the same
//! process, so the machine's state (external displays, heat) divides out.
//!
//! Method as developed (gate 12's, changed before the first run;
//! `docs/results/m13-gate13-development.md`):
//! - The corners' records lit by the cache write their colours and shares
//!   to the corners' words in the lighting pass (its 29th buffer); the
//!   others keep a scatter pass (their targets need two buffers). Records
//!   are plain words with their count apart.
//! - A resolved pixel's quarter sample is lit from its corners' cached
//!   shares (bilinearly, with its own point, normal and albedo, as bins
//!   are) where the corners showing its object agree on the light's share
//!   within 0.02; else lit in full by the cache (shadow edges).
//! - The rect light's unshadowed light (the renderer's own shading, every
//!   path): its unclipped case (the whole light above the point's horizon)
//!   summed unrolled with each corner's direction normalized once, without
//!   arrays indexed at run time; the same sums in the same order.
//!
//! Scene: the million-boulder field with its ground, its light and a sky,
//! 1280×720; gate 12's views (standing at 1.7 m, 150 m up looking 45°
//! down).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Colour:** as gate 12's: relative RMSE of the RGB image from the
//!    reference at most 0.5× one ray's, in both views (the reference: per
//!    pixel the mean of 16×16 samples, each one ray shaded by the
//!    renderer's own shading with its cache settled; one ray: the centre
//!    sample shaded the same way).
//! 2. **Cost:** the cache-lit path's settled frame at most 1.05× gate 11's
//!    path's (the same passes lit by its model, no cache) in both views,
//!    both settled in one process and timed in five interleaved rounds
//!    (each round, lit then model: 20 frames to warm up, then the median of
//!    21 frames each waited for); the criterion on the median of the
//!    rounds' ratios. Timed first.
//!
//! Amended before any run: the stub's rounds were the median of 11 frames
//! with no warm-up; development found the GPU's clock takes about nine
//! frames to step after a switch between the renderers, so 11 frames
//! measured the switch.
//!
//! Disclosed before the run: developed at full size on the gate's scene,
//! speed on the gate's views, accuracy on other views (0.422× and 0.426×).
//! The ratio in development varied from run to run by about ±0.02 (the
//! machine with external displays, warm): standing 1.04–1.07×, from 150 m
//! 1.00–1.04×; criterion 2 standing is near its bar.
//!
//! Reported: both paths' frame times (medians over rounds) with the
//! machine's control taken just before (gate 4's path); the 20 m view
//! settled; the renderer's own path (one ray a pixel, the cache).
//!
//! `cargo run --release -p room3d --bin m13_gate13`
//!
//! `M13_DEV=1M` runs development checks at full size: `M13_SPEED=1` the
//! interleaved ratio on the gate's views (three rounds; `M13_VIEW=standing`
//! or `150` for one); `M13_ACC=1` colour on other views (standing turned
//! 20°, 120 m up at 40° turned −20°).

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
/// Coverage pages baked a frame while settling (gates 11–12's).
const PAGE_BUDGET: u32 = 512;
/// A round of the interleaved timing: frames run to warm up, then frames
/// timed (their median).
const WARM: usize = 20;
const ROUND: usize = 21;

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

/// A renderer with the lighting cache: the sub-pixel path (`sub`, profile
/// stage 20 with pixel cones) or one ray a pixel (gate 12's).
fn cached(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    sub: bool,
) -> InstancedRenderer {
    kernel.set_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_profile_stage(if sub { 20 } else { 0 });
    kernel.set_pixel_cones(sub);
    let r = InstancedRenderer::with_cache_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        lights,
        &field.scene,
        origin,
        // A fixed bake budget for the sub-pixel path (gate 12's): its camera
        // passes alone take most of a frame target.
        if sub {
            CacheOptions {
                frame_target_ms: 0.0,
                frame_budget_ms: 0.0,
                ..CacheOptions::default()
            }
        } else {
            CacheOptions::default()
        },
    );
    kernel.set_pixel_cones(false);
    kernel.set_profile_stage(0);
    r
}

/// Gate 11's path: the same sub-pixel passes lit by its model (one sun, a
/// constant ambient, the sun's visibility cached in world space), no
/// lighting cache and no sky.
fn model(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
) -> InstancedRenderer {
    let sky = kernel.sky();
    kernel.set_sky(Vec3::ZERO);
    kernel.set_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_profile_stage(20);
    kernel.set_pixel_cones(true);
    let r = InstancedRenderer::new_lights(
        kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        1,
        lights,
        &field.scene,
        origin,
    );
    kernel.set_pixel_cones(false);
    kernel.set_profile_stage(0);
    kernel.set_sky(sky);
    r
}

/// Frames until `quiet` frames in a row ask for and bake no cache page and
/// ask for no coverage page (at most `max`): the frames run (gate 12's).
fn settle_cached(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    quiet: usize,
    max: usize,
) -> usize {
    let mut calm = 0;
    for k in 1..=max {
        r.frame(camera, target);
        r.wait();
        let s = r.cache_stats();
        let pages = r.m13_page_counters()[6];
        if s.is_some_and(|s| s.requests == 0 && s.pages_baked == 0) && pages == 0 {
            calm += 1;
            if calm >= quiet {
                return k;
            }
        } else {
            calm = 0;
        }
    }
    max
}

/// Frames until a frame asks for no coverage page and misses the sun's
/// cache nowhere (at most `max`): the frames run (gate 11's).
fn settle_model(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    max: usize,
) -> usize {
    for k in 1..=max {
        r.frame(camera, target);
        r.wait();
        if r.m13_page_counters()[6] == 0 && r.m13_work_counters()[1] == 0 {
            return k;
        }
    }
    max
}

fn frame(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Vec<[f32; 3]> {
    r.frame(camera, target);
    r.finish();
    r.read_colors()
}

/// The median of `n` frames each waited for, in milliseconds.
fn median_ms(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    n: usize,
) -> f64 {
    let mut t: Vec<f64> = (0..n)
        .map(|_| {
            let s = Instant::now();
            r.frame(camera, target);
            r.wait();
            s.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[n / 2]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The two paths settled at `camera`, then `rounds` interleaved rounds (lit
/// then model: WARM frames, then the median of ROUND frames each): the
/// rounds' lit and model medians, and the frames each took to settle.
#[allow(clippy::type_complexity)]
fn interleaved(
    lit: &mut InstancedRenderer,
    model: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    rounds: usize,
) -> (Vec<(f64, f64)>, usize, usize) {
    let n_lit = settle_cached(lit, camera, target, 8, 3000);
    let n_model = settle_model(model, camera, target, 200);
    let times = (0..rounds)
        .map(|_| {
            // (Frames to warm up first: after a switch the GPU's clock takes
            // about nine frames to step, development.)
            for _ in 0..WARM {
                lit.frame(camera, target);
                lit.wait();
            }
            let a = median_ms(lit, camera, target, ROUND);
            for _ in 0..WARM {
                model.frame(camera, target);
                model.wait();
            }
            let b = median_ms(model, camera, target, ROUND);
            (a, b)
        })
        .collect();
    (times, n_lit, n_model)
}

struct Views {
    ground: Camera,
    flat0: Vec3,
}

impl Views {
    fn new(ground: Camera) -> Self {
        let forward = (ground.target - ground.eye).normalize();
        Views {
            ground,
            flat0: Vec3::new(forward.x, forward.y, 0.0).normalize(),
        }
    }

    fn at(&self, height: f32, down_degrees: f32, turn_degrees: f32) -> Camera {
        let d = down_degrees.to_radians();
        let (s, c) = turn_degrees.to_radians().sin_cos();
        let f = self.flat0;
        let flat = Vec3::new(f.x * c - f.y * s, f.x * s + f.y * c, 0.0);
        let eye = Vec3::new(self.ground.eye.x, self.ground.eye.y, height);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: self.ground.vertical_fov_degrees,
        }
    }
}

/// The reference (16×16 samples a pixel, shaded by the renderer's own
/// path with its cache settled) and one ray (the centre sample), and the
/// renderer's own path's settled frame and the frames it took to settle.
fn reference(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, f64, usize) {
    let mut one = cached(kernel, field, origin, lights, false);
    let frames = settle_cached(&mut one, camera, target, 8, 3000);
    let t = median_ms(&mut one, camera, target, 11);
    let single = frame(&mut one, camera, target);
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    for s in 0..256 {
        let (dx, dy) = (
            ((s % 16) as f32 + 0.5) / 16.0 - 0.5,
            ((s / 16) as f32 + 0.5) / 16.0 - 0.5,
        );
        one.set_pixel_offset(dx, dy);
        for (r, p) in sum.iter_mut().zip(frame(&mut one, camera, target)) {
            for (v, c) in r.iter_mut().zip(p) {
                *v += c / 256.0;
            }
        }
    }
    (sum, single, t, frames)
}

fn mean(px: &[[f32; 3]]) -> [f64; 3] {
    let n = px.len() as f64;
    [0, 1, 2].map(|c| px.iter().map(|p| p[c] as f64).sum::<f64>() / n)
}

fn save(out_dir: &Path, name: &str, px: &[[f32; 3]], scale: f32) {
    let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in px.iter().enumerate() {
        let c = p.map(|v| ((v / scale).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
        image.set(i % W as usize, i / W as usize, c);
    }
    image
        .save(&out_dir.join(format!("m13_gate13_{name}.png")))
        .expect("save");
}

fn scale_of(px: &[[f32; 3]]) -> f32 {
    let mut l: Vec<f32> = px.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
    l.sort_by(f32::total_cmp);
    l[l.len() * 99 / 100].max(1e-6)
}

fn main() {
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return;
    };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir).expect("out");
    let dev = std::env::var("M13_DEV").ok();
    let field = Field::new(
        if dev.as_deref() == Some("1") {
            10_000
        } else {
            1_000_000
        },
        DVec3::ZERO,
    );
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let lights = [Light::white(light, field.radiance)];
    kernel.set_sky(field.sky());
    let view_target = target(&kernel);
    let views = Views::new(ground);
    let gate_views = [
        ("standing", views.at(1.7, 0.0, 0.0)),
        ("150 m up", views.at(150.0, 45.0, 0.0)),
    ];
    if dev.is_some() {
        let only = std::env::var("M13_VIEW").ok();
        if std::env::var_os("M13_SPEED").is_some() {
            for (name, camera) in gate_views {
                if only.as_ref().is_some_and(|v| !name.starts_with(v.as_str())) {
                    continue;
                }
                let mut lit = cached(&kernel, &field, origin, &lights, true);
                lit.set_m13_page_budget(PAGE_BUDGET);
                let mut m = model(&kernel, &field, origin, &lights);
                m.set_m13_page_budget(PAGE_BUDGET);
                let (rounds, n_lit, n_model) =
                    interleaved(&mut lit, &mut m, camera, &view_target, 3);
                // Where the lit frame's time goes: the GPU's camera passes
                // (timestamps), the cache's planning, waiting (11 frames).
                {
                    let mut g = Vec::new();
                    let mut plan = Vec::new();
                    let mut wait = Vec::new();
                    let mut wall = Vec::new();
                    for _ in 0..11 {
                        let s = Instant::now();
                        lit.frame(camera, &view_target);
                        lit.wait();
                        wall.push(s.elapsed().as_secs_f64() * 1000.0);
                        if let Some(c) = lit.cache_stats() {
                            g.push(c.gpu_render_ms as f64 + c.gpu_bake_ms as f64);
                            plan.push(c.planning.as_secs_f64() * 1000.0);
                            wait.push(c.waiting.as_secs_f64() * 1000.0);
                        }
                    }
                    let mut g2 = Vec::new();
                    for _ in 0..11 {
                        let s = Instant::now();
                        m.frame(camera, &view_target);
                        m.wait();
                        g2.push(s.elapsed().as_secs_f64() * 1000.0);
                    }
                    println!(
                        "{name}: lit wall {:.2}, GPU (timestamps) {:.2}, planning {:.2}, waiting {:.2} ms; model wall {:.2} ms",
                        median(wall),
                        median(g),
                        median(plan),
                        median(wait),
                        median(g2)
                    );
                }
                // `M13_ALT=1`: the paths alternating frame by frame (100
                // pairs): the median ratio of adjacent frames.
                if std::env::var_os("M13_ALT").is_some() {
                    let mut ratios = Vec::new();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    for _ in 0..100 {
                        let s = Instant::now();
                        lit.frame(camera, &view_target);
                        lit.wait();
                        let tl = s.elapsed().as_secs_f64() * 1000.0;
                        let s = Instant::now();
                        m.frame(camera, &view_target);
                        m.wait();
                        let tm = s.elapsed().as_secs_f64() * 1000.0;
                        ratios.push(tl / tm);
                        a.push(tl);
                        b.push(tm);
                    }
                    println!(
                        "{name}: alternating: lit {:.2} ms, model {:.2} ms, median ratio {:.3}×",
                        median(a),
                        median(b),
                        median(ratios)
                    );
                }
                // `M13_SEQ=1`: 60 frames of each in order.
                if std::env::var_os("M13_SEQ").is_some() {
                    // Lit after lit, after a pause, then after the model.
                    let seq = |r: &mut InstancedRenderer, n: usize| -> Vec<f64> {
                        (0..n)
                            .map(|_| {
                                let s = Instant::now();
                                r.frame(camera, &view_target);
                                r.wait();
                                (s.elapsed().as_secs_f64() * 10000.0).round() / 10.0
                            })
                            .collect()
                    };
                    println!("{name}: lit after lit {:?}", seq(&mut lit, 20));
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    println!("{name}: lit after a pause {:?}", seq(&mut lit, 20));
                    seq(&mut m, 30);
                    println!("{name}: lit after the model {:?}", seq(&mut lit, 20));
                    for (label, r) in [("lit", &mut lit), ("model", &mut m)] {
                        let t: Vec<f64> = (0..60)
                            .map(|_| {
                                let s = Instant::now();
                                r.frame(camera, &view_target);
                                r.wait();
                                (s.elapsed().as_secs_f64() * 10000.0).round() / 10.0
                            })
                            .collect();
                        println!("{name}: {label} frames {t:?}");
                    }
                }
                println!(
                    "{name}: pixels listed for samples: lit {}, model {}; pixels tracing bin rays: lit {}, model {}",
                    lit.m13_page_counters()[31],
                    m.m13_page_counters()[31],
                    lit.m13_work_counters()[0],
                    m.m13_work_counters()[0]
                );
                let hits = lit.m13_hit_counts();
                // The records by pass, target and where their rays went.
                {
                    let words = lit.m13_hit_records();
                    let mut tally = std::collections::BTreeMap::new();
                    for (i, r) in words.as_chunks::<5>().0.iter().enumerate() {
                        let pass = if (i as u32) < hits[1] { "a" } else { "b" };
                        let to = ["corner", "bin", "quad", "region"][(r[4] >> 28) as usize & 3];
                        let at = ["corner", "bin", "quarter", "centre"][(r[3] >> 30) as usize];
                        *tally.entry((pass, to, at)).or_insert(0u32) += 1;
                    }
                    println!("{name}: records by pass, target, ray: {tally:?}");
                }
                println!(
                    "{name} (development): lit {:?} ms, model {:?} ms; ratio {:.3}× (median of {} rounds); settled in {n_lit} and {n_model} frames; records {} ({} after the corners)",
                    rounds
                        .iter()
                        .map(|r| (r.0 * 100.0).round() / 100.0)
                        .collect::<Vec<_>>(),
                    rounds
                        .iter()
                        .map(|r| (r.1 * 100.0).round() / 100.0)
                        .collect::<Vec<_>>(),
                    median(rounds.iter().map(|r| r.0 / r.1).collect()),
                    rounds.len(),
                    hits[0],
                    hits[0].saturating_sub(hits[1])
                );
            }
        }
        if std::env::var_os("M13_ACC").is_some() {
            for (name, camera) in [
                ("standing_dev", views.at(1.7, 0.0, 20.0)),
                ("120m_dev", views.at(120.0, 40.0, -20.0)),
            ] {
                if only.as_ref().is_some_and(|v| !name.starts_with(v.as_str())) {
                    continue;
                }
                let mut lit = cached(&kernel, &field, origin, &lights, true);
                lit.set_m13_page_budget(PAGE_BUDGET);
                settle_cached(&mut lit, camera, &view_target, 8, 3000);
                let ours = frame(&mut lit, camera, &view_target);
                drop(lit);
                let (reference, single, _, _) =
                    reference(&kernel, &field, origin, &lights, camera, &view_target);
                let (e_ours, e_one) = (
                    relative_rmse(&ours, &reference),
                    relative_rmse(&single, &reference),
                );
                println!(
                    "{name} (development): colour {:.2}% from the reference, one ray {:.2}% ({:.3}×); means reference {:?}, ours {:?}",
                    100.0 * e_ours,
                    100.0 * e_one,
                    e_ours / e_one,
                    mean(&reference),
                    mean(&ours)
                );
                let scale = scale_of(&reference);
                save(&out_dir, &format!("{name}_ours"), &ours, scale);
                save(&out_dir, &format!("{name}_reference"), &reference, scale);
            }
        }
        return;
    }
    gate(
        &kernel,
        &field,
        origin,
        &lights,
        &views,
        gate_views,
        &view_target,
        &out_dir,
    );
}

#[allow(clippy::too_many_arguments)]
fn gate(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    views: &Views,
    gate_views: [(&str, Camera); 2],
    view_target: &wgpu::TextureView,
    out_dir: &Path,
) {
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    let mut pass = true;
    // 2. Cost, first; the lit images kept for criterion 1.
    let mut ours = Vec::new();
    for (name, camera) in gate_views {
        let mut lit = cached(kernel, field, origin, lights, true);
        lit.set_m13_page_budget(PAGE_BUDGET);
        let mut m = model(kernel, field, origin, lights);
        m.set_m13_page_budget(PAGE_BUDGET);
        let (rounds, n_lit, n_model) = interleaved(&mut lit, &mut m, camera, view_target, 5);
        let ratio = median(rounds.iter().map(|r| r.0 / r.1).collect());
        let c2 = ratio <= 1.05 && n_lit < 3000 && n_model < 200;
        pass &= c2;
        say(format!(
            "{name}: 2. the cache-lit path {:.3}× gate 11's (at most 1.05×): {}; lit {:.2} ms, model {:.2} ms (medians of 5 rounds; rounds' ratios {:?}); settled in {n_lit} and {n_model} frames",
            ratio,
            verdict(c2),
            median(rounds.iter().map(|r| r.0).collect()),
            median(rounds.iter().map(|r| r.1).collect()),
            rounds
                .iter()
                .map(|r| (r.0 / r.1 * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        ));
        drop(m);
        ours.push(frame(&mut lit, camera, view_target));
    }
    {
        let camera = views.at(20.0, 10.0, 0.0);
        let mut lit = cached(kernel, field, origin, lights, true);
        lit.set_m13_page_budget(PAGE_BUDGET);
        let frames = settle_cached(&mut lit, camera, view_target, 8, 3000);
        let t = median_ms(&mut lit, camera, view_target, 11);
        say(format!(
            "20 m up, 10° down (reported): the cache-lit path's settled frame {t:.2} ms; settled in {frames} frames"
        ));
    }
    // 1. Colour, and the renderer's own path's frame (reported).
    for (k, (name, camera)) in gate_views.into_iter().enumerate() {
        let (reference, single, t_one, frames) =
            reference(kernel, field, origin, lights, camera, view_target);
        let e_ours = relative_rmse(&ours[k], &reference);
        let e_one = relative_rmse(&single, &reference);
        let c1 = e_ours <= 0.5 * e_one;
        pass &= c1;
        let (mr, mo) = (mean(&reference), mean(&ours[k]));
        say(format!(
            "{name}: 1. colour {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most 0.5×): {}; mean colour: reference ({:.4}, {:.4}, {:.4}), ours ({:.4}, {:.4}, {:.4})",
            100.0 * e_ours,
            100.0 * e_one,
            e_ours / e_one,
            verdict(c1),
            mr[0],
            mr[1],
            mr[2],
            mo[0],
            mo[1],
            mo[2]
        ));
        say(format!(
            "{name}: the renderer's own path (one ray a pixel, the cache) settled in {frames} frames, {t_one:.2} ms (reported)"
        ));
        let slug = name.replace(' ', "_");
        let scale = scale_of(&reference);
        save(out_dir, &format!("{slug}_reference"), &reference, scale);
        save(out_dir, &format!("{slug}_ours"), &ours[k], scale);
        save(out_dir, &format!("{slug}_one_ray"), &single, scale);
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate13.txt"), report).expect("report");
}
