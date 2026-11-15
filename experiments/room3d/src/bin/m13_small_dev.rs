//! M13 development (not a gate): where the sub-pixel path's colour error
//! lies by how much of each pixel small instances cover, against the
//! renderer's own supersampled shading (DESIGN §3.4, "Into the renderer",
//! part C: small instances' lighting measured before it is designed).
//!
//! Per view, pixels are split by the share small entries cover (state word
//! 28 = 2 writes it): none (under 5%), some (5–50%), most (50–95%), all
//! (95% or more). Per class: our error and one ray's (relative RMSE from
//! the reference), their ratio, our mean over the reference's, and the
//! class's share of our squared error.
//!
//! `cargo run --release -p room3d --bin m13_small_dev` (full size; `M13_DEV=1`
//! on 10,000 boulders). `M13_LIGHTS=4` uses gate 14's four lights.

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, RectLight, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
/// Coverage pages baked a frame while settling (gates 11–13's).
const PAGE_BUDGET: u32 = 512;

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
/// stage 20 with pixel cones, a fixed bake budget) or one ray a pixel
/// (gates 12–13's).
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
    if sub {
        r.set_m13_page_budget(PAGE_BUDGET);
    }
    r
}

/// Frames until `quiet` frames in a row ask for and bake no cache page and
/// ask for no coverage page (at most `max`): the frames run.
fn settle(
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

fn frame(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> Vec<[f32; 3]> {
    r.frame(camera, target);
    r.finish();
    r.read_colors()
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

    /// The camera's line turned by `turn_degrees` (positive: left) at
    /// `height`, looking `down_degrees` down (negative: up).
    fn at(&self, height: f32, down_degrees: f32, turn_degrees: f32) -> Camera {
        let d = down_degrees.to_radians();
        let eye = Vec3::new(self.ground.eye.x, self.ground.eye.y, height);
        Camera {
            eye,
            target: eye + self.flat(turn_degrees) * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: self.ground.vertical_fov_degrees,
        }
    }

    fn flat(&self, turn_degrees: f32) -> Vec3 {
        let (s, c) = turn_degrees.to_radians().sin_cos();
        let f = self.flat0;
        Vec3::new(f.x * c - f.y * s, f.x * s + f.y * c, 0.0)
    }
}

/// The gate's four lights relative to `origin` (see the doc comment): the
/// field's sun first.
fn gate_lights(field: &Field, origin: DVec3, views: &Views) -> Vec<Light> {
    let (_, sun) = field.relative(origin);
    let main = Light::white(sun, field.radiance);
    // The low sun: 0.5° across, centre 0.3° up, 5° right, 5 km from the
    // eye, facing it.
    let distance = 5000.0f32;
    // (Development, `M13_SUNSET=1`: its centre on the eye's level, so the
    // ground's far edge cuts it.)
    let e = if std::env::var_os("M13_SUNSET").is_some() {
        0.0f32
    } else {
        0.3f32.to_radians()
    };
    let toward = views.flat(-5.0) * e.cos() + Vec3::Z * e.sin();
    let size = 2.0 * distance * 0.25f32.to_radians().tan();
    let w = -toward;
    let a = Vec3::Z.cross(w).normalize() * size;
    let b = w.cross(a).normalize() * size;
    let ground_eye = Vec3::new(views.ground.eye.x, views.ground.eye.y, 0.0);
    let middle = ground_eye + toward * distance;
    let tint = Vec3::new(1.0, 0.6, 0.35);
    let low = Light::rect(
        RectLight {
            corner: middle - (a + b) * 0.5,
            edge_u: a,
            edge_v: b,
            one_sided: true,
        },
        tint / (tint.element_sum() / 3.0) * 10000.0,
    );
    let lamp = field.colored_lights(origin, 1000.0)[2];
    let point = field.sphere_lights(origin)[0];
    vec![main, low, lamp, point]
}

/// The reference (16×16 samples a pixel, shaded by the renderer's own
/// path with its cache settled) and one ray (the centre sample), and the
/// frames the renderer's own path took to settle.
fn reference(
    kernel: &MaskConeKernel,
    field: &Field,
    origin: DVec3,
    lights: &[Light],
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, usize) {
    let mut one = cached(kernel, field, origin, lights, false);
    let frames = settle(&mut one, camera, target, 8, 3000);
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
    (sum, single, frames)
}

fn mean(px: &[[f32; 3]]) -> [f64; 3] {
    let n = px.len().max(1) as f64;
    [0, 1, 2].map(|c| px.iter().map(|p| p[c] as f64).sum::<f64>() / n)
}

fn main() {
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return;
    };
    let small = std::env::var("M13_DEV").ok().as_deref() == Some("1");
    let field = Field::new(if small { 10_000 } else { 1_000_000 }, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    kernel.set_sky(field.sky());
    let views = Views::new(ground);
    let lights = if std::env::var("M13_LIGHTS").ok().as_deref() == Some("4") {
        gate_lights(&field, origin, &views)
    } else {
        vec![Light::white(sun, field.radiance)]
    };
    let view_target = target(&kernel);
    let only = std::env::var("M13_VIEW").ok();
    for (name, camera) in [
        ("standing", views.at(1.7, 0.0, 0.0)),
        ("150 m up", views.at(150.0, 45.0, 0.0)),
        ("20 m up", views.at(20.0, 10.0, 0.0)),
    ] {
        if only.as_ref().is_some_and(|v| !name.starts_with(v.as_str())) {
            continue;
        }
        let t0 = Instant::now();
        let mut lit = cached(&kernel, &field, origin, &lights, true);
        let n = settle(&mut lit, camera, &view_target, 8, 3000);
        let ours = frame(&mut lit, camera, &view_target);
        lit.set_m13_state_word(28, 2);
        let class = frame(&mut lit, camera, &view_target);
        lit.set_m13_state_word(28, 0);
        drop(lit);
        let (reference, single, n_one) =
            reference(&kernel, &field, origin, &lights, camera, &view_target);
        let total: f64 = ours
            .iter()
            .zip(&reference)
            .map(|(a, b)| (0..3).map(|c| ((a[c] - b[c]) as f64).powi(2)).sum::<f64>())
            .sum();
        println!(
            "{name}: whole image ours {:.2}%, one ray {:.2}% ({:.3}×); settled in {n} and {n_one} frames ({:.0} s)",
            100.0 * relative_rmse(&ours, &reference),
            100.0 * relative_rmse(&single, &reference),
            relative_rmse(&ours, &reference) / relative_rmse(&single, &reference),
            t0.elapsed().as_secs_f64()
        );
        for (label, lo, hi) in [
            ("none", 0.0, 0.05),
            ("some", 0.05, 0.5),
            ("most", 0.5, 0.95),
            ("all", 0.95, 2.0),
        ] {
            let mask: Vec<bool> = class.iter().map(|c| c[0] >= lo && c[0] < hi).collect();
            let pick = |px: &[[f32; 3]]| -> Vec<[f32; 3]> {
                px.iter()
                    .zip(&mask)
                    .filter(|(_, m)| **m)
                    .map(|(p, _)| *p)
                    .collect()
            };
            let (o, s, r) = (pick(&ours), pick(&single), pick(&reference));
            if r.is_empty() {
                continue;
            }
            let sq: f64 = o
                .iter()
                .zip(&r)
                .map(|(a, b)| (0..3).map(|c| ((a[c] - b[c]) as f64).powi(2)).sum::<f64>())
                .sum();
            let (mo, mr) = (mean(&o), mean(&r));
            println!(
                "  {label:>4} ({:>6} px): ours {:.2}%, one ray {:.2}% ({:.3}×); mean ours/reference {:.4}; share of our squared error {:.1}%",
                r.len(),
                100.0 * relative_rmse(&o, &r),
                100.0 * relative_rmse(&s, &r),
                relative_rmse(&o, &r) / relative_rmse(&s, &r),
                (mo[0] + mo[1] + mo[2]) / (mr[0] + mr[1] + mr[2]),
                100.0 * sq / total
            );
        }
    }
}
