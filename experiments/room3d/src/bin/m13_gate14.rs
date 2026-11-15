//! M13 gate 14: lights seen directly on the sub-pixel path.
//!
//! The sub-pixel path (`RendererOptions::sub_pixel`, gates 5–13) colours a
//! pixel from its bins: small instances from coverage maps and pages in
//! front, then each bin's large object, the pixel's background or the sky.
//! A light seen directly is not drawn there (one ray a pixel draws it where
//! the pixel's centre sees it in front of its hit). Drawn on this path, a
//! light must cover the bins it covers (partial pixels at its edges) and
//! stay behind what lies in front of it, small instances included.
//!
//! Method (designed before the stub; DESIGN §3.4, "Lights seen directly"):
//! - In the pixels' pass, after the pixel's layers are known (its large
//!   objects' bins and distances, the background's 1/t across the pixel),
//!   each light whose bounding sphere the pixel's cone can meet is tested
//!   at the 16 bins' centres (`rect_distance`, the one-ray path's test). A
//!   bin sees the light where it is nearer than what the bin shows behind
//!   small entries (its large object, the background, else the sky).
//! - Small entries (map records, walked nodes, pages) behind the light in
//!   those bins are dropped (`g9_in_front`, by the nearest light's
//!   distance over the pixel); those in front cover it by their coverage.
//! - The resolve gives those bins the light's radiance (with a spot's
//!   falloff), in place of what lies behind it.
//!
//! Scene: the million-boulder field with its ground, sky and four lights:
//! the field's sun (30° up behind the camera, the light small instances
//! take), a low sun ahead (0.5° across, its centre 0.3° above the horizon,
//! 5° right of the camera's line, 5 km away), the lamp of M5 gate 6 (1.5 m
//! square, 6 m up, facing down) and the point light of M5 gate 8 (10 cm
//! across, 3 m up). 1280×720. Two views, both along the camera's line
//! looking 3° up: low (the eye 0.4 m up, so boulders 100 m–1.4 km away,
//! sub-pixel to a few pixels, stand against the low sun's lower part) and
//! standing (1.7 m).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Near the lights:** over the pixels within 2 px of a light's
//!    projected bounding box, the relative RMSE of the RGB image from the
//!    reference at most 0.5× one ray's, in both views (the reference: per
//!    pixel the mean of 16×16 samples, each one ray shaded by the renderer's
//!    own shading with its cache settled and lights drawn as that path
//!    draws them; one ray: the centre sample shaded the same way).
//! 2. **Cost:** the settled frame with lights drawn at most 1.03× the same
//!    renderer's with drawing off (state word 23), in both views, timed in
//!    five interleaved rounds (each round, drawn then off: 20 frames to warm
//!    up, then the median of 21 frames each waited for); the criterion on
//!    the median of the rounds' ratios. Timed first.
//!
//! Reported: the whole image's and the rest's colour error against one
//! ray's; the near-light pixels' mean colour against the reference's; the
//! frames to settle; gate 13's standing view's cost ratio (no light in
//! view).
//!
//! `cargo run --release -p room3d --bin m13_gate14`
//!
//! `M13_DEV=1` runs the checks on a field of 10,000 boulders, `M13_DEV=1M`
//! at full size on other views (low turned 3° left, standing looking 6°
//! up); `M13_VIEW=low` or `standing` for one.
//!
//! Development after the run (not part of the gate):
//! - `M13_SUNSET=1`: the low sun's centre on the eye's level.
//! - `M13_FLOOR=1`: the sun and a 40 m light panel 5 cm above the ground
//!   under the boulders, facing up (`M13_FLOOR_H`: its height;
//!   `M13_FLOOR_DOWN=1`: facing down), seen from 150 m and 20 m up.
//!   `M13_NO_PANEL=1` leaves the panel out.
//! - `M13_REF_ONLY=1`: only the one-ray path settles.
//! - `M13_LIGHTS_OFF=1`: drawing lights off.
//! - `M13_TRACE=1`: frame times while settling.
//!
//! The panel 5 cm up makes frames take seconds on both paths: see DESIGN
//! §8, "Wide light cones".

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, Emitter, InstancedRenderer, Light, MaskConeKernel, RectLight, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
/// Coverage pages baked a frame while settling (gates 11–13's).
const PAGE_BUDGET: u32 = 512;
/// A round of the interleaved timing: frames run to warm up, then frames
/// timed (their median).
const WARM: usize = 20;
const ROUND: usize = 21;
/// The pixels' pass's state word that turns drawing lights off (1).
const LIGHTS_OFF: u32 = 23;

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
    let trace = std::env::var_os("M13_TRACE").is_some();
    for k in 1..=max {
        let t = Instant::now();
        r.frame(camera, target);
        r.wait();
        if trace && (k <= 40 || k % 100 == 0) {
            let s = r.cache_stats();
            eprintln!(
                "frame {k}: {:.1} ms, baked {:?}, requests {:?}",
                t.elapsed().as_secs_f64() * 1000.0,
                s.map(|s| s.pages_baked),
                s.map(|s| s.requests)
            );
        }
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

/// `rounds` interleaved rounds on a settled renderer (lights drawn, then
/// off: WARM frames, then the median of ROUND frames each): the rounds'
/// medians, drawn and off.
fn interleaved(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    rounds: usize,
) -> Vec<(f64, f64)> {
    let times = (0..rounds)
        .map(|_| {
            let mut pair = [0.0; 2];
            for (off, t) in pair.iter_mut().enumerate() {
                r.set_m13_state_word(LIGHTS_OFF, off as u32);
                for _ in 0..WARM {
                    r.frame(camera, target);
                    r.wait();
                }
                *t = median_ms(r, camera, target, ROUND);
            }
            (pair[0], pair[1])
        })
        .collect();
    r.set_m13_state_word(LIGHTS_OFF, 0);
    times
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

/// Pixels within 2 px of a light's projected bounding box (a rectangle's
/// corners, a sphere's bounding cube), as a mask.
fn near_lights(lights: &[Light], camera: Camera) -> Vec<bool> {
    let mut mask = vec![false; (W * H) as usize];
    for [x0, y0, x1, y1] in light_boxes(lights, camera).into_iter().flatten() {
        for y in y0..=y1 {
            for x in x0..=x1 {
                mask[(y as u32 * W + x as u32) as usize] = true;
            }
        }
    }
    mask
}

/// Each light's projected bounding box widened by 2 px and clipped to the
/// screen (x0, y0, x1, y1 inclusive; none behind the eye or empty).
fn light_boxes(lights: &[Light], camera: Camera) -> Vec<Option<[i64; 4]>> {
    let forward = (camera.target - camera.eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let up = right.cross(forward);
    let tz = (camera.vertical_fov_degrees.to_radians() * 0.5).tan();
    let aspect = W as f32 / H as f32;
    let mut boxes = Vec::new();
    for l in lights {
        let points: Vec<Vec3> = match l.emitter {
            Emitter::Rect(r) => vec![
                r.corner,
                r.corner + r.edge_u,
                r.corner + r.edge_v,
                r.corner + r.edge_u + r.edge_v,
            ],
            Emitter::Sphere(s) => (0..8)
                .map(|k| {
                    s.center
                        + Vec3::new(
                            if k & 1 == 0 { -1.0 } else { 1.0 },
                            if k & 2 == 0 { -1.0 } else { 1.0 },
                            if k & 4 == 0 { -1.0 } else { 1.0 },
                        ) * s.radius
                })
                .collect(),
        };
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        let mut behind = false;
        for p in points {
            let d = p - camera.eye;
            let z = d.dot(forward);
            if z <= 0.0 {
                behind = true;
                break;
            }
            let sx = d.dot(right) / z;
            let sy = d.dot(up) / z;
            let x = (sx / (tz * aspect) + 1.0) * 0.5 * W as f32 - 0.5;
            let y = (1.0 - sy / tz) * 0.5 * H as f32 - 0.5;
            lo = [lo[0].min(x), lo[1].min(y)];
            hi = [hi[0].max(x), hi[1].max(y)];
        }
        let x0 = (lo[0] - 2.0).floor().max(0.0) as i64;
        let y0 = (lo[1] - 2.0).floor().max(0.0) as i64;
        let x1 = (hi[0] + 2.0).ceil().min(W as f32 - 1.0) as i64;
        let y1 = (hi[1] + 2.0).ceil().min(H as f32 - 1.0) as i64;
        boxes.push((!behind && x0 <= x1 && y0 <= y1).then_some([x0, y0, x1, y1]));
    }
    boxes
}

/// Development: per light in view, its box's mean colour (reference, ours,
/// one ray) and errors, and its crop (8× enlarged, scaled to its largest
/// reference value) saved as `<name>_light<i>_{reference,ours,one_ray}`.
fn per_light(
    out_dir: &Path,
    name: &str,
    lights: &[Light],
    camera: Camera,
    images: [&[[f32; 3]]; 3],
) {
    for (i, b) in light_boxes(lights, camera).into_iter().enumerate() {
        let Some([x0, y0, x1, y1]) = b else { continue };
        let (bw, bh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let crop = |px: &[[f32; 3]]| -> Vec<[f32; 3]> {
            (0..bw * bh)
                .map(|k| px[((y0 as usize + k / bw) * W as usize) + x0 as usize + k % bw])
                .collect()
        };
        let [r, o, s] = images.map(crop);
        let top = r.iter().flatten().fold(0.0f32, |a, &b| a.max(b)).max(1e-6);
        // Intensity-weighted centroids (pixels in the box).
        let centroid = |px: &[[f32; 3]]| {
            let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
            for (k, p) in px.iter().enumerate() {
                let v = (p[0] + p[1] + p[2]) as f64;
                sx += v * (k % bw) as f64;
                sy += v * (k / bw) as f64;
                sw += v;
            }
            (
                (sx / sw * 100.0).round() / 100.0,
                (sy / sw * 100.0).round() / 100.0,
            )
        };
        println!(
            "{name} light {i}: centroids reference {:?}, ours {:?}, one ray {:?}",
            centroid(&r),
            centroid(&o),
            centroid(&s)
        );
        println!(
            "{name} light {i} ({bw}×{bh} px): means reference {:.3?}, ours {:.3?}, one ray {:.3?}; error ours {:.2}%, one ray {:.2}%",
            mean(&r),
            mean(&o),
            mean(&s),
            100.0 * relative_rmse(&o, &r),
            100.0 * relative_rmse(&s, &r)
        );
        for (label, px) in [("reference", &r), ("ours", &o), ("one_ray", &s)] {
            let mut image = flatland::img::Image::new(8 * bw, 8 * bh, [0; 3]);
            for y in 0..8 * bh {
                for x in 0..8 * bw {
                    let p = px[(y / 8) * bw + x / 8];
                    image.set(
                        x,
                        y,
                        p.map(|v| {
                            ((v / top).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
                        }),
                    );
                }
            }
            image
                .save(&out_dir.join(format!("m13_gate14_{name}_light{i}_{label}.png")))
                .expect("save");
        }
    }
}

fn pick(px: &[[f32; 3]], mask: &[bool], inside: bool) -> Vec<[f32; 3]> {
    px.iter()
        .zip(mask)
        .filter(|(_, m)| **m == inside)
        .map(|(p, _)| *p)
        .collect()
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
    eprintln!("reference: the one-ray path settled in {frames} frames");
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

fn save(out_dir: &Path, name: &str, px: &[[f32; 3]], scale: f32) {
    let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in px.iter().enumerate() {
        let c = p.map(|v| ((v / scale).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
        image.set(i % W as usize, i / W as usize, c);
    }
    image
        .save(&out_dir.join(format!("m13_gate14_{name}.png")))
        .expect("save");
}

fn scale_of(px: &[[f32; 3]]) -> f32 {
    let mut l: Vec<f32> = px.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
    l.sort_by(f32::total_cmp);
    l[l.len() * 99 / 100].max(1e-6)
}

/// Colour against the reference: (near the lights, ours / one ray), (the
/// whole image), (the rest), each as (ours, one ray) relative RMSE, and the
/// near pixels' means (reference, ours).
#[allow(clippy::type_complexity)]
fn errors(
    ours: &[[f32; 3]],
    single: &[[f32; 3]],
    reference: &[[f32; 3]],
    mask: &[bool],
) -> ([(f64, f64); 3], [f64; 3], [f64; 3]) {
    let part = |inside: bool| {
        let r = pick(reference, mask, inside);
        (
            relative_rmse(&pick(ours, mask, inside), &r),
            relative_rmse(&pick(single, mask, inside), &r),
        )
    };
    (
        [
            part(true),
            (
                relative_rmse(ours, reference),
                relative_rmse(single, reference),
            ),
            part(false),
        ],
        mean(&pick(reference, mask, true)),
        mean(&pick(ours, mask, true)),
    )
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
    let (ground, _) = field.relative(origin);
    kernel.set_sky(field.sky());
    let views = Views::new(ground);
    let mut lights = gate_lights(&field, origin, &views);
    // Development after the gate (`M13_FLOOR=1`): the field's sun and a
    // light panel on the ground under the boulders, 40 m square, 5 cm up,
    // facing up, 150 m ahead, seen from 150 m up (45° down) and from 20 m up
    // (20° down), so small instances stand in front of a light.
    let floor = std::env::var_os("M13_FLOOR").is_some();
    if floor {
        let f = views.flat(0.0);
        let left = Vec3::Z.cross(f);
        // (`M13_FLOOR_H`: its height; `M13_FLOOR_DOWN=1`: facing down.)
        let height: f32 = std::env::var("M13_FLOOR_H")
            .ok()
            .and_then(|h| h.parse().ok())
            .unwrap_or(0.05);
        let down = std::env::var_os("M13_FLOOR_DOWN").is_some();
        let middle = Vec3::new(views.ground.eye.x, views.ground.eye.y, height) + f * 150.0;
        let (u, v) = (f * 40.0, left * 40.0);
        let (u, v) = if (u.cross(v).z > 0.0) != down {
            (u, v)
        } else {
            (v, u)
        };
        lights.truncate(1);
        if std::env::var_os("M13_NO_PANEL").is_none() {
            lights.push(Light::rect(
                RectLight {
                    corner: middle - (u + v) * 0.5,
                    edge_u: u,
                    edge_v: v,
                    one_sided: true,
                },
                Vec3::new(0.5, 0.45, 0.35),
            ));
        }
    }
    let view_target = target(&kernel);
    let gate_views = [
        ("low", views.at(0.4, -3.0, 0.0)),
        ("standing", views.at(1.7, -3.0, 0.0)),
    ];
    if let Some(dev) = dev {
        let only = std::env::var("M13_VIEW").ok();
        let dev_views = if floor {
            [
                ("floor_150", views.at(150.0, 45.0, 0.0)),
                ("floor_20", views.at(20.0, 20.0, 0.0)),
            ]
        } else if dev == "1" {
            gate_views
        } else {
            [
                ("low", views.at(0.4, -3.0, 3.0)),
                ("standing", views.at(1.7, -6.0, 0.0)),
            ]
        };
        for (name, camera) in dev_views {
            if only.as_ref().is_some_and(|v| !name.starts_with(v.as_str())) {
                continue;
            }
            let t0 = Instant::now();
            if std::env::var_os("M13_REF_ONLY").is_some() {
                let mut one = cached(&kernel, &field, origin, &lights, false);
                let n = settle(&mut one, camera, &view_target, 8, 3000);
                eprintln!("{name}: the one-ray path settled in {n} frames");
                continue;
            }
            let mut lit = cached(&kernel, &field, origin, &lights, true);
            if std::env::var_os("M13_LIGHTS_OFF").is_some() {
                lit.set_m13_state_word(LIGHTS_OFF, 1);
            }
            let n = settle(&mut lit, camera, &view_target, 8, 3000);
            eprintln!("{name}: settled ({:.0} s)", t0.elapsed().as_secs_f64());
            let rounds = interleaved(&mut lit, camera, &view_target, 3);
            eprintln!("{name}: timed ({:.0} s)", t0.elapsed().as_secs_f64());
            let ours = frame(&mut lit, camera, &view_target);
            drop(lit);
            let (reference, single, n_one) =
                reference(&kernel, &field, origin, &lights, camera, &view_target);
            eprintln!("{name}: reference ({:.0} s)", t0.elapsed().as_secs_f64());
            let mask = near_lights(&lights, camera);
            let (e, mr, mo) = errors(&ours, &single, &reference, &mask);
            println!(
                "{name} (development): near the lights ({} px) {:.2}% vs one ray {:.2}% ({:.3}×); whole {:.3}×; rest {:.3}×; near means reference {mr:.3?}, ours {mo:.3?}; drawn/off {:.3}× ({:?}); settled in {n} and {n_one} frames",
                mask.iter().filter(|m| **m).count(),
                100.0 * e[0].0,
                100.0 * e[0].1,
                e[0].0 / e[0].1,
                e[1].0 / e[1].1,
                e[2].0 / e[2].1,
                median(rounds.iter().map(|r| r.0 / r.1).collect()),
                rounds
                    .iter()
                    .map(|r| ((r.0 * 100.0).round() / 100.0, (r.1 * 100.0).round() / 100.0))
                    .collect::<Vec<_>>(),
            );
            let scale = scale_of(&reference);
            save(&out_dir, &format!("{name}_dev_ours"), &ours, scale);
            save(
                &out_dir,
                &format!("{name}_dev_reference"),
                &reference,
                scale,
            );
            save(&out_dir, &format!("{name}_dev_one_ray"), &single, scale);
            per_light(
                &out_dir,
                &format!("{name}_dev"),
                &lights,
                camera,
                [&reference, &ours, &single],
            );
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
    // 2. Cost, first; the images kept for criterion 1.
    let mut ours = Vec::new();
    for (name, camera) in gate_views {
        let mut lit = cached(kernel, field, origin, lights, true);
        let n = settle(&mut lit, camera, view_target, 8, 3000);
        let rounds = interleaved(&mut lit, camera, view_target, 5);
        let ratio = median(rounds.iter().map(|r| r.0 / r.1).collect());
        let c2 = ratio <= 1.03 && n < 3000;
        pass &= c2;
        say(format!(
            "{name}: 2. lights drawn {:.3}× drawing off (at most 1.03×): {}; drawn {:.2} ms, off {:.2} ms (medians of 5 rounds; rounds' ratios {:?}); settled in {n} frames",
            ratio,
            verdict(c2),
            median(rounds.iter().map(|r| r.0).collect()),
            median(rounds.iter().map(|r| r.1).collect()),
            rounds
                .iter()
                .map(|r| (r.0 / r.1 * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        ));
        ours.push(frame(&mut lit, camera, view_target));
    }
    {
        // Gate 13's standing view (no light in view), reported.
        let camera = views.at(1.7, 0.0, 0.0);
        let mut lit = cached(kernel, field, origin, lights, true);
        let n = settle(&mut lit, camera, view_target, 8, 3000);
        let rounds = interleaved(&mut lit, camera, view_target, 3);
        say(format!(
            "gate 13's standing view (reported): lights drawn {:.3}× drawing off, drawn {:.2} ms (medians of 3 rounds); settled in {n} frames",
            median(rounds.iter().map(|r| r.0 / r.1).collect()),
            median(rounds.iter().map(|r| r.0).collect()),
        ));
    }
    // 1. Colour near the lights.
    for (k, (name, camera)) in gate_views.into_iter().enumerate() {
        let (reference, single, frames) =
            reference(kernel, field, origin, lights, camera, view_target);
        let mask = near_lights(lights, camera);
        let (e, mr, mo) = errors(&ours[k], &single, &reference, &mask);
        let c1 = e[0].0 <= 0.5 * e[0].1;
        pass &= c1;
        say(format!(
            "{name}: 1. near the lights ({} px) {:.2}% from the reference, one ray {:.2}% ({:.3}×, at most 0.5×): {}",
            mask.iter().filter(|m| **m).count(),
            100.0 * e[0].0,
            100.0 * e[0].1,
            e[0].0 / e[0].1,
            verdict(c1),
        ));
        say(format!(
            "{name} (reported): the whole image {:.2}% vs one ray {:.2}% ({:.3}×); the rest {:.2}% vs {:.2}% ({:.3}×); near the lights' mean colour: reference ({:.3}, {:.3}, {:.3}), ours ({:.3}, {:.3}, {:.3}); the one-ray path settled in {frames} frames",
            100.0 * e[1].0,
            100.0 * e[1].1,
            e[1].0 / e[1].1,
            100.0 * e[2].0,
            100.0 * e[2].1,
            e[2].0 / e[2].1,
            mr[0],
            mr[1],
            mr[2],
            mo[0],
            mo[1],
            mo[2],
        ));
        let scale = scale_of(&reference);
        save(out_dir, &format!("{name}_reference"), &reference, scale);
        save(out_dir, &format!("{name}_ours"), &ours[k], scale);
        save(out_dir, &format!("{name}_one_ray"), &single, scale);
    }
    say(format!("Verdict: {}", verdict(pass)));
    std::fs::write(out_dir.join("m13_gate14.txt"), report).expect("report");
}
