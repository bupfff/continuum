//! Development (M5 gate 15, receivers next to large lights): the cache's
//! settled image near M5 gate 14's light panel, with and without rays to
//! wide lights (`CacheOptions::wide_rays`), against the cached Cycles
//! references of gate 14's development views. Not a gate.
//!
//! The million-boulder field, its sun and the panel; 320×180, supersampled
//! 4×4 (each sub-pixel offset settled, a fixed bake budget). `WD_VIEW`:
//! `near` (gate 14's development near view, default) or `40m`.

use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, RectLight, wgpu,
};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::field::Field;

const W: u32 = 320;
const H: u32 = 180;

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

fn kernel_with(guards: bool) -> Option<MaskConeKernel> {
    let kernel = MaskConeKernel::new()?;
    kernel.set_wide_cones(guards);
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    Some(kernel)
}

/// The camera's line on the ground (unit, horizontal) and a camera on it.
struct Views {
    ground: Camera,
    flat0: Vec3,
}

impl Views {
    fn new(ground: Camera) -> Self {
        let f = (ground.target - ground.eye).normalize();
        Views {
            ground,
            flat0: Vec3::new(f.x, f.y, 0.0).normalize(),
        }
    }

    fn flat(&self, turn_degrees: f32) -> Vec3 {
        let (s, c) = turn_degrees.to_radians().sin_cos();
        let f = self.flat0;
        Vec3::new(f.x * c - f.y * s, f.x * s + f.y * c, 0.0)
    }

    /// `ahead` m along the line, `height` up, turned `turn` (left) and
    /// looking `down` degrees down.
    fn at(&self, ahead: f32, height: f32, down: f32, turn: f32) -> Camera {
        let d = down.to_radians();
        let g = self.ground.eye;
        let eye = Vec3::new(g.x, g.y, 0.0) + self.flat0 * ahead + Vec3::Z * height;
        Camera {
            eye,
            target: eye + self.flat(turn) * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: self.ground.vertical_fov_degrees,
        }
    }
}

/// The panel: 40 m square, 5 cm up, facing up, 130–170 m ahead.
fn panel(views: &Views) -> Light {
    let f = views.flat0;
    let left = Vec3::Z.cross(f);
    let g = views.ground.eye;
    let middle = Vec3::new(g.x, g.y, 0.05) + f * 150.0;
    let (u, v) = (f * 40.0, left * 40.0);
    let (u, v) = if u.cross(v).z > 0.0 { (u, v) } else { (v, u) };
    Light::rect(
        RectLight {
            corner: middle - (u + v) * 0.5,
            edge_u: u,
            edge_v: v,
            one_sided: true,
        },
        Vec3::new(0.5, 0.45, 0.35),
    )
}

/// Per pixel at `ss × ss` samples (per-pixel cones), box filtered to
/// `w × h`, and which pixels (with their eight neighbours) have a sample
/// that sees a light (M5 gate 6's).
fn supersampled(
    kernel: &MaskConeKernel,
    camera: Camera,
    (w, h): (u32, u32),
    ss: u32,
    lights: &[Light],
    field: &Field,
    origin: DVec3,
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let (big, _, _) =
        kernel.render_instanced_lights(camera, w * ss, h * ss, lights, &field.scene, origin, 1);
    let mut px = vec![[0.0f32; 3]; (w * h) as usize];
    let mut sees = vec![false; (w * h) as usize];
    let n = (ss * ss) as f32;
    for y in 0..(h * ss) as usize {
        for x in 0..(w * ss) as usize {
            let p = big[y * (w * ss) as usize + x];
            let i = (y / ss as usize) * w as usize + x / ss as usize;
            if lights.iter().any(|l| l.radiance.to_array() == p) {
                sees[i] = true;
            }
            for c in 0..3 {
                px[i][c] += p[c] / n;
            }
        }
    }
    let mut apart = sees.clone();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !sees[(y * w as i32 + x) as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                        apart[(ny * w as i32 + nx) as usize] = true;
                    }
                }
            }
        }
    }
    (px, apart)
}

/// Relative RMSE over the pixels not set apart.
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

fn settled_supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> (Vec<[f32; 3]>, usize) {
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    let mut frames = 0;
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        let mut calm = 0;
        for _ in 0..20_000 {
            r.frame(camera, target);
            frames += 1;
            if k == 0 && frames <= 40 && frames % 5 == 0 {
                let (n, runs) = r.wide_listed();
                println!("  frame {frames}: {n} texels listed for wide rays (pass runs: {runs})");
            }
            if r.cache_stats()
                .is_some_and(|s| s.requests == 0 && s.pages_baked == 0)
            {
                calm += 1;
                if calm >= 16 {
                    break;
                }
            } else {
                calm = 0;
            }
        }
        r.finish();
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    (sum, frames)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = kernel_with(true).expect("GPU");
    let start = Instant::now();
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, sun) = field.relative(origin);
    let views = Views::new(ground);
    let both = [Light::white(sun, field.radiance), panel(&views)];
    // `WD_PANEL=1`: the panel alone.
    let alone = std::env::var_os("WD_PANEL").is_some();
    let lights: &[Light] = if alone { &both[1..] } else { &both };
    println!("field built in {:.0} s", start.elapsed().as_secs_f64());
    let which = std::env::var("WD_VIEW").unwrap_or_else(|_| "near".into());
    let (camera, dir) = if which == "40m" {
        (
            views.at(80.0, 40.0, 30.0, 0.0),
            "m5_gate14_cycles_40mupdevelopment",
        )
    } else {
        (
            views.at(110.0, 8.0, 15.0, 30.0),
            "m5_gate14_cycles_neardevelopment",
        )
    };
    let small = (W, H);
    let cycles = render_instanced_lights(
        &field.scene,
        origin,
        lights,
        Vec3::ZERO,
        camera,
        small,
        4096,
        0,
        1,
        &out_dir.join(if alone {
            format!("{dir}_panel")
        } else {
            dir.to_string()
        }),
    )?;
    // Pixels seeing a light, and their neighbours, set apart (per-pixel
    // cones at 4x4 find them).
    let (per_pixel, apart) = supersampled(&kernel, camera, small, 4, lights, &field, origin);
    println!(
        "{which}: per-pixel cones vs Cycles over surfaces {:.2}%",
        100.0 * surfaces_rmse(&per_pixel, &cycles.pixels, &apart)
    );
    let view = target(&kernel);
    for wide in [true, false] {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            W,
            H,
            1,
            lights,
            &field.scene,
            origin,
            CacheOptions {
                frame_target_ms: 0.0,
                wide_rays: wide,
                indirect: false,
                ..Default::default()
            },
        );
        let s = Instant::now();
        let (img, frames) = settled_supersampled(&mut r, camera, &view);
        let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
            let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                let v =
                    p.map(|v| ((v / 0.1).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
                image.set(i % W as usize, i / W as usize, v);
            }
            image.save(&out_dir.join(format!("m5_wide_dev_{which}_{name}.png")))
        };
        save(if wide { "rays" } else { "cones" }, &img)?;
        let raw = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
            let bytes: Vec<u8> = px
                .iter()
                .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
                .collect();
            std::fs::write(
                out_dir.join(format!("m5_wide_dev_{which}_{name}.f32")),
                bytes,
            )
        };
        raw(if wide { "rays" } else { "cones" }, &img)?;
        raw("cycles", &cycles.pixels)?;
        raw("per_pixel", &per_pixel)?;
        let apart_bytes: Vec<u8> = apart.iter().map(|a| *a as u8).collect();
        std::fs::write(
            out_dir.join(format!("m5_wide_dev_{which}_apart.u8")),
            apart_bytes,
        )?;
        save("cycles", &cycles.pixels)?;
        save("per_pixel", &per_pixel)?;
        let sum = |px: &[[f32; 3]]| -> f64 {
            px.iter()
                .zip(&apart)
                .filter(|(_, a)| !**a)
                .map(|(p, _)| (p[0] + p[1] + p[2]) as f64)
                .sum()
        };
        println!(
            "{which}: surfaces' sums: cache {:.1}, Cycles {:.1}, per-pixel {:.1}",
            sum(&img),
            sum(&cycles.pixels),
            sum(&per_pixel)
        );
        println!(
            "{which}: the cache {} wide rays vs Cycles over surfaces {:.2}% ({frames} frames, {:.0} s)",
            if wide { "with" } else { "without" },
            100.0 * surfaces_rmse(&img, &cycles.pixels, &apart),
            s.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
