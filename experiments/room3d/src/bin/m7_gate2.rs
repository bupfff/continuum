//! M7 gate 2: texture filtering for the area integral.
//!
//! M7 gate 1 found that on the textured scene, one ray per pixel is 7.11%
//! from the area-integrated image, and pixel cones did not help: the error
//! is texture filtering, not silhouettes. Each pixel samples its texture at
//! one mip chosen for its footprint's longest axis, stretched by the viewing
//! angle, which blurs across the shorter axis (isotropic trilinear).
//!
//! Method: the pixel's footprint on the surface from ray differentials (the
//! camera ray's change per pixel, projected onto the triangle's plane),
//! mapped to texture space through the triangle's texture coordinates.
//! The texture is read with four bilinear taps at the centers of the
//! footprint's 2×2 quarters, each at the mip of a quarter's longest axis:
//! a box filter over the pixel, anisotropic where the footprint is.
//!
//! Scene: the textured scene of M6 (sun and sky), its camera, 1280×720,
//! per-pixel lighting (mask cones, no cache), one ray per pixel.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Quality:** relative RMSE from the 16×16 area-integrated reference
//!    (M7 gate 1's: 256 frames over sub-pixel offsets, textures sampled for
//!    1/16 of a pixel) at most 0.8× that of the isotropic filter, measured
//!    in the same run. Pixels where a reference sample sees a light, and
//!    their neighbours, are set apart (as in M7 gate 1).
//! 2. **Speed:** the warm cached frame on the same scene at most 1.15× the
//!    isotropic filter's (median of 30 frames, each waited for).
//!
//! `cargo run --release -p room3d --bin m7_gate2`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use flatland::img::Image;
use room3d::textured::Textured;
use room3d::to_rgb;

const W: u32 = 1280;
const H: u32 = 720;
const SS: u32 = 16;

type Pixels = Vec<[f32; 3]>;

/// Relative RMS of `b − a` over the mean of `a`, pixels not set apart.
fn change(a: &[[f32; 3]], b: &[[f32; 3]], apart: &[bool]) -> f64 {
    let (mut err, mut sum, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for ((x, y), &skip) in a.iter().zip(b).zip(apart) {
        if skip {
            continue;
        }
        for c in 0..3 {
            sum += x[c] as f64;
            err += ((y[c] - x[c]) as f64).powi(2);
            n += 1.0;
        }
    }
    (err / n).sqrt() / (sum / n)
}

fn save(path: &Path, pixels: &[[f32; 3]]) -> std::io::Result<()> {
    let mut image = Image::new(W as usize, H as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % W as usize, i / W as usize, to_rgb(Vec3::from_array(*p)));
    }
    image.save(path)
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
    kernel.set_hardware_rays(true);
    let t = Textured::new();
    kernel.set_sky(t.sky);
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
    let camera: Camera = t.camera;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    // 2. Speed, first: warm cached frames, each filter.
    let warm = |boxed: bool| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        );
        r.set_texture_box(boxed);
        let mut frames = 0;
        loop {
            r.frame(camera, &target);
            frames += 1;
            let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
            if (latest >= 32
                && r.cache_log()[latest - 32..latest]
                    .iter()
                    .all(|s| s.requests == 0 && s.pages_baked == 0))
                || frames > 20_000
            {
                break;
            }
        }
        r.finish();
        let mut times: Vec<Duration> = (0..30)
            .map(|_| {
                let start = Instant::now();
                r.frame(camera, &target);
                r.wait();
                start.elapsed()
            })
            .collect();
        times.sort();
        times[15]
    };
    let iso_time = warm(false);
    let box_time = warm(true);
    let c2 = ms(box_time) <= 1.15 * ms(iso_time);
    say(format!(
        "2. speed: warm {:.2} ms boxed, {:.2} ms isotropic ({:.2}×, at most 1.15×): {}",
        ms(box_time),
        ms(iso_time),
        ms(box_time) / ms(iso_time),
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 1. Quality against the area-integrated reference.
    let mut r =
        InstancedRenderer::new_lights(&kernel, format, W, H, 1, &t.lights, &t.scene, t.origin);
    // The reference: M7 gate 1's (isotropic filter at 1/16 of a pixel).
    r.set_texture_box(false);
    r.set_footprint(1.0 / SS as f32);
    let n = (W * H) as usize;
    let mut sum = vec![[0.0f64; 3]; n];
    let mut sees = vec![false; n];
    for j in 0..SS {
        for i in 0..SS {
            let offset = |k: u32| (k as f32 + 0.5) / SS as f32 - 0.5;
            r.set_pixel_offset(offset(i), offset(j));
            r.frame(camera, &target);
            for (p, c) in r.read_colors().iter().enumerate() {
                if c.iter().any(|&v| v > 100.0) {
                    sees[p] = true;
                }
                for k in 0..3 {
                    sum[p][k] += c[k] as f64;
                }
            }
        }
    }
    let reference: Pixels = sum
        .iter()
        .map(|s| s.map(|v| (v / (SS * SS) as f64) as f32))
        .collect();
    let mut apart = sees.clone();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if !sees[(y * W as i32 + x) as usize] {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < W as i32 && ny < H as i32 {
                        apart[(ny * W as i32 + nx) as usize] = true;
                    }
                }
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    let mut one = |boxed: bool| {
        r.set_texture_box(boxed);
        r.frame(camera, &target);
        r.read_colors()
    };
    let iso = one(false);
    let boxed = one(true);
    let (e_iso, e_box) = (
        change(&reference, &iso, &apart),
        change(&reference, &boxed, &apart),
    );
    let c1 = e_box <= 0.8 * e_iso;
    say(format!(
        "1. quality: boxed {:.2}%, isotropic {:.2}% from the 16×16 reference ({:.2}×, at most 0.8×; {} pixels set apart): {}",
        100.0 * e_box,
        100.0 * e_iso,
        e_box / e_iso,
        apart.iter().filter(|&&a| a).count(),
        if c1 { "PASS" } else { "FAIL" }
    ));
    save(&out_dir.join("m7_gate2_reference.png"), &reference)?;
    save(&out_dir.join("m7_gate2_isotropic.png"), &iso)?;
    save(&out_dir.join("m7_gate2_boxed.png"), &boxed)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m7_gate2.txt"), report)?;
    Ok(())
}
