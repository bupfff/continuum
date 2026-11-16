//! M19 gate 1: bloom and upscaling.
//!
//! Method (DESIGN §3.4, "Bloom and upscaling"): bloom (`Bloom`, the
//! default: strength 0.04, radius 2% of the image's height) mixes into the
//! display (1 − strength) of each pixel plus strength of the image at a
//! quarter of its size (4 × 4 means) blurred by a Gaussian across and down
//! (weights renormalized at the edges), sampled bilinearly, in linear light;
//! upscaling (`Upscale::CatmullRom`) draws an image rendered at a lower
//! resolution with Catmull and Rom's bicubic clamped to its four nearest
//! pixels' range.
//!
//! Scene: M6's textured scene with foliage, its camera, sun and sky, cache
//! settled, drawn to a 1280×720 `Rgba32Float` target (linear light; no
//! auto-exposure, exposure 1). Timings: medians of 40 warm frames, each
//! waited for, interleaved in five rounds where two are compared.
//!
//! The bars were set with development numbers known (DESIGN §3.4: Catmull-
//! Rom 0.87× nearest's error, +0.29 ms, the frame 0.50× native; bloom
//! 0.25 ms).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Bloom's image:** the drawn target within 1e-3 (relative RMSE) of
//!    bloom computed on the CPU from `read_colors`, and the image's total
//!    light within 1% of the unbloomed image's.
//! 2. **Bloom's cost:** at most 0.3 ms added to the native frame.
//! 3. **Upscaling:** rendered at 640×360 (scale 2), Catmull-Rom's image at
//!    most 0.9× nearest's error (relative RMSE) against the scene rendered
//!    at 1280×720; its frame at most 0.35 ms more than nearest's and at
//!    most 0.55× the native frame.
//!
//! Reported: bilinear's error and frame.
//!
//! Development mode (`M19_DEV=1`): fewer frames timed (10 a round).

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum_gpu::{
    Bloom, CacheOptions, Camera, InstancedRenderer, MaskConeKernel, Upscale, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return;
        }
    }
}

fn read(kernel: &MaskConeKernel, texture: &wgpu::Texture) -> Vec<[f32; 3]> {
    let device = kernel.device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (W * H * 16) as u64,
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
                bytes_per_row: Some(W * 16),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    kernel.queue().submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    buffer
        .slice(..)
        .get_mapped_range()
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| {
            let f = |k: usize| f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
            [f(0), f(4), f(8)]
        })
        .collect()
}

/// Bloom computed on the CPU from linear colours (`w × h`), as the display
/// does it.
fn cpu_bloom(colors: &[[f32; 3]], w: usize, h: usize, bloom: &Bloom) -> Vec<[f64; 3]> {
    let (qw, qh) = (w.div_ceil(4), h.div_ceil(4));
    let mut quarter = vec![[0.0f64; 3]; qw * qh];
    for qy in 0..qh {
        for qx in 0..qw {
            let (mut sum, mut n) = ([0.0f64; 3], 0.0);
            for j in 0..4 {
                for i in 0..4 {
                    let (x, y) = (4 * qx + i, 4 * qy + j);
                    if x < w && y < h {
                        for c in 0..3 {
                            sum[c] += colors[y * w + x][c] as f64;
                        }
                        n += 1.0;
                    }
                }
            }
            quarter[qy * qw + qx] = sum.map(|v| v / n);
        }
    }
    let sigma = (bloom.radius as f64 * h as f64 / 4.0).max(0.25);
    let taps = (3.0 * sigma).ceil() as i64;
    let blur = |src: &[[f64; 3]], down: bool| -> Vec<[f64; 3]> {
        let mut out = vec![[0.0f64; 3]; qw * qh];
        for y in 0..qh as i64 {
            for x in 0..qw as i64 {
                let (mut sum, mut weight) = ([0.0f64; 3], 0.0);
                for k in -taps..=taps {
                    let (sx, sy) = if down { (x, y + k) } else { (x + k, y) };
                    if sx < 0 || sy < 0 || sx >= qw as i64 || sy >= qh as i64 {
                        continue;
                    }
                    let wgt = (-0.5 * (k * k) as f64 / (sigma * sigma)).exp();
                    for c in 0..3 {
                        sum[c] += wgt * src[sy as usize * qw + sx as usize][c];
                    }
                    weight += wgt;
                }
                out[y as usize * qw + x as usize] = sum.map(|v| v / weight);
            }
        }
        out
    };
    let blurred = blur(&blur(&quarter, false), true);
    let s = bloom.strength as f64;
    let mut out = vec![[0.0f64; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let u = ((x as f64 + 0.5) / 4.0 - 0.5, (y as f64 + 0.5) / 4.0 - 0.5);
            let (ix, iy) = (u.0.floor(), u.1.floor());
            let (fx, fy) = (u.0 - ix, u.1 - iy);
            let mut b = [0.0f64; 3];
            for k in 0..4i64 {
                let qx = (ix as i64 + (k & 1)).clamp(0, qw as i64 - 1) as usize;
                let qy = (iy as i64 + (k >> 1)).clamp(0, qh as i64 - 1) as usize;
                let wgt = if k & 1 == 1 { fx } else { 1.0 - fx }
                    * if k >> 1 == 1 { fy } else { 1.0 - fy };
                for c in 0..3 {
                    b[c] += wgt * blurred[qy * qw + qx][c];
                }
            }
            for c in 0..3 {
                out[y * w + x][c] = (1.0 - s) * colors[y * w + x][c] as f64 + s * b[c];
            }
        }
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M19_DEV").is_some();
    let frames = if dev { 10 } else { 40 };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        std::io::stdout().flush().ok();
        report.push_str(&line);
        report.push('\n');
    };
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    say(format!(
        "M19 gate 1{}: {W}x{H}, adapter {}",
        if dev { " (development mode)" } else { "" },
        kernel.adapter_name
    ));
    let format = wgpu::TextureFormat::Rgba32Float;
    let texture = kernel.device().create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target = texture.create_view(&Default::default());
    let t = Textured::new().with_foliage();
    kernel.set_sky(t.sky);
    let make = |scale: u32| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            scale,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        );
        settle(&mut r, t.camera, &target);
        r
    };
    let timed = |r: &mut InstancedRenderer| {
        let mut v: Vec<f64> = (0..frames)
            .map(|_| {
                let s = Instant::now();
                r.frame(t.camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        median(&mut v)
    };
    let mut native = make(1);
    let mut half = make(2);

    // 2. Bloom's cost, and 3. upscaling's (timed first).
    let (mut bloom_added, mut cr_added, mut cr_ratio, mut bl) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut rounds = String::new();
    for _ in 0..5 {
        native.set_bloom(None);
        let plain = timed(&mut native);
        native.set_bloom(Some(Bloom::default()));
        let bloomed = timed(&mut native);
        native.set_bloom(None);
        half.set_upscale(Upscale::Nearest);
        let nearest = timed(&mut half);
        half.set_upscale(Upscale::CatmullRom);
        let cr = timed(&mut half);
        half.set_upscale(Upscale::Bilinear);
        bl.push(timed(&mut half));
        bloom_added.push(bloomed - plain);
        cr_added.push(cr - nearest);
        cr_ratio.push(cr / plain);
        rounds.push_str(&format!(" {plain:.3}/{bloomed:.3}/{nearest:.3}/{cr:.3}"));
    }
    let (b_add, c_add, c_ratio, b_ms) = (
        median(&mut bloom_added),
        median(&mut cr_added),
        median(&mut cr_ratio),
        median(&mut bl),
    );
    let c2 = b_add <= 0.3;
    say(format!(
        "2. bloom's cost: +{b_add:.3} ms (at most 0.3): {} (rounds, native/bloom/nearest/Catmull-Rom ms:{rounds})",
        verdict(c2)
    ));

    // 1. Bloom's image.
    native.set_bloom(None);
    native.frame(t.camera, &target);
    native.wait();
    let colors = native.read_colors();
    let plain_drawn = read(&kernel, &texture);
    native.set_bloom(Some(Bloom::default()));
    native.frame(t.camera, &target);
    native.wait();
    let drawn = read(&kernel, &texture);
    let want = cpu_bloom(&colors, W as usize, H as usize, &Bloom::default());
    let (mut num, mut den, mut total_in, mut total_out) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (k, p) in drawn.iter().enumerate() {
        for c in 0..3 {
            num += (p[c] as f64 - want[k][c]).powi(2);
            den += want[k][c].powi(2);
            total_in += plain_drawn[k][c] as f64;
            total_out += p[c] as f64;
        }
    }
    let rel = (num / den).sqrt();
    let kept = total_out / total_in;
    let c1 = rel <= 1e-3 && (kept - 1.0).abs() <= 0.01;
    say(format!(
        "1. bloom's image: {rel:.2e} from the CPU's (at most 1e-3), total light {kept:.5} of the unbloomed (within 1%): {}",
        verdict(c1)
    ));

    // 3. Upscaling's image.
    native.set_bloom(None);
    native.frame(t.camera, &target);
    native.wait();
    let reference = read(&kernel, &texture);
    let mut errors = Vec::new();
    for up in [Upscale::Nearest, Upscale::Bilinear, Upscale::CatmullRom] {
        half.set_upscale(up);
        half.frame(t.camera, &target);
        half.wait();
        errors.push(relative_rmse(&read(&kernel, &texture), &reference));
    }
    let c3 = errors[2] <= 0.9 * errors[0] && c_add <= 0.35 && c_ratio <= 0.55;
    say(format!(
        "3. upscaling from 640x360: Catmull-Rom {:.2}% from the native render, nearest {:.2}% ({:.3}x, at most 0.9x), bilinear {:.2}%; Catmull-Rom +{c_add:.3} ms over nearest (at most 0.35), its frame {c_ratio:.3}x native (at most 0.55); bilinear's frame {b_ms:.3} ms: {}",
        100.0 * errors[2],
        100.0 * errors[0],
        errors[2] / errors[0],
        100.0 * errors[1],
        verdict(c3)
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {}: {}",
        verdict(c1),
        verdict(c2),
        verdict(c3),
        verdict(c1 && c2 && c3)
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m19_gate1_dev.txt"
        } else {
            "m19_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
