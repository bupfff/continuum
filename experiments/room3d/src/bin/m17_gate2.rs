//! M17 gate 2: auto-exposure at a quarter of the pixels.
//!
//! Gate 1 failed on cost only (the meter added 0.168 ms at 720p against
//! 0.15; PBR Neutral matched Blender's view, the adaptation had no
//! overshoot). The meter's histogram now takes every other pixel in each
//! direction. Criteria as gate 1's, unchanged; the CPU's rule takes the
//! same pixels.
//!
//! Method (DESIGN §3.4, "Display"): the display pass maps the renderer's
//! linear light with Khronos PBR Neutral (`ToneMap::PbrNeutral`, the
//! published formula; `Renderer`'s default) to an sRGB target; the
//! auto-exposure meter (`AutoExposure`) histograms the frame's log2
//! luminance in half stops at every other pixel in each direction and sets the exposure that shows the mean
//! between its 50th and 95th percentiles at 0.18, approached with a time
//! constant (0.5 s), only that number carried between frames.
//!
//! Views: M6's textured scene by day (sun and sky) and at night with its
//! five lamps (`with_lamps`), 1280×720, cache settled. For the tone map,
//! each view's exposure is fixed at the meter's rule computed on the CPU
//! from its linear image; Blender 5.2 applies its own "Khronos PBR
//! Neutral" view (and AgX and Standard, reported) to the same linear
//! image times that exposure (`cycles/view_transform.py`), against our
//! display's output read from an `Rgba8UnormSrgb` target.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost** (first): on the day view, the warm frame with
//!    auto-exposure at most 0.15 ms more than without (medians of five
//!    interleaved rounds of 30 frames, each waited for).
//! 2. **Tone map:** in both views, our PBR Neutral display values within
//!    1/255 of Blender's on average and 3/255 at the 99th percentile (in
//!    sRGB-encoded units, every channel of every pixel; ours are 8-bit).
//! 3. **Adaptation:** on the day view settled with auto-exposure, the
//!    sun's radiance drops to a sixteenth; the meter's exposure each
//!    frame, against its value 4 s later, never passes it by more than
//!    0.5% (no overshoot) and is within 2% of it 2.5 s after the drop
//!    (five time constants).
//!
//! Reported: our AgX (an approximation of the original) against Blender's
//! AgX; the maximum differences; the exposures.
//!
//! Development mode (`M17_DEV=1`): 640×360.
//!
//! Implementation notes: gate 1's binary with the CPU's rule subsampled.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum_gpu::{
    AutoExposure, CacheOptions, Camera, InstancedRenderer, MaskConeKernel, ToneMap, wgpu,
};
use room3d::cycles::blender_views;
use room3d::textured::Textured;

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
    eprintln!("WARNING: not settled after 20,000 frames");
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The meter's rule on the CPU: the exposure showing the mean of log2
/// luminance between the 50th and 95th percentiles (half-stop bins, every
/// other pixel in each direction of a `width`-wide image) at 0.18.
fn metered(colors: &[[f32; 3]], width: u32) -> f32 {
    let mut hist = [0u32; 64];
    for (k, c) in colors.iter().enumerate() {
        if !(k as u32 % width).is_multiple_of(2) || !(k as u32 / width).is_multiple_of(2) {
            continue;
        }
        let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let b = if l <= 0.0 {
            0
        } else {
            ((l.log2() + 16.0) * 2.0).clamp(0.0, 63.0) as usize
        };
        hist[b] += 1;
    }
    let total: u32 = hist.iter().sum();
    let (lo, hi) = (total as f64 * 0.5, total as f64 * 0.95);
    let (mut seen, mut weight, mut sum) = (0.0f64, 0.0f64, 0.0f64);
    for (b, &n) in hist.iter().enumerate() {
        let take = ((seen + n as f64).min(hi) - seen.max(lo)).max(0.0);
        sum += take * ((b as f64 + 0.5) * 0.5 - 16.0);
        weight += take;
        seen += n as f64;
    }
    (0.18 / (sum / weight).exp2()) as f32
}

/// The target's pixels (`Rgba8UnormSrgb`, read as stored: sRGB-encoded),
/// rows from the top.
fn read_target(kernel: &MaskConeKernel, texture: &wgpu::Texture, w: u32, h: u32) -> Vec<[f32; 3]> {
    let device = kernel.device();
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("target readback"),
        size: (row * h) as u64,
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
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    kernel.queue().submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let k = (y * row + 4 * x) as usize;
            out.push([
                data[k] as f32 / 255.0,
                data[k + 1] as f32 / 255.0,
                data[k + 2] as f32 / 255.0,
            ]);
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M17_DEV").is_some();
    let (w, h) = if dev { (640, 360) } else { (1280, 720) };
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
        "M17 gate 2{}: {w}x{h}, adapter {}",
        if dev { " (development mode)" } else { "" },
        kernel.adapter_name
    ));
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let texture = kernel.device().create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
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
    let day = Textured::new();
    let night = Textured::new().with_lamps();
    let make = |t: &Textured| {
        kernel.set_sky(t.sky);
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &t.lights,
            &t.scene,
            t.origin,
            CacheOptions::default(),
        );
        r.set_tone_map(ToneMap::PbrNeutral);
        settle(&mut r, t.camera, &target);
        r
    };

    // 1. Cost.
    let mut r = make(&day);
    let timed = |r: &mut InstancedRenderer| {
        let mut v: Vec<f64> = (0..30)
            .map(|_| {
                let s = Instant::now();
                r.frame(day.camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        median(&mut v)
    };
    let mut diffs = Vec::new();
    let mut rounds = String::new();
    for _ in 0..5 {
        r.set_auto_exposure(None);
        let off = timed(&mut r);
        r.set_auto_exposure(Some(AutoExposure::default()));
        let on = timed(&mut r);
        diffs.push(on - off);
        rounds.push_str(&format!(" {on:.3}/{off:.3}"));
    }
    let added = median(&mut diffs);
    let c1 = added <= 0.15;
    say(format!(
        "1. cost: auto-exposure adds {added:.3} ms (at most 0.15): {} (rounds, with/without ms:{rounds})",
        verdict(c1)
    ));

    // 2. Tone map against Blender's views.
    let mut c2 = true;
    for (name, t, renderer) in [("day", &day, Some(r)), ("night", &night, None)] {
        let mut r = renderer.unwrap_or_else(|| make(t));
        r.set_auto_exposure(None);
        r.set_exposure(1.0);
        r.frame(t.camera, &target);
        r.finish();
        let linear = r.read_colors();
        let e = metered(&linear, w);
        let exposed: Vec<[f32; 3]> = linear.iter().map(|p| p.map(|c| c * e)).collect();
        let mut by_tone = Vec::new();
        for tone in [ToneMap::PbrNeutral, ToneMap::Agx] {
            r.set_tone_map(tone);
            r.set_exposure(e);
            r.frame(t.camera, &target);
            r.finish();
            by_tone.push(read_target(&kernel, &texture, w, h));
        }
        let views = blender_views(&exposed, w, h, &out_dir.join(format!("m17_gate2_{name}")))?;
        let compare = |ours: &[[f32; 3]], theirs: &[[f32; 3]]| {
            let mut d: Vec<f32> = ours
                .iter()
                .zip(theirs)
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .collect();
            let mean = d.iter().map(|&x| x as f64).sum::<f64>() / d.len() as f64;
            d.sort_by(f32::total_cmp);
            (mean, d[d.len() * 99 / 100], d[d.len() - 1])
        };
        let (mn, p99, mx) = compare(&by_tone[0], &views[0].1);
        let (amn, ap99, amx) = compare(&by_tone[1], &views[1].1);
        let ok = mn <= 1.0 / 255.0 && p99 <= 3.0 / 255.0;
        c2 &= ok;
        say(format!(
            "2. {name} (exposure {e:.4}): PBR Neutral against Blender's: mean {:.3}/255 (at most 1), 99th percentile {:.2}/255 (at most 3), largest {:.1}/255: {}; AgX approximation against Blender's AgX: mean {:.2}/255, 99th {:.1}/255, largest {:.1}/255",
            mn * 255.0,
            p99 * 255.0,
            mx * 255.0,
            verdict(ok),
            amn * 255.0,
            ap99 * 255.0,
            amx * 255.0
        ));
    }

    // 3. Adaptation after a step.
    let mut r = make(&day);
    r.set_auto_exposure(Some(AutoExposure::default()));
    for _ in 0..60 {
        r.frame(day.camera, &target);
    }
    let before = r.metered_exposure().0;
    let sun = day.lights[0].radiance;
    r.set_light_radiance(0, sun / 16.0);
    let start = Instant::now();
    let mut trace: Vec<(f64, f32)> = Vec::new();
    while start.elapsed().as_secs_f64() < 4.0 {
        r.frame(day.camera, &target);
        r.wait();
        trace.push((start.elapsed().as_secs_f64(), r.metered_exposure().0));
    }
    let last = trace.last().unwrap().1;
    let rising = last > before;
    let overshoot = trace
        .iter()
        .map(|&(_, e)| {
            if rising {
                e / last - 1.0
            } else {
                last / e - 1.0
            }
        })
        .fold(f32::MIN, f32::max);
    let at = trace
        .iter()
        .find(|&&(t, _)| t >= 2.5)
        .map_or(f32::NAN, |&(_, e)| (e / last - 1.0).abs());
    let c3 = overshoot <= 0.005 && at <= 0.02;
    say(format!(
        "3. adaptation: exposure {before:.4} before the drop, {last:.4} 4 s after ({} frames); largest overshoot {:.3}% (at most 0.5%), off by {:.3}% at 2.5 s (at most 2%): {}",
        trace.len(),
        100.0 * overshoot.max(0.0),
        100.0 * at,
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
            "m17_gate1_dev.txt"
        } else {
            "m17_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
