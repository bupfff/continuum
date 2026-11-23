//! Development (the user's viewer session, 2026-11-22): the field as the
//! viewer builds it (a million boulders, the sun, no sky, bounce light,
//! anti-aliasing, bloom, PBR Neutral), settled at the viewer's start; then
//! the camera turns up toward the sun over `VR_TURN` frames (90), holds
//! `VR_HOLD` (30), turns back as it came, and stays. Each frame prints the
//! displayed image's mean (sRGB bytes made linear) over the lower half (the
//! ground) and, after the return, the share of pixels at most half as
//! bright as in the settled frame there. Frames named in `VR_SAVE`
//! (`a,b,…`; negative counts from the end) are saved to `VR_OUT`
//! (`out/viewer_repro`). Not a gate.
//!
//! `cargo run --release -p room3d --bin viewer_repro`

use std::path::Path;
use std::time::Instant;

use continuum::glam::DVec3;
use continuum_gpu::{Camera, Light, MaskConeKernel, Renderer, RendererOptions, wgpu};
use room3d::field::Field;

fn num(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The displayed image (sRGB bytes, RGBA rows).
fn grab(r: &Renderer, texture: &wgpu::Texture) -> Vec<u8> {
    let device = r.kernel().device();
    let queue = r.kernel().queue();
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grab"),
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
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU");
    let data = slice.get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&data[y * row as usize..y * row as usize + 4 * w as usize]);
    }
    out
}

fn linear(b: u8) -> f32 {
    let c = b as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn main() {
    let (w, h) = (num("VR_W", 1470) as u32, num("VR_H", 923) as u32);
    let turn = num("VR_TURN", 90);
    let hold = num("VR_HOLD", 30);
    let after = num("VR_AFTER", 240);
    let out = std::env::var("VR_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("out/viewer_repro"));
    std::fs::create_dir_all(&out).expect("the output folder");
    let field = Field::new(num("VR_BOULDERS", 1_000_000), DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (mut camera, light) = field.relative(origin);
    // `VR_UP=meters`, `VR_DOWN=degrees`: the start view raised and tilted
    // down.
    if let Ok(up) = std::env::var("VR_UP").map(|v| v.parse::<f32>().expect("meters")) {
        camera.eye.z += up;
        camera.target.z += up;
    }
    if let Ok(down) = std::env::var("VR_DOWN").map(|v| v.parse::<f32>().expect("degrees")) {
        let d = (camera.target - camera.eye).normalize();
        let right = d.cross(continuum::glam::Vec3::Z).normalize();
        camera.target = camera.eye + continuum::glam::Quat::from_axis_angle(right, -down.to_radians()) * d;
    }
    let kernel = MaskConeKernel::new().expect("GPU");
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let t = Instant::now();
    let mut r = Renderer::new(
        kernel,
        format,
        (w, h),
        &field.scene,
        origin,
        &[Light::white(light, field.radiance)],
        RendererOptions {
            cache_options: continuum_gpu::CacheOptions {
                sky_lag: true,
                // `VR_FRAME_BUDGET`: the frame period's cap (ms; 0: none).
                // `VR_FAR=1`: far bounce hits lit by rays.
                far_rays: std::env::var_os("VR_FAR").is_some(),
                // `VR_PPT`: pixels a texel spans.
                pixels_per_texel: std::env::var("VR_PPT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(continuum_gpu::CacheOptions::default().pixels_per_texel),
                frame_budget_ms: std::env::var("VR_FRAME_BUDGET")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(continuum_gpu::CacheOptions::default().frame_budget_ms),
                ..Default::default()
            },
            anti_aliasing: std::env::var("VR_AA").as_deref() != Ok("0"),
            bounce: std::env::var("VR_BOUNCE").as_deref() != Ok("0"),
            bloom: (std::env::var("VR_BLOOM").as_deref() != Ok("0"))
                .then(continuum_gpu::Bloom::default),
            tone_map: continuum_gpu::ToneMap::PbrNeutral,
            // `VR_AUTO=0`: a fixed exposure (for comparing frames).
            auto_exposure: (std::env::var("VR_AUTO").as_deref() != Ok("0"))
                .then(continuum_gpu::AutoExposure::default),
            ..Default::default()
        },
    );
    r.wait();
    println!("renderer {:.1} s", t.elapsed().as_secs_f64());
    let texture = r.kernel().device().create_texture(&wgpu::TextureDescriptor {
        label: Some("display"),
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
    let view = texture.create_view(&Default::default());
    let start_dir = (camera.target - camera.eye).normalize();
    let sun = light.corner + 0.5 * (light.edge_u + light.edge_v);
    let sun_dir = (sun - camera.eye).normalize();
    let pose = |s: f32| -> Camera {
        let d = start_dir.lerp(sun_dir, s).normalize();
        Camera {
            eye: camera.eye,
            target: camera.eye + d,
            vertical_fov_degrees: camera.vertical_fov_degrees,
        }
    };
    println!(
        "sun {:.1}° above the view's start direction",
        start_dir.angle_between(sun_dir).to_degrees()
    );
    let render = |r: &mut Renderer, cam: Camera| {
        let f = Instant::now();
        r.render(cam, &view);
        r.wait();
        let ms = f.elapsed().as_secs_f64() * 1000.0;
        if ms > 5000.0 {
            eprintln!("a frame took {ms:.0} ms: stopping");
            std::process::exit(3);
        }
        ms
    };
    // Settle at the start.
    let mut settled_at = None;
    for f in 0..num("VR_SETTLE", 600) {
        render(&mut r, pose(0.0));
        if std::env::var_os("VR_STATS2").is_some() && f >= 1000 && f % 25 == 0 {
            let st = r.stats().unwrap_or_default();
            println!(
                "settle {f}: baked {} requests {} at_wanted {}/{} rebaked {} sky_listed {} sky_rebaked {} deferred {} invalidated {} missed {} evicted {} resident {} bounce {} incomplete {}",
                st.pages_baked, st.requests, st.at_wanted, st.samples, st.rebaked_pages, st.sky_listed, st.sky_rebaked,
                st.deferred, st.invalidated, st.missed, st.evicted, st.resident, st.indirect_samples, st.indirect_incomplete
            );
        }
        if std::env::var_os("VR_STATS").is_some() && f % 50 == 0 {
            let st = r.stats().unwrap_or_default();
            println!(
                "settle {f}: resident {} baked {} requests {} fallback {} bounce {} incomplete {} budget {} cap {} spent {} bake {:.2} ms",
                st.resident,
                st.pages_baked,
                st.requests,
                st.fallback_pixels,
                st.indirect_samples,
                st.indirect_incomplete,
                st.budget,
                st.budget_cap,
                st.spent,
                st.gpu_bake_ms
            );
        }
        if f > 32 && r.is_settled(32) {
            settled_at = Some(f);
            break;
        }
    }
    println!("settled at the start: {settled_at:?}");
    // `VR_PITCH=1`: from the settled start, the view tilts up a step at a
    // time (2°, `VR_STEP` frames each), printing the meter's exposure, the
    // share of pixels not black, and the CPU's trimmed-mean exposure
    // (black pixels skipped), then stops.
    if std::env::var_os("VR_PITCH").is_some() {
        let right = start_dir.cross(continuum::glam::Vec3::Z).normalize();
        let steps = num("VR_STEP", 60);
        for k in 0..=25 {
            let a = (2.0 * k as f32).to_radians();
            let d = continuum::glam::Quat::from_axis_angle(right, a) * start_dir;
            let cam = Camera {
                eye: camera.eye,
                target: camera.eye + d,
                vertical_fov_degrees: camera.vertical_fov_degrees,
            };
            for _ in 0..steps {
                render(&mut r, cam);
            }
            let (e, _) = r.metered_exposure();
            let colors = r.read_colors();
            let mut hist = [0u32; 64];
            let mut sampled = 0u32;
            for (i, c) in colors.iter().enumerate() {
                let (x, y) = (i as u32 % w, i as u32 / w);
                if x % 2 != 0 || y % 2 != 0 {
                    continue;
                }
                sampled += 1;
                let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                if l > 0.0 {
                    hist[((l.log2() + 16.0) * 2.0).clamp(0.0, 63.0) as usize] += 1;
                }
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
            let want = if weight > 0.0 { 0.18 / (sum / weight).exp2() } else { f64::NAN };
            let dark = colors
                .iter()
                .filter(|c| {
                    let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                    l > 0.0 && l < 1e-4
                })
                .count();
            println!(
                "pitch {:2}°: exposure {e:.4}, CPU's {want:.4}, not black {:.1}%, faint (0 < l < 1e-4) {:.1}%",
                2 * k,
                100.0 * total as f64 / sampled as f64,
                100.0 * dark as f64 / colors.len() as f64
            );
        }
        return;
    }
    // `VR_PRE`: frames more at the start, the last of them the reference.
    for _ in 0..num("VR_PRE", 0) {
        render(&mut r, pose(0.0));
    }
    let reference = grab(&r, &texture);
    let lower = |img: &[u8]| -> f32 {
        let mut sum = 0.0;
        for y in h / 2..h {
            for x in 0..w {
                let i = 4 * (y * w + x) as usize;
                sum += (linear(img[i]) + linear(img[i + 1]) + linear(img[i + 2])) / 3.0;
            }
        }
        sum / ((h - h / 2) * w) as f32
    };
    let dark_share = |img: &[u8]| -> f32 {
        let mut dark = 0usize;
        for i in (0..img.len()).step_by(4) {
            let a: f32 = (0..3).map(|k| linear(reference[i + k])).sum();
            let b: f32 = (0..3).map(|k| linear(img[i + k])).sum();
            if a > 0.01 && b < 0.5 * a {
                dark += 1;
            }
        }
        dark as f32 / (img.len() / 4) as f32
    };
    let save_list: Vec<i64> = std::env::var("VR_SAVE")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let total = 2 * turn + hold + after;
    let save = |f: usize, img: &[u8]| {
        if save_list
            .iter()
            .any(|&k| (k >= 0 && k as usize == f) || (k < 0 && (total as i64 + k) as usize == f))
        {
            let path = out.join(format!("frame_{f:04}.png"));
            let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = 4 * (y * w as usize + x);
                    image.set(x, y, [img[i], img[i + 1], img[i + 2]]);
                }
            }
            image.save(&path).expect("the frame");
        }
    };
    {
        let path = out.join("settled.png");
        let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = 4 * (y * w as usize + x);
                image.set(x, y, [reference[i], reference[i + 1], reference[i + 2]]);
            }
        }
        image.save(&path).expect("the frame");
    }
    println!("settled: lower half {:.4}", lower(&reference));
    for f in 0..total {
        let s = if f < turn {
            (f + 1) as f32 / turn as f32
        } else if f < turn + hold {
            1.0
        } else if f < 2 * turn + hold {
            1.0 - (f - turn - hold + 1) as f32 / turn as f32
        } else {
            0.0
        };
        let ms = render(&mut r, pose(s));
        let img = grab(&r, &texture);
        save(f, &img);
        let back = f >= 2 * turn + hold;
        let st = r.stats().unwrap_or_default();
        if std::env::var_os("VR_STATS").is_some() {
            println!(
                "  invalidated {} sky_invalidated {} missed {} deferred {} bounce {} incomplete {} requests {} fallback {}",
                st.invalidated,
                st.sky_invalidated,
                st.missed,
                st.deferred,
                st.indirect_samples,
                st.indirect_incomplete,
                st.requests,
                st.fallback_pixels
            );
            println!(
                "  budget {} cap {} spent {} by mip {:?} bake {:.2} ms render {:.2} ms planning {:.2} ms waiting {:.2} ms",
                st.budget,
                st.budget_cap,
                st.spent,
                st.spent_by_mip,
                st.gpu_bake_ms,
                st.gpu_render_ms,
                st.planning.as_secs_f64() * 1000.0,
                st.waiting.as_secs_f64() * 1000.0
            );
        }
        println!(
            "frame {f:4} turn {s:.2}: {ms:6.1} ms, fallback {}, resident {}, evicted {}, baked {}, lower half {:.4}{}",
            st.fallback_pixels,
            st.resident,
            st.evicted,
            st.pages_baked,
            lower(&img),
            if back {
                format!(", darker than settled {:.2}%", 100.0 * dark_share(&img))
            } else {
                String::new()
            }
        );
    }
}
