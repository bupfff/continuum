//! Development tool for alpha blending's speed (M10 gate 16, not a gate):
//! each camera pass of the glass scene and of the same scene with the panes
//! opaque, submitted and waited for alone (`frame_profiled`), medians of 30
//! warm frames.
//!
//! `cargo run --release -p room3d --bin blend_dev`

use std::path::Path;
use std::time::{Duration, Instant};

use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 frames rendered since the call requested and baked
/// nothing; the frames it took.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> usize {
    let start = r.cache_log().len();
    for k in 1..=50_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return k;
        }
    }
    panic!("the view never settled");
}

/// The median of 30 warm frames, each waited for.
fn warm(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> Duration {
    let mut times: Vec<Duration> = (0..30)
        .map(|_| {
            let start = Instant::now();
            r.frame(camera, target);
            r.wait();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[15]
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
    let t = Textured::new().with_glass(true);
    let mut flat = Textured::new().with_glass(true);
    flat.scene.geometry_blend.clear();
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
    let renderer = |colored: bool| {
        let scene = if colored { &t } else { &flat };
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &scene.lights,
            &scene.scene,
            scene.origin,
            CacheOptions {
                indirect: true,
                ..Default::default()
            },
        )
    };
    for blend in [false, true] {
        let mut r = renderer(blend);
        settle(&mut r, t.camera, &target);
        let p = ms(warm(&mut r, t.camera, &target));
        let mut cpu: Vec<f64> = (0..30)
            .map(|_| {
                r.wait();
                let start = Instant::now();
                r.frame(t.camera, &target);
                let c = ms(start.elapsed());
                r.wait();
                c
            })
            .collect();
        cpu.sort_by(f64::total_cmp);
        println!(
            "{}: frame() on the CPU {:.2} ms (median)",
            if blend { "blended" } else { "opaque" },
            cpu[15]
        );
        let log = r.cache_log();
        let mut gpu: Vec<f32> = log[log.len() - 30..]
            .iter()
            .map(|s| s.gpu_render_ms)
            .collect();
        gpu.sort_by(f32::total_cmp);
        println!(
            "{}: GPU camera passes {:.2} ms (median of the warm frames)",
            if blend { "blended" } else { "opaque" },
            gpu[15]
        );
        let mut rows: Vec<[f64; 5]> = (0..30)
            .map(|_| {
                let f = r.frame_profiled(t.camera, &target);
                [
                    ms(f.passes[0]),
                    ms(f.passes[1]),
                    ms(f.passes[2]),
                    ms(f.passes[3]),
                    ms(f.readback),
                ]
            })
            .collect();
        let median = |k: usize| {
            rows.sort_by(|a, b| a[k].total_cmp(&b[k]));
            rows[15][k]
        };
        let m: Vec<f64> = (0..5).map(median).collect();
        println!(
            "{}: warm {p:.2} ms; passes alone {:.2} / {:.2} / {:.2} / {:.2} ms, readback {:.2} ms",
            if blend { "blended" } else { "opaque" },
            m[0],
            m[1],
            m[2],
            m[3],
            m[4]
        );
    }
    Ok(())
}
