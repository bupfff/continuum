//! Development tool for M10 gate 17's speed miss (not a gate): warm frames
//! of the vertex-colored textured scene and of the same scene with colors
//! ignored, alternated (plain, colored, plain, colored) to expose drift.
//! `VCOLOR_DEV=boulders` or `=block` colors only that geometry;
//! `VCOLOR_GATE_ORDER=1` measures in the gate's order instead.
//!
//! `cargo run --release -p room3d --bin vcolor_dev`

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
    let t = Textured::new().with_vertex_colors();
    let mut t = t;
    match std::env::var("VCOLOR_DEV").as_deref() {
        Ok("boulders") => t.scene.geometry_colors[2] = None,
        Ok("block") => t.scene.geometry_colors[1] = None,
        _ => {}
    }
    let mut flat = Textured::new().with_vertex_colors();
    flat.scene.geometry_colors.clear();
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
    if std::env::var_os("VCOLOR_GATE_ORDER").is_some() {
        // Gate 17's order: plain settled and timed, then colored.
        let mut plain = renderer(false);
        settle(&mut plain, t.camera, &target);
        let p = ms(warm(&mut plain, t.camera, &target));
        let mut colored = renderer(true);
        settle(&mut colored, t.camera, &target);
        let c = ms(warm(&mut colored, t.camera, &target));
        println!(
            "gate order: plain {p:.2} ms, colored {c:.2} ms ({:.3}×)",
            c / p
        );
        let p2 = ms(warm(&mut plain, t.camera, &target));
        println!("then plain again {p2:.2} ms");
        return Ok(());
    }
    let mut plain = renderer(false);
    let mut colored = renderer(true);
    settle(&mut plain, t.camera, &target);
    settle(&mut colored, t.camera, &target);
    for round in 0..2 {
        let p = ms(warm(&mut plain, t.camera, &target));
        let c = ms(warm(&mut colored, t.camera, &target));
        println!(
            "round {round}: plain {p:.2} ms, colored {c:.2} ms ({:.3}×)",
            c / p
        );
    }
    Ok(())
}
