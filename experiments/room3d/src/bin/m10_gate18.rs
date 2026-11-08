//! M10 gate 18: vertex colors' speed (gate 17 re-measured).
//!
//! Gate 17 passed accuracy (1.29% from Cycles, 0.02× the error with the
//! colors ignored) and failed speed: 3.49 ms against 3.05 ms (1.15×, limit
//! 1.1×). Development runs after it (`vcolor_dev`, disclosed in
//! `m10-gate17.md`, seen before this gate was written) measured 0.998–1.009×
//! in four tries, including gate 17's own order: its colored warm frames
//! met a transient. The kernel is unchanged since gate 17, so its accuracy
//! result stands; this gate measures speed alone, with a method that a
//! single transient cannot decide.
//!
//! Scene, renderers and settling are gate 17's (M6's textured scene with
//! `Textured::with_vertex_colors`, 1280×720, bounce light on; the
//! comparison is the same scene with the colors ignored).
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed:** both renderers settled, then five rounds, each the median
//!    of 30 warm frames (each waited for) of the colors ignored and then of
//!    the colored scene; the median over rounds of the colored-to-ignored
//!    ratio at most 1.1×.
//!
//! `cargo run --release -p room3d --bin m10_gate18`

use std::fmt::Write as _;
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
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let mut plain = renderer(false);
    let mut colored = renderer(true);
    let frames = (
        settle(&mut plain, t.camera, &target),
        settle(&mut colored, t.camera, &target),
    );
    let mut ratios = Vec::new();
    for round in 0..5 {
        let p = ms(warm(&mut plain, t.camera, &target));
        let c = ms(warm(&mut colored, t.camera, &target));
        ratios.push(c / p);
        say(format!(
            "round {round}: ignored {p:.2} ms, colored {c:.2} ms ({:.3}×)",
            c / p
        ));
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[2];
    let c1 = median <= 1.1;
    say(format!(
        "1. speed: median ratio {median:.3}× over five rounds (at most 1.1×); settled in {} and {} frames: {}",
        frames.0,
        frames.1,
        if c1 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m10_gate18.txt"), report)?;
    Ok(())
}
