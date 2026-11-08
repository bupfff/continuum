//! M10 gate 9: exactly back-to-back faces (gate 8 re-run after a fix).
//!
//! Gate 8 passed on speed and failed on accuracy (5.01% from Cycles, 3.78%
//! from the control): Metal's hardware ray queries ignore
//! `CULL_BACK_FACING` (naga 30 sets no triangle cull mode), so a camera ray
//! could hit either of a card's coincident faces, and the cache read the
//! back face's texels (turned away from the sun: unshadowed) for a pixel on
//! the front. Fix: a camera ray whose hardware hit is a back face is traced
//! again with its candidates tested for facing (the nearest front face, as
//! the software `camera_hit` finds). Criteria and scene are gate 8's.
//!
//! Double-sided cards (foliage, fences) and double-sided glTF materials are
//! a front face and a back face on the same plane. With the back faces
//! exactly behind the front ones, the cache's bakes were 100× slower or
//! more (M10 gate 6), so cards and glTF import put them 1 mm behind.
//!
//! Cause, found in development (`backface_dev`, disclosed): bounce light
//! alone is slow (sun and sky bakes are not; 1,793 µs a baked texel
//! against 2.67 µs with 1 mm, opaque cards, sun, sky and bounce). A mesh
//! without normals gets smooth normals summed over every triangle at a
//! vertex's position; a front face and its back face cancel, the shading
//! normal is zero, and the shader's `normalize` makes it NaN. Bounce rays
//! from those texels have NaN directions, which pass every box's slab test
//! and visit the whole scene.
//!
//! Gate 8's fix (commit `e876b61`):
//! - smooth normals sum only the triangles at a position that face the
//!   same side as the corner's own (positive dot of face normals);
//! - the shader takes the geometric normal where an interpolated normal is
//!   degenerate (zero or NaN);
//! - bounce rays: of two hits at the same distance within rounding, the one
//!   facing the ray wins (a back face counts as "inside geometry", no
//!   light, so a coincident pair seen from behind must not pick the front);
//! - cards and glTF double-sided materials put the back faces exactly
//!   behind (no 1 mm offset).
//!
//! Scene: M10 gate 7's (the textured scene of M6, sun and sky, its camera,
//! 1280×720, bounce light on; shrubs of crossed leaf cards and a lattice
//! fence, alpha-masked at 0.5), with the cards' back faces exactly behind
//! the front ones; the control is the same scene with them 1 mm behind.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Speed** (measured first): the exactly back-to-back scene settles
//!    from cold within 5,000 frames, its GPU bake time summed over the
//!    frames until it settles is at most 1.5× the control's, and its warm
//!    frame is at most 1.25× the control's (median of 30 frames each
//!    waited for).
//! 2. **Accuracy:** its settled image, supersampled 4×4, is within 5%
//!    relative RMSE of M10 gate 7's Cycles reference (the control's
//!    geometry; reused from `out/m10_gate7_cycles`), and within 0.5% of the
//!    control's settled image supersampled the same way.
//!
//! Reported, not gated: frames and wall time to settle; frames whose bake
//! was not timed on the GPU (left out of the sum).
//!
//! Details fixed before the first run: as M10 gate 7 (settled: 32 quiet
//! frames after the call; 4×4 supersampling over sub-pixel offsets, each
//! settled; Cycles at 4,096 samples per pixel, its one-pixel box filter).
//!
//! `cargo run --release -p room3d --bin m10_gate9`

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use continuum::glam::Vec3;
use continuum_gpu::{CacheOptions, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until 32 consecutive frames requested and baked nothing; the
/// frames it took, or None past `cap`.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
    cap: usize,
) -> Option<usize> {
    let mut frames = 0;
    loop {
        r.frame(camera, target);
        frames += 1;
        let log = r.cache_log();
        let latest = r
            .cache_stats()
            .map_or(0, |s| s.frame as usize)
            .min(log.len());
        if latest >= 32
            && log[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return Some(frames);
        }
        if frames >= cap {
            return None;
        }
    }
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

/// The view supersampled 4×4: each sub-pixel offset's frame after it
/// settles, averaged.
fn supersampled(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> Vec<[f32; 3]> {
    let mut sum = vec![[0.0f32; 3]; (W * H) as usize];
    r.set_footprint(0.25);
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        settle(r, camera, target, 50_000).expect("the view never settled");
        r.finish();
        for (s, p) in sum.iter_mut().zip(r.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    r.set_footprint(1.0);
    sum
}

/// A cold start: frames to settle (None past 5,000), wall time, the GPU
/// bake time summed over the timed frames, and the untimed frames.
struct Cold {
    frames: Option<usize>,
    seconds: f64,
    bake_ms: f64,
    untimed: usize,
}

fn cold(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> Cold {
    let start = Instant::now();
    let frames = settle(r, camera, target, 5_000);
    r.finish();
    let seconds = start.elapsed().as_secs_f64();
    let log = r.cache_log();
    Cold {
        frames,
        seconds,
        bake_ms: log
            .iter()
            .filter(|s| s.timed)
            .map(|s| s.gpu_bake_ms as f64)
            .sum(),
        untimed: log.iter().filter(|s| !s.timed).count(),
    }
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
    let exact = Textured::new().with_foliage_back(0.0);
    let control = Textured::new().with_foliage_back(0.001);
    kernel.set_sky(exact.sky);
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
    let renderer = |scene: &Textured| {
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
    let save = |name: &str, px: &[[f32; 3]]| -> std::io::Result<()> {
        let mut image = flatland::img::Image::new(W as usize, H as usize, [0; 3]);
        for (i, p) in px.iter().enumerate() {
            image.set(
                i % W as usize,
                i / W as usize,
                room3d::to_rgb(Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("m10_gate9_{name}.png")))
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    // 1. Speed, first.
    let mut base = renderer(&control);
    let base_cold = cold(&mut base, control.camera, &target);
    let base_warm = warm(&mut base, control.camera, &target);
    let mut ours = renderer(&exact);
    let ours_cold = cold(&mut ours, exact.camera, &target);
    let ours_warm = warm(&mut ours, exact.camera, &target);
    let c1 = ours_cold.frames.is_some()
        && base_cold.frames.is_some()
        && ours_cold.bake_ms <= 1.5 * base_cold.bake_ms
        && ms(ours_warm) <= 1.25 * ms(base_warm);
    let frames = |c: &Cold| c.frames.map_or("did not settle in 5000".into(), |f| f.to_string());
    say(format!(
        "1. speed: bake to settle {:.0} ms GPU exactly back-to-back, {:.0} ms with 1 mm ({:.2}×, at most 1.5×); warm {:.2} ms and {:.2} ms ({:.2}×, at most 1.25×); settled in {} and {} frames ({:.1} s and {:.1} s; untimed frames {} and {}): {}",
        ours_cold.bake_ms,
        base_cold.bake_ms,
        ours_cold.bake_ms / base_cold.bake_ms,
        ms(ours_warm),
        ms(base_warm),
        ms(ours_warm) / ms(base_warm),
        frames(&ours_cold),
        frames(&base_cold),
        ours_cold.seconds,
        base_cold.seconds,
        ours_cold.untimed,
        base_cold.untimed,
        if c1 { "PASS" } else { "FAIL" }
    ));

    // 2. Accuracy (only if both settled).
    let mut c2 = false;
    if ours_cold.frames.is_some() && base_cold.frames.is_some() {
        let ours_px = supersampled(&mut ours, exact.camera, &target);
        let base_px = supersampled(&mut base, control.camera, &target);
        let reference = render_instanced_lights(
            &control.scene,
            control.origin,
            &control.lights,
            control.sky,
            control.camera,
            (W, H),
            4096,
            1,
            1,
            &out_dir.join("m10_gate7_cycles"),
        )?;
        let to_cycles = relative_rmse(&ours_px, &reference.pixels);
        let base_to_cycles = relative_rmse(&base_px, &reference.pixels);
        let to_base = relative_rmse(&ours_px, &base_px);
        c2 = to_cycles <= 0.05 && to_base <= 0.005;
        say(format!(
            "2. accuracy: {:.2}% from Cycles (at most 5%; the control {:.2}%), {:.3}% from the control (at most 0.5%): {}",
            100.0 * to_cycles,
            100.0 * base_to_cycles,
            100.0 * to_base,
            if c2 { "PASS" } else { "FAIL" }
        ));
        save("ours", &ours_px)?;
        save("control", &base_px)?;
    } else {
        say("2. accuracy: not measured (a scene did not settle): FAIL".into());
    }
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m10_gate9.txt"), report)?;
    Ok(())
}
