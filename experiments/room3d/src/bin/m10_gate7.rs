//! M10 gate 7: alpha cutouts (gate 6 re-run after a cache fix).
//!
//! Gate 6 never reached a criterion: its control (the same scene with the
//! cards opaque) never settled. The CPU computed a chart's texel grid one
//! texel smaller than the shader did (1.4 m × scale 1.2 at 100 texels per
//! metre: 168 against the shader's f32 169), refused the pages the shader
//! asked for just past its own grid, and they were asked for every frame.
//! The plan now allows that extra texel. Criteria and scene are gate 6's.
//!
//! Foliage, fences and chains in glTF are cards whose base color alpha
//! (alphaMode MASK) cuts their shape. Rendered opaque, they are solid
//! rectangles that cast rectangular shadows.
//!
//! Method: the cut becomes geometry, so every path stays exact with no
//! change: camera rays, light and sky cones, the cache and bounce light.
//! - `InstancedScene::geometry_alpha_cutoff`: per geometry, a cutoff for
//!   its base color texture's alpha.
//! - Before upload, each masked triangle is replaced by the opaque part of
//!   its texture: the contour of the bilinear alpha field at the cutoff
//!   (marching squares on texel centers, crossings interpolated linearly),
//!   at the mip where the triangle spans at most 128 texels, clipped to the
//!   triangle in texture space and mapped back through its barycentric
//!   coordinates (positions, normals, texture coordinates).
//! - Runs of fully opaque cells merge into rectangles.
//!
//! Scene: the textured scene of M6 (sun and sky), its camera, 1280×720,
//! bounce light on. Added: shrubs made of crossed double-sided leaf cards
//! and a lattice fence panel, both alpha-masked (cutoff 0.5) with textures
//! generated here.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Accuracy:** the settled cached image, supersampled 4×4, within 5%
//!    relative RMSE of instanced Cycles masking the same cards by the same
//!    alpha (a Transparent BSDF where alpha is below the cutoff; one
//!    bounce, 4,096 samples per pixel), and at most 0.5× the error of the
//!    same scene with the cards opaque against that reference.
//! 2. **Speed:** the warm frame at most 1.5× the opaque cards' (median of
//!    30 frames each waited for, bounce light on in both).
//!
//! Reported, not gated: triangles before and after cutting; the cut's CPU
//! time.
//!
//! Changed before the first full-size run, after development runs: each
//! card's back faces sit 1 mm behind its front ones. With exactly
//! back-to-back faces the cache's bakes were about 100× slower (opaque
//! cards too, so not the cut; cause not yet established).
//!
//! Details fixed before the first run: as M10 gate 2 (settled: 32 quiet
//! frames after the call; 4×4 supersampling over sub-pixel offsets, each
//! settled; Cycles at 4,096 samples per pixel, its one-pixel box filter).
//!
//! `cargo run --release -p room3d --bin m10_gate7`
//!
//! `M10_DEV=1` saves the settled image alone (no Cycles): a development
//! check, whose numbers are not the gate's.

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
/// frames it took.
fn settle(
    r: &mut InstancedRenderer,
    camera: continuum_gpu::Camera,
    target: &wgpu::TextureView,
) -> usize {
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
            return frames;
        }
        assert!(frames < 50_000, "the view never settled");
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
        settle(r, camera, target);
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
    let t = Textured::new().with_foliage();
    let mut flat = Textured::new().with_foliage();
    flat.scene.geometry_alpha_cutoff.clear();
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
        image.save(&out_dir.join(format!("m10_gate7_{name}.png")))
    };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };

    if std::env::var_os("M10_DEV").is_some() {
        println!("DEVELOPMENT RUN: not the gate's numbers");
        let start = Instant::now();
        let cut = t.scene.with_cutouts();
        let n = cut.geometries.len();
        println!(
            "cut in {:.0} ms: {} and {} triangles",
            ms(start.elapsed()),
            cut.geometries[n - 2].triangles.len(),
            cut.geometries[n - 1].triangles.len()
        );
        let which = if std::env::var("M10_DEV").as_deref() == Ok("opaque") {
            &flat
        } else {
            &t
        };
        let mut r = renderer(which);
        let begin = Instant::now();
        for k in 0..400 {
            r.frame(t.camera, &target);
            if k % 50 == 49 {
                r.finish();
                let s = r.cache_stats().unwrap_or_default();
                println!(
                    "frame {}: {:.1} s, requests {}, baked {}, budget {}, bake {:.1} ms",
                    k + 1,
                    begin.elapsed().as_secs_f64(),
                    s.requests,
                    s.pages_baked,
                    s.budget,
                    s.gpu_bake_ms
                );
            }
        }
        let frames = settle(&mut r, t.camera, &target);
        r.finish();
        save("dev", &r.read_colors())?;
        println!("settled in {frames} frames; saved out/m10_gate7_dev.png");
        return Ok(());
    }

    // Reported: the cut's cost and size.
    let start = Instant::now();
    let cut = t.scene.with_cutouts();
    let cut_ms = ms(start.elapsed());
    let triangles = |s: &continuum_gpu::InstancedScene| -> usize {
        let n = s.geometries.len();
        s.geometries[n - 2..]
            .iter()
            .map(|g| g.triangles.len())
            .sum()
    };
    say(format!(
        "reported: cut {} card triangles into {} in {cut_ms:.0} ms (CPU)",
        triangles(&t.scene),
        triangles(&cut)
    ));
    drop(cut);

    // 2. Speed, first.
    let mut plain = renderer(&flat);
    let plain_frames = settle(&mut plain, t.camera, &target);
    let plain_warm = warm(&mut plain, t.camera, &target);
    let mut mapped = renderer(&t);
    let mapped_frames = settle(&mut mapped, t.camera, &target);
    let mapped_warm = warm(&mut mapped, t.camera, &target);
    let c2 = ms(mapped_warm) <= 1.5 * ms(plain_warm);
    say(format!(
        "2. speed: warm {:.2} ms with cutouts, {:.2} ms opaque cards ({:.2}×, at most 1.5×); settled in {mapped_frames} and {plain_frames} frames: {}",
        ms(mapped_warm),
        ms(plain_warm),
        ms(mapped_warm) / ms(plain_warm),
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 1. Accuracy against Cycles masking the cards.
    let with = supersampled(&mut mapped, t.camera, &target);
    let without = supersampled(&mut plain, t.camera, &target);
    let cycles = |scene: &Textured, dir: &str| {
        render_instanced_lights(
            &scene.scene,
            scene.origin,
            &scene.lights,
            scene.sky,
            scene.camera,
            (W, H),
            4096,
            1,
            1,
            &out_dir.join(dir),
        )
    };
    let reference = cycles(&t, "m10_gate7_cycles")?;
    let ours = relative_rmse(&with, &reference.pixels);
    let ignoring = relative_rmse(&without, &reference.pixels);
    let c1 = ours <= 0.05 && ours <= 0.5 * ignoring;
    say(format!(
        "1. accuracy against Cycles masking the cards ({:.0} s, {}): {:.2}% (at most 5%), opaque cards {:.2}% (ours at most 0.5×: {:.2}×): {}",
        reference.seconds,
        reference.device,
        100.0 * ours,
        100.0 * ignoring,
        ours / ignoring,
        if c1 { "PASS" } else { "FAIL" }
    ));
    save("ours", &with)?;
    save("opaque", &without)?;
    save("cycles", &reference.pixels)?;
    say(format!(
        "Verdict: 1 {}, 2 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c1 && c2 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(out_dir.join("m10_gate7.txt"), report)?;
    Ok(())
}
