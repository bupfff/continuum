//! M5 gate 9: sky warm-up, the sky later.
//!
//! Views with a sky warm slowly: in M5 gate 5, 13–95× the sun-only cache's
//! frames, and still flights with sky show 21–22% fallback at 720p (M9 gate
//! 3). A sky sample costs ten sun texels, so a page with sky costs about 11×
//! one without. Until a pixel's page arrives it shows unshadowed light: no
//! sun shadows at all.
//!
//! Method (`CacheOptions::sky_later`): a page the pixels ask for is baked
//! with its lights only, its sky shown as the open sky over the pixel's
//! normal (the guess M9 gate 3 already shows on moved pages), and listed to
//! have its sky baked alone from the next frame, within what each frame's
//! budget leaves after new pages. Sun shadows arrive about 11× sooner; the
//! sky follows. Bounce samples treat a page whose sky is a guess as missing.
//! The converged cache is the same.
//!
//! Scene: the million-boulder field (sun 3 km, `Field::sky`), M5 gate 5's
//! four viewpoints at 1280×720, cache defaults except a fixed bake budget of
//! 11,000 units a frame (`frame_target_ms` 0; about gate 5's flights' mean
//! budget), so frame counts do not depend on the machine's heat. Prefilter
//! 1 bin with object-level prefiltering, sky prefilter ×8, hardware camera
//! rays.
//!
//! Runs per viewpoint: the baseline warmed once (its warm image is the
//! reference), then warmed again and the sky-later cache warmed, each read
//! back every 20th frame.
//!
//! Pass criterion, fixed before the first run, at every viewpoint:
//! 1. **Same result:** the sky-later warm image within 0.1% relative RMSE
//!    of the baseline's.
//! 2. **Frames until warm** (16 quiet frames, as gate 5) at most 1.2× the
//!    baseline's.
//! 3. **Warm-up quality:** relative RMSE against the baseline's warm image,
//!    averaged over the frames read back up to the baseline's frames until
//!    warm, at most 0.5× the baseline's (its second run) over the same
//!    frames.
//!
//! Reported, not gated: the share of pixels shown with fallback (no page,
//! or a guessed sky) averaged over the same frames; the frames until the
//! image is within 5% of the reference.
//!
//! `cargo run --release -p room3d --bin m5_gate9`

use std::fmt::Write as _;
use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const SKY_PREFILTER: f32 = 8.0;
const BUDGET: u32 = 11_000;
const EVERY: usize = 20;

/// A warm-up from an empty cache: frames until 16 quiet frames, the warm
/// image, and every `EVERY`th frame's error against `reference` (if given)
/// with its fallback share.
struct WarmUp {
    frames: Option<usize>,
    image: Vec<[f32; 3]>,
    samples: Vec<(usize, f64, f64)>,
}

fn warm_up(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
    reference: Option<&[[f32; 3]]>,
) -> WarmUp {
    let mut samples = Vec::new();
    let mut frames = None;
    for k in 1..=60_000usize {
        r.frame(camera, target);
        if let Some(reference) = reference
            && k % EVERY == 0
        {
            r.finish();
            let image = r.read_colors();
            let fallback = r.cache_stats().map_or(0, |s| s.fallback_pixels) as f64 / (W * H) as f64;
            samples.push((k, relative_rmse(&image, reference), fallback));
        }
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            frames = Some(latest - 16);
            break;
        }
    }
    r.finish();
    WarmUp {
        frames,
        image: r.read_colors(),
        samples,
    }
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(SKY_PREFILTER);
    kernel.set_hardware_rays(true);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    kernel.set_sky(field.sky());
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
    let cached = |sky_later: bool| {
        InstancedRenderer::with_cache(
            &kernel,
            format,
            W,
            H,
            1,
            light,
            &field.scene,
            origin,
            field.radiance,
            CacheOptions {
                bake_budget: BUDGET,
                frame_target_ms: 0.0,
                sky_later,
                ..Default::default()
            },
        )
    };
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let view = |ahead: f32, height: f32, down: f32| {
        let d = down.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let views = [
        ("standing", view(0.0, 1.7, 0.0)),
        ("10 m up", view(0.0, 10.0, 15.0)),
        ("40 m up", view(0.0, 40.0, 30.0)),
        ("150 m up", view(0.0, 150.0, 45.0)),
    ];

    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let mut pass = [true; 3];
    for (name, camera) in views {
        let reference = warm_up(&mut cached(false), camera, &target, None);
        let Some(base_frames) = reference.frames else {
            say(format!("{name}: the baseline never warmed: FAIL"));
            pass = [false; 3];
            continue;
        };
        let base = warm_up(&mut cached(false), camera, &target, Some(&reference.image));
        let later = warm_up(&mut cached(true), camera, &target, Some(&reference.image));
        // Over the frames read back up to the baseline's frames until warm;
        // a run that warmed sooner counts its last error (zero work left)
        // for the frames after.
        let mean = |w: &WarmUp, column: fn(&(usize, f64, f64)) -> f64| -> f64 {
            let at = |k: usize| {
                w.samples
                    .iter()
                    .take_while(|s| s.0 <= k)
                    .last()
                    .map_or(0.0, column)
            };
            let ks: Vec<usize> = (1..=base_frames / EVERY).map(|i| i * EVERY).collect();
            ks.iter().map(|&k| at(k)).sum::<f64>() / ks.len().max(1) as f64
        };
        let within = |w: &WarmUp| {
            w.samples
                .iter()
                .find(|s| s.1 <= 0.05)
                .map_or("never".to_string(), |s| s.0.to_string())
        };
        let same = relative_rmse(&later.image, &reference.image);
        let c1 = same <= 0.001;
        let c2 = later.frames.is_some_and(|f| f as f64 <= 1.2 * base_frames as f64);
        let (e_base, e_later) = (mean(&base, |s| s.1), mean(&later, |s| s.1));
        let c3 = e_later <= 0.5 * e_base;
        pass[0] &= c1;
        pass[1] &= c2;
        pass[2] &= c3;
        let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
        say(format!(
            "{name}: 1. same result {:.4}% (at most 0.1%): {}; 2. frames until warm {} against {} ({:.2}×, at most 1.2×; the baseline's second run {}): {}; 3. mean error while warming {:.2}% against {:.2}% ({:.2}×, at most 0.5×): {}; reported: fallback {:.1}% against {:.1}%, within 5% at frame {} against {}",
            100.0 * same,
            verdict(c1),
            later.frames.map_or("never".into(), |f| f.to_string()),
            base_frames,
            later.frames.map_or(f64::NAN, |f| f as f64 / base_frames as f64),
            base.frames.map_or("never".into(), |f| f.to_string()),
            verdict(c2),
            100.0 * e_later,
            100.0 * e_base,
            e_later / e_base,
            verdict(c3),
            100.0 * mean(&later, |s| s.2),
            100.0 * mean(&base, |s| s.2),
            within(&later),
            within(&base),
        ));
    }
    let verdict = |c: bool| if c { "PASS" } else { "FAIL" };
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        verdict(pass[0]),
        verdict(pass[1]),
        verdict(pass[2]),
        verdict(pass.iter().all(|&p| p))
    ));
    std::fs::write(out_dir.join("m5_gate9.txt"), report)?;
    Ok(())
}
