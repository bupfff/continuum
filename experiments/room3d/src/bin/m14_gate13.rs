//! M14 gate 13: gate 12 with a fixed bake budget.
//!
//! Gate 12 (coarser rock levels shade with the finest's normals) passed
//! criteria 1 and 2 and failed criterion 3 at one view by 0.0045%: the
//! resident (finest-only) images, which the change does not touch, varied
//! with the time-based bake budget's timing. With a fixed budget they were
//! identical (diagnostic after gate 12).
//!
//! The same scene, renderers and criteria as gate 12, every renderer with a
//! fixed bake budget (`frame_target_ms` 0, the default 20,000 units):
//!
//! Pass criteria, fixed before the run:
//! 1. **Shading against the finest:** the colour difference (relative RMSE)
//!    streamed against resident with the finest's normals at most 0.75× that
//!    with the levels' own, at each of the three flights' last views.
//! 2. **Production:** a rock's levels 1 and 2 produced in at most 1.5× the
//!    time (medians of five interleaved rounds over 64 rocks).
//! 3. **The finest unchanged:** the resident renderers' images identical
//!    with and without (a first pass rendered and discarded).
//!
//! **Disclosed (before the run):** gate 12's diagnostic with the fixed budget
//! (`M14_FIXED=1`): 0.649×, 0.641×, 0.561×; 0.87×; resident images identical.
//!
//! `M14_DEV=1`: a development check on 36 rocks; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m14_gate13`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, StreamOptions, wgpu};
use flatland::rng::Pcg32;
use room3d::cycles::relative_rmse;
use room3d::rocks::{Detail, Rocks, Shape};

const W: u32 = 1280;
const H: u32 = 720;

/// Renders until the cache and streaming are quiet for 16 frames (at most
/// 4,000), as M14 gate 1.
fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    for f in 1..=4000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        let cache_quiet = latest >= 16
            && r.cache_log()[latest - 16..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0);
        let log = r.stream_log();
        let stream_quiet = log.len() < 16
            || log[log.len() - 16..]
                .iter()
                .all(|s| s.loads == 0 && s.evictions == 0);
        if cache_quiet && stream_quiet && f >= 16 {
            r.finish();
            return f;
        }
    }
    r.finish();
    4000
}

/// 4×4 sub-pixel offsets averaged, one frame each (as M14 gate 1).
fn supersampled(
    r: &mut InstancedRenderer,
    camera: Camera,
    target: &wgpu::TextureView,
) -> Vec<[f32; 3]> {
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    for k in 0..16 {
        let (dx, dy) = ((k % 4) as f32 * 0.25 - 0.375, (k / 4) as f32 * 0.25 - 0.375);
        r.set_pixel_offset(dx, dy);
        r.frame(camera, target);
        r.finish();
        for (c, s) in color.iter_mut().zip(r.read_colors()) {
            for i in 0..3 {
                c[i] += s[i] / 16.0;
            }
        }
    }
    r.set_pixel_offset(0.0, 0.0);
    color
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M14_DEV").is_some();
    let side = if dev { 6 } else { 16 };
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if dev {
        say("DEVELOPMENT RUN: 36 rocks; not the gate's numbers".into());
    }
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_sky(Vec3::ZERO);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());

    // 2. Production, first (CPU only, before the GPU work heats the machine).
    let produce = |finest: bool| {
        let mut rng = Pcg32::new(14, 1);
        let shapes: Vec<Shape> = (0..64)
            .map(|_| {
                Shape::random(
                    &mut rng,
                    Detail {
                        finest_normals: finest,
                        ..Default::default()
                    },
                )
            })
            .collect();
        let start = Instant::now();
        for s in &shapes {
            std::hint::black_box(s.surface(1));
            std::hint::black_box(s.surface(2));
        }
        start.elapsed().as_secs_f64() * 1e3 / shapes.len() as f64
    };
    // Five interleaved rounds, the medians (CPU timings wander).
    let _ = produce(false);
    let (mut own, mut fin) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        own.push(produce(false));
        fin.push(produce(true));
    }
    own.sort_by(f64::total_cmp);
    fin.sort_by(f64::total_cmp);
    let (own_ms, finest_ms) = (own[2], fin[2]);
    let c2 = finest_ms <= 1.5 * own_ms;
    say(format!(
        "2. production: levels 1 and 2 of a rock in {finest_ms:.2} ms with the finest's normals against {own_ms:.2} ms ({:.2}×; at most 1.5×): {}",
        finest_ms / own_ms,
        if c2 { "PASS" } else { "FAIL" }
    ));

    // 1 and 3.
    let mut images: Vec<Vec<Vec<[f32; 3]>>> = Vec::new();
    let same_twice = std::env::var_os("M14_SAME").is_some();
    // A first pass, discarded: a renderer's first instance in a process
    // settles to images a little different from later ones (development).
    let order: &[bool] = &[false, false, true];
    for &finest in order {
        let finest = finest && !same_twice;
        let rocks = Rocks::with_detail(
            side,
            6.0,
            Detail {
                finest_normals: finest,
                ..Default::default()
            },
        );
        let half = rocks.half as f32;
        let make = |options: StreamOptions| {
            InstancedRenderer::with_streaming(
                &kernel,
                format,
                W,
                H,
                1,
                &[rocks.sun],
                &rocks.scene,
                DVec3::ZERO,
                Some(CacheOptions {
                    frame_target_ms: 0.0,
                    ..Default::default()
                }),
                options,
            )
        };
        let mut streamed = make(StreamOptions {
            budget_triangles: 5 << 19,
            ..Default::default()
        });
        let mut resident = make(StreamOptions {
            resident: true,
            ..Default::default()
        });
        let mut set = Vec::new();
        for k in 0..3 {
            let last = room3d::rocks::flight(k, 1.0, half, 50.0);
            settle(&mut resident, last, &target);
            set.push(supersampled(&mut resident, last, &target));
            settle(&mut streamed, last, &target);
            set.push(supersampled(&mut streamed, last, &target));
        }
        images.push(set);
    }
    if images.len() == 3 {
        images.remove(0);
    }
    let mut c1 = true;
    let mut c3 = true;
    for k in 0..3 {
        let (own_res, own_str) = (&images[0][2 * k], &images[0][2 * k + 1]);
        let (fin_res, fin_str) = (&images[1][2 * k], &images[1][2 * k + 1]);
        let before = relative_rmse(own_str, own_res);
        let after = relative_rmse(fin_str, fin_res);
        let ok = after <= 0.75 * before;
        c1 &= ok;
        let same = relative_rmse(fin_res, own_res);
        c3 &= same == 0.0;
        say(format!(
            "1. view {k}: streamed against resident {:.3}% with the finest's normals, {:.3}% with the levels' own ({:.3}×; at most 0.75×): {}; 3. the resident images {:.6}% apart",
            100.0 * after,
            100.0 * before,
            after / before,
            if ok { "PASS" } else { "FAIL" },
            100.0 * same
        ));
    }
    say(format!(
        "1. shading against the finest: {}",
        if c1 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "3. the finest unchanged: {}",
        if c3 { "PASS" } else { "FAIL" }
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {} → {}",
        if c1 { "PASS" } else { "FAIL" },
        if c2 { "PASS" } else { "FAIL" },
        if c3 { "PASS" } else { "FAIL" },
        if c1 && c2 && c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m14_gate13_dev.txt"
        } else {
            "m14_gate13.txt"
        }),
        report,
    )?;
    Ok(())
}
