//! M22 gate 5: the time budget steered by the frame period's median.
//!
//! Gate 4 kept the time budget's GPU model only for frames whose
//! timestamps arrive in order; on this device the frame period then steers
//! the budget, and a lone stall in it (a caller reading the image back, now
//! counted since M5 gate 14) cut the budget: M14 gate 11's readback warm-up
//! failed (1.38–1.50×), and while a view's pages were missing the cap
//! starved the bake (gate 4's settling).
//!
//! Method (DESIGN §3.4): the budget follows the median of the last three
//! frame periods (sustained waits count, a lone one does not); where frames
//! with the bake at its floor already overrun the frame budget or target,
//! frames may take a quarter more than those (`CacheOptions::floor_overrun`,
//! 0.25), so the bake still progresses.
//!
//! Pass criterion, fixed before the run (bars set with development known):
//! 1. **The frame budget** (M14 gate 11's criterion 1, its binary run in
//!    full with this code, `docs/results/m22-gate5-m14.txt`): each flight's
//!    mean and 99th percentile at most 16.7 ms, worst at most 33.3 ms.
//! 2. **Readbacks** (M14 gate 11's criterion 2, the same run): each view
//!    warms in at most 1.25× its undisturbed frames reading back every 20th
//!    frame (development 1.06–1.09× on its development scenes).
//! 3. **Settling** (this binary): `Renderer` with its defaults (bounce on)
//!    on M10 gate 22's night scene at 1280×720 settles in at most 0.8× gate
//!    4's 1,570 frames (development 1,018–1,096). The day scene and gate 4's
//!    new view are reported, with their settling without bounce.
//!
//! Rehearsal (`M22_REHEARSE=1`): the night scene only.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Camera, MaskConeKernel, Renderer, RendererOptions, wgpu};
use room3d::textured::Textured;

fn target(r: &Renderer, w: u32, h: u32) -> wgpu::TextureView {
    r.kernel()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn settle(r: &mut Renderer, camera: Camera, view: &wgpu::TextureView) -> (usize, f64) {
    let start = Instant::now();
    let mut frames = 0;
    while frames < 20_000 && !(frames > 32 && r.is_settled(32)) {
        r.render(camera, view);
        frames += 1;
    }
    (frames, start.elapsed().as_secs_f64())
}

fn main() -> std::io::Result<()> {
    let rehearse = std::env::var_os("M22_REHEARSE").is_some();
    let (w, h) = (1280u32, 720u32);
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        std::io::stdout().flush().ok();
        report.push_str(&line);
        report.push('\n');
    };
    say(format!(
        "M22 gate 5{}: {w}x{h} (criteria 1-2 are M14 gate 11's run)",
        if rehearse { " (rehearsal)" } else { "" }
    ));
    let mut c3 = true;
    for name in ["(b) night", "(a) day", "(c) materials, far side"] {
        if rehearse && name != "(b) night" {
            continue;
        }
        let (t, camera) = match name {
            "(b) night" => {
                let t = Textured::new().with_lamps();
                let c = t.camera;
                (t, c)
            }
            "(c) materials, far side" => (
                Textured::new().with_materials(),
                Camera {
                    eye: Vec3::new(6.0, 22.0, 3.0),
                    target: Vec3::new(-2.0, 8.0, 0.0),
                    vertical_fov_degrees: 45.0,
                },
            ),
            _ => {
                let t = Textured::new();
                let c = t.camera;
                (t, c)
            }
        };
        let make = |bounce: bool| {
            let kernel = MaskConeKernel::new().expect("GPU");
            Renderer::new(
                kernel,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                (w, h),
                &t.scene,
                t.origin,
                &t.lights,
                RendererOptions {
                    sky: (t.sky != Vec3::ZERO).then_some(t.sky),
                    bounce,
                    ..Default::default()
                },
            )
        };
        let mut on = make(true);
        let v_on = target(&on, w, h);
        let (f_on, s_on) = settle(&mut on, camera, &v_on);
        drop(on);
        let mut off = make(false);
        let v_off = target(&off, w, h);
        let (f_off, s_off) = settle(&mut off, camera, &v_off);
        drop(off);
        if name == "(b) night" {
            let ratio = f_on as f64 / 1570.0;
            c3 = ratio <= 0.8;
            say(format!(
                "3. {name}: settled with bounce in {f_on} frames ({s_on:.1} s), gate 4 1,570 ({ratio:.3}x, at most 0.8x): {}; without bounce {f_off} ({s_off:.1} s), gate 4 898",
                if c3 { "PASS" } else { "FAIL" }
            ));
        } else {
            say(format!(
                "{name} (reported): settled with bounce in {f_on} frames ({s_on:.1} s), without {f_off} ({s_off:.1} s): {:.2}x",
                f_on as f64 / f_off.max(1) as f64
            ));
        }
    }
    say(format!(
        "Verdict (criterion 3): {}",
        if c3 { "PASS" } else { "FAIL" }
    ));
    std::fs::write(
        out_dir.join(if rehearse {
            "m22_gate5_rehearsal.txt"
        } else {
            "m22_gate5.txt"
        }),
        report,
    )?;
    Ok(())
}
