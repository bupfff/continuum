//! M22 gate 1: bounce light with movers, through the ledger.
//!
//! Method (DESIGN §3.4, "Bounce light with movers"): each frame something
//! moves, pages holding bounce samples are charged the share of their view
//! the moves can change (moved boxes, the box where moved shadows may fall,
//! a page's own object); past ε (0.01) their bounce samples alone re-bake,
//! listings expiring after 16 frames; at rest a drain re-bakes what is
//! charged above ε / 100 and what was baked while something moved.
//!
//! Scene: M6's textured scene, sun only (movers under a sky are their own
//! gap), bounce light on, a fixed bake budget (`frame_target_ms` 0), 640×360.
//! Flights: (a) boulders 1–6 at 0.02 m a frame along +x for 120 frames
//! (developed); (b), not used in development: boulders 1–12 at 0.04 m a
//! frame for 120 frames.
//!
//! Measures: `verify_bounce` re-bakes every resident page's bounce samples
//! in place from the direct light the cache holds and counts pages that
//! changed by more than ε of the largest bounce value (restoring the
//! cache); `rebake_bounce` leaves them re-baked, so the image can be
//! compared with the same renderer's (no history between them).
//!
//! The bars were set with development numbers known (DESIGN §3.4: pages
//! over ε 0.49–0.63× the renderer without re-bakes; the image 0.457× its
//! distance; at rest 50 pages over ε against 137 for the scene settled
//! without motion; moving frames 1.10× with bounce light off).
//!
//! Pass criterion, fixed before the first run, for each flight:
//! 1. **While moving:** at frames 30, 60, 90 and 120, pages over ε at most
//!    0.7× a renderer whose ledger never re-bakes (ε 1e9).
//! 2. **The image:** at the end of the motion, the image's relative RMSE
//!    from its bounce re-baked in place at most 0.6× that renderer's.
//! 3. **At rest:** on a second flight, stopped and settled, every charge
//!    drained (the largest 0) and at most as many pages over ε as the scene
//!    settled without motion at the final poses.
//! 4. **Cost:** the moving frames' median at most 1.2× the same flight with
//!    bounce light off (frames interleaved).
//!
//! Development mode (`M22_DEV=1`): 320×180, 30 frames, checks every 15.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::textured::Textured;

const EPS: f32 = 0.01;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> bool {
    let start = r.cache_log().len();
    for _ in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0 && s.indirect_samples == 0)
        {
            return true;
        }
    }
    false
}

struct Flight {
    name: &'static str,
    movers: Vec<usize>,
    speed: f64,
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M22_DEV").is_some();
    let (w, h) = if dev { (320u32, 180u32) } else { (640, 360) };
    let frames: usize = if dev { 30 } else { 120 };
    let every: usize = if dev { 15 } else { 30 };
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
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
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
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let mut t = Textured::new();
    t.sky = Vec3::ZERO;
    kernel.set_sky(t.sky);
    say(format!(
        "M22 gate 1{}: {w}x{h}, {frames} moving frames, adapter {}",
        if dev { " (development mode)" } else { "" },
        kernel.adapter_name
    ));
    let flights = [
        Flight {
            name: "(a) six boulders at 0.02 m a frame",
            movers: (1..=6).collect(),
            speed: 0.02,
        },
        Flight {
            name: "(b) twelve boulders at 0.04 m a frame",
            movers: (1..=12).collect(),
            speed: 0.04,
        },
    ];
    let mut all = [true; 4];
    let mut unsettled = 0;
    for fl in &flights {
        let options = |indirect: bool, eps: f32| CacheOptions {
            indirect,
            bounce_epsilon: eps,
            frame_target_ms: 0.0,
            ..Default::default()
        };
        // A renderer with the movers placed `at` metres along +x from the
        // start.
        let make = |indirect: bool, eps: f32, at: f64| {
            let mut scene = t.scene.clone();
            for &k in &fl.movers {
                scene.instances[k].position += DVec3::new(at, 0.0, 0.0);
            }
            InstancedRenderer::with_movers(
                &kernel,
                format,
                w,
                h,
                1,
                &t.lights,
                &scene,
                t.origin,
                Some(options(indirect, eps)),
                &fl.movers,
                &[],
            )
        };
        let place = |r: &mut InstancedRenderer, at: f64| {
            for &k in &fl.movers {
                let i = &t.scene.instances[k];
                r.set_instance_transform(k, i.rotation, i.position + DVec3::new(at, 0.0, 0.0));
            }
        };
        // Flight 1: ours, without re-bakes, and bounce off (for the cost).
        let mut ours = make(true, EPS, 0.0);
        let mut stale = make(true, 1e9, 0.0);
        let mut off = make(false, EPS, 0.0);
        for r in [&mut ours, &mut stale, &mut off] {
            if !settle(r, t.camera, &target) {
                unsettled += 1;
            }
        }
        let (mut t_ours, mut t_off) = (Vec::new(), Vec::new());
        let mut c1 = true;
        for f in 1..=frames {
            let at = fl.speed * f as f64;
            for r in [&mut ours, &mut stale, &mut off] {
                place(r, at);
            }
            let s = Instant::now();
            ours.frame(t.camera, &target);
            ours.wait();
            t_ours.push(s.elapsed().as_secs_f64() * 1000.0);
            let s = Instant::now();
            off.frame(t.camera, &target);
            off.wait();
            t_off.push(s.elapsed().as_secs_f64() * 1000.0);
            stale.frame(t.camera, &target);
            if f % every == 0 {
                let (co, cs) = (ours.verify_bounce(EPS), stale.verify_bounce(EPS));
                let ratio = co.over_epsilon as f64 / cs.over_epsilon.max(1) as f64;
                let ok = ratio <= 0.7;
                c1 &= ok;
                say(format!(
                    "1. {} frame {f}: pages over eps {} of {} against {} without re-bakes ({ratio:.3}x, at most 0.7x): {}; largest change {:.3e} of {:.3e}, largest charge {:.2}, missed {}",
                    fl.name,
                    co.over_epsilon,
                    co.pages,
                    cs.over_epsilon,
                    verdict(ok),
                    co.largest_change,
                    co.largest_value,
                    co.largest_charge,
                    co.missed
                ));
            }
        }
        // 2. The image against its bounce re-baked in place.
        let image_vs_rebaked = |r: &mut InstancedRenderer| {
            r.finish();
            let shown = r.read_colors();
            r.rebake_bounce(EPS);
            r.frame(t.camera, &target);
            r.finish();
            relative_rmse(&shown, &r.read_colors())
        };
        let (io, is) = (image_vs_rebaked(&mut ours), image_vs_rebaked(&mut stale));
        let c2 = io <= 0.6 * is;
        say(format!(
            "2. {}: the image from its bounce re-baked in place {:.3}%, without re-bakes {:.3}% ({:.3}x, at most 0.6x): {}",
            fl.name,
            100.0 * io,
            100.0 * is,
            io / is,
            verdict(c2)
        ));
        // 4. Cost.
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (mo, mf) = (median(&mut t_ours), median(&mut t_off));
        let c4 = mo <= 1.2 * mf;
        say(format!(
            "4. {}: moving frames' median {mo:.2} ms, with bounce light off {mf:.2} ms ({:.3}x, at most 1.2x): {}",
            fl.name,
            mo / mf,
            verdict(c4)
        ));
        drop((ours, stale, off));
        // 3. At rest, on a second flight.
        let mut rest = make(true, EPS, 0.0);
        if !settle(&mut rest, t.camera, &target) {
            unsettled += 1;
        }
        for f in 1..=frames {
            place(&mut rest, fl.speed * f as f64);
            rest.frame(t.camera, &target);
        }
        if !settle(&mut rest, t.camera, &target) {
            unsettled += 1;
        }
        let cr = rest.verify_bounce(EPS);
        drop(rest);
        let mut still = make(true, EPS, fl.speed * frames as f64);
        if !settle(&mut still, t.camera, &target) {
            unsettled += 1;
        }
        let cs = still.verify_bounce(EPS);
        let c3 = cr.largest_charge == 0.0 && cr.over_epsilon <= cs.over_epsilon;
        say(format!(
            "3. {}: stopped and settled, pages over eps {} of {} (largest charge {:.3}, largest change {:.3e} of {:.3e}), the scene settled without motion {} of {}: {}",
            fl.name,
            cr.over_epsilon,
            cr.pages,
            cr.largest_charge,
            cr.largest_change,
            cr.largest_value,
            cs.over_epsilon,
            cs.pages,
            verdict(c3)
        ));
        for (k, ok) in [c1, c2, c3, c4].into_iter().enumerate() {
            all[k] &= ok;
        }
    }
    say(format!(
        "renderers that did not settle within 20,000 frames: {unsettled}"
    ));
    say(format!(
        "Verdict: 1 {}, 2 {}, 3 {}, 4 {}: {}",
        verdict(all[0]),
        verdict(all[1]),
        verdict(all[2]),
        verdict(all[3]),
        verdict(all.iter().all(|&x| x))
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m22_gate1_dev.txt"
        } else {
            "m22_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
