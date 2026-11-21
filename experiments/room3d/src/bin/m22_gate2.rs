//! M22 gate 2: bounce light with movers, through the ledger, timed right.
//!
//! Gate 1 failed on cost (2.163× and 2.075× bounce light off), but its
//! timing was at fault, found after the run: the renderer without re-bakes
//! rendered untimed and was not waited for, so its GPU work ran into the
//! next timed frame of ours. Waiting for every renderer, the same flights
//! measured 1.18× and 1.145× (`m22_cost_dev`, DESIGN §3.4). This gate times
//! every renderer's frame to completion, and adds flights. Since gate 1,
//! moved shadows charge pages by each mover's shadow regions, not by one
//! box around every page a moved shadow may touch.
//!
//! Method (DESIGN §3.4, "Bounce light with movers"): each frame something
//! moves, pages holding bounce samples are charged the share of their view
//! the moves can change; past ε (0.01) their bounce samples alone re-bake;
//! at rest a drain re-bakes what is charged above ε / 100 and what was
//! baked while something moved.
//!
//! Scene: M6's textured scene, sun only (movers under a sky are their own
//! gap), bounce light on, a fixed bake budget (`frame_target_ms` 0), 640×360.
//! Flights, 120 frames each: (a) boulders 1–6 at 0.02 m a frame along +x;
//! (b) boulders 1–12 at 0.04 m a frame along +x; (c) boulders 13–24 at 0.03
//! m a frame along −y (all three seen in development); (d), not used in
//! development: boulders 25–30 at 0.05 m a frame along (+x, +y)/√2.
//!
//! Measures: `verify_bounce` re-bakes every resident page's bounce samples
//! in place from the direct light the cache holds and counts pages that
//! changed by more than ε of the largest bounce value (restoring the
//! cache); `rebake_bounce` leaves them re-baked, so the image can be
//! compared with the same renderer's (no history between them).
//!
//! Pass criterion, fixed before the run, for each flight (gate 1's bars):
//! 1. **While moving:** at frames 30, 60, 90 and 120, pages over ε at most
//!    0.7× a renderer whose ledger never re-bakes (ε 1e9).
//! 2. **The image:** at the end of the motion, the image's relative RMSE
//!    from its bounce re-baked in place at most 0.6× that renderer's.
//! 3. **At rest:** on a second flight, stopped and settled, every charge
//!    drained (the largest 0) and at most as many pages over ε as the scene
//!    settled without motion at the final poses.
//! 4. **Cost:** the moving frames' median at most 1.2× the same flight with
//!    bounce light off, every renderer's frame waited for, ours and bounce
//!    off timed back to back each frame.
//!
//! Rehearsal (`M22_REHEARSE=1`): flights (a)–(c) as the gate measures
//! them at full size, 60 frames, checks at 30 and 60. Development mode
//! (`M22_DEV=1`): 320×180, 30 frames, checks every 15. Both only run flight
//! (d) for 8 frames at 320×180, its numbers not printed.

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
    step: DVec3,
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M22_DEV").is_some();
    let rehearse = std::env::var_os("M22_REHEARSE").is_some();
    let (w, h) = if dev { (320u32, 180u32) } else { (640, 360) };
    let frames: usize = if dev {
        30
    } else if rehearse {
        60
    } else {
        120
    };
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
        "M22 gate 2{}: {w}x{h}, {frames} moving frames, adapter {}",
        if dev {
            " (development mode)"
        } else if rehearse {
            " (rehearsal)"
        } else {
            ""
        },
        kernel.adapter_name
    ));
    let flights = [
        Flight {
            name: "(a) six boulders at 0.02 m a frame",
            movers: (1..=6).collect(),
            step: DVec3::new(0.02, 0.0, 0.0),
        },
        Flight {
            name: "(b) twelve boulders at 0.04 m a frame",
            movers: (1..=12).collect(),
            step: DVec3::new(0.04, 0.0, 0.0),
        },
        Flight {
            name: "(c) twelve other boulders at 0.03 m a frame along -y",
            movers: (13..=24).collect(),
            step: DVec3::new(0.0, -0.03, 0.0),
        },
        Flight {
            name: "(d) boulders 25-30 at 0.05 m a frame diagonally",
            movers: (25..=30).collect(),
            step: DVec3::new(0.05, 0.05, 0.0) / 2f64.sqrt(),
        },
    ];
    let mut all = [true; 4];
    let mut unsettled = 0;
    for (fi, fl) in flights.iter().enumerate() {
        // Rehearsals and development only run flight (d), small and
        // briefly, unseen.
        let smoke = (rehearse || dev) && fi == 3;
        if smoke {
            smoke_run(&kernel, &t, fl);
            say(format!(
                "{}: ran 8 frames at 320x180 (numbers not shown)",
                fl.name
            ));
            continue;
        }
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
                scene.instances[k].position += fl.step * at;
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
                r.set_instance_transform(k, i.rotation, i.position + fl.step * at);
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
            let at = f as f64;
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
            // Untimed, but waited for: its work must not run into the next
            // timed frame (gate 1's fault).
            stale.frame(t.camera, &target);
            stale.wait();
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
            place(&mut rest, f as f64);
            rest.frame(t.camera, &target);
        }
        if !settle(&mut rest, t.camera, &target) {
            unsettled += 1;
        }
        let cr = rest.verify_bounce(EPS);
        drop(rest);
        let mut still = make(true, EPS, frames as f64);
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
            "m22_gate2_dev.txt"
        } else if rehearse {
            "m22_gate2_rehearsal.txt"
        } else {
            "m22_gate2.txt"
        }),
        report,
    )?;
    Ok(())
}

/// Flight (d) in rehearsals and development: eight frames at 320×180, to catch a crash,
/// nothing measured.
fn smoke_run(kernel: &MaskConeKernel, t: &Textured, fl: &Flight) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = kernel
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 320,
                height: 180,
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
    let mut r = InstancedRenderer::with_movers(
        kernel,
        format,
        320,
        180,
        1,
        &t.lights,
        &t.scene,
        t.origin,
        Some(CacheOptions {
            indirect: true,
            bounce_epsilon: EPS,
            frame_target_ms: 0.0,
            ..Default::default()
        }),
        &fl.movers,
        &[],
    );
    for f in 0..8 {
        for &k in &fl.movers {
            let i = &t.scene.instances[k];
            r.set_instance_transform(k, i.rotation, i.position + fl.step * f as f64);
        }
        r.frame(t.camera, &target);
    }
    r.finish();
}
