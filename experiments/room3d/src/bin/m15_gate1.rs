//! M15 gate 1: anti-aliasing on the default path, in motion.
//!
//! Method (DESIGN §3.4, "Anti-aliasing on the default path"):
//! `CacheOptions::anti_aliasing` with hardware camera rays. Edge pixels (a
//! neighbour, left, right, up or down, sees another object, light or the
//! sky, or a point more than half a pixel footprint off the pixel's own
//! triangle's plane) trace the outer four samples of a fixed pattern of
//! eight, then the inner four where those did not all see the pixel's own
//! surface, within a ray budget of a ray a pixel (the default). A
//! sample on the pixel's own surface takes its colour; any other is shaded
//! from the cache.
//!
//! Views: M6's textured scene with foliage (`Textured::new().with_foliage()`,
//! its camera, sun and sky), and the million-boulder field from standing
//! height (its ground camera, sun only). 1280×720, release build, cache
//! settled (32 quiet frames).
//!
//! The reference: the cached image at 16×16 sub-pixel offsets, each
//! settled, textures read for 1/16 of a pixel (as M7's gates). Geometric
//! edge pixels: those whose 256 reference samples see more than one object
//! or light, or one object at distances more than 5% apart. Errors are
//! relative RMSE (over the image's mean) over the pixels named.
//!
//! Pass criterion, fixed before the first run, in both views:
//! 1. **Cost** (timed first): anti-aliasing adds at most 3.3 ms (a fifth of
//!    a 60 fps frame) to one ray's warm frame: medians of five interleaved
//!    rounds of 20 frames each, each frame waited for.
//! 2. **Edges:** the anti-aliased image's error over geometric edge pixels
//!    at most 0.5× one ray's.
//! 3. **Elsewhere:** its error over the other pixels at most one ray's.
//! 4. **Motion:** the camera turns a quarter pixel a frame for six frames,
//!    each frame settled, each with its own reference. Over the pixels that
//!    are geometric edges in any frame, the RMS of the difference between
//!    each frame-to-frame change and the reference's at most 0.5× one
//!    ray's.
//!
//! Reported, not gated: the frames' ratio; edge pixels listed, samples
//! traced and shaded; M7's silhouette rays (`edge_rays`, four quarters)
//! on the same measures; the whole image's error; each method's
//! frame-to-frame change over the reference's.
//!
//! Development mode (`M15_DEV=1`): 640×360, 100,000 boulders, 4×4
//! reference offsets, three frames of motion.
//!
//! Implementation notes (not criteria): the static criteria (2, 3) use the
//! motion's first frame, which is the view's own camera; both views are
//! timed before any reference is made, so the timings see the machine
//! before the long reference work.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::field::Field;
use room3d::textured::Textured;

type Image = Vec<[f32; 3]>;

struct Setup {
    w: u32,
    h: u32,
    offsets: u32,
    frames: usize,
}

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) -> usize {
    let start = r.cache_log().len();
    for n in 0..20_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return n;
        }
    }
    panic!("never settled");
}

struct View<'a> {
    name: &'static str,
    scene: &'a InstancedScene,
    origin: DVec3,
    lights: Vec<Light>,
    sky: Vec3,
    camera: Camera,
}

fn options(method: usize) -> CacheOptions {
    match method {
        0 => CacheOptions::default(),
        1 => CacheOptions {
            anti_aliasing: true,
            ..Default::default()
        },
        _ => CacheOptions {
            edge_rays: true,
            ..Default::default()
        },
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> std::io::Result<()> {
    let dev = std::env::var_os("M15_DEV").is_some();
    let setup = if dev {
        Setup {
            w: 640,
            h: 360,
            offsets: 4,
            frames: 3,
        }
    } else {
        Setup {
            w: 1280,
            h: 720,
            offsets: 16,
            frames: 6,
        }
    };
    let (w, h) = (setup.w, setup.h);
    let n = (w * h) as usize;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        std::io::stdout().flush().ok();
        report.push_str(&line);
        report.push('\n');
    };
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    say(format!(
        "M15 gate 1{}: {w}x{h}, adapter {}",
        if dev { " (development mode)" } else { "" },
        kernel.adapter_name
    ));
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
    let textured = Textured::new().with_foliage();
    let field = Field::new(if dev { 100_000 } else { 1_000_000 }, DVec3::ZERO);
    let field_origin = field.origin_near_camera();
    let (ground, sun) = field.relative(field_origin);
    let views = [
        View {
            name: "textured scene with foliage",
            scene: &textured.scene,
            origin: textured.origin,
            lights: textured.lights.clone(),
            sky: textured.sky,
            camera: textured.camera,
        },
        View {
            name: "field standing",
            scene: &field.scene,
            origin: field_origin,
            lights: vec![Light::white(sun, field.radiance)],
            sky: Vec3::ZERO,
            camera: ground,
        },
    ];
    let make = |view: &View, method: usize| {
        kernel.set_sky(view.sky);
        InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            w,
            h,
            1,
            &view.lights,
            view.scene,
            view.origin,
            options(method),
        )
    };
    let timed = |r: &mut InstancedRenderer, camera: Camera, frames: usize| -> Vec<f64> {
        (0..frames)
            .map(|_| {
                let s = Instant::now();
                r.frame(camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect()
    };

    // 1. Cost, both views first.
    let mut pass = [true; 2];
    for (v, view) in views.iter().enumerate() {
        let mut one = make(view, 0);
        let mut aa = make(view, 1);
        settle(&mut one, view.camera, &target);
        settle(&mut aa, view.camera, &target);
        let (mut t1, mut ta) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            t1.extend(timed(&mut one, view.camera, 20));
            ta.extend(timed(&mut aa, view.camera, 20));
        }
        aa.finish();
        let counters = aa.aa_counters();
        drop(aa);
        drop(one);
        let mut quarters = make(view, 2);
        settle(&mut quarters, view.camera, &target);
        let mut tq = timed(&mut quarters, view.camera, 40);
        drop(quarters);
        let (m1, ma, mq) = (median(&mut t1), median(&mut ta), median(&mut tq));
        let ok = ma - m1 <= 3.3;
        pass[v] &= ok;
        say(format!(
            "{}: 1. cost: one ray {m1:.2} ms, anti-aliased {ma:.2} ms: adds {:.2} ms (at most 3.3; {:.3}x): {}; M7 quarters {mq:.2} ms ({:.3}x, measured after)",
            view.name,
            ma - m1,
            ma / m1,
            if ok { "PASS" } else { "FAIL" },
            mq / m1
        ));
        say(format!(
            "   listed {} edge pixels ({:.2}%), {} samples a pixel in stage 0, {} traced ({:.3} a pixel), {} shaded ({:.1}%), {} pixels done after stage 0",
            counters[0],
            100.0 * counters[0] as f64 / n as f64,
            counters[1],
            counters[3],
            counters[3] as f64 / n as f64,
            counters[2],
            100.0 * counters[2] as f64 / counters[3].max(1) as f64,
            counters[4]
        ));
    }

    // 2–4. Quality, standing and turning.
    let pixel_angle =
        |view: &View| 2.0 * (view.camera.vertical_fov_degrees.to_radians() * 0.5).tan() / h as f32;
    for (v, view) in views.iter().enumerate() {
        let start = Instant::now();
        let turn = |f: usize| {
            let yaw = 0.25 * pixel_angle(view) * f as f32;
            let d = view.camera.target - view.camera.eye;
            let (sn, cs) = yaw.sin_cos();
            Camera {
                eye: view.camera.eye,
                target: view.camera.eye + Vec3::new(d.x * cs - d.y * sn, d.x * sn + d.y * cs, d.z),
                vertical_fov_degrees: view.camera.vertical_fov_degrees,
            }
        };
        let mut renderers: Vec<InstancedRenderer> = (0..3).map(|m| make(view, m)).collect();
        // Per frame: the three methods' images and the reference; per
        // frame, its geometric edge pixels.
        let mut images: Vec<[Image; 4]> = Vec::new();
        let mut edges: Vec<Vec<bool>> = Vec::new();
        for f in 0..setup.frames {
            let camera = turn(f);
            let mut shots: Vec<Image> = Vec::new();
            for r in renderers.iter_mut() {
                settle(r, camera, &target);
                r.finish();
                shots.push(r.read_colors());
            }
            let one = &mut renderers[0];
            let mut sum = vec![[0.0f64; 3]; n];
            let mut first: Vec<(u32, u32, f32, f32)> = Vec::new();
            let mut edge = vec![false; n];
            let k = setup.offsets;
            one.set_footprint(1.0 / k as f32);
            for s in 0..k * k {
                let offset = |i: u32| (i as f32 + 0.5) / k as f32 - 0.5;
                one.set_pixel_offset(offset(s % k), offset(s / k));
                settle(one, camera, &target);
                one.finish();
                for (a, p) in sum.iter_mut().zip(one.read_colors()) {
                    for c in 0..3 {
                        a[c] += p[c] as f64 / (k * k) as f64;
                    }
                }
                let hits = one.read_hits();
                if first.is_empty() {
                    first = hits.iter().map(|&(o, _, d, l)| (o, l, d, d)).collect();
                }
                for ((fr, e), &(o, _, d, l)) in first.iter_mut().zip(edge.iter_mut()).zip(&hits) {
                    if o != fr.0 || l != fr.1 {
                        *e = true;
                    } else if o != u32::MAX && l == 0 {
                        fr.2 = fr.2.min(d);
                        fr.3 = fr.3.max(d);
                        if fr.3 > 1.05 * fr.2 {
                            *e = true;
                        }
                    }
                }
            }
            one.set_footprint(1.0);
            one.set_pixel_offset(0.0, 0.0);
            let reference: Image = sum.iter().map(|s| s.map(|x| x as f32)).collect();
            let [a, b, c]: [Image; 3] = shots.try_into().expect("three methods");
            images.push([a, b, c, reference]);
            edges.push(edge);
            println!(
                "   ({}: frame {f} done, {:.0} s)",
                view.name,
                start.elapsed().as_secs_f64()
            );
        }
        drop(renderers);
        let save = |name: &str, px: &Image| -> std::io::Result<()> {
            let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                image.set(
                    i % w as usize,
                    i / w as usize,
                    room3d::to_rgb(Vec3::from_array(*p)),
                );
            }
            image.save(&out_dir.join(format!("m15_gate1_{v}_{name}.png")))
        };
        for (m, name) in ["one", "aa", "quarters", "reference"].iter().enumerate() {
            save(name, &images[0][m])?;
        }
        let mean = |img: &Image| {
            img.iter()
                .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
                .sum::<f64>()
                / n as f64
        };
        // Static: frame 0.
        let m0 = mean(&images[0][3]);
        let rmse = |img: &Image, mask: &dyn Fn(usize) -> bool| {
            let (mut e, mut count) = (0.0f64, 0usize);
            for p in (0..n).filter(|&p| mask(p)) {
                for c in 0..3 {
                    e += ((img[p][c] - images[0][3][p][c]) as f64).powi(2);
                }
                count += 1;
            }
            100.0 * (e / (3 * count.max(1)) as f64).sqrt() / m0
        };
        let on_edge = |p: usize| edges[0][p];
        let off_edge = |p: usize| !edges[0][p];
        let all = |_: usize| true;
        let edge_share = 100.0 * edges[0].iter().filter(|e| **e).count() as f64 / n as f64;
        let e: Vec<[f64; 3]> = (0..3)
            .map(|m| {
                [
                    rmse(&images[0][m], &on_edge),
                    rmse(&images[0][m], &off_edge),
                    rmse(&images[0][m], &all),
                ]
            })
            .collect();
        let ok2 = e[1][0] <= 0.5 * e[0][0];
        let ok3 = e[1][1] <= e[0][1];
        pass[v] &= ok2 && ok3;
        say(format!(
            "{}: 2. edges ({edge_share:.2}% of pixels): one ray {:.2}%, anti-aliased {:.2}% ({:.3}x, at most 0.5x): {}; M7 quarters {:.2}% ({:.3}x)",
            view.name,
            e[0][0],
            e[1][0],
            e[1][0] / e[0][0],
            if ok2 { "PASS" } else { "FAIL" },
            e[2][0],
            e[2][0] / e[0][0]
        ));
        say(format!(
            "{}: 3. elsewhere: one ray {:.3}%, anti-aliased {:.3}% ({:.3}x, at most 1x): {}; M7 quarters {:.3}%; the whole image: {:.3}% / {:.3}% / {:.3}%",
            view.name,
            e[0][1],
            e[1][1],
            e[1][1] / e[0][1],
            if ok3 { "PASS" } else { "FAIL" },
            e[2][1],
            e[0][2],
            e[1][2],
            e[2][2]
        ));
        // Motion.
        let any_edge: Vec<bool> = (0..n).map(|p| edges.iter().any(|e| e[p])).collect();
        let mean_all = images.iter().map(|i| mean(&i[3])).sum::<f64>() / images.len() as f64;
        let motion = |m: usize| {
            let (mut off, mut change, mut truth, mut count) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for f in 0..setup.frames - 1 {
                for p in (0..n).filter(|&p| any_edge[p]) {
                    for c in 0..3 {
                        let dm = (images[f + 1][m][p][c] - images[f][m][p][c]) as f64;
                        let dr = (images[f + 1][3][p][c] - images[f][3][p][c]) as f64;
                        off += (dm - dr).powi(2);
                        change += dm * dm;
                        truth += dr * dr;
                    }
                    count += 1;
                }
            }
            (
                100.0 * (off / (3 * count.max(1)) as f64).sqrt() / mean_all,
                (change / truth.max(1e-30)).sqrt(),
            )
        };
        let t: Vec<(f64, f64)> = (0..3).map(motion).collect();
        let ok4 = t[1].0 <= 0.5 * t[0].0;
        pass[v] &= ok4;
        say(format!(
            "{}: 4. motion ({} frames, {:.2}% of pixels edges in some frame): change off the reference's: one ray {:.2}%, anti-aliased {:.2}% ({:.3}x, at most 0.5x): {}; M7 quarters {:.2}% ({:.3}x); change over the reference's: {:.2} / {:.2} / {:.2}",
            view.name,
            setup.frames,
            100.0 * any_edge.iter().filter(|e| **e).count() as f64 / n as f64,
            t[0].0,
            t[1].0,
            t[1].0 / t[0].0,
            if ok4 { "PASS" } else { "FAIL" },
            t[2].0,
            t[2].0 / t[0].0,
            t[0].1,
            t[1].1,
            t[2].1
        ));
        say(format!(
            "   ({} quality in {:.0} s)",
            view.name,
            start.elapsed().as_secs_f64()
        ));
    }
    say(format!(
        "Verdict: {}",
        if pass.iter().all(|p| *p) {
            "PASS"
        } else {
            "FAIL"
        }
    ));
    std::fs::write(
        out_dir.join(if dev {
            "m15_gate1_dev.txt"
        } else {
            "m15_gate1.txt"
        }),
        report,
    )?;
    Ok(())
}
