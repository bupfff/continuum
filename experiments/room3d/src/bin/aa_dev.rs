//! Development (M15): anti-aliasing on the default path against one ray
//! per pixel and M7's silhouette rays, on the textured scene with foliage
//! (`textured`, the default), the field at standing height (`field`) or
//! from 150 m up (`field150`). Not a gate.
//!
//! `AA_W`/`AA_H` set the size (640×360); `AA_RAYS` the ray budget (1.0);
//! `AA_REF=0` skips the 16×16 reference (timings and counters only);
//! `AA_SAVE=dir` writes the images.

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{
    CacheOptions, Camera, InstancedRenderer, InstancedScene, Light, MaskConeKernel, wgpu,
};
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::textured::Textured;

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
    eprintln!("warning: never settled");
    20_000
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let w: u32 = env("AA_W", 640);
    let h: u32 = env("AA_H", 360);
    let rays: f32 = env("AA_RAYS", 1.0);
    let reference_on = env("AA_REF", 1u32) == 1;
    let save_dir = std::env::var("AA_SAVE").ok();
    let view = std::env::args().nth(1).unwrap_or_else(|| "textured".into());
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
    let t = Textured::new().with_foliage();
    let field = view
        .starts_with("field")
        .then(|| Field::new(env("AA_BOULDERS", 100_000usize), DVec3::ZERO));
    let (scene, origin, lights, sky, camera): (&InstancedScene, DVec3, Vec<Light>, Vec3, Camera) =
        if let Some(f) = &field {
            let origin = f.origin_near_camera();
            let (ground, sun) = f.relative(origin);
            let cam = if view == "field150" {
                let forward = (ground.target - ground.eye).normalize();
                let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
                let d = 45f32.to_radians();
                let eye = Vec3::new(ground.eye.x, ground.eye.y, 150.0);
                Camera {
                    eye,
                    target: eye + flat * d.cos() - Vec3::Z * d.sin(),
                    vertical_fov_degrees: ground.vertical_fov_degrees,
                }
            } else {
                ground
            };
            (
                &f.scene,
                origin,
                vec![Light::white(sun, f.radiance)],
                Vec3::ZERO,
                cam,
            )
        } else {
            (&t.scene, t.origin, t.lights.clone(), t.sky, t.camera)
        };
    kernel.set_sky(sky);
    let make = |options: CacheOptions| {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel, format, w, h, 1, &lights, scene, origin, options,
        );
        let frames = settle(&mut r, camera, &target);
        (r, frames)
    };
    let timed = |r: &mut InstancedRenderer| {
        let mut times: Vec<f64> = (0..40)
            .map(|_| {
                let s = Instant::now();
                r.frame(camera, &target);
                r.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        times.sort_by(f64::total_cmp);
        r.finish();
        (times[20], r.read_colors())
    };
    if std::env::args().any(|a| a == "profile") {
        let (mut r, _) = make(CacheOptions {
            anti_aliasing: true,
            aa_rays: rays,
            ..Default::default()
        });
        let mut per: Vec<Vec<f64>> = Vec::new();
        for _ in 0..30 {
            let p = r.frame_profiled(camera, &target);
            for (k, d) in p.passes.iter().enumerate().filter(|(_, d)| !d.is_zero()) {
                if per.len() <= k {
                    per.push(Vec::new());
                }
                per[k].push(d.as_secs_f64() * 1000.0);
            }
        }
        for v in &mut per {
            v.sort_by(f64::total_cmp);
        }
        println!(
            "camera passes (median ms, each waited for): {:?}",
            per.iter()
                .map(|v| (v[15] * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        return;
    }
    if std::env::args().any(|a| a == "pan") {
        // A slow turn: the camera yaws a quarter pixel a frame for
        // `AA_FRAMES` frames. Each frame's image from one ray and from
        // anti-aliasing against the frame's 16×16 reference: how the
        // frame-to-frame change differs from the reference's, over the
        // pixels whose reference samples see more than one surface in any
        // frame.
        let frames: usize = env("AA_FRAMES", 6);
        let pixel_angle = 2.0 * (camera.vertical_fov_degrees.to_radians() * 0.5).tan() / h as f32;
        let at = |f: usize| {
            let yaw = 0.25 * pixel_angle * f as f32;
            let d = camera.target - camera.eye;
            let (sn, cs) = yaw.sin_cos();
            Camera {
                eye: camera.eye,
                target: camera.eye + Vec3::new(d.x * cs - d.y * sn, d.x * sn + d.y * cs, d.z),
                vertical_fov_degrees: camera.vertical_fov_degrees,
            }
        };
        let (mut one, _) = make(CacheOptions::default());
        let (mut aa, _) = make(CacheOptions {
            anti_aliasing: true,
            aa_rays: rays,
            ..Default::default()
        });
        let (mut quarters, _) = make(CacheOptions {
            edge_rays: true,
            ..Default::default()
        });
        let n = (w * h) as usize;
        let mut images: Vec<[Vec<[f32; 3]>; 4]> = Vec::new();
        let mut edge = vec![false; n];
        for f in 0..frames {
            let cam = at(f);
            let shot = |r: &mut InstancedRenderer| {
                settle(r, cam, &target);
                r.finish();
                r.read_colors()
            };
            let i1 = shot(&mut one);
            let ia = shot(&mut aa);
            let iq = shot(&mut quarters);
            let mut reference = vec![[0.0f64; 3]; n];
            let mut first: Vec<(u32, u32, f32, f32)> = Vec::new();
            one.set_footprint(1.0 / 16.0);
            for k in 0..256u32 {
                let offset = |i: u32| (i as f32 + 0.5) / 16.0 - 0.5;
                one.set_pixel_offset(offset(k % 16), offset(k / 16));
                settle(&mut one, cam, &target);
                one.finish();
                for (s, p) in reference.iter_mut().zip(one.read_colors()) {
                    for c in 0..3 {
                        s[c] += p[c] as f64 / 256.0;
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
            let reference: Vec<[f32; 3]> = reference.iter().map(|s| s.map(|x| x as f32)).collect();
            eprintln!("frame {f} done");
            images.push([i1, ia, iq, reference]);
        }
        let mean = images
            .iter()
            .map(|i| {
                i[3].iter()
                    .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
                    .sum::<f64>()
                    / n as f64
            })
            .sum::<f64>()
            / frames as f64;
        let measure = |m: usize, mask: &dyn Fn(usize) -> bool| {
            let (mut te, mut change, mut truth, mut err, mut count) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
            for f in 0..frames {
                for p in (0..n).filter(|&p| mask(p)) {
                    for c in 0..3 {
                        let e = (images[f][m][p][c] - images[f][3][p][c]) as f64;
                        err += e * e;
                        if f + 1 < frames {
                            let dm = (images[f + 1][m][p][c] - images[f][m][p][c]) as f64;
                            let dr = (images[f + 1][3][p][c] - images[f][3][p][c]) as f64;
                            te += (dm - dr) * (dm - dr);
                            change += dm * dm;
                            truth += dr * dr;
                        }
                    }
                    count += 1;
                }
            }
            let pairs = count as f64 * (frames - 1) as f64 / frames as f64;
            (
                100.0 * (err / (3.0 * count as f64)).sqrt() / mean,
                100.0 * (te / (3.0 * pairs)).sqrt() / mean,
                (change / truth).sqrt(),
            )
        };
        let edges = |p: usize| edge[p];
        let all = |_: usize| true;
        println!(
            "{view} {w}x{h}, a turn of {frames} frames at a quarter pixel a frame; geometric edge pixels {:.2}%",
            100.0 * edge.iter().filter(|e| **e).count() as f64 / n as f64
        );
        for (name, mask) in [("edges", &edges as &dyn Fn(usize) -> bool), ("all", &all)] {
            let (e1, t1, c1) = measure(0, mask);
            let (ea, ta, ca) = measure(1, mask);
            let (eq, tq, cq) = measure(2, mask);
            println!(
                "{name}: error one ray {e1:.2}%, aa {ea:.2}% ({:.3}x), quarters {eq:.2}% ({:.3}x); frame-to-frame change off the reference's: one ray {t1:.2}%, aa {ta:.2}% ({:.3}x), quarters {tq:.2}% ({:.3}x); change / the reference's: one ray {c1:.2}, aa {ca:.2}, quarters {cq:.2}",
                ea / e1,
                eq / e1,
                ta / t1,
                tq / t1
            );
        }
        return;
    }
    let (mut one, f1) = make(CacheOptions::default());
    let (t1, i1) = timed(&mut one);
    let (mut aa, fa) = make(CacheOptions {
        anti_aliasing: true,
        aa_rays: rays,
        ..Default::default()
    });
    let (ta, ia) = timed(&mut aa);
    let counters = aa.aa_counters();
    let (mut edges, _) = make(CacheOptions {
        edge_rays: true,
        ..Default::default()
    });
    let (te, ie) = timed(&mut edges);
    let pixels = (w * h) as f64;
    println!(
        "{view} {w}x{h}: settled in {f1} / {fa} frames; warm frame: one ray {t1:.2} ms, aa {ta:.2} ms ({:.3}x), M7 quarters {te:.2} ms ({:.3}x)",
        ta / t1,
        te / t1
    );
    println!(
        "aa: {} edge pixels ({:.2}%), {} samples each, {} traced ({:.3} a pixel), {} shaded ({:.1}% of traced)",
        counters[0],
        100.0 * counters[0] as f64 / pixels,
        counters[1],
        counters[3],
        counters[3] as f64 / pixels,
        counters[2],
        100.0 * counters[2] as f64 / counters[3].max(1) as f64
    );
    println!(
        "aa: {} pixels ({:.1}%) all of whose samples saw its own surface; {:.1}% of samples did",
        counters[4],
        100.0 * counters[4] as f64 / counters[0].max(1) as f64,
        100.0 * counters[5] as f64 / counters[3].max(1) as f64
    );
    let save = |name: &str, px: &[[f32; 3]]| {
        if let Some(dir) = &save_dir {
            let mut image = flatland::img::Image::new(w as usize, h as usize, [0; 3]);
            for (i, p) in px.iter().enumerate() {
                image.set(
                    i % w as usize,
                    i / w as usize,
                    room3d::to_rgb(Vec3::from_array(*p)),
                );
            }
            image
                .save(&std::path::Path::new(dir).join(format!("aa_{view}_{name}.png")))
                .expect("save");
        }
    };
    save("one", &i1);
    save("aa", &ia);
    save("quarters", &ie);
    if !reference_on {
        return;
    }
    let mut reference = vec![[0.0f64; 3]; (w * h) as usize];
    one.set_footprint(1.0 / 16.0);
    let start = Instant::now();
    // Geometric edges: pixels whose 256 samples see more than one object
    // or light, or one object at distances more than 5% apart.
    let mut first: Vec<(u32, u32, f32, f32)> = Vec::new();
    let mut edge = vec![false; (w * h) as usize];
    for k in 0..256u32 {
        let offset = |i: u32| (i as f32 + 0.5) / 16.0 - 0.5;
        one.set_pixel_offset(offset(k % 16), offset(k / 16));
        settle(&mut one, camera, &target);
        one.finish();
        for (s, p) in reference.iter_mut().zip(one.read_colors()) {
            for c in 0..3 {
                s[c] += p[c] as f64 / 256.0;
            }
        }
        let hits = one.read_hits();
        if first.is_empty() {
            first = hits.iter().map(|&(o, _, d, l)| (o, l, d, d)).collect();
        }
        for ((f, e), &(o, _, d, l)) in first.iter_mut().zip(edge.iter_mut()).zip(&hits) {
            if o != f.0 || l != f.1 {
                *e = true;
            } else if o != u32::MAX && l == 0 {
                f.2 = f.2.min(d);
                f.3 = f.3.max(d);
                if f.3 > 1.05 * f.2 {
                    *e = true;
                }
            }
        }
    }
    let reference: Vec<[f32; 3]> = reference.iter().map(|s| s.map(|x| x as f32)).collect();
    save("reference", &reference);
    // Where one ray is visibly off the area integral (more than 5% of the
    // image's mean): the aliased pixels, the same set for every method.
    let mean = reference
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64 / 3.0)
        .sum::<f64>()
        / pixels;
    let mask: Vec<bool> = i1
        .iter()
        .zip(&reference)
        .map(|(a, b)| {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)) as f64
                / 3.0;
            d.sqrt() > 0.05 * mean
        })
        .collect();
    let masked_by = |img: &[[f32; 3]], mask: &[bool]| {
        let (mut num, mut n) = (0.0f64, 0usize);
        for ((a, b), m) in img.iter().zip(&reference).zip(mask) {
            if *m {
                for c in 0..3 {
                    num += ((a[c] - b[c]) as f64).powi(2);
                }
                n += 1;
            }
        }
        100.0 * (num / (3 * n.max(1)) as f64).sqrt() / mean
    };
    let masked = |img: &[[f32; 3]]| masked_by(img, &mask);
    let inside: Vec<bool> = edge.iter().map(|e| !e).collect();
    let e = |img: &[[f32; 3]]| 100.0 * relative_rmse(img, &reference);
    println!(
        "from the 16x16 reference ({:.0} s): one ray {:.3}%, aa {:.3}% ({:.3}x), quarters {:.3}% ({:.3}x)",
        start.elapsed().as_secs_f64(),
        e(&i1),
        e(&ia),
        e(&ia) / e(&i1),
        e(&ie),
        e(&ie) / e(&i1)
    );
    println!(
        "over the {} aliased pixels ({:.2}%): one ray {:.2}%, aa {:.2}% ({:.3}x), quarters {:.2}% ({:.3}x)",
        mask.iter().filter(|m| **m).count(),
        100.0 * mask.iter().filter(|m| **m).count() as f64 / pixels,
        masked(&i1),
        masked(&ia),
        masked(&ia) / masked(&i1),
        masked(&ie),
        masked(&ie) / masked(&i1)
    );
    let g = |img: &[[f32; 3]]| masked_by(img, &edge);
    let ng = |img: &[[f32; 3]]| masked_by(img, &inside);
    println!(
        "over the {} geometric edge pixels ({:.2}%): one ray {:.2}%, aa {:.2}% ({:.3}x), quarters {:.2}% ({:.3}x); elsewhere: one ray {:.3}%, aa {:.3}%, quarters {:.3}%",
        edge.iter().filter(|e| **e).count(),
        100.0 * edge.iter().filter(|e| **e).count() as f64 / pixels,
        g(&i1),
        g(&ia),
        g(&ia) / g(&i1),
        g(&ie),
        g(&ie) / g(&i1),
        ng(&i1),
        ng(&ia),
        ng(&ie)
    );
}
