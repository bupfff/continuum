//! Development: silhouette rays through ray queries against the compute
//! path (M7 gate 4), on the textured scene at 640×360, or the field from
//! 150 m up with 100,000 boulders (`field`). Not a gate.

use std::time::{Duration, Instant};

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, Camera, InstancedRenderer, Light, MaskConeKernel, wgpu};
use room3d::cycles::relative_rmse;
use room3d::field::Field;
use room3d::textured::Textured;

const W: u32 = 640;
const H: u32 = 360;

fn settle(r: &mut InstancedRenderer, camera: Camera, target: &wgpu::TextureView) {
    let start = r.cache_log().len();
    for _ in 0..60_000 {
        r.frame(camera, target);
        let latest = r.cache_stats().map_or(0, |s| s.frame as usize);
        if latest >= start + 32
            && r.cache_log()[latest - 32..latest]
                .iter()
                .all(|s| s.requests == 0 && s.pages_baked == 0)
        {
            return;
        }
    }
    panic!("never settled");
}

fn main() {
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
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
    let field_view = std::env::args().any(|a| a == "field");
    let t = Textured::new();
    let field = field_view.then(|| Field::new(100_000, DVec3::ZERO));
    let (scene, origin, lights, sky, camera) = if let Some(f) = &field {
        let origin = f.origin_near_camera();
        let (ground, sun) = f.relative(origin);
        let forward = (ground.target - ground.eye).normalize();
        let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
        let d = 45f32.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, 150.0);
        let cam = Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
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
    let run = |hw: bool, edge_rays: bool| {
        kernel.set_hardware_rays(hw);
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            scene,
            origin,
            CacheOptions {
                edge_rays,
                ..Default::default()
            },
        );
        settle(&mut r, camera, &target);
        let mut times: Vec<Duration> = (0..30)
            .map(|_| {
                let s = Instant::now();
                r.frame(camera, &target);
                r.wait();
                s.elapsed()
            })
            .collect();
        times.sort();
        r.finish();
        let edges = r.cache_stats().map_or(0, |s| s.edge_pixels);
        (
            r.read_colors(),
            times[15].as_secs_f64() * 1000.0,
            edges,
            r.hardware_rays(),
        )
    };
    if std::env::args().any(|a| a == "quality") {
        kernel.set_hardware_rays(true);
        let make = |edge_rays: bool, edge_samples: u32| {
            let mut r = InstancedRenderer::with_cache_lights(
                &kernel,
                format,
                W,
                H,
                1,
                &lights,
                scene,
                origin,
                CacheOptions {
                    edge_rays,
                    edge_samples,
                    ..Default::default()
                },
            );
            settle(&mut r, camera, &target);
            r
        };
        let timed = |r: &mut InstancedRenderer| {
            let mut t: Vec<f64> = (0..30)
                .map(|_| {
                    let s = Instant::now();
                    r.frame(camera, &target);
                    r.wait();
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            t.sort_by(f64::total_cmp);
            r.finish();
            (t[15], r.read_colors())
        };
        let mut one = make(false, 4);
        let (t1, i1) = timed(&mut one);
        let (t4, i4) = timed(&mut make(true, 4));
        let (t2, i2) = timed(&mut make(true, 2));
        let mut reference = vec![[0.0f64; 3]; (W * H) as usize];
        one.set_footprint(1.0 / 16.0);
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
        }
        let reference: Vec<[f32; 3]> = reference.iter().map(|s| s.map(|x| x as f32)).collect();
        let e = |i: &[[f32; 3]]| 100.0 * relative_rmse(i, &reference);
        println!(
            "from the area integral: one ray {:.2}% ({t1:.2} ms), four quarters {:.2}% ({t4:.2} ms, {:.2}x), two quarters {:.2}% ({t2:.2} ms, {:.2}x)",
            e(&i1),
            e(&i4),
            t4 / t1,
            e(&i2),
            t2 / t1
        );
        return;
    }
    if std::env::args().any(|a| a == "profile") {
        kernel.set_hardware_rays(true);
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            scene,
            origin,
            CacheOptions {
                edge_rays: true,
                ..Default::default()
            },
        );
        settle(&mut r, camera, &target);
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
                .map(|v| (v[15] * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
        return;
    }
    let (one, t_one, _, _) = run(true, false);
    let (hw, t_hw, e_hw, is_hw) = run(true, true);
    let (cp, t_cp, e_cp, _) = run(false, true);
    let (one_sw, _, _, _) = run(false, false);
    println!(
        "one ray, hardware vs software camera rays: {:.4}%",
        100.0 * relative_rmse(&one, &one_sw)
    );
    println!(
        "hardware rays {is_hw}; warm: one ray {t_one:.2} ms, hardware edges {t_hw:.2} ms ({:.2}x), compute edges {t_cp:.2} ms ({:.2}x); edge pixels {e_hw} and {e_cp}; hardware vs compute {:.4}%; hardware vs one ray {:.3}%",
        t_hw / t_one,
        t_cp / t_one,
        100.0 * relative_rmse(&hw, &cp),
        100.0 * relative_rmse(&hw, &one)
    );
}
