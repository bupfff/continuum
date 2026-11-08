//! Development: what the slowest frames of the field's flight with three
//! colored lights have in common (M5 gate 6's lights, M5 gate 8's path,
//! 1280×720). Not a gate.

use std::time::Instant;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{CacheOptions, CacheStats, Camera, InstancedRenderer, MaskConeKernel, wgpu};
use room3d::field::Field;

const W: u32 = 1280;
const H: u32 = 720;
const FLIGHT: usize = 600;

/// A profiled frame: its index, total bake time, and each bake pass's label
/// and time (ms).
type Row = (usize, f64, Vec<(&'static str, f64)>);

fn main() {
    let work = std::env::args().any(|a| a == "work");
    let kernel = if work {
        MaskConeKernel::new_with(continuum_gpu::KernelOptions {
            count_work: true,
            ..Default::default()
        })
    } else {
        MaskConeKernel::new()
    }
    .expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(true);
    kernel.set_sky(Vec3::ZERO);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, _) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0).to_vec();
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
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let camera = |f: usize| {
        let t = f as f32 / (FLIGHT - 1) as f32;
        let (ahead, height, down) = (300.0 * t, 1.7 + 148.3 * t, (45.0 * t).to_radians());
        let eye = Vec3::new(ground.eye.x, ground.eye.y, height) + flat * ahead;
        Camera {
            eye,
            target: eye + flat * down.cos() - Vec3::Z * down.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    if std::env::args().any(|a| a == "passes") {
        // Every frame profiled: each bake pass submitted and waited for
        // alone.
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            &field.scene,
            origin,
            CacheOptions::default(),
        );
        let mut rows: Vec<Row> = Vec::new();
        for f in 0..FLIGHT {
            let p = r.frame_profiled(camera(f), &target);
            let passes: Vec<(&'static str, f64)> = p
                .bake_passes
                .iter()
                .filter(|(l, _)| !l.is_empty())
                .map(|(l, d)| (*l, d.as_secs_f64() * 1000.0))
                .collect();
            let total = passes.iter().map(|x| x.1).sum::<f64>();
            rows.push((f, total, passes));
        }
        let mut mean: std::collections::BTreeMap<&str, f64> = Default::default();
        for (_, _, passes) in &rows {
            for (l, d) in passes {
                *mean.entry(l).or_default() += d / FLIGHT as f64;
            }
        }
        println!("mean per frame: {mean:.2?}");
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (f, total, passes) in rows.iter().take(10) {
            let pages = r.cache_log()[*f].pages_baked;
            println!("frame {f}: bake {total:.2} ms ({pages} pages): {passes:.2?}");
        }
        return;
    }
    for run in 0..2 {
        let mut r = InstancedRenderer::with_cache_lights(
            &kernel,
            format,
            W,
            H,
            1,
            &lights,
            &field.scene,
            origin,
            CacheOptions::default(),
        );
        let mut times = Vec::new();
        for f in 0..FLIGHT {
            let s = Instant::now();
            r.frame(camera(f), &target);
            times.push(s.elapsed().as_secs_f64() * 1000.0);
        }
        r.finish();
        let log: Vec<CacheStats> = r.cache_log().to_vec();
        // Per frame: GPU bake time, units planned, and units per mip.
        let mut csv = String::from(
            "frame,timed,bake_ms,spent,pages,m0,m1,m2,m3,m4,m5,m6,m7,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11\n",
        );
        for (i, s) in log.iter().enumerate() {
            csv += &format!(
                "{i},{},{},{},{},{},{}\n",
                s.timed as u32,
                s.gpu_bake_ms,
                s.spent,
                s.pages_baked,
                s.spent_by_mip.map(|x| x.to_string()).join(","),
                s.work.map(|x| x.to_string()).join(",")
            );
        }
        std::fs::write(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("out/spikes_run{}.csv", run + 1)),
            csv,
        )
        .unwrap();
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let p95 = sorted[FLIGHT * 95 / 100];
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        // A frame waits for the GPU work of the frame two before: pair each
        // frame's wall time with that frame's record.
        let summary = |sel: &dyn Fn(usize) -> bool| {
            let idx: Vec<usize> = (2..FLIGHT).filter(|&i| sel(i)).collect();
            let n = idx.len().max(1) as f64;
            let m = |f: &dyn Fn(&CacheStats) -> f64| {
                idx.iter().map(|&i| f(&log[i - 2])).sum::<f64>() / n
            };
            format!(
                "{} frames: wall {:.2} ms; planning {:.2} ms, evicting {:.2} ms, waiting {:.2} ms; GPU bake {:.2} ms, camera {:.2} ms; pages baked {:.0}, evicted {:.0}, budget {:.0}",
                idx.len(),
                idx.iter().map(|&i| times[i]).sum::<f64>() / n,
                m(&|s| ms(s.planning)),
                m(&|s| ms(s.evicting)),
                m(&|s| ms(s.waiting)),
                m(&|s| s.gpu_bake_ms as f64),
                m(&|s| s.gpu_render_ms as f64),
                m(&|s| s.pages_baked as f64),
                m(&|s| s.evicted as f64),
                m(&|s| s.budget as f64),
            )
        };
        println!(
            "run {}: p95 {p95:.2} ms, p99 {:.2} ms, worst {:.2} ms",
            run + 1,
            sorted[FLIGHT * 99 / 100 - 1],
            sorted[FLIGHT - 1]
        );
        println!("  slowest 5%: {}", summary(&|i| times[i] >= p95));
        println!("  the rest:   {}", summary(&|i| times[i] < p95));
        let worst: Vec<usize> = {
            let mut v: Vec<usize> = (0..FLIGHT).collect();
            v.sort_by(|&a, &b| times[b].total_cmp(&times[a]));
            v.truncate(8);
            v
        };
        for i in worst.into_iter().filter(|&i| i >= 2) {
            let s = &log[i - 2];
            println!(
                "  frame {i} (record of frame {}): {:.1} ms; planning {:.2}, evicting {:.2}, waiting {:.2}; bake {:.2}, camera {:.2}; baked {}, evicted {}, timed {}",
                i - 2,
                times[i],
                ms(s.planning),
                ms(s.evicting),
                ms(s.waiting),
                s.gpu_bake_ms,
                s.gpu_render_ms,
                s.pages_baked,
                s.evicted,
                s.timed
            );
        }
    }
}
