//! Development check (not a gate): the field's three colored lights per pixel
//! against Cycles at a small size, to check the reference pipeline's light
//! list. `cargo run --release -p room3d --bin lights_dev -- [W H spp]`, with
//! `BOULDERS=n` and `LIGHTS=0,2` to narrow it down.

use std::path::Path;

use continuum::glam::DVec3;
use continuum_gpu::MaskConeKernel;
use flatland::img::Image;
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::field::Field;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h, spp) = (
        *args.first().unwrap_or(&320),
        *args.get(1).unwrap_or(&180),
        *args.get(2).unwrap_or(&256),
    );
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(
        std::env::var("PREFILTER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0),
    );
    // Samples per pixel along each axis.
    let ss: u32 = std::env::var("SS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let boulders: usize = std::env::var("BOULDERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let field = Field::new(boulders, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, _) = field.relative(origin);
    let all = field.colored_lights(origin, 1000.0);
    // LIGHTS picks some, as "0,2".
    let lights: Vec<_> = match std::env::var("LIGHTS") {
        Ok(v) => v
            .split(',')
            .filter_map(|k| k.parse::<usize>().ok())
            .map(|k| all[k])
            .collect(),
        Err(_) => all.to_vec(),
    };
    let (big, t, _) =
        kernel.render_instanced_lights(camera, w * ss, h * ss, &lights, &field.scene, origin, 1);
    let mut ours = vec![[0.0f32; 3]; (w * h) as usize];
    // Pixels where a subsample sees a light (its color is the light's
    // radiance): emitters are hundreds of times brighter than surfaces, so
    // they are scored apart.
    let mut emitter = vec![false; (w * h) as usize];
    for y in 0..(h * ss) as usize {
        for x in 0..(w * ss) as usize {
            let p = big[y * (w * ss) as usize + x];
            let i = (y / ss as usize) * w as usize + x / ss as usize;
            if lights.iter().any(|l| l.radiance.to_array() == p) {
                emitter[i] = true;
            }
            for c in 0..3 {
                ours[i][c] += p[c] / (ss * ss) as f32;
            }
        }
    }
    let reference = render_instanced_lights(
        &field.scene,
        origin,
        &lights,
        continuum::glam::Vec3::ZERO,
        camera,
        (w, h),
        spp,
        0,
        1,
        &out_dir.join("lights_dev_cycles"),
    )?;
    // One pixel around them too: an emitter's edge can cover less of a pixel
    // than one of its 16 samples.
    let near = emitter.clone();
    for y in 0..h as usize {
        for x in 0..w as usize {
            if near[y * w as usize + x] {
                for (dx, dy) in [
                    (-1i32, 0i32),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                ] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                        emitter[ny as usize * w as usize + nx as usize] = true;
                    }
                }
            }
        }
    }
    let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
        img.iter()
            .zip(&emitter)
            .filter(|(_, e)| !**e)
            .map(|(p, _)| *p)
            .collect()
    };
    let dropped = emitter.iter().filter(|e| **e).count();
    let mean = |img: &[[f32; 3]]| img.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    println!(
        "  surfaces: ours {:+.2}% brighter than Cycles",
        100.0 * (mean(&keep(&ours)) / mean(&keep(&reference.pixels)) - 1.0)
    );
    let mut worst: Vec<(f32, usize)> = ours
        .iter()
        .zip(&reference.pixels)
        .enumerate()
        .filter(|(i, _)| !emitter[*i])
        .map(|(i, (a, b))| {
            (
                (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs(),
                i,
            )
        })
        .collect();
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(d, i) in worst.iter().take(6) {
        println!(
            "  ({}, {}): ours {:?}, Cycles {:?} (|Δ| {d:.3})",
            i % w as usize,
            i / w as usize,
            ours[i],
            reference.pixels[i]
        );
    }
    println!(
        "three lights, {w}×{h}, 4×4 per pixel ({:.1} s) vs Cycles {spp} spp ({:.1} s): surfaces {:.2}% ({dropped} emitter pixels apart), all pixels {:.2}%",
        t.as_secs_f64(),
        reference.seconds,
        100.0 * relative_rmse(&keep(&ours), &keep(&reference.pixels)),
        100.0 * relative_rmse(&ours, &reference.pixels)
    );
    for (tag, img) in [("ours", &ours), ("cycles", &reference.pixels)] {
        let mut image = Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in img.iter().enumerate() {
            image.set(
                i % w as usize,
                i / w as usize,
                to_rgb(continuum::glam::Vec3::from_array(*p)),
            );
        }
        image.save(&out_dir.join(format!("lights_dev_{tag}.png")))?;
    }
    Ok(())
}
