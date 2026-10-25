//! Development check (not a gate): M5 gate 6's accuracy section against the
//! Cycles references that gate rendered (read from `out/m5_gate6_cycles_*`,
//! so Blender does not run): the field's three colored lights per pixel,
//! 16×16 samples per pixel at 320×180, over surfaces (pixels whose samples
//! see a light, and their neighbours, set apart), each light alone, and
//! with the sky.
//! `cargo run --release -p room3d --bin lights_cycles_check`

use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Light, MaskConeKernel};
use room3d::cycles::relative_rmse;
use room3d::field::Field;

/// The 320×180 reference in a gate 6 work directory, as RGB.
fn reference(dir: &Path) -> Vec<[f32; 3]> {
    let file = std::fs::read_dir(dir)
        .expect("gate 6's Cycles references")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.extension().is_some_and(|x| x == "f32")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("320x180_4096spp"))
        })
        .expect("a 320×180 reference");
    let raw = std::fs::read(file).expect("reference");
    let values: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    values.as_chunks::<3>().0.to_vec()
}

fn main() {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_hardware_rays(false);
    let field = Field::new(1_000_000, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (ground, _) = field.relative(origin);
    let lights = field.colored_lights(origin, 1000.0);
    let (w, h, ss) = (320u32, 180u32, 16u32);
    let score = |lights: &[Light], sky: Vec3, tag: &str| {
        kernel.set_sky(sky);
        let (big, _, _) =
            kernel.render_instanced_lights(ground, w * ss, h * ss, lights, &field.scene, origin, 1);
        let mut px = vec![[0.0f32; 3]; (w * h) as usize];
        let mut sees = vec![false; (w * h) as usize];
        for y in 0..(h * ss) as usize {
            for x in 0..(w * ss) as usize {
                let p = big[y * (w * ss) as usize + x];
                let i = (y / ss as usize) * w as usize + x / ss as usize;
                if lights.iter().any(|l| l.radiance.to_array() == p) {
                    sees[i] = true;
                }
                for c in 0..3 {
                    px[i][c] += p[c] / (ss * ss) as f32;
                }
            }
        }
        let mut apart = sees.clone();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                if !sees[(y * w as i32 + x) as usize] {
                    continue;
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                            apart[(ny * w as i32 + nx) as usize] = true;
                        }
                    }
                }
            }
        }
        let cycles = reference(&out_dir.join(format!("m5_gate6_cycles_{tag}")));
        let keep = |img: &[[f32; 3]]| -> Vec<[f32; 3]> {
            img.iter()
                .zip(&apart)
                .filter(|(_, x)| !**x)
                .map(|(p, _)| *p)
                .collect()
        };
        println!(
            "{tag}: {:.2}% over surfaces vs Cycles",
            100.0 * relative_rmse(&keep(&px), &keep(&cycles))
        );
    };
    score(&lights, Vec3::ZERO, "lights");
    for (k, name) in ["sun", "fill", "lamp"].iter().enumerate() {
        score(&lights[k..k + 1], Vec3::ZERO, name);
    }
    score(&lights, field.sky(), "lights_sky");
}
