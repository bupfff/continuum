//! Check (not a gate): sky cones with cube faces in world axes or in the
//! frame of the object they start on, against Cycles. A uniform sky's light
//! does not depend on the cube's orientation, but binned coverage does a
//! little; the view cache's own-occlusion masks need faces in the object's
//! frame. Million-boulder field, sun at 1 km (gate 3's setup), sky only,
//! 640×360, 4×4 per pixel, against instanced Cycles at 1,024 samples per
//! pixel, at the standing viewpoint and 40 m up.

use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, MaskConeKernel};
use room3d::cycles::{relative_rmse, render_instanced};
use room3d::field::Field;

const W: u32 = 640;
const H: u32 = 360;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let field = Field::new(1_000_000, DVec3::ZERO).with_sun_at(1000.0);
    let origin = field.origin_near_camera();
    let (ground, light) = field.relative(origin);
    let sky = field.sky();
    kernel.set_sky(sky);
    let forward = (ground.target - ground.eye).normalize();
    let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
    let up40 = {
        let d = 30f32.to_radians();
        let eye = Vec3::new(ground.eye.x, ground.eye.y, 40.0);
        Camera {
            eye,
            target: eye + flat * d.cos() - Vec3::Z * d.sin(),
            vertical_fov_degrees: ground.vertical_fov_degrees,
        }
    };
    let factors: Vec<f32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let factors = if factors.is_empty() {
        vec![8.0]
    } else {
        factors
    };
    for (name, camera) in [("standing", ground), ("40 m up", up40)] {
        for &factor in &factors {
            kernel.set_sky_prefilter(factor);
            println!("sky prefilter ×{factor}:");
            let supersampled = |object_frame: bool| {
                kernel.set_sky_object_frame(object_frame);
                let (big, t, _) = kernel.render_instanced(
                    camera,
                    W * 4,
                    H * 4,
                    light,
                    &field.scene,
                    origin,
                    0.0,
                    1,
                );
                let mut px = vec![[0.0f32; 3]; (W * H) as usize];
                for y in 0..(H * 4) as usize {
                    for x in 0..(W * 4) as usize {
                        let p = big[y * (W * 4) as usize + x];
                        let o = &mut px[(y / 4) * W as usize + x / 4];
                        for c in 0..3 {
                            o[c] += p[c] / 16.0;
                        }
                    }
                }
                (px, t)
            };
            let (world, tw) = supersampled(false);
            let (object, to) = supersampled(true);
            let tag = name.replace(' ', "_");
            let reference = render_instanced(
                &field.scene,
                origin,
                light,
                0.0,
                sky,
                camera,
                (W, H),
                1024,
                0,
                1,
                &out_dir.join(format!("sky_frame_check_{tag}_cycles")),
            )?;
            println!(
                "{name}: world axes {:.2}% from Cycles ({:.0} s), object frame {:.2}% ({:.0} s), between them {:.2}%",
                100.0 * relative_rmse(&world, &reference.pixels),
                tw.as_secs_f64(),
                100.0 * relative_rmse(&object, &reference.pixels),
                to.as_secs_f64(),
                100.0 * relative_rmse(&object, &world)
            );
            let mean =
                |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
            let truth = mean(&reference.pixels);
            println!(
                "  mean brightness against Cycles: world {:+.2}%, object frame {:+.2}%",
                100.0 * (mean(&world) / truth - 1.0),
                100.0 * (mean(&object) / truth - 1.0)
            );
        }
    }
    Ok(())
}
