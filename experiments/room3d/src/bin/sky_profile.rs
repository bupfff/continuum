//! Where a sky cone's time goes on the boulder field (not a gate): sky-only
//! renders at profiling stages (1: no cones, 2: cone setup and weights, 0:
//! full), prefiltered with object-level prefiltering.
//! `cargo run --release -p room3d --bin sky_profile -- [boulders]`

use continuum::glam::DVec3;
use continuum_gpu::MaskConeKernel;
use room3d::field::Field;

fn main() {
    let count = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000usize);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(1.0);
    let field = Field::new(count, DVec3::ZERO).with_sun_at(1000.0);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    kernel.set_sky(field.sky());
    for stage in [1u32, 2, 3, 4, 0] {
        kernel.set_profile_stage(stage);
        let (_, time, _) =
            kernel.render_instanced(camera, 320, 180, light, &field.scene, origin, 0.0, 2);
        println!(
            "stage {stage}: {:.1} ms ({:.1} µs per pixel)",
            time.as_secs_f64() * 1e3,
            time.as_secs_f64() * 1e6 / (320.0 * 180.0)
        );
    }
}
