//! Sky cones' prefilter factor against cost and error on the boulder field
//! (not a gate): sky-only renders at 320×180, compared with factor 1.
//! `cargo run --release -p room3d --bin sky_prefilter_sweep -- [boulders]`

use continuum::glam::DVec3;
use continuum_gpu::MaskConeKernel;
use room3d::cycles::relative_rmse;
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
    let mut base = None;
    for factor in [1.0f32, 2.0, 4.0, 8.0] {
        kernel.set_sky_prefilter(factor);
        let (pixels, time, _) =
            kernel.render_instanced(camera, 320, 180, light, &field.scene, origin, 0.0, 1);
        let base = base.get_or_insert_with(|| pixels.clone());
        println!(
            "sky prefilter ×{factor}: {:.1} µs per pixel, {:.2}% from ×1",
            time.as_secs_f64() * 1e6 / (320.0 * 180.0),
            100.0 * relative_rmse(&pixels, base)
        );
    }
}
