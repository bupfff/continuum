//! Development check for object-level prefiltering (not a gate): sky-only and
//! sun-only renders of the boulder field with and without it.
//! `cargo run --release -p room3d --bin object_prefilter_check -- [boulders]`

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::MaskConeKernel;
use room3d::cycles::relative_rmse;
use room3d::field::Field;

fn main() {
    let count = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20_000usize);
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(1.0);
    let field = Field::new(count, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    for (name, radiance, sky) in [
        ("sun only", field.radiance, Vec3::ZERO),
        ("sky only", 0.0, Vec3::new(0.03, 0.04, 0.05)),
    ] {
        kernel.set_sky(sky);
        let mut images = Vec::new();
        for on in [false, true] {
            kernel.set_object_prefilter(on);
            let (pixels, time, _) =
                kernel.render_instanced(camera, 320, 180, light, &field.scene, origin, radiance, 2);
            println!(
                "{name}, object prefilter {on}: {:.1} ms",
                time.as_secs_f64() * 1e3
            );
            images.push(pixels);
        }
        println!(
            "{name}: with vs without {:.3}%",
            100.0 * relative_rmse(&images[1], &images[0])
        );
    }
}
