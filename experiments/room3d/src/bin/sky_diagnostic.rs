//! Diagnostics for sky light (not a gate): the outdoor scene's per-pixel
//! exact renders with the sun only, the sky only and both, saved as raw linear
//! RGB (f32) for comparison with the Cycles references of M5 gate 1.
//! `cargo run --release -p room3d --bin sky_diagnostic`

use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::MaskConeKernel;
use room3d::outdoor::Outdoor;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    let scene = Outdoor::new();
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(0.0);
    for (name, radiance, sky) in [
        ("sun", scene.radiance, Vec3::ZERO),
        ("sky", 0.0, scene.sky),
        ("both", scene.radiance, scene.sky),
    ] {
        kernel.set_sky(sky);
        let (pixels, time, _) = kernel.render_instanced(
            scene.camera,
            640,
            360,
            scene.light,
            &scene.scene,
            DVec3::ZERO,
            radiance,
            1,
        );
        let bytes: Vec<u8> = pixels
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(out_dir.join(format!("sky_diagnostic_{name}.f32")), bytes)?;
        println!("{name}: {:.1} s", time.as_secs_f64());
    }
    Ok(())
}
