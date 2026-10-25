//! Checks the instanced Cycles reference (geometry-node instancing) against
//! the flattened one on M5 gate 2's outdoor scene (not a gate).
//! `cargo run --release -p room3d --bin cycles_instancing_check`

use std::path::Path;

use continuum::glam::DVec3;
use room3d::cycles::{relative_rmse, render_instanced, render_with_sky};
use room3d::outdoor::Outdoor;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let scene = Outdoor::with_sun_at(1000.0);
    let (surfaces, albedo) = scene.flattened();
    let size = (640, 360);
    let flat = render_with_sky(
        &surfaces,
        &albedo,
        scene.light,
        scene.radiance,
        scene.sky,
        scene.camera,
        size,
        1024,
        0,
        1,
        &out_dir.join("cycles_check_flat"),
    )?;
    let inst = render_instanced(
        &scene.scene,
        DVec3::ZERO,
        scene.light,
        scene.radiance,
        scene.sky,
        scene.camera,
        size,
        1024,
        0,
        2,
        &out_dir.join("cycles_check_instanced"),
    )?;
    println!(
        "flattened {:.1} s, instanced {:.1} s; instanced vs flattened {:.3}% (different seeds)",
        flat.seconds,
        inst.seconds,
        100.0 * relative_rmse(&inst.pixels, &flat.pixels)
    );
    Ok(())
}
