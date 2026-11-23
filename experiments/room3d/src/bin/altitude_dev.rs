//! Development (not a gate): the basics scorecard's field seen from above
//! (`AD_UP` meters higher, 40; tilted `AD_DOWN` degrees down, 15), measured
//! as the scorecard measures (`scorecard::quality`: 640×360, 4×4
//! supersampled, a fixed bake budget, at most 600 frames) against Cycles.
//! Far boulders show coarse pages there, whose texels light their whole area
//! from one point (the viewer's dark blocks, 2026-11-22).
//!
//! `cargo run --release -p room3d --bin altitude_dev`

use std::path::Path;

use continuum::glam::{Quat, Vec3};
use room3d::scorecard::{field_case, quality};

fn main() -> std::io::Result<()> {
    let num = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let mut case = field_case();
    case.name = "field_high";
    let up = num("AD_UP", 40.0);
    let down = num("AD_DOWN", 15.0);
    let c = &mut case.camera;
    let d = (c.target - c.eye).normalize();
    let right = d.cross(Vec3::Z).normalize();
    c.eye.z += up;
    c.target = c.eye + Quat::from_axis_angle(right, -down.to_radians()) * d;
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/scorecard");
    std::fs::create_dir_all(&out)?;
    let q = quality(&case, &out)?;
    println!(
        "field from {up} m up, {down}° down: {:.2}% from Cycles, pixels off {:.2}%, settled {:?}, means {:.4} / {:.4}",
        100.0 * q.rmse,
        100.0 * q.off,
        q.settled,
        q.mean_ours,
        q.mean_cycles
    );
    Ok(())
}
