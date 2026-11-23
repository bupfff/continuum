//! Development (not a gate): the basics scorecard's field seen from above
//! (`AD_UP` meters higher, 40; tilted `AD_DOWN` degrees down, 15), measured
//! as the scorecard measures (`scorecard::quality`: 640×360, 4×4
//! supersampled, a fixed bake budget, at most 600 frames) against Cycles.
//! Far boulders show coarse pages there, whose texels light their whole area
//! from one point (the viewer's dark blocks, 2026-11-22). `AD_DIRECT=1`:
//! without bounce light, here and in Cycles. `SC_AREA=1`: with
//! `CacheOptions::area_texels`.
//!
//! `cargo run --release -p room3d --bin altitude_dev`

use std::path::Path;

use room3d::scorecard::{field_from_above, quality};

fn main() -> std::io::Result<()> {
    let num = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (up, down) = (num("AD_UP", 40.0), num("AD_DOWN", 15.0));
    let case = field_from_above(up, down, std::env::var_os("AD_DIRECT").is_some());
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
