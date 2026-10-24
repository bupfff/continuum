//! Renders the instanced boulder field (per-pixel prefiltered mask cones). Not a
//! gate. `cargo run --release -p room3d --bin render_field -- [boulders] [width height]`.

use std::path::Path;

use continuum::glam::{DVec3, Vec3};
use continuum_gpu::MaskConeKernel;
use flatland::img::Image;
use room3d::field::Field;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let count = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10_000);
    let w = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(960u32);
    let h = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(540u32);
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    let field = Field::new(count, DVec3::ZERO);
    let origin = field.origin_near_camera();
    let (camera, light) = field.relative(origin);
    let (pixels, time, bytes) =
        kernel.render_instanced(camera, w, h, light, &field.scene, origin, field.radiance, 3);
    let mut image = Image::new(w as usize, h as usize, [0; 3]);
    for (i, p) in pixels.iter().enumerate() {
        image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
    }
    let name = format!("render_field_{count}.png");
    image.save(&out_dir.join(&name))?;
    println!(
        "{} boulders, {} triangles: {:.1} ms at {w}×{h} (best of 3), {:.0} MB on the GPU. Wrote out/{name}.",
        field.boulders(),
        field.triangles(),
        time.as_secs_f64() * 1000.0,
        bytes as f64 / 1e6
    );
    Ok(())
}
