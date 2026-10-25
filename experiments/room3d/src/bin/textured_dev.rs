//! Development check (not a gate): the textured scene through glTF and back,
//! and against Cycles at a small size.
//! `cargo run --release -p room3d --bin textured_dev -- [W H ss spp]`

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::MaskConeKernel;
use flatland::img::Image;
use room3d::cycles::{relative_rmse, render_instanced_lights};
use room3d::textured::Textured;
use room3d::to_rgb;

fn main() -> std::io::Result<()> {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h) = (*args.first().unwrap_or(&320), *args.get(1).unwrap_or(&180));
    let ss = *args.get(2).unwrap_or(&4);
    let spp = *args.get(3).unwrap_or(&256);
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let t = Textured::new();
    let glb = out_dir.join("textured.glb");
    let start = Instant::now();
    continuum_gltf::save(&t.scene, &glb)?;
    let imported = continuum_gltf::load(&glb).map_err(std::io::Error::other)?;
    println!(
        "round trip {:.2} s, {} bytes: {:?}",
        start.elapsed().as_secs_f64(),
        std::fs::metadata(&glb)?.len(),
        imported.stats
    );
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_hardware_rays(false);
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    kernel.set_sky(t.sky);
    let render = |scene: &continuum_gpu::InstancedScene, ss: u32| {
        let (big, _, _) =
            kernel.render_instanced_lights(t.camera, w * ss, h * ss, &t.lights, scene, t.origin, 1);
        let mut px = vec![[0.0f32; 3]; (w * h) as usize];
        for y in 0..(h * ss) as usize {
            for x in 0..(w * ss) as usize {
                let p = big[y * (w * ss) as usize + x];
                let o = &mut px[(y / ss as usize) * w as usize + x / ss as usize];
                for c in 0..3 {
                    o[c] += p[c] / (ss * ss) as f32;
                }
            }
        }
        px
    };
    let original = render(&t.scene, 1);
    let back = render(&imported.scene, 1);
    println!(
        "per pixel, original vs imported: {:.5}%",
        100.0 * relative_rmse(&back, &original)
    );
    let ours = render(&imported.scene, ss);
    let reference = render_instanced_lights(
        &imported.scene,
        t.origin,
        &t.lights,
        t.sky,
        t.camera,
        (w, h),
        spp,
        0,
        1,
        &out_dir.join("textured_dev_cycles"),
    )?;
    let mean = |px: &[[f32; 3]]| px.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
    println!(
        "imported, {ss}×{ss} per pixel vs Cycles {spp} spp: {:.2}% (ours {:+.2}% brighter)",
        100.0 * relative_rmse(&ours, &reference.pixels),
        100.0 * (mean(&ours) / mean(&reference.pixels) - 1.0)
    );
    for (tag, img) in [("ours", &ours), ("cycles", &reference.pixels)] {
        let mut image = Image::new(w as usize, h as usize, [0; 3]);
        for (i, p) in img.iter().enumerate() {
            image.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*p)));
        }
        image.save(&out_dir.join(format!("textured_dev_{tag}.png")))?;
    }
    Ok(())
}
