//! M3 gate 2: general meshes in the lighting cache.
//!
//! The M2 lighting cache put texels on box faces. It now takes any mesh with a
//! chart layout (each triangle has coordinates in a rectangular chart; texels
//! and probes are placed through those coordinates) and smooth shading normals.
//!
//! Scene: the showcase (static meshes only, movers removed) at 1920×1080 from
//! the first key of its camera path.
//!
//! Pass criterion, fixed before the first run: (1) the cached render (direct
//! light only) is within 2% relative RMSE of the per-pixel mask-cone render of
//! the same frame, which shades with the same interpolated normals; (2) the
//! cached render takes at most 16.7 ms (best of 5, warm). Bake time and texel
//! count are reported.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Lightmap, MaskConeKernel};
use flatland::img::Image;
use room3d::showcase::{Showcase, TEXEL};
use room3d::to_rgb;

const W: u32 = 1920;
const H: u32 = 1080;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let scene = Showcase::new().without_movers();
    let meshes = scene.meshes();
    let triangles: usize = meshes.iter().map(Vec::len).sum();
    let camera = scene.camera(0.0);
    let start = Instant::now();
    let mut lightmap = Lightmap::new(&scene.surfaces, TEXEL, |_| true);
    let layout = start.elapsed();
    let bake = kernel.bake(&mut lightmap, scene.light, &meshes);
    let real = lightmap
        .receivers
        .iter()
        .filter(|r| r.1 != Vec3::ZERO)
        .count();
    let (cached, t_cached) = kernel.render_cached(
        camera,
        W,
        H,
        scene.light,
        &meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
        5,
    );
    let (exact, t_exact) = kernel.render_per_pixel(
        camera,
        W,
        H,
        scene.light,
        &meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
        3,
    );
    let values = |img: &[[f32; 3]]| img.iter().flatten().copied().collect::<Vec<f32>>();
    let (a, b) = (values(&cached), values(&exact));
    let mean = b.iter().map(|&x| x as f64).sum::<f64>() / b.len() as f64;
    let rmse = (a
        .iter()
        .zip(&b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
        / mean;
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let pass = ms(t_cached) <= 16.7 && rmse <= 0.02;

    let gap = 8;
    let (w, h) = (W as usize / 2, H as usize / 2);
    let mut sheet = Image::new(3 * w + 4 * gap, h + 2 * gap, [28; 3]);
    for y in 0..h {
        for x in 0..w {
            let px = 2 * y * W as usize + 2 * x;
            let (c, e) = (Vec3::from_array(cached[px]), Vec3::from_array(exact[px]));
            sheet.set(gap + x, gap + y, to_rgb(c));
            sheet.set(2 * gap + w + x, gap + y, to_rgb(e));
            let d = ((c - e).abs().max_element() / (mean as f32) * 255.0 * 4.0).min(255.0) as u8;
            sheet.set(3 * gap + 2 * w + x, gap + y, [d, d, d]);
        }
    }
    sheet.save(&out_dir.join("m3_gate2.png"))?;

    let mut out = String::new();
    writeln!(out, "# M3 gate 2 results\n").unwrap();
    writeln!(
        out,
        "Showcase without movers: {} meshes, {triangles} triangles, on {} at {W}×{H}. \
         Lightmap: {} texel slots ({real} on surfaces) about {TEXEL} m apart; layout {:.0} ms, bake {:.0} ms.\n",
        meshes.len(),
        kernel.adapter_name,
        lightmap.texels(),
        ms(layout),
        ms(bake)
    )
    .unwrap();
    writeln!(
        out,
        "- Cached render: {:.2} ms (best of 5).\n\
         - Per-pixel mask cones with the same normals: {:.1} ms (best of 3).\n\
         - Relative RMSE (RMSE over the reference's mean, linear RGB): {:.2}%.",
        ms(t_cached),
        ms(t_exact),
        100.0 * rmse
    )
    .unwrap();
    writeln!(
        out,
        "\n## Verdict\n\nWithin 2% RMSE and at most 16.7 ms → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m3_gate2_results.md"), &out)?;
    Ok(())
}
