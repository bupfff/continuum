//! M2 gate 4: a world-space lighting cache on the GPU.
//!
//! Direct light on static surfaces changes only when a light or mover changes,
//! so it is baked into lightmap texels (one mask cone per texel, through the
//! hierarchy). A frame is then a camera ray per pixel plus a bilinear lookup;
//! dynamic geometry (none in this scene) would still get a cone per pixel.
//!
//! Pass criterion, fixed before the first run: (1) the cached render at
//! 1920×1080 takes at most 16.7 ms (best of 5) on the M4 Air; (2) its relative
//! RMSE against the per-pixel cone render of the same frame is at most 2%.
//! Bake time is reported.
//!
//! Attempt 1 failed (2) at 3.78%. The error map showed two sources: faces
//! pointing away from the light were baked with light (the bake lacked the
//! renderer's facing rule, a bug, now fixed for every receiver), and
//! interpolation stripes where shadows have structure finer than texels. Attempt
//! 2 fixes only the bug; texel size is unchanged.

use std::fmt::Write as _;
use std::path::Path;

use continuum::glam::Vec3;
use continuum_gpu::{Camera, Lightmap, MaskConeKernel, RectLight};
use flatland::img::Image;
use room3d::{render_scene, to_rgb};

const TEXEL: f32 = 0.005;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let scene = render_scene();
    let l = &scene.room.light;
    let light = RectLight {
        corner: l.corner,
        edge_u: l.edge_u,
        edge_v: l.edge_v,
    };
    let camera = Camera {
        eye: scene.eye,
        target: scene.target,
        vertical_fov_degrees: scene.vertical_fov_degrees,
    };
    let (w, h) = (1920u32, 1080u32);

    let mut lightmap = Lightmap::boxes(&scene.room.meshes, TEXEL, |_| true);
    let bake = kernel.bake(&mut lightmap, light, &scene.room.meshes);
    let (cached, t_cached) = kernel.render_cached(
        camera,
        w,
        h,
        light,
        &scene.room.meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
        5,
    );
    let (exact, t_exact) = kernel.render(
        camera,
        w,
        h,
        light,
        &scene.room.meshes,
        &scene.albedo,
        scene.radiance,
        true,
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
    let ms = t_cached.as_secs_f64() * 1000.0;
    let pass = ms <= 16.7 && rmse <= 0.02;

    let mut img = Image::new(w as usize, h as usize, [0; 3]);
    for (i, c) in cached.iter().enumerate() {
        img.set(i % w as usize, i / w as usize, to_rgb(Vec3::from_array(*c)));
    }
    img.save(&out_dir.join("m2_gate4_cached_1080p.png"))?;
    // Where the error is: signed difference (red: cache darker, blue: brighter), 10% of mean at full scale.
    let mut err = Image::new(w as usize, h as usize, [0; 3]);
    for (i, (c, e)) in cached.iter().zip(&exact).enumerate() {
        let d = (Vec3::from_array(*c) - Vec3::from_array(*e)).dot(Vec3::ONE) / 3.0;
        let m = ((d.abs() / (0.1 * mean as f32)).min(1.0).sqrt() * 255.0) as u8;
        err.set(
            i % w as usize,
            i / w as usize,
            if d < 0.0 {
                [m, m / 4, m / 6]
            } else {
                [m / 5, m / 2, m]
            },
        );
    }
    err.save(&out_dir.join("m2_gate4_error.png"))?;

    let mut out = String::new();
    writeln!(out, "# M2 gate 4 results\n").unwrap();
    writeln!(
        out,
        "Render scene, {} (wgpu). Lightmap: {} texels of about {TEXEL} on {} static meshes.\n\n\
         - Bake (one mask cone per texel, hierarchy): {:.1} ms.\n\
         - Cached render at {w}×{h}: {ms:.2} ms (best of 5).\n\
         - Per-pixel cone render at {w}×{h} for comparison: {:.1} ms.\n\
         - Relative RMSE of the cached render against per-pixel cones: {:.2}%.\n\n## Verdict\n\n\
         At most 16.7 ms at 1080p and within 2% RMSE → {}.",
        kernel.adapter_name,
        lightmap.texels(),
        scene.room.meshes.len(),
        bake.as_secs_f64() * 1000.0,
        t_exact.as_secs_f64() * 1000.0,
        rmse * 100.0,
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("m2_gate4_results.md"), &out)?;
    Ok(())
}
