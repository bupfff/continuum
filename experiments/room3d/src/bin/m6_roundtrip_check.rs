//! Diagnostic (not a gate): where M6 gate 1's round-trip difference comes
//! from. Renders the textured scene per pixel exactly as the gate's
//! criterion 1 does (640×360, one sample per pixel, compute rays) and swaps
//! one imported part at a time into the original scene: instance order,
//! instance transforms, geometries, textures. For each it prints the
//! relative RMSE against the original and how concentrated the error is.
//! `cargo run --release -p room3d --bin m6_roundtrip_check`

use std::path::Path;

use continuum_gpu::{InstancedScene, MaskConeKernel};
use room3d::cycles::relative_rmse;
use room3d::textured::Textured;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let kernel = MaskConeKernel::new().expect("GPU");
    kernel.set_prefilter(1.0);
    kernel.set_sky_prefilter(8.0);
    let t = Textured::new();
    kernel.set_sky(t.sky);
    kernel.set_hardware_rays(false);
    let glb = out_dir.join("m6_roundtrip_check.glb");
    continuum_gltf::save(&t.scene, &glb)?;
    let imported = continuum_gltf::load(&glb).map_err(std::io::Error::other)?.scene;

    // Imported instances, matched to the original's by position.
    let n = t.scene.instances.len();
    assert_eq!(imported.instances.len(), n);
    let matched: Vec<usize> = t
        .scene
        .instances
        .iter()
        .map(|o| {
            (0..n)
                .min_by(|&a, &b| {
                    let d = |k: usize| (imported.instances[k].position - o.position).length();
                    d(a).total_cmp(&d(b))
                })
                .unwrap()
        })
        .collect();
    println!(
        "imported instance order: {:?}",
        &matched[..matched.len().min(8)]
    );
    let mut rotation_error = 0.0f32;
    let mut scale_error = 0.0f32;
    let mut albedo_error = 0.0f32;
    for (o, &k) in t.scene.instances.iter().zip(&matched) {
        let i = &imported.instances[k];
        assert_eq!(i.geometry, o.geometry);
        rotation_error = rotation_error.max(
            (i.rotation - o.rotation)
                .to_cols_array()
                .iter()
                .fold(0.0f32, |m, x| m.max(x.abs())),
        );
        scale_error = scale_error.max((i.scale / o.scale - 1.0).abs());
        albedo_error = albedo_error.max((i.albedo - o.albedo).abs().max_element());
    }
    println!(
        "largest differences: rotation entry {rotation_error:.2e}, scale {scale_error:.2e}, albedo {albedo_error:.2e}"
    );
    for (g, (o, i)) in t
        .scene
        .geometries
        .iter()
        .zip(&imported.geometries)
        .enumerate()
    {
        let same_triangles = o.triangles == i.triangles;
        let same_normals = o.normals == i.normals;
        let same_tex = o.tex == i.tex;
        println!(
            "geometry {g}: {} → {} triangles; triangles equal {same_triangles}, normals equal {same_normals}, tex equal {same_tex}",
            o.triangles.len(),
            i.triangles.len()
        );
    }
    let textures_equal = t
        .scene
        .textures
        .iter()
        .zip(&imported.textures)
        .all(|(a, b)| a.rgba == b.rgba);
    println!("textures equal: {textures_equal}");

    let (w, h) = (640, 360);
    let render = |scene: &InstancedScene| {
        kernel
            .render_instanced_lights(t.camera, w, h, &t.lights, scene, t.origin, 1)
            .0
    };
    let original = render(&t.scene);
    let report = |name: &str, scene: &InstancedScene| {
        let px = render(scene);
        let total: f64 = px
            .iter()
            .zip(&original)
            .map(|(a, b)| (0..3).map(|c| ((a[c] - b[c]) as f64).powi(2)).sum::<f64>())
            .sum();
        let mut errors: Vec<(f64, usize)> = px
            .iter()
            .zip(&original)
            .enumerate()
            .map(|(k, (a, b))| {
                (
                    (0..3).map(|c| ((a[c] - b[c]) as f64).powi(2)).sum::<f64>(),
                    k,
                )
            })
            .collect();
        errors.sort_by(|a, b| b.0.total_cmp(&a.0));
        let differing = errors.iter().filter(|e| e.0 > 0.0).count();
        let top: f64 = errors.iter().take(20).map(|e| e.0).sum();
        let worst: Vec<String> = errors
            .iter()
            .take(4)
            .map(|&(_, k)| {
                let (a, b) = (px[k], original[k]);
                format!(
                    "({}, {}) {:.3} vs {:.3}",
                    k % w as usize,
                    k / w as usize,
                    a[0] + a[1] + a[2],
                    b[0] + b[1] + b[2]
                )
            })
            .collect();
        println!(
            "{name}: {:.5}%, {differing} pixels differ, top 20 hold {:.0}% of the squared error; worst {}",
            100.0 * relative_rmse(&px, &original),
            if total > 0.0 { 100.0 * top / total } else { 0.0 },
            worst.join(", ")
        );
    };
    report("original again", &t.scene);
    report("imported", &imported);
    let mut reordered = t.scene.clone();
    for (o, &k) in t.scene.instances.iter().zip(&matched) {
        reordered.instances[k] = *o;
    }
    report("original, imported instance order", &reordered);
    let mut transforms = t.scene.clone();
    for (o, &k) in transforms.instances.iter_mut().zip(&matched) {
        let i = &imported.instances[k];
        o.rotation = i.rotation;
        o.scale = i.scale;
        o.position = i.position;
    }
    report("original, imported transforms", &transforms);
    let mut geometries = t.scene.clone();
    geometries.geometries = imported.geometries.clone();
    report("original, imported geometries", &geometries);
    let mut textures = t.scene.clone();
    textures.textures = imported.textures.clone();
    report("original, imported textures", &textures);
    Ok(())
}
