//! Checks the Cycles reference pipeline on a scene whose answer is known: a
//! floor under the showcase's skylight, seen by a wide camera that also sees
//! the light. Direct light on an open floor is exact in both renderers (no
//! occluders, no bounce), and the light's front shows its radiance. Not a gate:
//! it validates units, camera mapping and the one-sided emitter before gates
//! rely on the pipeline.

use std::path::Path;

use continuum::glam::Vec3;
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Lightmap, MaskConeKernel};
use room3d::cycles::{self, relative_rmse};
use room3d::showcase::Showcase;

const W: u32 = 480;
const H: u32 = 360;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let kernel = MaskConeKernel::new().expect("GPU");
    let showcase = Showcase::new();
    let floor = Surface::from_triangles(vec![
        [
            Vec3::ZERO,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.0, 4.0, 0.0),
        ],
        [
            Vec3::ZERO,
            Vec3::new(5.0, 4.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ],
    ]);
    let surfaces = vec![floor];
    let albedo = vec![Vec3::new(0.7, 0.6, 0.5)];
    let meshes = vec![surfaces[0].triangles.clone()];
    let camera = Camera {
        eye: Vec3::new(2.5, -1.0, 1.2),
        target: Vec3::new(2.5, 1.0, 1.75),
        vertical_fov_degrees: 90.0,
    };
    let mut lightmap = Lightmap::new(&surfaces, 0.01, |_| true);
    kernel.bake(&mut lightmap, showcase.light, &meshes);
    let (ours, _) = kernel.render_cached(
        camera,
        W,
        H,
        showcase.light,
        &meshes,
        &albedo,
        showcase.radiance,
        &lightmap,
        1,
    );
    let reference = cycles::render(
        &surfaces,
        &albedo,
        showcase.light,
        showcase.radiance,
        camera,
        (W, H),
        1024,
        1,
        0,
        &out_dir.join("cycles"),
    )?;
    let theirs = &reference.pixels;
    let emitter: Vec<usize> = (0..ours.len())
        .filter(|&i| ours[i][0] >= showcase.radiance * 0.999)
        .collect();
    let floor: Vec<usize> = (0..ours.len())
        .filter(|&i| ours[i][0] > 0.0 && ours[i][0] < showcase.radiance * 0.5)
        .collect();
    let pick = |img: &[[f32; 3]], idx: &[usize]| idx.iter().map(|&i| img[i]).collect::<Vec<_>>();
    let mean = |img: &[[f32; 3]], idx: &[usize]| {
        idx.iter().map(|&i| img[i][0] as f64).sum::<f64>() / idx.len().max(1) as f64
    };
    println!(
        "Cycles on {} in {:.1} s.\nEmitter pixels: {} (ours {:.3}, Cycles mean {:.3}).\n\
         Floor pixels: {}; mean red ours {:.5}, Cycles {:.5} (ratio {:.4}); relative RMSE {:.2}%.\n\
         Pixels Cycles sees as emitter but we do not: {}.",
        reference.device,
        reference.seconds,
        emitter.len(),
        showcase.radiance,
        mean(theirs, &emitter),
        floor.len(),
        mean(&ours, &floor),
        mean(theirs, &floor),
        mean(theirs, &floor) / mean(&ours, &floor),
        100.0 * relative_rmse(&pick(&ours, &floor), &pick(theirs, &floor)),
        (0..ours.len())
            .filter(
                |&i| theirs[i][0] > showcase.radiance * 0.5 && ours[i][0] < showcase.radiance * 0.5
            )
            .count()
    );
    Ok(())
}
