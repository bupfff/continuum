//! Renders the showcase at a few points along its camera path, from the live
//! renderer's caches (direct light, moving objects, one bounce). Not a gate.

use std::path::Path;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::{Lightmap, LiveRenderer, MaskConeKernel, wgpu};
use flatland::img::Image;
use room3d::showcase::{PATH_FRAMES, PROBES, Showcase, TEXEL};
use room3d::to_rgb;

const W: u32 = 960;
const H: u32 = 540;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let mut scene = Showcase::new();
    let meshes = scene.meshes();
    let triangles: usize = meshes.iter().map(Vec::len).sum();
    let start = Instant::now();
    let mut lightmap =
        Lightmap::new(&scene.surfaces, TEXEL, |m| m < scene.first_mover).with_probes(PROBES);
    let layout = start.elapsed();
    let bake = kernel.bake(&mut lightmap, scene.light, &meshes);
    let start = Instant::now();
    let mut renderer = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        scene.light,
        &meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
    );
    renderer.wait();
    println!(
        "{} meshes, {triangles} triangles; {} texels in {} blocks, {} probes. Layout {:.0} ms, bake {:.0} ms, caches {:.0} ms.",
        meshes.len(),
        lightmap.texels(),
        renderer.blocks(),
        renderer.probes(),
        layout.as_secs_f64() * 1000.0,
        bake.as_secs_f64() * 1000.0,
        start.elapsed().as_secs_f64() * 1000.0
    );
    let shots = [0.0, PATH_FRAMES as f32 * 0.5, PATH_FRAMES as f32];
    let gap = 8;
    let mut sheet = Image::new(
        W as usize + 2 * gap,
        shots.len() * (H as usize + gap) + gap,
        [28; 3],
    );
    let mut frame = 0.0;
    for (row, &shot) in shots.iter().enumerate() {
        // Advance the movers frame by frame up to the shot, as the viewer would.
        while frame < shot {
            frame += 1.0;
            scene.set_frame(frame);
            let changes: Vec<_> = (scene.first_mover..scene.surfaces.len())
                .map(|m| (m, &scene.surfaces[m]))
                .collect();
            renderer.update_surfaces(&changes);
        }
        renderer.wait();
        lightmap.irradiance = renderer.read_texels();
        lightmap.indirect = renderer.read_probes();
        let (pixels, _) = kernel.render_cached(
            scene.camera(shot),
            W,
            H,
            scene.light,
            &scene.meshes(),
            &scene.albedo,
            scene.radiance,
            &lightmap,
            1,
        );
        for (i, p) in pixels.iter().enumerate() {
            sheet.set(
                gap + i % W as usize,
                gap + row * (H as usize + gap) + i / W as usize,
                to_rgb(Vec3::from_array(*p)),
            );
        }
    }
    sheet.save(&out_dir.join("render_showcase.png"))?;
    println!("Wrote out/render_showcase.png: frames {shots:?} of the camera path.");
    Ok(())
}
