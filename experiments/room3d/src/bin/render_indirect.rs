//! Direct light only beside direct plus one bounce of indirect light, from the
//! live renderer's caches (lightmap texels and probe G-buffers). Not a gate.

use std::path::Path;

use continuum::glam::Vec3;
use continuum_gpu::{Camera, Lightmap, LiveRenderer, MaskConeKernel, RectLight, wgpu};
use flatland::img::Image;
use room3d::{render_scene, to_rgb};

const W: u32 = 960;
const H: u32 = 540;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let Some(kernel) = MaskConeKernel::new() else {
        println!("No GPU adapter available.");
        return Ok(());
    };
    let mut scene = render_scene();
    let first = scene.add_movers(0.0);
    let l = &scene.room.light;
    let light = RectLight {
        corner: l.corner,
        edge_u: l.edge_u,
        edge_v: l.edge_v,
        one_sided: false,
    };
    let camera = Camera {
        eye: scene.eye,
        target: scene.target,
        vertical_fov_degrees: scene.vertical_fov_degrees,
    };
    let mut lightmap =
        Lightmap::from_meshes(&scene.room.meshes, 0.005, |m| m < first).with_probes(0.04);
    kernel.bake(&mut lightmap, light, &scene.room.meshes);
    let renderer = LiveRenderer::new(
        &kernel,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        W,
        H,
        light,
        &scene.room.meshes,
        &scene.albedo,
        scene.radiance,
        &lightmap,
    );
    let probes = renderer.read_probes();
    let valid = probes.iter().filter(|p| p[3] > 0.0).count();
    let gap = 8;
    let mut sheet = Image::new(2 * W as usize + 3 * gap, H as usize + 2 * gap, [28; 3]);
    for (i, indirect) in [Vec::new(), probes].into_iter().enumerate() {
        lightmap.indirect = indirect;
        let (pixels, _) = kernel.render_cached(
            camera,
            W,
            H,
            light,
            &scene.room.meshes,
            &scene.albedo,
            scene.radiance,
            &lightmap,
            1,
        );
        for (px, p) in pixels.iter().enumerate() {
            sheet.set(
                gap + i * (W as usize + gap) + px % W as usize,
                gap + px / W as usize,
                to_rgb(Vec3::from_array(*p)),
            );
        }
    }
    sheet.save(&out_dir.join("render_indirect.png"))?;
    println!(
        "Wrote out/render_indirect.png ({W}×{H}): direct only (left), direct plus one bounce (right). \
         {} probes, {valid} valid.",
        renderer.probes()
    );
    Ok(())
}
