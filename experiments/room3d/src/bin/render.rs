//! A first picture: a camera view of the gate 1 room, lit directly by the area
//! light, with mask cones at every visible surface point. Beside it, the same
//! view with 16 jittered light samples per pixel, which costs about the same per
//! pixel (M1 gate 1), to show the difference in noise at equal cost.
//!
//! Not a gate. Direct light only; surfaces that see the light only partly above
//! their horizon are approximated (Lambert's formula assumes the light is above).

use std::path::Path;
use std::time::Instant;

use continuum::bvh::Bvh;
use continuum::cone3::ray_triangle;
use continuum::glam::Vec3;
use flatland::img::Image;
use flatland::incremental::par_map;
use flatland::rng::Pcg32;
use room3d::{render_scene, to_rgb};

const WIDTH: usize = 480;
const HEIGHT: usize = 360;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let scene = render_scene();
    let l = &scene.room.light;
    let bvh = Bvh::build(&scene.room.meshes);
    let hits = scene.primary_hits(WIDTH, HEIGHT, threads);

    let start = Instant::now();
    let masked = scene.masked(&bvh, &hits, threads);
    let mask_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let sampled: Vec<Vec3> = par_map(WIDTH * HEIGHT, threads, 256, |px| {
        let Some(h) = &hits[px] else {
            return Vec3::ZERO;
        };
        let Some(cone) = scene.light_cone(h) else {
            return Vec3::ZERO;
        };
        let candidates = bvh.candidates(&cone).0;
        let mut rng = Pcg32::new(px as u64, 3);
        let (k, light_normal) = (4, l.normal());
        let mut sum = 0.0;
        for c in 0..k * k {
            let (s, t) = (
                ((c % k) as f32 + rng.next_f32()) / k as f32,
                ((c / k) as f32 + rng.next_f32()) / k as f32,
            );
            let d = l.point(s, t) - h.point;
            let blocked = candidates
                .iter()
                .flat_map(|c| &c.1)
                .any(|tri| ray_triangle(h.point, d, tri).is_some_and(|x| x < 1.0));
            if !blocked {
                let r2 = d.length_squared();
                let r = r2.sqrt();
                sum += (h.normal.dot(d) / r).max(0.0) * (light_normal.dot(-d).abs() / r) / r2;
            }
        }
        scene.shade(h, sum * l.area() / (k * k) as f32)
    });
    let rays_ms = start.elapsed().as_secs_f64() * 1000.0;

    let gap = 8;
    let mut sheet = Image::new(2 * WIDTH + 3 * gap, HEIGHT + 2 * gap, [28; 3]);
    for (i, img) in [&masked, &sampled].iter().enumerate() {
        for px in 0..WIDTH * HEIGHT {
            sheet.set(
                gap + i * (WIDTH + gap) + px % WIDTH,
                gap + px / WIDTH,
                to_rgb(img[px]),
            );
        }
    }
    sheet.save(&out_dir.join("render.png"))?;
    println!(
        "Wrote out/render.png ({WIDTH}×{HEIGHT}). Left: mask cones, {mask_ms:.0} ms. Right: 16 jittered rays, {rays_ms:.0} ms. \
         {threads} threads, shared hierarchy culling included in both."
    );
    Ok(())
}
