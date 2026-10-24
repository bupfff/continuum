//! A first picture: a camera view of the gate 1 room, lit directly by the area
//! light, with mask cones at every visible surface point. Beside it, the same
//! view with 16 jittered light samples per pixel, which costs about the same per
//! pixel (M1 gate 1), to show the difference in noise at equal cost.
//!
//! Not a gate. Direct light only; surfaces that see the light only partly above
//! their horizon are approximated (Lambert's formula assumes the light is above).

use std::path::Path;
use std::time::Instant;

use continuum::CoverageMask;
use continuum::bvh::Bvh;
use continuum::cone3::{PyramidCone, box_mesh, ray_triangle};
use continuum::glam::Vec3;
use flatland::img::Image;
use flatland::incremental::par_map;
use flatland::rng::Pcg32;
use room3d::{Tri, gate1};

const WIDTH: usize = 480;
const HEIGHT: usize = 360;
const RADIANCE: f32 = 9.0;

struct Hit {
    point: Vec3,
    normal: Vec3,
    albedo: Vec3,
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut room = gate1();
    let props = room.meshes.len();
    // Floor and two walls for context (they never sit between a surface and the light).
    room.meshes.push(box_mesh(
        Vec3::new(-0.5, -0.5, -0.02),
        Vec3::new(1.5, 1.5, 0.0),
    ));
    room.meshes.push(box_mesh(
        Vec3::new(-0.5, 1.0, 0.0),
        Vec3::new(1.5, 1.02, 1.2),
    ));
    room.meshes.push(box_mesh(
        Vec3::new(-0.02, -0.5, 0.0),
        Vec3::new(0.0, 1.5, 1.2),
    ));
    let albedo = |mesh: usize| match mesh {
        m if m < 32 => Vec3::new(0.55, 0.42, 0.30), // slats: wood
        32 => Vec3::new(0.62, 0.48, 0.34),          // table top
        33 => Vec3::new(0.25, 0.42, 0.70),          // cube
        m if m == props => Vec3::new(0.72, 0.72, 0.70), // floor
        _ => Vec3::new(0.78, 0.74, 0.68),           // walls
    };
    let bvh = Bvh::build(&room.meshes);
    let tris: Vec<(usize, Tri)> = room
        .meshes
        .iter()
        .enumerate()
        .flat_map(|(m, mesh)| mesh.iter().map(move |&t| (m, t)))
        .collect();
    let l = &room.light;

    let eye = Vec3::new(1.05, -0.55, 0.95);
    let forward = (Vec3::new(0.45, 0.55, 0.18) - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let up = right.cross(forward);
    let half = (50f32.to_radians() * 0.5).tan();
    let primary = |px: usize| -> Option<Hit> {
        let (x, y) = (px % WIDTH, px / WIDTH);
        let sx =
            ((x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0) * half * WIDTH as f32 / HEIGHT as f32;
        let sy = (1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0) * half;
        let dir = (forward + right * sx + up * sy).normalize();
        let (mut best, mut hit) = (f32::MAX, None);
        for (m, t) in &tris {
            if let Some(d) = ray_triangle(eye, dir, t)
                && d < best
            {
                best = d;
                hit = Some((*m, *t));
            }
        }
        let (m, t) = hit?;
        let mut normal = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
        if normal.dot(dir) > 0.0 {
            normal = -normal;
        }
        Some(Hit {
            point: eye + dir * best + normal * 1e-4,
            normal,
            albedo: albedo(m),
        })
    };
    let hits: Vec<Option<Hit>> = par_map(WIDTH * HEIGHT, threads, 256, primary);

    let lit = |h: &Hit| -> Option<PyramidCone> {
        let to_light = l.corner + (l.edge_u + l.edge_v) * 0.5 - h.point;
        if h.normal.dot(to_light) <= 0.0 {
            return None;
        }
        PyramidCone::toward_rect_light(h.point, l.corner, l.edge_u, l.edge_v, h.normal)
    };
    let shade =
        |h: &Hit, irradiance: f32| h.albedo * (RADIANCE * irradiance / std::f32::consts::PI);

    let start = Instant::now();
    let masked: Vec<Vec3> = par_map(WIDTH * HEIGHT, threads, 256, |px| {
        let Some(h) = &hits[px] else {
            return Vec3::ZERO;
        };
        let Some(cone) = lit(h) else {
            return Vec3::ZERO;
        };
        let mut m = CoverageMask::EMPTY;
        for (_, tris) in bvh.candidates(&cone).0 {
            cone.add_mesh(&mut m, &tris);
        }
        shade(h, cone.visible(&m))
    });
    let mask_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let sampled: Vec<Vec3> = par_map(WIDTH * HEIGHT, threads, 256, |px| {
        let Some(h) = &hits[px] else {
            return Vec3::ZERO;
        };
        let Some(cone) = lit(h) else {
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
        shade(h, sum * l.area() / (k * k) as f32)
    });
    let rays_ms = start.elapsed().as_secs_f64() * 1000.0;

    let to_rgb = |c: Vec3| {
        c.to_array()
            .map(|v| ((v / (v + 0.6)).powf(1.0 / 2.2) * 255.0).round() as u8)
    };
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
