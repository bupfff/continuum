//! How much brighter a coarse rock level shades than the finest it stands
//! for (development, not a gate): for each camera level, the length of the
//! average of the finest surface's unit normals over each of the level's
//! grid cells. Lambert shading is linear in the normal, so the cell's
//! average fine shading is shading by that average; a unit normal shades
//! brighter by about 1 / length (where nothing clamps).
//!
//! `cargo run --release -p room3d --bin normals_dev`

use continuum::glam::Vec3;
use flatland::rng::Pcg32;
use room3d::rocks::{Detail, Shape};

fn main() {
    let mut rng = Pcg32::new(14, 1);
    let shapes: Vec<Shape> = (0..64)
        .map(|_| Shape::random(&mut rng, Detail::default()))
        .collect();
    // The finest surface's normal at (u, v), by central differences.
    let normal = |s: &Shape, u: f32, v: f32| {
        let h = 1e-4;
        let du = s.point(u + h, v, 4) - s.point(u - h, v, 4);
        let dv = s.point(u, v + h, 4) - s.point(u, v - h, 4);
        let n = du.cross(dv);
        if n.length() > 0.0 {
            n.normalize()
        } else {
            Vec3::ZERO
        }
    };
    // The analytic finest normal against central differences.
    let mut worst = 0.0f32;
    for s in shapes.iter().take(8) {
        for i in 0..64 {
            for j in 1..32 {
                let uv = continuum::glam::Vec2::new(i as f32 / 64.0 + 0.003, j as f32 / 32.0);
                if let Some(a) = s.finest_normal(uv) {
                    let b = normal(s, uv.x, uv.y);
                    worst = worst.max(a.angle_between(b));
                }
            }
        }
    }
    println!(
        "analytic against finite-difference finest normals: largest angle {:.4}°",
        worst.to_degrees()
    );
    // What averaging costs a level's production (M14 gate 12).
    for averaged in [false, true] {
        let mut rng = Pcg32::new(14, 1);
        let shapes: Vec<Shape> = (0..16)
            .map(|_| {
                Shape::random(
                    &mut rng,
                    Detail {
                        finest_normals: averaged,
                        ..Default::default()
                    },
                )
            })
            .collect();
        let mut row = format!("averaged {averaged}: per rock");
        for level in 0..4usize {
            let start = std::time::Instant::now();
            for s in &shapes {
                std::hint::black_box(s.surface(level));
            }
            row += &format!(
                " level {level} {:.2} ms;",
                start.elapsed().as_secs_f64() * 1e3 / shapes.len() as f64
            );
        }
        println!("{row}");
    }
    for level in 0..4usize {
        // Level `level` (the camera's level `level - 1`; 0 is the lighting
        // level): a grid of 32·2^level by 16·2^level cells.
        let (nu, nv) = (32usize << level, 16usize << level);
        let (mut sum, mut worst, mut cells) = (0.0f64, 1.0f32, 0usize);
        for s in shapes.iter().take(16) {
            for i in (0..nu).step_by((nu / 32).max(1)) {
                for j in 1..nv - 1 {
                    let mut avg = Vec3::ZERO;
                    let k = 8;
                    for a in 0..k {
                        for b in 0..k {
                            let u = (i as f32 + (a as f32 + 0.5) / k as f32) / nu as f32;
                            let v = (j as f32 + (b as f32 + 0.5) / k as f32) / nv as f32;
                            avg += normal(s, u, v);
                        }
                    }
                    let len = (avg / (k * k) as f32).length();
                    sum += len as f64;
                    worst = worst.min(len);
                    cells += 1;
                }
            }
        }
        println!(
            "level {level} ({nu}×{nv} cells): average normal length {:.4} (brighter by {:.2}%), shortest {:.3}",
            sum / cells as f64,
            100.0 * (cells as f64 / sum - 1.0),
            worst
        );
    }
}
