//! Simplified rock levels against the analytic ones (development, not a
//! gate; CPU only): stated errors, edge lengths, time.
//!
//! `cargo run --release -p room3d --bin simplify_dev -- [rocks=N]`

use std::time::Instant;

use continuum::mesh::Surface;
use continuum::simplify;
use room3d::rocks::{Detail, Rocks};

fn edges(s: &Surface) -> (f32, f32, f32) {
    let mut e: Vec<f32> = s
        .triangles
        .iter()
        .flat_map(|t| {
            [
                (t[1] - t[0]).length(),
                (t[2] - t[1]).length(),
                (t[0] - t[2]).length(),
            ]
        })
        .collect();
    e.sort_by(f32::total_cmp);
    let mean = e.iter().sum::<f32>() / e.len() as f32;
    (mean, e[e.len() * 95 / 100], *e.last().unwrap())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("rocks=").map(|v| v.parse().unwrap()))
        .unwrap_or(4);
    let rocks = Rocks::with_detail(
        2,
        6.0,
        Detail {
            finest_normals: true,
            ..Default::default()
        },
    );
    for g in 1..=count.min(rocks.positions.len()) {
        let l = rocks.scene.geometry_levels[g].as_ref().unwrap();
        let finest = &l.finer[2].surface;
        let start = Instant::now();
        let ours = simplify::levels(finest, &[16_384, 4_096, 1_024]);
        let ms = start.elapsed().as_secs_f64() * 1e3;
        println!("rock {g}: {ms:.0} ms; finest edges {:?}", edges(finest));
        let analytic = [
            (&l.finer[1].surface, l.finer[1].error),
            (&l.finer[0].surface, l.finer[0].error),
            (&rocks.scene.geometries[g], l.base_error),
        ];
        for (k, (o, (a, ae))) in ours.iter().zip(analytic).enumerate() {
            let dense =
                simplify::sampled_distance(&o.surface.triangles, &finest.triangles, 64, k as u64);
            println!(
                "  {} triangles: ours error {:.4} (distance {:.4}, dense {:.4}), edges mean/95%/max {:?}; analytic {} triangles error {:.4}, edges {:?}",
                o.surface.triangles.len(),
                o.error,
                o.distance,
                dense,
                edges(&o.surface),
                a.triangles.len(),
                ae,
                edges(a)
            );
        }
    }
}
