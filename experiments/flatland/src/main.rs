//! Flatland gate 1: do mask cones beat rays on error per unit of cost?
//!
//! Renders the fluence from an area light over a 2D scene with several methods,
//! compares each against the analytic answer, and writes images plus a results
//! table to `out/`.

mod geom;
mod img;
mod methods;
mod rng;
mod scene;

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use geom::{Cone, Vec2, v2};
use rng::Pcg32;
use scene::Scene;

const SIZE: usize = 384;
/// Absolute fluence error (radians of visible light) shown at full intensity in error maps.
const ERROR_SCALE: f32 = 0.05;

struct Render {
    name: String,
    fluence: Vec<f32>,
    ns_per_pixel: f64,
}

fn pixel_center(idx: usize) -> Vec2 {
    let (i, j) = (idx % SIZE, idx / SIZE);
    v2(
        (i as f32 + 0.5) / SIZE as f32,
        1.0 - (j as f32 + 0.5) / SIZE as f32,
    )
}

/// Runs `visible` for every pixel, single-threaded so timings compare fairly.
fn render(name: &str, scene: &Scene, mut visible: impl FnMut(&Cone, usize) -> f32) -> Render {
    let start = Instant::now();
    let fluence = (0..SIZE * SIZE)
        .map(|idx| match Cone::toward(pixel_center(idx), scene.light) {
            Some(cone) => scene.radiance * cone.angle * visible(&cone, idx),
            None => 0.0,
        })
        .collect();
    let ns_per_pixel = start.elapsed().as_nanos() as f64 / (SIZE * SIZE) as f64;
    Render {
        name: name.to_string(),
        fluence,
        ns_per_pixel,
    }
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

/// Root-mean-square difference, relative to the reference's mean fluence.
fn rel_rmse(a: &[f32], reference: &[f32]) -> f64 {
    let mse = a
        .iter()
        .zip(reference)
        .map(|(&x, &r)| ((x - r) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    mse.sqrt() / mean(reference)
}

fn max_abs(a: &[f32], reference: &[f32]) -> f32 {
    a.iter()
        .zip(reference)
        .map(|(&x, &r)| (x - r).abs())
        .fold(0.0, f32::max)
}

/// Mean signed error and RMS error inside a box of the unit square, both relative
/// to the reference's mean there.
fn region_error(a: &[f32], reference: &[f32], min: Vec2, max: Vec2) -> (f64, f64) {
    let inside = |idx: &usize| {
        let p = pixel_center(*idx);
        p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y
    };
    let (mut err, mut sq, mut sum, mut n) = (0.0, 0.0, 0.0, 0.0);
    for idx in (0..a.len()).filter(inside) {
        let d = (a[idx] - reference[idx]) as f64;
        err += d;
        sq += d * d;
        sum += reference[idx] as f64;
        n += 1.0;
    }
    let mean_ref = sum / n;
    (err / n / mean_ref, (sq / n).sqrt() / mean_ref)
}

fn main() -> std::io::Result<()> {
    let scene = scene::gate1();
    let occ = &scene.occluders;
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out)?;

    let setup = render("Cone setup only (overhead in every row)", &scene, |_, _| {
        1.0
    });
    let mut scratch = Vec::new();
    let exact = render("Exact (2D-only reference)", &scene, |c, _| {
        methods::exact(c, occ, &mut scratch)
    });
    let mask = render("Mask cone, 64 bins", &scene, |c, _| methods::mask(c, occ));
    let fixed64 = render("64 rays, fixed (one per bin)", &scene, |c, _| {
        methods::rays(c, occ, 64, None)
    });
    let transmittance = render("Transmittance cone, per occluder", &scene, |c, _| {
        methods::transmittance(c, occ)
    });
    let layered = render(
        "Transmittance cone, per layer (voxel-cone-tracing style)",
        &scene,
        |c, _| methods::layered_transmittance(c, occ, &scene.layers, &mut scratch),
    );

    let jittered = |n: u32, frame: u64| {
        render(&format!("{n} rays, jittered"), &scene, move |c, idx| {
            methods::rays(c, occ, n, Some(&mut Pcg32::new(idx as u64, frame)))
        })
    };
    let ray_counts = [4, 16, 64, 256];
    let jittered_runs: Vec<(Render, Render)> = ray_counts
        .iter()
        .map(|&n| (jittered(n, 0), jittered(n, 1)))
        .collect();

    // Results table.
    let ref_f = &exact.fluence;
    let cost_64 = jittered_runs[2].0.ns_per_pixel;
    let mut table = String::new();
    writeln!(
        table,
        "| Method | Rel. RMSE | Max abs error | Flicker | ns/pixel | Cost vs 64 jittered rays |"
    )
    .unwrap();
    writeln!(table, "|---|---|---|---|---|---|").unwrap();
    writeln!(
        table,
        "| {} | — | — | — | {:.0} | {:.2}× |",
        setup.name,
        setup.ns_per_pixel,
        setup.ns_per_pixel / cost_64
    )
    .unwrap();
    let mut row = |r: &Render, flicker: Option<f64>| {
        let flicker = flicker.map_or("0 (deterministic)".to_string(), |f| {
            format!("{:.2}%", f * 100.0)
        });
        writeln!(
            table,
            "| {} | {:.2}% | {:.4} | {} | {:.0} | {:.2}× |",
            r.name,
            rel_rmse(&r.fluence, ref_f) * 100.0,
            max_abs(&r.fluence, ref_f),
            flicker,
            r.ns_per_pixel,
            r.ns_per_pixel / cost_64
        )
        .unwrap();
    };
    row(&exact, None);
    row(&mask, None);
    row(&fixed64, None);
    for (a, b) in &jittered_runs {
        row(a, Some(rel_rmse(&a.fluence, &b.fluence)));
    }
    row(&transmittance, None);
    row(&layered, None);

    let mismatched = mask
        .fluence
        .iter()
        .zip(&fixed64.fluence)
        .filter(|(a, b)| (*a - *b).abs() > 1e-5)
        .count();
    let aligned = (v2(0.10, 0.45), v2(0.45, 0.62));
    let offset = (v2(0.55, 0.45), v2(0.90, 0.62));
    let mut report = String::new();
    writeln!(report, "# Flatland gate 1 results\n").unwrap();
    writeln!(
        report,
        "{SIZE}×{SIZE} pixels, {} occluders, single-threaded release build.\n",
        occ.len()
    )
    .unwrap();
    report.push_str(&table);
    writeln!(
        report,
        "\nFlicker: relative RMS difference between two frames with different random seeds.\n"
    )
    .unwrap();
    writeln!(
        report,
        "Mask cone vs 64 fixed rays: {mismatched} of {} pixels differ (expected: only float ties at bin edges).\n",
        SIZE * SIZE
    )
    .unwrap();
    writeln!(
        report,
        "## Correlation error below each fence (relative to the reference's mean there)\n"
    )
    .unwrap();
    writeln!(report, "| Method | Aligned slats: bias | Aligned slats: RMSE | Offset slats: bias | Offset slats: RMSE |").unwrap();
    writeln!(report, "|---|---|---|---|---|").unwrap();
    for r in [&mask, &transmittance, &layered] {
        let (ab, ar) = region_error(&r.fluence, ref_f, aligned.0, aligned.1);
        let (ob, or) = region_error(&r.fluence, ref_f, offset.0, offset.1);
        writeln!(
            report,
            "| {} | {:+.1}% | {:.1}% | {:+.1}% | {:.1}% |",
            r.name,
            ab * 100.0,
            ar * 100.0,
            ob * 100.0,
            or * 100.0
        )
        .unwrap();
    }
    print!("{report}");
    std::fs::write(out.join("results.md"), &report)?;

    // Images: top row renders, bottom row signed error maps.
    let shaded = |r: &Render| {
        let mut im = img::tonemap(&r.fluence, SIZE);
        img::draw_scene(&mut im, &scene);
        im
    };
    let error = |r: &Render| {
        let mut im = img::signed_error(&r.fluence, ref_f, SIZE, ERROR_SCALE);
        img::draw_scene(&mut im, &scene);
        im
    };
    let rays16 = &jittered_runs[1].0;
    let top = [
        shaded(&exact),
        shaded(&mask),
        shaded(rays16),
        shaded(&layered),
    ];
    let mut diagram = img::Image::new(SIZE, SIZE, [0; 3]);
    img::draw_scene(&mut diagram, &scene);
    let bottom = [diagram, error(&mask), error(rays16), error(&layered)];

    let gap = 8;
    let mut sheet = img::Image::new(4 * SIZE + 5 * gap, 2 * SIZE + 3 * gap, [28; 3]);
    for (i, (t, b)) in top.iter().zip(&bottom).enumerate() {
        let x = gap + i * (SIZE + gap);
        sheet.blit(t, x, gap);
        sheet.blit(b, x, 2 * gap + SIZE);
    }
    sheet.save(&out.join("sheet.png"))?;
    for (name, im) in [
        "exact",
        "mask64",
        "rays16_jittered",
        "layered_transmittance",
    ]
    .iter()
    .zip(&top)
    {
        im.save(&out.join(format!("{name}.png")))?;
    }
    println!("\nWrote images to {}", out.display());
    Ok(())
}
