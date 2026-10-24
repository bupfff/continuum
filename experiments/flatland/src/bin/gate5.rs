//! Flatland gate 5: does magnitude deferral pay off for wide cones once
//! visibility is continuous?
//!
//! Receivers gather a sky through 8 cones of 45° each, so every moving object is
//! a sliver of a wide cone. ε = 0 measures what dependency regions alone save
//! (only exact invalidation); the gate asks how much more the ledger saves by
//! deferring changes too small to matter. Binary and continuous masks, and two
//! ways of charging ledgers, run side by side under the same policy.
//!
//! Pass criterion, fixed before the first run: with continuous masks at ε = 1%,
//! zero soundness violations and at most half the cone evaluations of ε = 0.
//! Attempt 1 (geometric bounds) failed it at 59.5%; attempt 2 adds projected
//! charges and is judged by the same criterion.

use std::fmt::Write as _;
use std::path::Path;

use flatland::anim;
use flatland::incremental::{
    self, Charging, Grid, Policy, Precomputed, RunConfig, TILE, Visibility,
};
use flatland::methods;

const SIZE: usize = 128;
/// ε values, as fractions of mean fluence at frame 0. 16% is a stress value that
/// forces heavy deferral to test the bounds; it is not usable for images.
const EPSILONS: [f32; 4] = [0.0, 0.01, 0.04, 0.16];
const STRESS_FROM: usize = 3;
/// Index into `EPSILONS` for the verdict and image sheets.
const HEADLINE: usize = 1;

/// Root-mean-square difference relative to the reference's mean, and max abs difference.
fn compare(a: &[f32], reference: &[f32]) -> (f64, f32) {
    let mean = reference.iter().map(|&r| r as f64).sum::<f64>() / reference.len() as f64;
    let mse = a
        .iter()
        .zip(reference)
        .map(|(&x, &r)| ((x - r) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    let max = a
        .iter()
        .zip(reference)
        .map(|(&x, &r)| (x - r).abs())
        .fold(0.0, f32::max);
    (mse.sqrt() / mean, max)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let s = anim::sky();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let grid = Grid { size: SIZE };

    let mut out = String::new();
    writeln!(out, "# Flatland gate 5 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} receivers, {TILE}×{TILE}-receiver tiles, 8 sky cones of 45° per receiver, \
         {} frames, release build on {threads} threads (reference and incremental alike). \
         Policy: per-tile budget (recompute the largest weighted balances until the rest fit in ε).\n",
        s.frames
    )
    .unwrap();

    // Accuracy of each estimator against the exact answer at frame 0.
    let pre = Precomputed::new(&s, grid, Visibility::Continuous, threads);
    let mut occluders = s.statics.clone();
    occluders.extend((s.movers)(0));
    let weights = s.weights(0);
    let exact = pre.compose(
        &pre.evaluate(|c| methods::exact(c, &occluders, &mut Vec::new())),
        &weights,
    );
    writeln!(out, "## Accuracy at frame 0 (vs exact)\n").unwrap();
    writeln!(out, "| Visibility | Rel. RMSE | Max abs error |").unwrap();
    writeln!(out, "|---|---|---|").unwrap();
    for vis in [Visibility::Binary, Visibility::Continuous] {
        let estimate = match vis {
            Visibility::Binary => pre.evaluate(|c| methods::mask(c, &occluders)),
            Visibility::Continuous => pre.evaluate(|c| methods::coverage(c, &occluders)),
        };
        let (rmse, max) = compare(&pre.compose(&estimate, &weights), &exact);
        writeln!(
            out,
            "| {} | {:.3}% | {:.5} |",
            vis.name(),
            rmse * 100.0,
            max
        )
        .unwrap();
    }
    writeln!(out).unwrap();

    let mut violations = 0;
    let mut work_ratio = Vec::new();
    let configs = [
        (Visibility::Binary, Charging::Bounds),
        (Visibility::Continuous, Charging::Bounds),
        (Visibility::Binary, Charging::Projected),
        (Visibility::Continuous, Charging::Projected),
    ];
    for (vis, charging) in configs {
        let cfg = RunConfig {
            grid,
            visibility: vis,
            policy: Policy::TileBudget,
            charging,
            threads,
            epsilons: EPSILONS.to_vec(),
            sheets: vec![HEADLINE],
            time_static_cache: true,
        };
        let r = incremental::run(&s, &cfg);
        let title = format!("{}, {}, {}", s.name, vis.name(), charging.name());
        incremental::report(&mut out, &title, &s, &cfg, &r, STRESS_FROM);
        violations += r.runs.iter().map(|(_, st)| st.violations).sum::<u64>();
        // Work at ε = 1% relative to ε = 0 (where only dependency regions help).
        work_ratio.push(r.recomputed_fraction(HEADLINE) / r.recomputed_fraction(0));
        let slug = |n: &str| n.split(' ').next().unwrap().to_string();
        let name = format!("gate5_{}_{}.png", slug(vis.name()), slug(charging.name()));
        incremental::save_sheet(&r.sheets[0], &out_dir.join(name))?;
    }

    let verdict = |ratio: f64| {
        if violations == 0 && ratio <= 0.5 {
            "PASS"
        } else {
            "FAIL"
        }
    };
    writeln!(out, "## Verdict\n").unwrap();
    writeln!(
        out,
        "Soundness: {violations} violations across every configuration, ε and frame.\n"
    )
    .unwrap();
    writeln!(
        out,
        "Work at ε = 1% as a fraction of ε = 0 (target ≤ 50% with continuous masks):\n"
    )
    .unwrap();
    for ((vis, charging), ratio) in configs.iter().zip(&work_ratio) {
        writeln!(
            out,
            "- {}, {}: {:.1}%",
            vis.name(),
            charging.name(),
            ratio * 100.0
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nAttempt 1 (continuous, geometric bounds) → {}. Attempt 2 (continuous, projected charges) → {}.",
        verdict(work_ratio[1]),
        verdict(work_ratio[3])
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("gate5_results.md"), &out)?;
    Ok(())
}
