//! Flatland gate 4: the worst case, everything moving.
//!
//! Design rule 1 says the incremental path must never cost meaningfully more than
//! a conventional full recompute, even when nothing can be reused. Every
//! occluder drifts every frame here, so caches of static work are empty and most
//! tiles are charged every frame.
//!
//! Pass criterion, fixed before the first run: in every configuration, at ε = 0
//! and ε = 1%, zero soundness violations and incremental time at most 1.1× the
//! full recompute time (same thread count).

use std::fmt::Write as _;
use std::path::Path;

use flatland::anim;
use flatland::incremental::{self, Charging, Grid, Policy, RunConfig, Visibility};

const EPSILONS: [f32; 2] = [0.0, 0.01];

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut out = String::new();
    writeln!(out, "# Flatland gate 4 results\n").unwrap();
    writeln!(
        out,
        "Every occluder drifts every frame. {threads} threads for both paths.\n"
    )
    .unwrap();
    writeln!(out, "| Scenario | Visibility | Charging | ε | Violations | Cones recomputed | Incremental ms/frame | Full ms/frame | Incremental / full |").unwrap();
    writeln!(out, "|---|---|---|---|---|---|---|---|---|").unwrap();

    let configs = [
        (
            anim::everything_moves(),
            256,
            Visibility::Binary,
            Charging::Bounds,
        ),
        (
            anim::everything_moves(),
            256,
            Visibility::Continuous,
            Charging::Bounds,
        ),
        (
            anim::sky_everything_moves(),
            128,
            Visibility::Continuous,
            Charging::Bounds,
        ),
        (
            anim::sky_everything_moves(),
            128,
            Visibility::Continuous,
            Charging::Projected,
        ),
    ];
    let mut pass = true;
    for (s, size, visibility, charging) in configs {
        let cfg = RunConfig {
            grid: Grid { size },
            visibility,
            policy: Policy::TileBudget,
            charging,
            threads,
            epsilons: EPSILONS.to_vec(),
            sheets: Vec::new(),
            time_static_cache: false,
        };
        let r = incremental::run(&s, &cfg);
        let frames = r.frames_timed as f64;
        let full_ms = r.full_time.as_secs_f64() * 1000.0 / frames;
        for (i, ((state, stats), &e)) in r.runs.iter().zip(&EPSILONS).enumerate() {
            let inc_ms = state.time.as_secs_f64() * 1000.0 / frames;
            let ratio = inc_ms / full_ms;
            pass &= stats.violations == 0 && ratio <= 1.1;
            writeln!(
                out,
                "| {} | {} | {} | {:.0}% | {} | {:.1}% | {:.2} | {:.2} | {:.2} |",
                s.name,
                visibility.name(),
                charging.name(),
                e * 100.0,
                stats.violations,
                r.recomputed_fraction(i) * 100.0,
                inc_ms,
                full_ms,
                ratio
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "\n## Verdict\n\nIncremental ≤ 1.1× full everywhere with zero violations → {}.",
        if pass { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("gate4_results.md"), &out)?;
    Ok(())
}
