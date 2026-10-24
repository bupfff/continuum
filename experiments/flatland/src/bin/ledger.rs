//! Flatland gates 2 and 3: is the error ledger sound, and how much work does it skip?
//!
//! Animates scenarios where occluders move and lights switch on and flicker.
//! Each frame, an incremental renderer reuses cached visibility unless a tile's
//! ledger says it could be more than ε off, while a full recompute runs
//! alongside as the reference (see `flatland::incremental`).
//!
//! - Gate 2 (soundness): the certified bound is never below the actual error.
//! - Gate 3 (the edge): the fraction of cone evaluations the ledger skips.
//!
//! Caches hold visible angle per light, so light intensity changes are exact and
//! free (lighting is linear in emitted power); only geometry motion is charged.

use std::fmt::Write as _;
use std::path::Path;

use flatland::anim;
use flatland::incremental::{self, Charging, Grid, Policy, RunConfig, TILE, Visibility};

const SIZE: usize = 256;
/// ε values, as fractions of the scenario's mean fluence at frame 0. Values from
/// `STRESS_FROM` on are far too large to use; they exist to force deferral so the
/// magnitude bounds, not just the dependency regions, get tested.
const EPSILONS: [f32; 5] = [0.0, 0.01, 0.04, 0.16, 0.64];
const STRESS_FROM: usize = 3;
/// Index into `EPSILONS` used for the gate 3 verdict and the main image sheet.
const HEADLINE: usize = 1;

fn main() -> std::io::Result<()> {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out_dir)?;
    let cfg = RunConfig {
        grid: Grid { size: SIZE },
        visibility: Visibility::Binary,
        policy: Policy::EqualShare,
        charging: Charging::Bounds,
        threads: 1,
        epsilons: EPSILONS.to_vec(),
        sheets: vec![HEADLINE, STRESS_FROM],
        time_static_cache: true,
    };
    let mut out = String::new();
    writeln!(out, "# Flatland gates 2 and 3 results\n").unwrap();
    writeln!(
        out,
        "{SIZE}×{SIZE} receivers, {TILE}×{TILE}-receiver tiles, 2 lights, 240 frames, \
         single-threaded release build. Speedups count charging and recomputing; \
         the soundness check is not timed.\n"
    )
    .unwrap();

    let mut gate2_violations = 0;
    let mut typical_skip = 0.0;
    for s in [anim::typical(), anim::everything_moves()] {
        let r = incremental::run(&s, &cfg);
        incremental::report(&mut out, s.name, &s, &cfg, &r, STRESS_FROM);
        gate2_violations += r.runs.iter().map(|(_, st)| st.violations).sum::<u64>();
        if s.slug == "typical" {
            typical_skip = 1.0 - r.recomputed_fraction(HEADLINE);
            let switch = s.phases[1].1.start as usize;
            writeln!(
                out,
                "Light switch (frame {switch}) at ε = {:.0}%: max error {:.5} vs ε {:.5}.\n",
                EPSILONS[HEADLINE] * 100.0,
                r.runs[HEADLINE].1.max_error_by_frame[switch],
                r.epsilons[HEADLINE]
            )
            .unwrap();
        }
        for (rows, suffix) in r.sheets.iter().zip(["", "_stress"]) {
            incremental::save_sheet(
                rows,
                &out_dir.join(format!("ledger_{}{suffix}.png", s.slug)),
            )?;
        }
    }

    writeln!(out, "## Verdicts\n").unwrap();
    writeln!(
        out,
        "- Gate 2 (soundness): {} violations across every scenario, ε and frame → {}.",
        gate2_violations,
        if gate2_violations == 0 {
            "PASS"
        } else {
            "FAIL"
        }
    )
    .unwrap();
    writeln!(
        out,
        "- Gate 3 (edge): typical scenario at ε = {:.0}% skips {:.1}% of cone evaluations (target ≥ 80%) → {}.",
        EPSILONS[HEADLINE] * 100.0,
        typical_skip * 100.0,
        if typical_skip >= 0.8 { "PASS" } else { "FAIL" }
    )
    .unwrap();
    print!("{out}");
    std::fs::write(out_dir.join("ledger_results.md"), &out)?;
    Ok(())
}
