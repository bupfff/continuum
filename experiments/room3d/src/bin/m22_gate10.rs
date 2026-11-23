//! M22 gate 10: coarse texels average their area, measured on what they
//! hold (basics item 1; gate 9's candidate, gate 9 parked unrun:
//! `docs/results/m22-gate9.md`).
//!
//! A texel stores its light at one point, its centre placed on the surface;
//! where it covers much of a curved surface that point stands for it badly
//! (each of the field's boulders is one chart: its top texel lies on its
//! side facing partly down, and a boulder shown at that mip is unlit, the
//! viewer's dark blocks, 2026-11-22). Candidate (`CacheOptions::
//! area_texels`): texels in the top three mips of their chart that curve (a
//! point of 16 over their area more than 30° from the centre's normal) take
//! each static light's ratio from hardware rays at those points (4 a point,
//! weighted by each point's unshadowed light) and their sky from the sky's
//! 192 rays spread over the same points.
//!
//! Gate 9 compared against Cycles at four bounces, where our one bounce
//! (which never settles on the field) leaves the image 3–12% short; this
//! gate compares what texels hold: direct light and the sky.
//!
//! Pass criteria, fixed before the run:
//! 1. **From above** (`altitude_dev`'s view: the scorecard's field 40 m
//!    higher, 15° down), bounce light off here and in Cycles: at most 0.85×
//!    the current's error from Cycles, and the mean's shortfall from Cycles'
//!    at most 0.8× the current's.
//! 2. **Elsewhere:** the scorecard's seven scenes against Cycles (as the
//!    scorecard measures, bounce on) each within 0.2 points of the current
//!    (Sponza 0.5), or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair`; run first): in
//!    every scene the settled frame's median at most 1.05× the current's,
//!    the moving frames' at most 1.10×, and every scene that settles now
//!    settling, within 1.25× its time.
//!
//! **Disclosed (before the run):** developed on the view from above and
//! the field's start view (`viewer_repro`). From above, without bounce:
//! current 13.83% (mean 3.8% short), candidate 10.42% (2.9% short); with
//! bounce against four-bounce Cycles 14.23% and 11.32%. Averaging the top
//! five mips instead of three: 10.82% with bounce, the mean unchanged. A
//! rehearsal with every coarse texel averaged (flat ones too) failed
//! criterion 2 on Sponza (25.53% → 27.55%: its walls' coarse texels, read
//! by bounce rays, averaged darker, the image 6% darker; without bounce
//! 0.9996×); the field passed (1.79% → 1.29%). So only texels that curve
//! are averaged (Sponza's mean then 1.0026× the current's; from above
//! unchanged, 10.42%).
//!
//! `M22_REHEARSE=1`: the view from above, Sponza and the field only, every
//! criterion measured the same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate10`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

/// The scene with the current defaults or the candidate.
fn configured(build: &Build, candidate: bool) -> Case {
    let mut case = build();
    case.area_texels = Some(candidate);
    case
}

fn main() -> std::io::Result<()> {
    let rehearse = std::env::var_os("M22_REHEARSE").is_some();
    let start = Instant::now();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let gate_out = root.join("m22_gate10");
    let card_out = root.join("scorecard");
    std::fs::create_dir_all(&gate_out)?;
    std::fs::create_dir_all(&card_out)?;
    let builds: Vec<_> = cases(&root.join("assets"))
        .into_iter()
        .filter(|(n, _)| !rehearse || *n == "sponza" || *n == "field")
        .collect();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if rehearse {
        say("REHEARSAL: the view from above, Sponza and the field only; not the gate's numbers".into());
    }
    let opt = |v: Option<(usize, f64)>| {
        v.map_or("not within 120 s".into(), |(f, t)| format!("{t:.2} s ({f})"))
    };
    // 3. Speed, side by side, first.
    say("| Scene | Settled frame: current, candidate (ratio) | Moving frame (ratio) | Settled at: current, candidate (ratio) |".into());
    say("|---|---|---|---|".into());
    let mut speed_ok = true;
    for (name, build) in &builds {
        let (a, b) = (configured(build, false), configured(build, true));
        let [cur, cand] = speed_pair(&a, &b, start, 120.0);
        let frame = cand.frame.0 / cur.frame.0;
        let moving = cand.motion.0 / cur.motion.0;
        speed_ok &= frame <= 1.05 && moving <= 1.10;
        let settled = match (cur.settled, cand.settled) {
            (Some((_, x)), Some((_, y))) => {
                speed_ok &= y <= 1.25 * x;
                format!("{:.2}×", y / x)
            }
            (Some(_), None) => {
                speed_ok = false;
                "never".into()
            }
            (None, _) => "n/a (the current never settles)".into(),
        };
        say(format!(
            "| {name} | {:.2}, {:.2} ms ({frame:.3}×) | {:.2}, {:.2} ms ({moving:.3}×) | {}, {} ({settled}) |",
            cur.frame.0,
            cand.frame.0,
            cur.motion.0,
            cand.motion.0,
            opt(cur.settled),
            opt(cand.settled)
        ));
    }
    // 1. From above, without bounce light.
    say(String::new());
    let above = |candidate: bool| {
        let mut case = field_from_above(40.0, 15.0, true);
        case.area_texels = Some(candidate);
        case
    };
    let q_cur = quality(&above(false), &card_out)?;
    for (from, to) in [("png", "above.current.png"), ("diff.png", "above.current.diff.png")] {
        let _ = std::fs::copy(card_out.join(format!("field_high_direct.{from}")), gate_out.join(to));
    }
    let q_cand = quality(&above(true), &card_out)?;
    for (from, to) in [
        ("png", "above.candidate.png"),
        ("diff.png", "above.candidate.diff.png"),
        ("cycles.png", "above.cycles.png"),
    ] {
        let _ = std::fs::copy(card_out.join(format!("field_high_direct.{from}")), gate_out.join(to));
    }
    let short = |q: &Quality| ((q.mean_cycles - q.mean_ours) / q.mean_cycles).max(0.0);
    let ratio = q_cand.rmse / q_cur.rmse;
    let short_ratio = if short(&q_cur) > 0.0 { short(&q_cand) / short(&q_cur) } else { 0.0 };
    let above_ok = ratio <= 0.85 && short_ratio <= 0.8;
    say("| From above, no bounce | Current | Candidate | Ratio | Bar |".into());
    say("|---|---|---|---|---|".into());
    say(format!(
        "| Against Cycles | {:.2}% | {:.2}% | {ratio:.3}× | at most 0.85× |",
        100.0 * q_cur.rmse,
        100.0 * q_cand.rmse
    ));
    say(format!(
        "| Mean short of Cycles' ({:.4}) | {:.2}% ({:.4}) | {:.2}% ({:.4}) | {short_ratio:.3}× | at most 0.8× |",
        q_cur.mean_cycles,
        100.0 * short(&q_cur),
        q_cur.mean_ours,
        100.0 * short(&q_cand),
        q_cand.mean_ours
    ));
    // 2. Elsewhere, against Cycles as the scorecard measures.
    say(String::new());
    say("| Scene | Current | Candidate | Bar |".into());
    say("|---|---|---|---|".into());
    let mut elsewhere_ok = true;
    for (name, build) in &builds {
        let cur = quality(&configured(build, false), &card_out)?;
        let cand = quality(&configured(build, true), &card_out)?;
        for (from, to) in [("png", "candidate.png"), ("diff.png", "candidate.diff.png")] {
            let _ = std::fs::copy(
                card_out.join(format!("{name}.{from}")),
                gate_out.join(format!("{name}.{to}")),
            );
        }
        let (a, b) = (100.0 * cur.rmse, 100.0 * cand.rmse);
        let slack = if *name == "sponza" { 0.5 } else { 0.2 };
        elsewhere_ok &= b <= a + slack;
        say(format!("| {name} | {a:.2}% | {b:.2}% | at most {:.2}% |", a + slack));
    }
    say(String::new());
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    say(format!("1. From above: {}", verdict(above_ok)));
    say(format!("2. Elsewhere: {}", verdict(elsewhere_ok)));
    say(format!("3. Speed: {}", verdict(speed_ok)));
    say(format!(
        "Gate: {} ({:.1} min)",
        verdict(above_ok && elsewhere_ok && speed_ok),
        start.elapsed().as_secs_f64() / 60.0
    ));
    std::fs::write(
        gate_out.join(if rehearse { "rehearsal.md" } else { "report.md" }),
        report,
    )?;
    Ok(())
}
