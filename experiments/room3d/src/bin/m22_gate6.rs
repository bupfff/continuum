//! M22 gate 6: bounce light past the first bounce by default (item 5,
//! "bounce by default and quicker to settle"; the basics scorecard's worst
//! row, DESIGN §3.8).
//!
//! The cache's one bounce leaves a closed room 41% dark (46.55% from Cycles
//! at 12 bounces); M10 gate 25's exact bounces fix it (4 bounces: 7.74%) but
//! their cascade never lets Sponza settle. The candidate: **two exact
//! bounces** (`CacheOptions::bounces` 2) **with the orders past them
//! estimated** (`CacheOptions::bounce_rest`): the pages a bounce sample's
//! rays hit hold one-bounce samples of their own, each scaled by 1 / (1 −
//! ρ), ρ its closure-weighted albedo (progressive radiosity's ambient term,
//! per sample; Cohen et al. 1988). Hits whose samples hold a bounce ask for
//! no more where bounce light is under `bounce_tail` of their light.
//!
//! Measured exactly as the basics scorecard measures (`room3d::scorecard`):
//! its seven scenes, each with the defaults (one bounce) and the candidate.
//!
//! Pass criteria, fixed before the run:
//! 1. **Speed** (run first; per scene, one bounce then the candidate): the
//!    settled frame's median at most 1.10× one bounce's, and the moving
//!    frames' median at most 1.15×, in every scene.
//! 2. **Settling:** every scene that settles with one bounce within 120 s
//!    settles with the candidate too, within 2× its time.
//! 3. **Indoors:** the room and the movers each at most 10% from Cycles
//!    (clamped relative RMSE; one bounce: 46.55%, 44.46%).
//! 4. **Elsewhere:** the field, the helmet, Fox and BrainStem within 0.2
//!    points of one bounce's error, Sponza within 2.0 points.
//!
//! **Disclosed (before the run):** developed with the scorecard on five of
//! the scenes (not Fox and BrainStem), the candidate against one bounce:
//! room 7.68% (46.55%), movers 5.99% (44.46%), field 1.33% (1.68%), helmet
//! 4.46% (4.46%), Sponza 31.32% (29.94%); Sponza's settled frame 22.2 ms
//! (22.4), settled at 56.5 s (35.0 s; 33–50 s across runs), the room's 2.2
//! s (1.7). Alternatives tried: 4 exact bounces (room 7.74%, Sponza never
//! settled, 34–40 ms frames), the estimate on one bounce alone (room 20.31%,
//! Sponza 33.47%).
//!
//! `M22_REHEARSE=1`: the room and Sponza only, every criterion measured the
//! same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate6`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

/// The scene with one bounce (the defaults before this gate) or the
/// candidate.
fn configured(build: &Build, candidate: bool) -> Case {
    let mut case = build();
    case.bounces = Some(if candidate { 2 } else { 1 });
    case.bounce_rest = Some(candidate);
    case
}

fn main() -> std::io::Result<()> {
    let rehearse = std::env::var_os("M22_REHEARSE").is_some();
    let start = Instant::now();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let gate_out = root.join("m22_gate6");
    let card_out = root.join("scorecard");
    std::fs::create_dir_all(&gate_out)?;
    std::fs::create_dir_all(&card_out)?;
    let builds: Vec<_> = cases(&root.join("assets"))
        .into_iter()
        .filter(|(n, _)| !rehearse || *n == "room" || *n == "sponza")
        .collect();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    if rehearse {
        say("REHEARSAL: the room and Sponza only; not the gate's numbers".into());
    }
    let opt = |v: Option<(usize, f64)>| {
        v.map_or("not within 120 s".into(), |(f, t)| {
            format!("{t:.2} s ({f})")
        })
    };
    // 1–2. Speed and settling, first.
    say("| Scene | Settled frame: one, two+rest (ratio) | Moving frame (ratio) | Settled at: one, two+rest (ratio) |".into());
    say("|---|---|---|---|".into());
    let (mut speed_ok, mut settle_ok) = (true, true);
    for (name, build) in &builds {
        let one = speed_capped(&configured(build, false), start, 120.0);
        let two = speed_capped(&configured(build, true), start, 120.0);
        let frame = two.frame.0 / one.frame.0;
        let moving = two.motion.0 / one.motion.0;
        speed_ok &= frame <= 1.10 && moving <= 1.15;
        let settled = match (one.settled, two.settled) {
            (Some((_, a)), Some((_, b))) => {
                settle_ok &= b <= 2.0 * a;
                format!("{:.2}×", b / a)
            }
            (Some(_), None) => {
                settle_ok = false;
                "never".into()
            }
            (None, _) => "n/a (one bounce never settles)".into(),
        };
        say(format!(
            "| {name} | {:.2}, {:.2} ms ({frame:.3}×) | {:.2}, {:.2} ms ({moving:.3}×) | {}, {} ({settled}) |",
            one.frame.0,
            two.frame.0,
            one.motion.0,
            two.motion.0,
            opt(one.settled),
            opt(two.settled)
        ));
    }
    // 3–4. Against Cycles.
    say(String::new());
    say("| Scene | One bounce | Two + rest | Bar |".into());
    say("|---|---|---|---|".into());
    let (mut indoor_ok, mut elsewhere_ok) = (true, true);
    for (name, build) in &builds {
        let one = quality(&configured(build, false), &card_out)?;
        for (from, to) in [("png", "one.png"), ("diff.png", "one.diff.png")] {
            let _ = std::fs::copy(
                card_out.join(format!("{name}.{from}")),
                gate_out.join(format!("{name}.{to}")),
            );
        }
        let two = quality(&configured(build, true), &card_out)?;
        for (from, to) in [
            ("png", "two.png"),
            ("diff.png", "two.diff.png"),
            ("cycles.png", "cycles.png"),
        ] {
            let _ = std::fs::copy(
                card_out.join(format!("{name}.{from}")),
                gate_out.join(format!("{name}.{to}")),
            );
        }
        let (a, b) = (100.0 * one.rmse, 100.0 * two.rmse);
        let bar = match *name {
            "room" | "movers" => {
                indoor_ok &= b <= 10.0;
                "at most 10%".to_string()
            }
            "sponza" => {
                elsewhere_ok &= b <= a + 2.0;
                format!("at most {:.2}%", a + 2.0)
            }
            _ => {
                elsewhere_ok &= b <= a + 0.2;
                format!("at most {:.2}%", a + 0.2)
            }
        };
        say(format!("| {name} | {a:.2}% | {b:.2}% | {bar} |"));
    }
    say(String::new());
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    say(format!("1. Speed: {}", verdict(speed_ok)));
    say(format!("2. Settling: {}", verdict(settle_ok)));
    say(format!("3. Indoors: {}", verdict(indoor_ok)));
    say(format!("4. Elsewhere: {}", verdict(elsewhere_ok)));
    say(format!(
        "Gate: {} ({:.1} min)",
        verdict(speed_ok && settle_ok && indoor_ok && elsewhere_ok),
        start.elapsed().as_secs_f64() / 60.0
    ));
    std::fs::write(
        gate_out.join(if rehearse {
            "rehearsal.md"
        } else {
            "report.md"
        }),
        report,
    )?;
    Ok(())
}
