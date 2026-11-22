//! M22 gate 7: gate 6's candidate (two exact bounces with the orders past
//! them estimated, `CacheOptions::bounces` 2 and `bounce_rest`) by default,
//! its speed measured side by side.
//!
//! Gate 6 failed on speed alone: settled frames 1.184× on Sponza and 1.225×
//! on the field, each measured after the one-bounce renderer, on a warmer
//! machine. After gate 6 (development, `gltf_dev`): Sponza settled with two
//! bounces bakes nothing (no request, page or bounce sample in 1,500
//! frames), yet its frames swing 22.4–27 ms; with one bounce, later, 25–31
//! ms: the machine's heat, not bounce work. The field never settles with
//! either, so its frames are bounded by the time budget's 14 ms target.
//!
//! Measured as the basics scorecard measures, with `room3d::scorecard::
//! speed_pair`: per scene, both renderers alive, their frames alternated
//! (warm-up and settling count each one's own frames' time; settled frames
//! in six alternated blocks of ten; the motion frame by frame), warm-up
//! capped at 120 s.
//!
//! Pass criteria, fixed before the run:
//! 1. **Speed** (run first): in every scene that settles with one bounce,
//!    the candidate's settled-frame median at most 1.10× one bounce's and
//!    its moving median at most 1.15×; in a scene that settles with
//!    neither, its frames within 1.10× or under the 14 ms target.
//! 2. **Settling:** every scene that settles with one bounce within 120 s
//!    settles with the candidate too, within 2× its time.
//! 3. **Indoors:** the room and the movers each at most 10% from Cycles
//!    (one bounce: 46.55%, 44.46%).
//! 4. **Elsewhere:** the field, the helmet, Fox and BrainStem within 0.2
//!    points of one bounce's error, Sponza within 2.0 points.
//!
//! **Disclosed (before the run):** gate 6's numbers (accuracy is measured
//! at a fixed bake budget, so 3 and 4 repeat it: room 7.68%, movers 5.99%,
//! field 1.32%, helmet 4.46%, Fox 1.21%, BrainStem 3.11%, Sponza 31.32%
//! against 29.94%); the side-by-side measure was written after gate 6
//! failed and is first run in this gate's rehearsal.
//!
//! `M22_REHEARSE=1`: the room and Sponza only, every criterion measured the
//! same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m22_gate7`

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
    let gate_out = root.join("m22_gate7");
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
        let [one, two] = speed_pair(
            &configured(build, false),
            &configured(build, true),
            start,
            120.0,
        );
        let frame = two.frame.0 / one.frame.0;
        let moving = two.motion.0 / one.motion.0;
        if one.settled.is_some() {
            speed_ok &= frame <= 1.10 && moving <= 1.15;
        } else if two.settled.is_none() {
            // Settled with neither: frames bounded by the time budget.
            speed_ok &= frame <= 1.10 || two.frame.0 < 14.0;
        }
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
