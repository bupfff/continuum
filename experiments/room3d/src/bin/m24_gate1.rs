//! M24 gate 1: two rays a bounce bin by default (basics item c, the rest of
//! Sponza's gap; development in `docs/results/m23-settling.md`, "Sponza's
//! blotches").
//!
//! Sponza's recessed walls show page-sized blotches: a recess sees the
//! sunlit courtyard through a few of a bounce sample's 192 bins, so one ray
//! a bin aliases differently from sample to sample, and with hits lit by
//! rays each hit's one sky ray adds noise. Two rays a bin (its diagonal
//! quarters, `CacheOptions::bounce_rays`) in development: Sponza 25.04% →
//! 23.62% from Cycles (four rays 23.43%, at 1.16–1.74× the settling), the
//! room 6.66% → 6.52%. Rejected on the way: hits' sky from resident pages
//! (Sponza 25.76%), bins refined only at bright edges, stratified sky rays.
//!
//! Measured exactly as the basics scorecard measures (`room3d::scorecard`),
//! `Renderer`'s defaults with one ray a bin (A) and two (B).
//!
//! Pass criteria, fixed before the gate run:
//! 1. **Sponza:** at least 1.0 point closer to Cycles than A.
//! 2. **Elsewhere against Cycles:** the other six scenes each within 0.2
//!    points of A, or better; the field from above (`field_from_above(40,
//!    15)`, bounce on, Cycles at four bounces) within 0.5 points, or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair_passes`, three
//!    passes of the motion, warm-up capped at 60 s; run first): in every
//!    scene B's settled frame's median at most 1.05× A's, the moving
//!    frames' at most 1.10×, every scene A settles B settling within 1.5×
//!    its time (the field then within 15 s, Sponza within a minute), and
//!    the field settling within 60 s.
//!
//! `M24G1_REHEARSE=1`: Sponza, the field and the view from above only,
//! every criterion measured the same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m24_gate1`

use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

/// A scene for the quality half: its name, the case with `hit_rays` given,
/// and its bar (points).
type QualityCase<'a> = (&'a str, Box<dyn Fn(bool) -> Case + 'a>, f64);

fn main() -> std::io::Result<()> {
    let start = Instant::now();
    let rehearse = std::env::var_os("M24G1_REHEARSE").is_some();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let card_out = root.join("scorecard");
    std::fs::create_dir_all(&card_out)?;
    let builds: Vec<_> = cases(&root.join("assets"))
        .into_iter()
        .filter(|(n, _)| !rehearse || ["sponza", "field"].contains(n))
        .collect();
    let side = |build: &Build, two: bool| {
        let mut case = build();
        case.bounce_rays = Some(if two { 2 } else { 1 });
        case
    };
    let secs = |v: Option<(usize, f64)>| v.map(|(_, t)| t);
    let show = |v: Option<f64>| v.map_or("never".to_string(), |t| format!("{t:.2} s"));
    println!(
        "M24 gate 1{}: one ray a bin (A) against two (B)",
        if rehearse { " (rehearsal)" } else { "" }
    );
    let mut pass = [true; 3];
    // 3. Speed, side by side, first.
    println!("\n## Speed (side by side, 1280×720, three motions)\n");
    for (name, build) in &builds {
        let [a, b] = speed_pair_passes(&side(build, false), &side(build, true), start, 60.0, 3);
        let frame = b.frame.0 / a.frame.0;
        let moving = b.motion.0 / a.motion.0;
        let settles = match (secs(a.settled), secs(b.settled)) {
            (Some(ta), Some(tb)) => tb <= 1.5 * ta,
            (Some(_), None) => false,
            (None, _) => true,
        };
        let ok = frame <= 1.05 && moving <= 1.10 && settles;
        pass[2] &= ok;
        println!(
            "{name} ({:.1} min of load): settled frame {:.2} / {:.2} ms ({frame:.3}×), moving {:.2} / {:.2} ms ({moving:.3}×), settled at {} / {}, again after the motion {} / {}, without a page in motion {:.2}% / {:.2}%: {}",
            a.load_min,
            a.frame.0,
            b.frame.0,
            a.motion.0,
            b.motion.0,
            show(secs(a.settled)),
            show(secs(b.settled)),
            show(secs(a.resettled)),
            show(secs(b.resettled)),
            a.motion_fallback.0,
            b.motion_fallback.0,
            if ok { "pass" } else { "FAIL" }
        );
        if *name == "field" {
            let ok = secs(b.settled).is_some_and(|t| t <= 60.0);
            pass[2] &= ok;
            println!(
                "  the field settles within 60 s: {}",
                if ok { "pass" } else { "FAIL" }
            );
        }
    }
    // 2. Against Cycles.
    println!("\n## Against Cycles (640×360 from 2560×1440)\n");
    let mut quality_cases: Vec<QualityCase> = builds
        .iter()
        .map(|(name, build)| {
            let bar = if *name == "sponza" { -1.0 } else { 0.2 };
            let f: Box<dyn Fn(bool) -> Case> = Box::new(move |on| side(build, on));
            (*name, f, bar)
        })
        .collect();
    quality_cases.push((
        "field from above",
        Box::new(|on| {
            let mut case = field_from_above(40.0, 15.0, false);
            case.bounce_rays = Some(if on { 2 } else { 1 });
            case
        }),
        0.5,
    ));
    for (name, case, bar) in &quality_cases {
        let qa = quality(&case(false), &card_out)?;
        let qb = quality(&case(true), &card_out)?;
        let ok = 100.0 * (qb.rmse - qa.rmse) <= *bar;
        pass[if *name == "sponza" { 0 } else { 1 }] &= ok;
        println!(
            "{name}: {:.2}% / {:.2}% (bar {bar:+} points), settled at {:?} / {:?}: {}",
            100.0 * qa.rmse,
            100.0 * qb.rmse,
            qa.settled,
            qb.settled,
            if ok { "pass" } else { "FAIL" }
        );
    }
    println!(
        "\nCriteria: 1 {}, 2 {}, 3 {}; {:.1} min",
        if pass[0] { "pass" } else { "FAIL" },
        if pass[1] { "pass" } else { "FAIL" },
        if pass[2] { "pass" } else { "FAIL" },
        start.elapsed().as_secs_f64() / 60.0
    );
    Ok(())
}
