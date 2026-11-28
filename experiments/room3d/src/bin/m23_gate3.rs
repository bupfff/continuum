//! M23 gate 3: bounce hits lit by rays, measured over three motions (basics
//! item 2, "the field's bounce light settling"; DESIGN §3.4 "M23 as built";
//! development in `docs/results/m23-settling.md`).
//!
//! Gate 2 (`docs/results/m23-gate2.md`) failed on Fox's moving frames alone
//! (1.138× against 1.10×). Repeated, the same pair measured 1.010× and
//! 0.989× for Fox, and BrainStem 1.009× then 1.195× (its side without the
//! candidate 34.18 then 29.10 ms): one pass of an animation's moving frames
//! moves more than the bar between runs. Over three passes: Fox 0.969× and
//! 0.994×, BrainStem 0.963× and 0.989×, the movers 1.006× and 1.030×.
//!
//! Changes since gate 2: the motion played three times
//! (`scorecard::speed_pair_passes`, the moving frames' median over all
//! three), and the ray-lit hits' shadow rays opaque and ending at the first
//! hit (`hw_segment_open`: nothing to skip, so the same answer without
//! visiting every candidate).
//!
//! The field never settled: a bounce sample's rays read the pages of what
//! they hit, the bake could not fit the pages asked for (dropped, asked for
//! again only by a retry that waited its turn behind ~100,000 others), and
//! under the frame cap ~100,000 hit pages take minutes to bake.
//!
//! Candidate (`CacheOptions::hit_rays`): every hit of a one-bounce sample
//! (with more bounces, the last level's) reads no page. Its direct light is
//! the unshadowed light at the hit times one hardware shadow ray toward each
//! light (movers seen too), its sky one ray cosine-distributed about its
//! normal; albedo and emission as before. Samples finish in their first
//! bake and ask for nothing. Both sides have this session's other changes
//! (pending hit requests, retry tickets, samples sharing a texel baked
//! once).
//!
//! Measured exactly as the basics scorecard measures (`room3d::scorecard`),
//! `Renderer`'s defaults with `hit_rays` off (A) and on (B).
//!
//! Pass criteria, fixed before the gate run:
//! 1. **The field settles:** with B within 60 s in the speed measurement's
//!    warm-up and again within 30 s after its walk; its pixels without a
//!    page in motion no more than A's plus 0.1 points.
//! 2. **Against Cycles:** the scorecard's seven scenes each within 0.2
//!    points of A (Sponza 0.5), or better; and the field from above
//!    (`field_from_above(40, 15)`, bounce on, Cycles at four bounces)
//!    within 0.5 points, or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair_passes`, three
//!    passes of the motion, warm-up capped at 60 s; run first): in every
//!    scene B's settled frame's median at most 1.05× A's, the moving
//!    frames' at most 1.10×, and every scene A settles B settling, within
//!    1.25× its time.
//!
//! `M23G3_REHEARSE=1`: the field, Fox and the view from above only, every
//! criterion measured the same way; not the gate's numbers.
//!
//! `cargo run --release -p room3d --bin m23_gate3`

use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

/// A scene for the quality half: its name, the case with `hit_rays` given,
/// and its bar (points).
type QualityCase<'a> = (&'a str, Box<dyn Fn(bool) -> Case + 'a>, f64);

fn main() -> std::io::Result<()> {
    let start = Instant::now();
    let rehearse = std::env::var_os("M23G3_REHEARSE").is_some();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let card_out = root.join("scorecard");
    std::fs::create_dir_all(&card_out)?;
    let builds: Vec<_> = cases(&root.join("assets"))
        .into_iter()
        .filter(|(n, _)| !rehearse || ["field", "fox"].contains(n))
        .collect();
    let side = |build: &Build, on: bool| {
        let mut case = build();
        case.hit_rays = Some(on);
        case
    };
    let secs = |v: Option<(usize, f64)>| v.map(|(_, t)| t);
    let show = |v: Option<f64>| v.map_or("never".to_string(), |t| format!("{t:.2} s"));
    println!(
        "M23 gate 3{}: hit_rays off (A) against on (B)",
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
            (Some(ta), Some(tb)) => tb <= 1.25 * ta,
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
            let ok = secs(b.settled).is_some_and(|t| t <= 60.0)
                && secs(b.resettled).is_some_and(|t| t <= 30.0)
                && b.motion_fallback.0 <= a.motion_fallback.0 + 0.1;
            pass[0] &= ok;
            println!(
                "  criterion 1 (the field settles): {}",
                if ok { "pass" } else { "FAIL" }
            );
        }
    }
    // 2. Against Cycles.
    println!("\n## Against Cycles (640×360 from 2560×1440)\n");
    let mut quality_cases: Vec<QualityCase> = builds
        .iter()
        .map(|(name, build)| {
            let bar = if *name == "sponza" { 0.5 } else { 0.2 };
            let f: Box<dyn Fn(bool) -> Case> = Box::new(move |on| side(build, on));
            (*name, f, bar)
        })
        .collect();
    quality_cases.push((
        "field from above",
        Box::new(|on| {
            let mut case = field_from_above(40.0, 15.0, false);
            case.hit_rays = Some(on);
            case
        }),
        0.5,
    ));
    for (name, case, bar) in &quality_cases {
        let qa = quality(&case(false), &card_out)?;
        let qb = quality(&case(true), &card_out)?;
        let ok = 100.0 * (qb.rmse - qa.rmse) <= *bar;
        pass[1] &= ok;
        println!(
            "{name}: {:.2}% / {:.2}% (bar +{bar}), settled at {:?} / {:?}: {}",
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
