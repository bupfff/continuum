//! M25 gate 1: coarse flat texels average their area (basics item b, "flying
//! high, new coarse ground pages show darker"; development in
//! `docs/results/m23-settling.md`, "Flying up: the dark band").
//!
//! Rising over the field, the ground shows pages coarser than wanted while
//! the wanted ones bake, and coarse ground texels' sun visibility, traced as
//! cones through the prefiltered hierarchy over many small boulders, comes
//! out dark in patches: after a 40 m rise 4.81% of pixels at most half as
//! bright as the same view settled. Area texels (M22 gate 10: 16 points'
//! hardware rays) covered only a chart's top three mips and texels that
//! curve.
//!
//! Candidate: flat texels average too (`CacheOptions::area_flat`), over a
//! chart's top ten mips (`area_mips` 10). In development the rise's dark
//! share 4.81% → 0.52%.
//!
//! A: `Renderer`'s defaults (`area_mips` 3, `area_flat` off); B: the
//! candidate.
//!
//! Pass criteria, fixed before the gate run:
//! 1. **Flying up:** `viewer_repro` (the viewer's field under its sun and
//!    sky, 1470×923, bounce, anti-aliasing, bloom, a fixed exposure, the
//!    11 ms cap) settled, then 40 m up in 240 frames: on arrival at most
//!    1.5% of pixels at most half as bright as the same view settled with
//!    B, and at most half A's share.
//! 2. **Against Cycles:** the scorecard's seven scenes each within 0.2
//!    points of A (Sponza 0.5), or better; the field from above
//!    (`field_from_above(40, 15)`, bounce on, Cycles at four bounces) within
//!    0.5 points, or better.
//! 3. **Speed** (side by side, `scorecard::speed_pair_passes`, three
//!    passes, warm-up capped at 60 s; run first): in every scene B's
//!    settled frame's median at most 1.05× A's, the moving frames' at most
//!    1.10×, every scene A settles B settling within 1.25× its time.
//!
//! Criterion 1 runs `target/release/viewer_repro` (build it first) with
//! `CONTINUUM_BOUNCE_FLAGS` and `CONTINUUM_AREA_MIPS` setting each side.
//! `M25G1_REHEARSE=1`: the field, the rise and the view from above only.
//!
//! `cargo run --release -p room3d --bin m25_gate1`

use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

/// A scene for the quality half: its name, the case with `hit_rays` given,
/// and its bar (points).
type QualityCase<'a> = (&'a str, Box<dyn Fn(bool) -> Case + 'a>, f64);

fn main() -> std::io::Result<()> {
    let start = Instant::now();
    let rehearse = std::env::var_os("M25G1_REHEARSE").is_some();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let card_out = root.join("scorecard");
    std::fs::create_dir_all(&card_out)?;
    let builds: Vec<_> = cases(&root.join("assets"))
        .into_iter()
        .filter(|(n, _)| !rehearse || ["field"].contains(n))
        .collect();
    let side = |build: &Build, on: bool| {
        let mut case = build();
        case.area = Some(if on { (10, true) } else { (3, false) });
        case
    };
    let secs = |v: Option<(usize, f64)>| v.map(|(_, t)| t);
    let show = |v: Option<f64>| v.map_or("never".to_string(), |t| format!("{t:.2} s"));
    println!(
        "M25 gate 1{}: area texels as before (A) against flat ones over ten mips (B)",
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
    }
    // 1. Flying up (after speed, before Cycles).
    println!("\n## Flying up (viewer_repro, 40 m in 240 frames)\n");
    let rise = |on: bool| -> Option<f64> {
        let out = std::process::Command::new(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/viewer_repro"),
        )
        .env(
            "CONTINUUM_BOUNCE_FLAGS",
            if on { "area_flat" } else { "-area_flat" },
        )
        .env("CONTINUUM_AREA_MIPS", if on { "10" } else { "3" })
        .env("VR_SKY", "1")
        .env("VR_RISE", "40")
        .env("VR_AUTO", "0")
        .env("VR_SETTLE", "3000")
        .output()
        .ok()?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let line = text.lines().find(|l| l.starts_with("rose"))?;
        println!("{}: {line}", if on { "B" } else { "A" });
        let at = line.find("as it arrived ")? + "as it arrived ".len();
        line[at..].split('%').next()?.trim().parse().ok()
    };
    let (ra, rb) = (rise(false), rise(true));
    let ok = match (ra, rb) {
        (Some(a), Some(b)) => b <= 1.5 && b <= 0.5 * a,
        _ => false,
    };
    pass[0] &= ok;
    println!(
        "criterion 1: {ra:?}% against {rb:?}%: {}",
        if ok { "pass" } else { "FAIL" }
    );
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
            case.area = Some(if on { (10, true) } else { (3, false) });
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
