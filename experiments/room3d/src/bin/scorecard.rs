//! The basics scorecard (phase 1 of DESIGN §3.8): a fixed set of scenes,
//! each measured the same way, run after each change that may move them.
//! Not a gate.
//!
//! Scenes: Sponza (the courtyard view, sun and sky), the million-boulder
//! field (its standing view, its sun at 1 km, its sky), the showcase's
//! closed room (one skylight panel, no sky), the same room with its two
//! movers (a bouncing ball, a spinning cube), DamagedHelmet (sun and sky),
//! and two animated characters (Fox, BrainStem: posed every frame over the
//! viewer's ground, under its sun and sky).
//!
//! **Speed** (first, for every scene, so timings are as cool as the
//! machine allows; each row states the minutes of load before it):
//! `Renderer` with its defaults at 1280×720, every frame waited for:
//! - warm-up: seconds (frames) from the first frame until under 1% of the
//!   pixels lack their page, and until settled (32 quiet frames; at most
//!   60 s);
//! - the settled frame: median and p90 of 60 frames;
//! - motion: 120 frames of a camera move (an orbit around an object, a
//!   walk and turn indoors and on the field) or of the animation or the
//!   movers playing (2 s at 60 fps): median and p99 frame, mean and worst share of pixels
//!   without their page; then held still: the image at the stop against
//!   the settled one (relative RMSE) and the seconds to settle again (at
//!   most 30 s);
//! - memory: `Renderer::gpu_bytes` after all that.
//!
//! **Quality:** the same scene at 2560×1440 with a fixed bake budget
//! (120,000 texels a frame), until settled or 600 frames, averaged 4×4
//! down to 640×360, against Cycles (Blender 5.2, cached under
//! `out/scorecard/`): relative RMSE with both clamped at 4× the reference's
//! mean (sun glints on smooth metal dominate otherwise), and the share of
//! pixels off by more than 10% of the mean in any channel. glTF files go
//! through Blender's own importer (`render_gltf_reference.py`); animated
//! characters and movers are compared at the pose where their motion
//! stopped.
//!
//! `SC_ONLY=sponza,field`: only these scenes. `SC_SPEED=0` / `SC_QUALITY=0`:
//! skip a half.
//!
//! `cargo run --release -p room3d --bin scorecard`

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

fn main() -> std::io::Result<()> {
    let run_start = Instant::now();
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/scorecard");
    std::fs::create_dir_all(&out)?;
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/assets");
    let only: Option<Vec<String>> = std::env::var("SC_ONLY")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let wanted = |name: &str| only.as_ref().is_none_or(|o| o.iter().any(|n| n == name));
    let builds = cases(&assets);
    let builds: Vec<_> = builds.into_iter().filter(|(n, _)| wanted(n)).collect();
    let mut report = String::new();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(report, "{line}").unwrap();
    };
    let opt = |v: Option<(usize, f64)>| {
        v.map_or("not within the cap".to_string(), |(f, t)| {
            format!("{t:.2} s ({f})")
        })
    };
    if std::env::var("SC_SPEED").as_deref() != Ok("0") {
        say(format!(
            "## Speed ({W}×{H}, `Renderer` defaults)\n\n| Scene | Load before | Warm-up: under 1% / settled | Settled frame (median, p90) | Motion frame (median, p99) | Without page in motion (mean, worst) | Stop vs settled | Settles again | GPU memory |\n|---|---|---|---|---|---|---|---|---|"
        ));
        for (_, build) in &builds {
            let case = build();
            let s = speed(&case, run_start);
            say(format!(
                "| {} | {:.1} min | {} / {} | {:.2}, {:.2} ms | {:.2}, {:.2} ms | {:.2}%, {:.2}% | {:.2}% | {} | {:.0} MB |",
                case.name,
                s.load_min,
                opt(s.under_1),
                opt(s.settled),
                s.frame.0,
                s.frame.1,
                s.motion.0,
                s.motion.1,
                s.motion_fallback.0,
                s.motion_fallback.1,
                100.0 * s.stop_error,
                opt(s.resettled),
                s.gpu_mb
            ));
        }
    }
    if std::env::var("SC_QUALITY").as_deref() != Ok("0") {
        say(format!(
            "\n## Against Cycles ({QW}×{QH}, ours {SS}×{SS} supersampled at a fixed budget)\n\n| Scene | Clamped relative RMSE | Pixels off > 10% | Settled at | Mean: ours, Cycles |\n|---|---|---|---|---|"
        ));
        for (_, build) in &builds {
            let case = build();
            let q = quality(&case, &out)?;
            say(format!(
                "| {} | {:.2}% | {:.2}% | {} | {:.4}, {:.4} |",
                case.name,
                100.0 * q.rmse,
                100.0 * q.off,
                q.settled
                    .map_or("not in 600 frames".to_string(), |f| format!("frame {f}")),
                q.mean_ours,
                q.mean_cycles
            ));
        }
    }
    say(format!(
        "\nTotal {:.1} min.",
        run_start.elapsed().as_secs_f64() / 60.0
    ));
    std::fs::write(out.join("scorecard.md"), report)?;
    Ok(())
}
