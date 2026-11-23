//! Development (not a gate): the basics scorecard's scenes with two values of
//! `CacheOptions::bounce_rays`, `PAIR_A` and `PAIR_B` (1 and 2), speed side
//! by side (`scorecard::speed_pair`, warm-up capped at 120 s) then each
//! against Cycles. `SC_ONLY=a,b`: only those scenes.
//!
//! `cargo run --release -p room3d --bin pair_dev`

use std::path::Path;
use std::time::Instant;

use room3d::scorecard::*;

fn main() -> std::io::Result<()> {
    let start = Instant::now();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    let card_out = root.join("scorecard");
    std::fs::create_dir_all(&card_out)?;
    let only: Option<Vec<String>> = std::env::var("SC_ONLY")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let value = |name: &str, default: u32| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (a, b) = (value("PAIR_A", 1), value("PAIR_B", 2));
    let builds: Vec<_> = cases(&root.join("assets"))
        .into_iter()
        .filter(|(n, _)| only.as_ref().is_none_or(|o| o.iter().any(|x| x == n)))
        .collect();
    let configured = |build: &Build, rays: u32| {
        let mut case = build();
        case.bounce_rays = Some(rays);
        case
    };
    let opt = |v: Option<(usize, f64)>| v.map_or("never".to_string(), |(_, t)| format!("{t:.2} s"));
    println!("bounce_rays {a} against {b}");
    for (name, build) in &builds {
        let [one, two] = speed_pair(&configured(build, a), &configured(build, b), start, 120.0);
        println!(
            "{name}: settled frame {:.2} / {:.2} ms ({:.3}×), moving {:.2} / {:.2} ms ({:.3}×), settled at {} / {}",
            one.frame.0,
            two.frame.0,
            two.frame.0 / one.frame.0,
            one.motion.0,
            two.motion.0,
            two.motion.0 / one.motion.0,
            opt(one.settled),
            opt(two.settled)
        );
    }
    for (name, build) in &builds {
        let qa = quality(&configured(build, a), &card_out)?;
        let qb = quality(&configured(build, b), &card_out)?;
        println!(
            "{name}: against Cycles {:.2}% / {:.2}%",
            100.0 * qa.rmse,
            100.0 * qb.rmse
        );
    }
    println!("{:.1} min", start.elapsed().as_secs_f64() / 60.0);
    Ok(())
}
