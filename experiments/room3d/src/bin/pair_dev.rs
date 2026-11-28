//! Development (not a gate): the basics scorecard's scenes with two values of
//! `CacheOptions::bounce_rays`, `PAIR_A` and `PAIR_B` (1 and 2), speed side
//! by side (`scorecard::speed_pair`, warm-up capped at 120 s) then each
//! against Cycles. `SC_ONLY=a,b`: only those scenes. `PAIR_OPTION=hit_rays`:
//! `CacheOptions::hit_rays` instead (the values 0 and 1); `one_level` the
//! same; `bounces` (`CacheOptions::bounces`). `SC_ONLY=field_sun` adds the
//! viewer's field (the sun alone), not one of the seven. `PAIR_PASSES=<n>`:
//! the motion played n times (`speed_pair_passes`). `SC_QUALITY=0`: speed
//! only.
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
    let mut all = cases(&root.join("assets"));
    all.push(("field_sun", Box::new(field_sun_case) as Build));
    let builds: Vec<_> = all
        .into_iter()
        .filter(|(n, _)| match &only {
            Some(o) => o.iter().any(|x| x == n),
            None => *n != "field_sun",
        })
        .collect();
    let option = std::env::var("PAIR_OPTION").unwrap_or_else(|_| "bounce_rays".into());
    let configured = |build: &Build, v: u32| {
        let mut case = build();
        match option.as_str() {
            "hit_rays" => case.hit_rays = Some(v != 0),
            "one_level" => case.one_level = Some(v != 0),
            "bounces" => case.bounces = Some(v),
            _ => case.bounce_rays = Some(v),
        }
        case
    };
    let opt = |v: Option<(usize, f64)>| v.map_or("never".to_string(), |(_, t)| format!("{t:.2} s"));
    println!("{option} {a} against {b}");
    for (name, build) in &builds {
        let passes = value("PAIR_PASSES", 1) as usize;
        let [one, two] = speed_pair_passes(
            &configured(build, a),
            &configured(build, b),
            start,
            120.0,
            passes,
        );
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
    for (name, build) in builds
        .iter()
        .filter(|_| std::env::var("SC_QUALITY").as_deref() != Ok("0"))
    {
        let qa = quality(&configured(build, a), &card_out)?;
        let qb = quality(&configured(build, b), &card_out)?;
        println!(
            "{name}: against Cycles {:.2}% / {:.2}%, means {:.4} / {:.4} (Cycles {:.4}), settled at {:?} / {:?}",
            100.0 * qa.rmse,
            100.0 * qb.rmse,
            qa.mean_ours,
            qb.mean_ours,
            qa.mean_cycles,
            qa.settled,
            qb.settled
        );
    }
    println!("{:.1} min", start.elapsed().as_secs_f64() / 60.0);
    Ok(())
}
