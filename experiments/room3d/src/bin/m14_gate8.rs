//! M14 gate 8: a sky and bounce light with streamed lighting levels.
//!
//! Gates 4–7 streamed lighting levels with the sun alone: the sky's and
//! bounce light's bakes did not yet mark pages that met an absent level.
//!
//! Method (developed before this stub, with `stream_dev` `PAGES` on 64 and
//! 256 rocks, every comparison exact; see below):
//! - the sky's bake marks a page (and lists the absent object) as the sun's
//!   does; every absent object a page met is listed once a frame (a hash
//!   bitset), so a page needing several levels learns them in one round;
//! - a streamed geometry's stub keeps the top four levels of its hierarchy
//!   (boxes, prefiltered coverage; gate 5 kept the root alone): wide cones
//!   (the sky's faces, coarse texels) stop within them as a full walk
//!   would, and a walk marks only below the stub's frontier;
//! - bounce light, like an absent page at a hit, counts a ray that meets an
//!   absent level as missing and retries the sample, the level wanted
//!   (after direct light's wants); the page stays shown. Direct light (sun
//!   and sky channels) stays certified; bounce converges;
//! - skies baked alone and bounce retries run only once their levels are
//!   resident (pinned then; wanted, not pinned, before: pins held by pages
//!   that cannot progress deadlocked the slots); a bounce retry no longer
//!   clears its page's marks.
//!
//! The sky cannot be certified against gate 7's reference (a region with a
//! margin): it sees rocks out to the horizon. So certification uses a field
//! small enough for a renderer with every lighting level resident.
//!
//! Scenes: (a) 96 × 96 = 9,216 unique rocks, lighting levels on demand,
//! lighting slots an eighth of them; (b) gate 7's 36,864 rocks and options.
//! Both with a uniform sky and bounce light, the sun, 1280×720, the view
//! cache, hardware camera rays; gate 4's three flights (over the central
//! 384 m of (b), over all of (a)).
//!
//! Pass criterion, fixed before the gate binary was written:
//! 1. **Certified** (a): in flight (every 60 frames) and settled at each
//!    flight's last view, every page shown here identical (1e-5) in its sun
//!    channel, and every page with its sky baked in its sky channel, to the
//!    same scene with every lighting level resident; at least 100,000 sun
//!    and 30,000 sky pages compared in all.
//! 2. **Bounce converges** (a): at each flight's last view, settled, the
//!    image's mean absolute difference from the all-resident renderer's
//!    settled image at most 1% of that image's mean.
//! 3. **Frame budget** (b): each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms (timed first, after the scene
//!    is built and the machine has idled two minutes).
//! 4. **Memory per unique rock** (b, 4,096 → 36,864 rocks, same pools, at
//!    creation): GPU at most 1/6 of a resident lighting level's, CPU heap
//!    at most 1/6 of a level as gate 4 kept it (the deeper stubs cost more).
//!
//! Reported: frames to settle, pages waiting, bounce samples retried.
//!
//! `cargo run --release -p room3d --bin m14_gate8`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    println!("M14 gate 8: not implemented yet");
}
