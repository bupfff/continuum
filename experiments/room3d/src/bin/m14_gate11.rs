//! M14 gate 11: the frame-period cap times the renderer's own frames.
//!
//! The cap on the bake budget (M14 gate 9, `CacheOptions::frame_budget_ms`)
//! measures the period between frames on the CPU. That period includes
//! whatever the caller does between frames: in M5 gate 13's development, a
//! warm-up that read the image back every 20th frame held the cap near its
//! floor (it shrinks in proportion to a long frame, and grows only while
//! frames take under 85% of the budget), and a view took up to 1,895 frames
//! to warm instead of 423.
//!
//! Now the cap measures the time spent inside the renderer's frame call
//! (which waits for the GPU when the GPU is behind), not the gap between
//! calls: what the caller does between frames, a readback or an engine's
//! own work or a wait for the display, no longer moves the bake budget.
//!
//! Pass criteria, fixed before the implementation:
//! 1. **The frame budget still holds** (run first): M14 gate 10's timed
//!    field and flights (36,864 unique rocks, camera and lighting levels on
//!    demand, a uniform sky and bounce light, default cache options, three
//!    1,800-frame flights at 1280×720 after the build and two minutes idle):
//!    each flight's mean and 99th percentile at most 16.7 ms, worst at most
//!    33.3 ms (gate 10's criterion 1).
//! 2. **The caller's stalls no longer slow warming:** at M5 gate 13's four
//!    viewpoints (the million-boulder field, sun and sky through rays,
//!    default options, 1280×720), a cache warmed from empty while the image
//!    is read back every 20th frame warms (16 quiet frames) in at most 1.25×
//!    the frames of one warmed undisturbed in the same run.
//!
//! `cargo run --release -p room3d --bin m14_gate11`

fn main() {
    println!("M14 gate 11: stub (criteria in the doc comment); not implemented yet");
}
