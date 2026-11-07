//! M9 gate 1: moving objects in the view-dependent lighting cache.
//!
//! The paradigm's second rule (apply what changed exactly, where that is
//! cheap) has run only in the lightmap path (M2 gates 5–6: one light, no
//! instancing, no sky). The view cache, which `Renderer` uses, assumes a
//! static scene. This gate brings moving objects into it.
//!
//! Method:
//! - **Scene structure.** Instances marked movable leave the static object
//!   hierarchy. The root's second child is a small subtree over them,
//!   rebuilt on the CPU each frame something moves; its nodes carry no
//!   prefiltered coverage (cones always descend to the movers' own triangle
//!   hierarchies, which move rigidly with them). Hardware camera rays query
//!   a second, small top-level structure of the movers, rebuilt each frame.
//! - **Exact invalidation.** A cached texel's light channel depends on a
//!   mover only if the mover's box meets the hull of the texel's apex and
//!   the light's bounding sphere (a light cone never leaves that hull). Each
//!   page keeps the object-space box of its texels' apexes. Each frame
//!   something moves, a GPU pass tests every resident page against every
//!   moved object's old and new boxes and every light: a page on a moved
//!   object, or whose hull meets a moved box, is stale.
//! - **Same-frame re-bake.** Stale pages read in the last two frames are
//!   re-baked in the same frame, before the camera passes (appended on the
//!   GPU to the frame's bake, with an indirect dispatch). Other stale pages
//!   are marked invalid at once (shading treats them as missing) and evicted
//!   when the CPU reads the list back, so no stale light is ever shown.
//!
//! Scene: the million-boulder field with gate 6 of M5's three colored lights
//! (`Field::colored_lights(origin, 1000.0)`), no sky, 1280×720, hardware
//! camera rays, cache defaults, prefilter 1 bin. Movers:
//! - the 64 boulders nearest the point 15 m ahead of the camera's start
//!   along the flight's direction, each circling a center 1.5 m from its
//!   start once every 240 frames (phases spread by index) while turning
//!   about its vertical axis at 1/60 radian per frame;
//! - a 4 × 2 × 1.6 m box (a new geometry) driving parallel to the flight,
//!   6 m to its left, from 5 m ahead of the camera's start, at 0.1 m per
//!   frame.
//! Motion advances per frame, not with the clock, so every run is the same.
//!
//! Flight: M5 gate 8's path (600 frames from standing height to 150 m up,
//! pitching down to 45°), from an empty cache.
//!
//! Control, before anything else: the sun-only warm frame at the standing
//! viewpoint (static scene) at most 3.3 ms, or the run is void.
//!
//! Pass criterion, fixed before the first run:
//! 1. **No stale light** (the cache is exact): over the first 300 frames of
//!    the flight with movers, after every 10th frame, every texel of every
//!    resident page that shading may read (not marked invalid) equals, bit
//!    for bit, its value re-baked from scratch with the scene as it then
//!    is. Zero differing texels in all 30 checks.
//! 2. **Work follows change:** in the timed flights with movers, pages
//!    re-baked because something moved average at most 15% of the pages
//!    read in the frame before (the alternative that ignores dependency
//!    regions re-bakes all of them).
//! 3. **Speed** (measured first, cool): three flights with movers,
//!    alternating with three control flights in which the movers stay
//!    still. Each flight with movers has a mean frame time at most 16.7 ms
//!    and at most 1.25× the mean of the control flights, a 99th percentile
//!    at most 33.3 ms, and fallback at most 10%.
//!
//! Reported, not gated: texels re-baked per frame for movers; pages marked
//! invalid; the controls' numbers; bake budgets and GPU times.
//!
//! `cargo run --release -p room3d --bin m9_gate1`

fn main() {
    unimplemented!("M9 gate 1: not yet implemented");
}
