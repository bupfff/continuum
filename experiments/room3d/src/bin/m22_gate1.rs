//! M22 gate 1: bounce light with movers, through the ledger.
//!
//! Method (DESIGN §3.4, "Bounce light with movers"): each frame something
//! moves, pages holding bounce samples are charged the share of their view
//! the moves can change (moved boxes, the box where moved shadows may fall,
//! a page's own object); past ε (0.01) their bounce samples alone re-bake,
//! listings expiring after 16 frames; at rest a drain re-bakes what is
//! charged above ε / 100 and what was baked while something moved.
//!
//! Scene: M6's textured scene, sun only (movers under a sky are their own
//! gap), bounce light on, a fixed bake budget (`frame_target_ms` 0), 640×360.
//! Flights: (a) boulders 1–6 at 0.02 m a frame along +x for 120 frames
//! (developed); (b), not used in development: boulders 1–12 at 0.04 m a
//! frame for 120 frames.
//!
//! Measures: `verify_bounce` re-bakes every resident page's bounce samples
//! in place from the direct light the cache holds and counts pages that
//! changed by more than ε of the largest bounce value (restoring the
//! cache); `rebake_bounce` leaves them re-baked, so the image can be
//! compared with the same renderer's (no history between them).
//!
//! The bars were set with development numbers known (DESIGN §3.4: pages
//! over ε 0.49–0.63× the renderer without re-bakes; the image 0.457× its
//! distance; at rest 50 pages over ε against 137 for the scene settled
//! without motion; moving frames 1.10× with bounce light off).
//!
//! Pass criterion, fixed before the first run, for each flight:
//! 1. **While moving:** at frames 30, 60, 90 and 120, pages over ε at most
//!    0.7× a renderer whose ledger never re-bakes (ε 1e9).
//! 2. **The image:** at the end of the motion, the image's relative RMSE
//!    from its bounce re-baked in place at most 0.6× that renderer's.
//! 3. **At rest:** on a second flight, stopped and settled, every charge
//!    drained (the largest 0) and at most as many pages over ε as the scene
//!    settled without motion at the final poses.
//! 4. **Cost:** the moving frames' median at most 1.2× the same flight with
//!    bounce light off (frames interleaved).
//!
//! Development mode (`M22_DEV=1`): 320×180, 30 frames, checks every 15.

fn main() {
    unimplemented!("M22 gate 1: written after this stub is committed");
}
