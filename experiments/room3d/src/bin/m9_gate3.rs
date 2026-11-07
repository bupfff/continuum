//! M9 gate 3: moving objects under a sky, through the error ledger.
//!
//! The cache refuses movers with a sky: a sky cone sees the whole
//! hemisphere, so exact invalidation would re-bake nearly every page each
//! frame. This is the ledger's case: bounded deferral within ε.
//!
//! Method:
//! - **Light channels** stay exact (gate 1's stale pass and same-frame
//!   re-bake).
//! - **Charges.** Each frame something moves, every page is charged, for
//!   each moved object, the solid angle its old and its new bounding sphere
//!   subtend from the nearest point of the page's box (π if inside): a bound
//!   on how much the move can change the page's sky light per unit sky
//!   radiance. Charges accumulate in the page's ledger until it re-bakes.
//! - **ε.** A page whose ledger passes half of ε × max(its smallest sky
//!   value at the bake, 0.05π) is listed for the CPU to re-bake its sky
//!   channel; one that passes ε is marked invalid (shown missing, evicted),
//!   so no shown sky value is further than ε from a re-bake.
//! - **Pages on moved objects** keep their exact light channels; their sky
//!   shows the cache's fallback guess, flagged, until a sky re-bake after
//!   they stop.
//!
//! Scene: the million-boulder field with M5 gate 6's three colored lights
//! and the field's sky, gate 1's 65 movers and motion, 1280×720, hardware
//! camera rays, cache defaults, ε = 1%. Flight: M5 gate 8's path.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Soundness:** over the first 300 frames, after every 10th, for every
//!    page shading may read (not invalid, not flagged): its light channels
//!    equal a re-bake bit for bit, and its sky channel's largest change is at
//!    most its ledger (plus 1e-5) and at most its ε threshold. Zero
//!    violations in 30 checks.
//! 2. **Work:** pages whose sky re-baked for movers average at most 25% of
//!    the pages charged at all (which exact invalidation would re-bake).
//! 3. **Speed** (first, cool): three flights with movers alternating with
//!    three with them still, sky on in both: each moving flight's mean at
//!    most 1.25× the still flights' mean and at most 20 ms, 99th percentile
//!    at most 40 ms, fallback (with flagged sky) at most 10%.
//!
//! `cargo run --release -p room3d --bin m9_gate3`

fn main() {
    unimplemented!("M9 gate 3: not yet implemented");
}
