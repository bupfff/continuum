//! M14 gate 4: lighting levels streamed, the cache still certified.
//!
//! Gate 3 streamed camera levels beyond CPU memory, but every geometry's
//! lighting level (what mask cones, the cache's bakes and bounce light see)
//! stayed resident with its hierarchy and its hardware structure: about
//! 350 B a triangle on the GPU, 1.36 GB for 4,096 rocks before the cache,
//! growing with every unique object.
//!
//! Method:
//! - The camera never sees lighting levels: each streamed geometry has an
//!   always-resident *proxy* level (tens of triangles) in the camera pool,
//!   its stand-in until camera levels arrive. Lighting levels need no
//!   hardware structure.
//! - Lighting levels live in a slot pool (triangles, normals, hierarchy
//!   nodes and items rebased on upload, node coverage, chart tables) within
//!   a lighting budget. An absent geometry's objects point at a resident
//!   *stub*: its hierarchy's root (bounds and prefiltered coverage). A cone
//!   that the root alone answers (missed, or splatted at its footprint)
//!   needs nothing more: the full walk would stop there too.
//! - A cone or bounce ray that would descend into a stub, or a texel on an
//!   absent geometry, makes its page incomplete: the page is not shown, the
//!   geometry is requested, and the page is baked again once every
//!   lighting level it needs is resident (those pinned while it waits, as
//!   M10 gate 24's bounce pages).
//!
//! Scene: gate 3's rock field (camera levels produced on demand), each rock
//! with a proxy level, lighting levels produced on demand too, a lighting
//! budget an eighth of all lighting levels; the sun; 1280×720, the view
//! cache, hardware camera rays; three flights.
//!
//! Pass criterion, fixed before the implementation:
//! 1. **Scale:** all lighting levels hold at least 8× the lighting budget's
//!    triangles; resident lighting triangles never exceed it.
//! 2. **Certified:** at each flight's last view, settled, every cache page
//!    resident both here and in a reference renderer (every lighting level
//!    resident, the same camera levels) holds the same texel values (within
//!    1e-5 relative), and no pixel in any frame of the flights was shaded
//!    from a page baked against a stub.
//! 3. **Memory:** GPU memory at creation at most half that of the same
//!    renderer with every lighting level resident (gate 3's way).
//! 4. **Frame budget:** each flight's mean and 99th percentile frame at
//!    most 16.7 ms, its worst at most 33.3 ms.
//!
//! Reported: lighting loads, incomplete pages and retries, pinned levels,
//! settled frames, the settled images' difference from the reference.
//!
//! `cargo run --release -p room3d --bin m14_gate4`
//!
//! `M14_DEV=1` runs a development check (fewer rocks, short flights), whose
//! numbers are not the gate's.

fn main() {
    unimplemented!("M14 gate 4: stub committed with its criteria before the implementation");
}
