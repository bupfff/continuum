//! M12 gate 1: deforming geometry.
//!
//! The first of the gaps the user ordered on 2026-11-15: geometry whose
//! vertices change every frame (skinned characters, morph targets, animated
//! scale, anything an application deforms). Design: DESIGN §3.4,
//! "Deforming geometry (M12 design)". In brief:
//! - `RendererOptions::deforming` lists geometries whose vertices change;
//!   their instances move as movers do. `Renderer::set_vertices` gives a
//!   pose (triangles and normals; topology, charts and texture coordinates
//!   stay the rest pose's).
//! - Per frame, for each deformed geometry: its triangles and normals
//!   uploaded; its triangle hierarchy refit on the CPU (the rest pose's
//!   tree, boxes and coverage samples recomputed); its hardware structure
//!   rebuilt; its new object-space box gives its instances' world boxes,
//!   which update the movers' subtree and structure.
//! - The cache: its instances count as moved (M9 G1's stale pass); its own
//!   pages are stale and re-bake in the same frame when read.
//!
//! Scene: the million-boulder field under M5 gate 6's three coloured
//! lights (`Field::colored_lights(origin, 1000.0)`), no sky, 1280×720,
//! hardware camera rays, prefilter 1 bin, cache defaults. A blob: a sphere
//! of 128 × 64 segments (16,384 triangles) 2 m across, its centre 1.2 m up,
//! 10 m ahead of the standing camera along its line, its radius
//! r(θ, φ, k) = 1 + 0.15 sin(3θ + 0.05k) sin(2φ) + 0.1 sin(5φ − 0.07k) at
//! frame k (normals from the surface's derivatives). The camera stands at
//! the field's camera, looking at the blob. Motion advances per frame.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Exact:** from an empty cache, over the first 300 frames with the
//!    blob deforming, after every 10th frame, every texel of every
//!    resident page that shading may read equals, bit for bit, its value
//!    re-baked from scratch with the scene as it then is
//!    (`verify_cache`). Zero differing texels in all 30 checks.
//! 2. **Accuracy:** at frame 150's pose, per-pixel cones (no cache), 16×16
//!    samples a pixel at 320×180 box filtered, within 3% relative RMSE of
//!    instanced Cycles (4,096 samples a pixel, direct light only) over
//!    surfaces. Pixels where a sample sees a light, and their eight
//!    neighbours, are scored apart (M5 gate 6's method).
//! 3. **Cost** (timed first): three runs of 600 frames from an empty
//!    cache with the blob deforming, alternating with three in which it
//!    stays at frame 0's pose. Each deforming run's mean frame at most
//!    1.25× the still runs' mean and at most 16.7 ms, its 99th percentile
//!    at most 33.3 ms, and fallback at most 10%.
//!
//! Reported: pages and texels re-baked per frame for the blob; the refit's
//! CPU time; per-pixel cones at frame 150 against a renderer built with
//! that pose as static geometry.
//!
//! `cargo run --release -p room3d --bin m12_gate1`
//!
//! `M12_DEV=1` runs on 10,000 boulders with 100-frame runs, one of each.

fn main() {
    println!("M12 gate 1: stub (criteria above); the implementation comes next.");
}
