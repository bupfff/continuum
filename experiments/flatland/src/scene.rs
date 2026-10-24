//! The gate 1 test scene, in a unit square with y pointing up.

use crate::geom::{Segment, seg};

use std::ops::Range;

pub struct Scene {
    pub light: Segment,
    pub radiance: f32,
    pub occluders: Vec<Segment>,
    /// Partition of `occluders` into depth layers, standing in for the voxels a
    /// voxel cone tracer would average occlusion inside.
    pub layers: Vec<Range<usize>>,
}

impl Scene {
    fn push_layer(&mut self, segments: impl IntoIterator<Item = Segment>) {
        let start = self.occluders.len();
        self.occluders.extend(segments);
        self.layers.push(start..self.occluders.len());
    }
}

/// A row of horizontal slats at height `y`, starting at `x0`.
fn slats(x0: f32, y: f32, count: usize, width: f32, gap: f32) -> impl Iterator<Item = Segment> {
    (0..count).map(move |i| {
        let x = x0 + i as f32 * (width + gap);
        seg(x, y, x + width, y)
    })
}

/// Built to stress occlusion correlation:
/// - left: two stacked rows of slats with aligned gaps, so seen from below the rows
///   often hide the same directions;
/// - right: two rows with offset gaps, so the rows often hide complementary directions;
/// - a wall with a doorway and a few tilted segments for ordinary penumbrae.
pub fn gate1() -> Scene {
    let mut scene = Scene {
        light: seg(0.38, 0.95, 0.62, 0.95),
        radiance: 1.0,
        occluders: Vec::new(),
        layers: Vec::new(),
    };
    scene.push_layer(slats(0.10, 0.70, 9, 0.02, 0.02));
    scene.push_layer(slats(0.10, 0.64, 9, 0.02, 0.02));
    scene.push_layer(slats(0.55, 0.70, 9, 0.02, 0.02));
    scene.push_layer(slats(0.57, 0.64, 9, 0.02, 0.02));
    for s in [
        seg(0.15, 0.40, 0.40, 0.40),
        seg(0.48, 0.40, 0.85, 0.40),
        seg(0.20, 0.25, 0.30, 0.15),
        seg(0.70, 0.30, 0.80, 0.22),
        seg(0.60, 0.50, 0.66, 0.56),
    ] {
        scene.push_layer([s]);
    }
    scene
}
