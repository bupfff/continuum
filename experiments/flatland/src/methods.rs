//! Ways to estimate the visible fraction of a light from a point.
//!
//! All methods return the unblocked fraction of the cone in `[0, 1]`; fluence is
//! `radiance · cone.angle · fraction`.

use std::ops::Range;

use continuum::{ConeMask, CoverageMask};

use crate::geom::{Cone, Segment, ray_hits};
use crate::rng::Pcg32;

/// Analytic answer: the measure of the union of blocked angular intervals.
/// Only affordable in 2D; in 3D this is a union of projected polygons on a sphere.
pub fn exact(cone: &Cone, occluders: &[Segment], scratch: &mut Vec<(f32, f32)>) -> f32 {
    scratch.clear();
    scratch.extend(occluders.iter().filter_map(|s| cone.project(s)));
    scratch.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut blocked = 0.0;
    let mut current: Option<(f32, f32)> = None;
    for &(a, b) in scratch.iter() {
        current = match current {
            Some((ca, cb)) if a <= cb => Some((ca, cb.max(b))),
            Some((ca, cb)) => {
                blocked += cb - ca;
                Some((a, b))
            }
            None => Some((a, b)),
        };
    }
    if let Some((ca, cb)) = current {
        blocked += cb - ca;
    }
    (1.0 - blocked).max(0.0)
}

/// One mask cone: each occluder sets the bins it covers.
pub fn mask(cone: &Cone, occluders: &[Segment]) -> f32 {
    mask_bits(cone, occluders).visible_fraction()
}

/// The direction mask of one cone against `occluders`.
pub fn mask_bits(cone: &Cone, occluders: &[Segment]) -> ConeMask {
    let mut m = ConeMask::EMPTY;
    for s in occluders {
        if let Some((a, b)) = cone.project(s) {
            m |= ConeMask::from_interval(a, b);
            if m.is_full() {
                break;
            }
        }
    }
    m
}

/// One mask cone with fractional edge bins (continuous visibility).
pub fn coverage(cone: &Cone, occluders: &[Segment]) -> f32 {
    let mut m = CoverageMask::EMPTY;
    add_coverage(&mut m, cone, occluders);
    m.visible_fraction()
}

/// Adds `occluders` to a coverage mask, stopping once it is full.
pub fn add_coverage(m: &mut CoverageMask, cone: &Cone, occluders: &[Segment]) {
    for s in occluders {
        if m.is_full() {
            return;
        }
        if let Some((a, b)) = cone.project(s) {
            m.add_interval(a, b);
        }
    }
}

/// Multiplies per-occluder transmittance, which assumes every occluder is
/// independent of every other, even ones side by side.
pub fn transmittance(cone: &Cone, occluders: &[Segment]) -> f32 {
    occluders
        .iter()
        .filter_map(|s| cone.project(s))
        .map(|(a, b)| 1.0 - (b - a))
        .product()
}

/// Voxel-cone-tracing-style occlusion: coverage is exact inside each layer (as if
/// averaged inside one voxel) but multiplied across layers, which assumes layers
/// at different depths are independent.
pub fn layered_transmittance(
    cone: &Cone,
    occluders: &[Segment],
    layers: &[Range<usize>],
    scratch: &mut Vec<(f32, f32)>,
) -> f32 {
    layers
        .iter()
        .map(|l| exact(cone, &occluders[l.clone()], scratch))
        .product()
}

/// `n` rays stratified across the cone. With `rng`, each ray is jittered inside its
/// stratum (a Monte Carlo estimate); without, rays sit at stratum centers.
pub fn rays(cone: &Cone, occluders: &[Segment], n: u32, mut rng: Option<&mut Pcg32>) -> f32 {
    let mut unblocked = 0;
    for k in 0..n {
        let offset = rng.as_mut().map_or(0.5, |r| r.next_f32());
        let dir = cone.direction((k as f32 + offset) / n as f32);
        let max_t = cone.light_distance(dir);
        if !occluders.iter().any(|s| ray_hits(cone.apex, dir, s, max_t)) {
            unblocked += 1;
        }
    }
    unblocked as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{seg, v2};
    use crate::scene;

    fn sample_points(count: u32) -> impl Iterator<Item = Cone> {
        let scene = scene::gate1();
        let mut rng = Pcg32::new(7, 0);
        (0..count).filter_map(move |_| {
            let p = v2(rng.next_f32(), rng.next_f32());
            Cone::toward(p, scene.light)
        })
    }

    #[test]
    fn exact_matches_dense_rays() {
        let scene = scene::gate1();
        let mut scratch = Vec::new();
        for cone in sample_points(400) {
            let e = exact(&cone, &scene.occluders, &mut scratch);
            let r = rays(&cone, &scene.occluders, 8192, None);
            assert!(
                (e - r).abs() < 2e-3,
                "exact {e} vs rays {r} at {:?}",
                cone.apex
            );
        }
    }

    #[test]
    fn mask_equals_one_ray_per_bin() {
        let scene = scene::gate1();
        let cones: Vec<_> = sample_points(2000).collect();
        let mismatches = cones
            .iter()
            .filter(|c| mask(c, &scene.occluders) != rays(c, &scene.occluders, 64, None))
            .count();
        // Only float ties at bin boundaries may differ.
        assert!(
            mismatches * 200 <= cones.len(),
            "{mismatches} of {} differ",
            cones.len()
        );
    }

    #[test]
    fn single_occluder_has_no_correlation_error() {
        let occluders = [seg(0.45, 0.6, 0.52, 0.6)];
        let light = seg(0.38, 0.95, 0.62, 0.95);
        let cone = Cone::toward(v2(0.5, 0.2), light).unwrap();
        let e = exact(&cone, &occluders, &mut Vec::new());
        assert!((transmittance(&cone, &occluders) - e).abs() < 1e-6);
    }

    #[test]
    fn occluders_behind_the_light_are_ignored() {
        let light = seg(0.38, 0.95, 0.62, 0.95);
        let cone = Cone::toward(v2(0.5, 0.2), light).unwrap();
        assert!(cone.project(&seg(0.3, 0.98, 0.7, 0.98)).is_none());
        assert!(cone.project(&seg(0.3, 0.5, 0.7, 0.5)).is_some());
    }
}
