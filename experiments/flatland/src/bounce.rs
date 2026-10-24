//! One-bounce indirect light for gate 6.
//!
//! Static segments are split into small reflective patches (both sides, albedo
//! [`ALBEDO`]); movers are black. Each patch side's outgoing radiance comes from
//! its direct light, kept per light so intensity changes stay free. A receiver
//! gathers, through each cone, the radiance of the nearest surface in every
//! direction.
//!
//! The gather here is an exact 2D angular sweep, not a mask: a binned gather with
//! partial coverage is order-dependent when bins hold several partial surfaces,
//! which would break the ledger's charge bounds. A depth-correct mask gather is a
//! 3D design question (see `docs/DESIGN.md`); this gate tests the ledger.

use std::f32::consts::PI;

use crate::geom::{Cone, Segment, Vec2, ray_distance};
use crate::methods;

/// Lights in the gate 6 scene; caches hold one value per light.
pub const LIGHTS: usize = 2;
pub const ALBEDO: f32 = 0.7;
/// Patch sides sample direct light this far off the surface.
const SIDE_OFFSET: f32 = 1e-3;

/// Splits segments into patches no longer than `max_len`.
pub fn patches(statics: &[Segment], max_len: f32) -> Vec<Segment> {
    statics
        .iter()
        .flat_map(|s| {
            let n = ((s.b - s.a).length() / max_len).ceil().max(1.0) as usize;
            (0..n).map(move |i| {
                let at = |f: f32| s.a + (s.b - s.a) * f;
                Segment {
                    a: at(i as f32 / n as f32),
                    b: at((i + 1) as f32 / n as f32),
                }
            })
        })
        .collect()
}

/// Which side of `s` the point `p` is on: 0 = left of a→b, 1 = right.
pub fn side_of(s: &Segment, p: Vec2) -> usize {
    if (s.b - s.a).cross(p - s.a) > 0.0 {
        0
    } else {
        1
    }
}

/// Index of a patch side's radiance for light `l` in a [`patch_radiance`] array.
pub fn radiance_index(patch: usize, side: usize, l: usize) -> usize {
    (patch * 2 + side) * LIGHTS + l
}

/// Outgoing radiance per unit light intensity of every patch side, for every light.
pub fn patch_radiance(patches: &[Segment], occluders: &[Segment], lights: &[Segment]) -> Vec<f32> {
    assert_eq!(lights.len(), LIGHTS);
    let mut out = vec![0.0; patches.len() * 2 * LIGHTS];
    for (j, p) in patches.iter().enumerate() {
        let mid = (p.a + p.b) * 0.5;
        let normal = (p.b - p.a).perp().normalized();
        for side in 0..2 {
            let sample = mid + normal * if side == 0 { SIDE_OFFSET } else { -SIDE_OFFSET };
            for (l, light) in lights.iter().enumerate() {
                if let Some(cone) = Cone::toward(sample, *light) {
                    let fluence = cone.angle * methods::coverage(&cone, occluders);
                    out[radiance_index(j, side, l)] = ALBEDO * fluence / PI;
                }
            }
        }
    }
    out
}

/// Something a gather can hit: a reflective patch, or a black mover.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub segment: Segment,
    pub patch: Option<usize>,
}

/// Radiance-weighted angle per light gathered through one cone: for every
/// direction, the radiance of the nearest surface, integrated over the cone.
///
/// Exact for surfaces that do not cross: between consecutive interval endpoints
/// the set of surfaces covering a direction is fixed, and so (without crossings)
/// is the nearest one, found along the sub-interval's middle direction.
pub fn gather(cone: &Cone, surfaces: &[Surface], radiance: &[f32]) -> [f32; LIGHTS] {
    let mut events: Vec<(f32, usize, bool)> = Vec::with_capacity(surfaces.len() * 2);
    for (i, s) in surfaces.iter().enumerate() {
        if let Some((a, b)) = cone.project(&s.segment)
            && b > a
        {
            events.push((a, i, true));
            events.push((b, i, false));
        }
    }
    events.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
    let mut out = [0.0; LIGHTS];
    let mut active: Vec<usize> = Vec::new();
    let mut at = 0.0;
    for (t, i, starts) in events {
        if t > at && !active.is_empty() {
            let dir = cone.direction((at + t) * 0.5);
            let nearest = active
                .iter()
                .filter_map(|&j| ray_distance(cone.apex, dir, &surfaces[j].segment).map(|d| (d, j)))
                .min_by(|x, y| x.0.total_cmp(&y.0));
            if let Some((_, j)) = nearest
                && let Some(patch) = surfaces[j].patch
            {
                let side = side_of(&surfaces[j].segment, cone.apex);
                let angle = (t - at) * cone.angle;
                for (l, o) in out.iter_mut().enumerate() {
                    *o += angle * radiance[radiance_index(patch, side, l)];
                }
            }
        }
        at = at.max(t);
        if starts {
            active.push(i);
        } else if let Some(pos) = active.iter().position(|&j| j == i) {
            active.swap_remove(pos);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::{Target, sky_targets};
    use crate::geom::{seg, v2};
    use crate::scene;

    fn surfaces() -> (Vec<Surface>, Vec<f32>) {
        let patches = patches(&scene::gate1().occluders, 0.04);
        let mut radiance = vec![0.0; patches.len() * 2 * LIGHTS];
        for (i, r) in radiance.iter_mut().enumerate() {
            *r = 0.05 + (i % 7) as f32 * 0.03;
        }
        let mut surfaces: Vec<Surface> = patches
            .iter()
            .enumerate()
            .map(|(j, &s)| Surface {
                segment: s,
                patch: Some(j),
            })
            .collect();
        surfaces.push(Surface {
            segment: seg(0.45, 0.33, 0.48, 0.33),
            patch: None,
        });
        (surfaces, radiance)
    }

    /// Dense rays through the same cone: the nearest hit's radiance per ray.
    fn ray_gather(
        cone: &Cone,
        surfaces: &[Surface],
        radiance: &[f32],
        rays: usize,
    ) -> [f32; LIGHTS] {
        let mut out = [0.0; LIGHTS];
        for r in 0..rays {
            let dir = cone.direction((r as f32 + 0.5) / rays as f32);
            let nearest = surfaces
                .iter()
                .enumerate()
                .filter_map(|(j, s)| ray_distance(cone.apex, dir, &s.segment).map(|d| (d, j)))
                .min_by(|x, y| x.0.total_cmp(&y.0));
            if let Some((_, j)) = nearest
                && let Some(patch) = surfaces[j].patch
            {
                let side = side_of(&surfaces[j].segment, cone.apex);
                for (l, o) in out.iter_mut().enumerate() {
                    *o += cone.angle / rays as f32 * radiance[radiance_index(patch, side, l)];
                }
            }
        }
        out
    }

    #[test]
    fn sweep_matches_dense_rays() {
        let (surfaces, radiance) = surfaces();
        for p in [v2(0.3, 0.55), v2(0.5, 0.2), v2(0.82, 0.8), v2(0.05, 0.5)] {
            for target in sky_targets(8) {
                let Target::Relative(_) = target else {
                    unreachable!()
                };
                let cone = Cone::toward(p, target.at(p)).unwrap();
                let a = gather(&cone, &surfaces, &radiance);
                let b = ray_gather(&cone, &surfaces, &radiance, 20000);
                for l in 0..LIGHTS {
                    assert!(
                        (a[l] - b[l]).abs() < 2e-3,
                        "{p:?}: sweep {} vs rays {}",
                        a[l],
                        b[l]
                    );
                }
            }
        }
    }

    #[test]
    fn gather_is_linear_in_radiance() {
        let (surfaces, radiance) = surfaces();
        let doubled: Vec<f32> = radiance.iter().map(|r| r * 2.0).collect();
        let cone = Cone::toward(v2(0.5, 0.5), sky_targets(8)[2].at(v2(0.5, 0.5))).unwrap();
        let (a, b) = (
            gather(&cone, &surfaces, &radiance),
            gather(&cone, &surfaces, &doubled),
        );
        for l in 0..LIGHTS {
            assert!((b[l] - 2.0 * a[l]).abs() < 1e-5);
        }
    }

    #[test]
    fn lit_side_is_brighter() {
        // A horizontal wall below a light: its top side sees the light, its bottom does not.
        let wall = [seg(0.2, 0.5, 0.8, 0.5)];
        let lights = [seg(0.4, 0.9, 0.6, 0.9), seg(0.4, 0.9, 0.6, 0.9)];
        let r = patch_radiance(&wall, &wall, &lights);
        let top = side_of(&wall[0], v2(0.5, 0.6));
        assert!(r[radiance_index(0, top, 0)] > 0.0);
        assert_eq!(r[radiance_index(0, 1 - top, 0)], 0.0);
    }
}
