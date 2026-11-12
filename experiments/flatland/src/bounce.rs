//! One-bounce indirect light for gate 6.
//!
//! Static segments are split into small reflective patches (both sides, albedo
//! [`ALBEDO`]); movers are black. Each patch side's outgoing radiance comes from
//! its direct light, kept per light so intensity changes stay free. A receiver
//! gathers, through each cone, the radiance of the nearest surface in every
//! direction.
//!
//! Two gathers are provided:
//!
//! - [`gather`]: an exact 2D angular sweep (gate 6). A binned gather with partial
//!   coverage is order-dependent when bins hold several partial surfaces, which
//!   would break the ledger's charge bounds.
//! - [`binned_gather`] over a [`StaticLayer`] (gate 6b): a cone G-buffer. Each
//!   bin caches the nearest static surface at its center direction and its
//!   depth; black movers are composited on top by depth. Movers and radiance
//!   changes then become exact, cheap updates instead of bounds.

use std::f32::consts::PI;

use continuum::ConeMask;

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

const BINS: usize = ConeMask::BINS as usize;
/// A bin that sees no static surface.
pub const NO_SURFACE: u16 = u16::MAX;

/// A cone G-buffer: per bin, the nearest static patch along the bin's center
/// direction and its distance. 384 bytes per cone.
#[derive(Clone, Copy, Debug)]
pub struct StaticLayer {
    pub id: [u16; BINS],
    pub depth: [f32; BINS],
}

/// Bins whose center directions fall inside `[a, b]` (normalized positions).
fn center_bins(a: f32, b: f32) -> std::ops::Range<usize> {
    let n = BINS as f32;
    let lo = (a * n - 0.5).ceil().max(0.0) as usize;
    let hi = ((b * n - 0.5).floor() + 1.0).clamp(0.0, n) as usize;
    lo..hi.max(lo)
}

fn bin_direction(cone: &Cone, bin: usize) -> Vec2 {
    cone.direction((bin as f32 + 0.5) / BINS as f32)
}

/// Builds the static layer of one cone: the nearest patch at every bin center.
pub fn static_layer(cone: &Cone, patches: &[Segment]) -> StaticLayer {
    let mut layer = StaticLayer {
        id: [NO_SURFACE; BINS],
        depth: [f32::INFINITY; BINS],
    };
    let dirs: [Vec2; BINS] = std::array::from_fn(|bin| bin_direction(cone, bin));
    for (j, patch) in patches.iter().enumerate() {
        let Some((a, b)) = cone.project(patch) else {
            continue;
        };
        for bin in center_bins(a, b) {
            if let Some(d) = ray_distance(cone.apex, dirs[bin], patch)
                && d < layer.depth[bin]
            {
                layer.depth[bin] = d;
                layer.id[bin] = j as u16;
            }
        }
    }
    layer
}

/// Radiance-weighted angle per light through one cone, sampled at bin centers:
/// the static layer's patch in each bin unless a (black) mover is nearer.
pub fn binned_gather(
    cone: &Cone,
    layer: &StaticLayer,
    patches: &[Segment],
    movers: &[Segment],
    radiance: &[f32],
) -> [f32; LIGHTS] {
    let mut blocked = 0u64;
    for m in movers {
        let Some((a, b)) = cone.project(m) else {
            continue;
        };
        for bin in center_bins(a, b) {
            if layer.id[bin] == NO_SURFACE || blocked & (1 << bin) != 0 {
                continue;
            }
            if let Some(d) = ray_distance(cone.apex, bin_direction(cone, bin), m)
                && d < layer.depth[bin]
            {
                blocked |= 1 << bin;
            }
        }
    }
    let bin_angle = cone.angle / BINS as f32;
    let mut out = [0.0; LIGHTS];
    for (bin, &id) in layer.id.iter().enumerate().take(BINS) {
        if id == NO_SURFACE || blocked & (1 << bin) != 0 {
            continue;
        }
        let side = side_of(&patches[id as usize], cone.apex);
        for (l, o) in out.iter_mut().enumerate() {
            *o += bin_angle * radiance[radiance_index(id as usize, side, l)];
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
    fn binned_gather_approximates_the_sweep() {
        let (surfaces, radiance) = surfaces();
        let patches: Vec<Segment> = surfaces
            .iter()
            .filter(|s| s.patch.is_some())
            .map(|s| s.segment)
            .collect();
        let movers: Vec<Segment> = surfaces
            .iter()
            .filter(|s| s.patch.is_none())
            .map(|s| s.segment)
            .collect();
        let (mut err, mut total) = (0.0, 0.0);
        for p in [v2(0.3, 0.55), v2(0.5, 0.2), v2(0.82, 0.8), v2(0.05, 0.5)] {
            for target in sky_targets(8) {
                let cone = Cone::toward(p, target.at(p)).unwrap();
                let exact = gather(&cone, &surfaces, &radiance);
                let binned = binned_gather(
                    &cone,
                    &static_layer(&cone, &patches),
                    &patches,
                    &movers,
                    &radiance,
                );
                for l in 0..LIGHTS {
                    err += (exact[l] - binned[l]).abs();
                    total += exact[l];
                }
            }
        }
        assert!(
            err / total < 0.03,
            "binned gather off by {:.2}%",
            100.0 * err / total
        );
    }

    #[test]
    fn a_mover_in_front_blacks_out_its_bins() {
        let wall = [seg(0.0, 0.8, 1.0, 0.8)];
        let radiance = vec![1.0; 2 * LIGHTS];
        let p = v2(0.5, 0.2);
        let cone = Cone::toward(p, seg(0.2, 0.9, 0.8, 0.9)).unwrap();
        let layer = static_layer(&cone, &wall);
        let open = binned_gather(&cone, &layer, &wall, &[], &radiance)[0];
        let covered = binned_gather(&cone, &layer, &wall, &[seg(0.3, 0.5, 0.7, 0.5)], &radiance)[0];
        let behind = binned_gather(
            &cone,
            &layer,
            &wall,
            &[seg(0.3, 0.85, 0.7, 0.85)],
            &radiance,
        )[0];
        assert!(covered < open * 0.5);
        assert_eq!(behind, open);
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
