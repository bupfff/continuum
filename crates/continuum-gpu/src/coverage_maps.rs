//! Directional coverage maps (M13 gate 3): each geometry's opacity seen from
//! a grid of directions, for pixels to read instead of walking a small
//! instance's triangles.
//!
//! Directions are the centres of an `DIRS × DIRS` octahedral grid (object
//! space). For each, the geometry's bounding sphere (centre `c`, radius `r`)
//! is seen orthographically along the direction through a square of side
//! `2r` split into `TEXELS × TEXELS` texels, each the share of its 4×4
//! sample points whose ray (along the direction) meets a triangle; then
//! mip levels by 2×2 means, down to one texel. `mask_cones.wgsl`'s
//! `m13_coverage_maps` reads them with the same conventions.
//!
//! Buffer layout (u32 words, floats as bits): the geometry count, then 8
//! words per geometry (its triangle hierarchy's root, the word offset of its
//! maps, `c`, `r`, unused), then the maps: per direction, the levels from
//! `TEXELS` texels across down to 1, rows from v = −1, as f32.

use continuum::glam::Vec3;
use continuum::mesh::Tri;

/// Directions per side of the octahedral grid.
pub(crate) const DIRS: u32 = 16;
/// Texels per side of a map's finest level.
pub(crate) const TEXELS: u32 = 16;
/// Samples per side of a texel.
const SAMPLES: u32 = 4;

/// Texels in one direction's levels (16² + 8² + 4² + 2² + 1).
pub(crate) fn texels_per_direction() -> u32 {
    let mut n = 0;
    let mut s = TEXELS;
    while s >= 1 {
        n += s * s;
        if s == 1 {
            break;
        }
        s /= 2;
    }
    n
}

fn sign(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

/// The unit direction at octahedral coordinates `p` in [−1, 1]².
pub(crate) fn oct_decode(p: [f32; 2]) -> Vec3 {
    let mut d = Vec3::new(p[0], p[1], 1.0 - p[0].abs() - p[1].abs());
    if d.z < 0.0 {
        let (x, y) = (d.x, d.y);
        d.x = (1.0 - y.abs()) * sign(x);
        d.y = (1.0 - x.abs()) * sign(y);
    }
    d.normalize()
}

/// A map's axes for direction `d`: `u` and `v` across the view.
pub(crate) fn basis(d: Vec3) -> (Vec3, Vec3) {
    let helper = if d.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let u = helper.cross(d).normalize();
    (u, d.cross(u))
}

/// The coverage maps of one geometry: per direction, its levels.
fn geometry_maps(tris: &[Tri], c: Vec3, r: f32) -> Vec<f32> {
    let s = (TEXELS * SAMPLES) as usize;
    let mut out = Vec::with_capacity((DIRS * DIRS * texels_per_direction()) as usize);
    let mut hit = vec![false; s * s];
    for j in 0..DIRS {
        for i in 0..DIRS {
            let d = oct_decode([
                (i as f32 + 0.5) / DIRS as f32 * 2.0 - 1.0,
                (j as f32 + 0.5) / DIRS as f32 * 2.0 - 1.0,
            ]);
            let (u, v) = basis(d);
            hit.fill(false);
            // Sample (a, b) sits at u, v = ((a + 0.5) / s · 2 − 1) r.
            let to_grid = |p: Vec3| {
                let q = p - c;
                [
                    (q.dot(u) / r + 1.0) * 0.5 * s as f32 - 0.5,
                    (q.dot(v) / r + 1.0) * 0.5 * s as f32 - 0.5,
                ]
            };
            for t in tris {
                let [a, b, e] = [to_grid(t[0]), to_grid(t[1]), to_grid(t[2])];
                let area = (b[0] - a[0]) * (e[1] - a[1]) - (b[1] - a[1]) * (e[0] - a[0]);
                if area.abs() < 1e-12 {
                    continue;
                }
                let lo = [
                    a[0].min(b[0]).min(e[0]).ceil().max(0.0) as usize,
                    a[1].min(b[1]).min(e[1]).ceil().max(0.0) as usize,
                ];
                let hi = [
                    (a[0].max(b[0]).max(e[0]).floor() as isize).min(s as isize - 1),
                    (a[1].max(b[1]).max(e[1]).floor() as isize).min(s as isize - 1),
                ];
                if hi[0] < lo[0] as isize || hi[1] < lo[1] as isize {
                    continue;
                }
                let edge = |p: [f32; 2], q: [f32; 2], x: f32, y: f32| {
                    ((q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])) * area.signum()
                };
                for y in lo[1]..=hi[1] as usize {
                    for x in lo[0]..=hi[0] as usize {
                        let (fx, fy) = (x as f32, y as f32);
                        if edge(a, b, fx, fy) >= 0.0
                            && edge(b, e, fx, fy) >= 0.0
                            && edge(e, a, fx, fy) >= 0.0
                        {
                            hit[y * s + x] = true;
                        }
                    }
                }
            }
            // The finest level: each texel the share of its samples hit.
            let n = TEXELS as usize;
            let k = SAMPLES as usize;
            let mut level: Vec<f32> = (0..n * n)
                .map(|t| {
                    let (tx, ty) = (t % n, t / n);
                    let mut count = 0;
                    for y in 0..k {
                        for x in 0..k {
                            count += hit[(ty * k + y) * s + tx * k + x] as u32;
                        }
                    }
                    count as f32 / (k * k) as f32
                })
                .collect();
            out.extend_from_slice(&level);
            let mut side = n;
            while side > 1 {
                let half = side / 2;
                level = (0..half * half)
                    .map(|t| {
                        let (x, y) = (t % half, t / half);
                        (level[2 * y * side + 2 * x]
                            + level[2 * y * side + 2 * x + 1]
                            + level[(2 * y + 1) * side + 2 * x]
                            + level[(2 * y + 1) * side + 2 * x + 1])
                            * 0.25
                    })
                    .collect();
                out.extend_from_slice(&level);
                side = half;
            }
        }
    }
    out
}

/// The maps buffer for geometries `geometries`, each with its triangle
/// hierarchy's root (None: not placed; left out).
pub(crate) fn build(geometries: &[(&[Tri], Option<u32>)]) -> Vec<u32> {
    let placed: Vec<(usize, u32)> = geometries
        .iter()
        .enumerate()
        .filter_map(|(g, (_, root))| root.map(|r| (g, r)))
        .collect();
    let header = 1 + 8 * placed.len();
    let mut words = vec![0u32; header];
    words[0] = placed.len() as u32;
    for (k, &(g, root)) in placed.iter().enumerate() {
        let tris = geometries[g].0;
        let (lo, hi) = tris.iter().flatten().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), &p| (lo.min(p), hi.max(p)),
        );
        let c = (lo + hi) * 0.5;
        let r = ((hi - lo) * 0.5).length().max(1e-6);
        let at = 1 + 8 * k;
        words[at] = root;
        words[at + 1] = words.len() as u32;
        words[at + 2] = c.x.to_bits();
        words[at + 3] = c.y.to_bits();
        words[at + 4] = c.z.to_bits();
        words[at + 5] = r.to_bits();
        words.extend(geometry_maps(tris, c, r).into_iter().map(f32::to_bits));
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube seen nearly along +z covers about its face: the finest
    /// level's mean is the face's area over the map's square, (2r)².
    #[test]
    fn a_cube_seen_head_on_covers_its_face() {
        let s = continuum::mesh::Surface::cuboid(Vec3::splat(-0.5), Vec3::splat(0.5));
        let words = build(&[(&s.triangles, Some(7))]);
        assert_eq!(words[0], 1);
        assert_eq!(words[1], 7);
        let offset = words[2] as usize;
        let r = f32::from_bits(words[6]);
        assert!((r - 0.75f32.sqrt()).abs() < 1e-5);
        // The direction nearest +z: octahedral centre cells around (0, 0).
        let per = texels_per_direction() as usize;
        let mut best = (0usize, -1.0f32);
        for j in 0..DIRS {
            for i in 0..DIRS {
                let d = oct_decode([
                    (i as f32 + 0.5) / DIRS as f32 * 2.0 - 1.0,
                    (j as f32 + 0.5) / DIRS as f32 * 2.0 - 1.0,
                ]);
                if d.z > best.1 {
                    best = ((j * DIRS + i) as usize, d.z);
                }
            }
        }
        let n = (TEXELS * TEXELS) as usize;
        let level: Vec<f32> = words[offset + best.0 * per..offset + best.0 * per + n]
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();
        let mean = level.iter().sum::<f32>() / n as f32;
        // Seen nearly head on, about a unit square in a (2r)² square.
        let expected = 1.0 / (4.0 * r * r);
        assert!((mean - expected).abs() < 0.06, "{mean} against {expected}");
        // The coarsest level is the finest's mean.
        let coarsest = f32::from_bits(words[offset + best.0 * per + per - 1]);
        assert!((coarsest - mean).abs() < 1e-4);
    }
}
