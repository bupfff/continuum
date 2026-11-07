//! Alpha cutouts (M10 gate 6): a masked triangle becomes the opaque part of
//! its texture, as geometry, so every path (camera rays, light and sky
//! cones, the cache, bounce light) sees the cut with no change.
//!
//! The opaque part is where the bilinear alpha field reaches the cutoff:
//! marching squares on texel centers (crossings interpolated linearly along
//! cell edges), at the mip where the triangle spans at most [`MAX_CELLS`]
//! texels, clipped to the triangle in texture space and mapped back through
//! its barycentric coordinates. Runs of fully opaque cells merge into
//! rectangles.

use continuum::glam::{Vec2, Vec3};
use continuum::mesh::{Surface, Tri};

use crate::Texture;

/// Texels a triangle spans at most along each texture axis at the mip its
/// cut is made at.
pub const MAX_CELLS: f32 = 128.0;

/// An alpha mip pyramid: box-filtered, read repeating.
struct AlphaMips {
    levels: Vec<(usize, usize, Vec<f32>)>,
}

impl AlphaMips {
    fn new(t: &Texture) -> AlphaMips {
        let (w, h) = (t.width.max(1) as usize, t.height.max(1) as usize);
        let base: Vec<f32> = (0..w * h)
            .map(|i| t.rgba.get(4 * i + 3).map_or(1.0, |&a| a as f32 / 255.0))
            .collect();
        let mut levels = vec![(w, h, base)];
        while let Some((w, h, a)) = levels.last()
            && (*w > 1 || *h > 1)
        {
            let (w, h) = (*w, *h);
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let mut next = vec![0.0; nw * nh];
            for y in 0..nh {
                for x in 0..nw {
                    let mut sum = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        sum += a[(2 * y + dy).min(h - 1) * w + (2 * x + dx).min(w - 1)];
                    }
                    next[y * nw + x] = sum / 4.0;
                }
            }
            levels.push((nw, nh, next));
        }
        AlphaMips { levels }
    }

    fn at(&self, level: usize, x: i64, y: i64) -> f32 {
        let (w, h, a) = &self.levels[level];
        a[y.rem_euclid(*h as i64) as usize * w + x.rem_euclid(*w as i64) as usize]
    }
}

/// `p` clipped to the half-plane left of `a → b` (times `side`).
fn clip(p: &[Vec2], a: Vec2, b: Vec2, side: f32) -> Vec<Vec2> {
    let d = |x: Vec2| side * (b - a).perp_dot(x - a);
    let mut out = Vec::with_capacity(p.len() + 2);
    for k in 0..p.len() {
        let (u, v) = (p[k], p[(k + 1) % p.len()]);
        let (du, dv) = (d(u), d(v));
        if du >= 0.0 {
            out.push(u);
        }
        if (du >= 0.0) != (dv >= 0.0) {
            out.push(u + (v - u) * (du / (du - dv)));
        }
    }
    out
}

/// `surface` with each triangle replaced by the part of it where `texture`'s
/// alpha (bilinear, repeating, through the triangle's texture coordinates)
/// is at least `cutoff`. Triangles without texture coordinates, or with
/// degenerate ones, are kept whole.
pub fn cut_out(surface: &Surface, texture: &Texture, cutoff: f32) -> Surface {
    if surface.tex.len() != surface.triangles.len() {
        return surface.clone();
    }
    let mips = AlphaMips::new(texture);
    let (tw, th) = (texture.width.max(1) as f32, texture.height.max(1) as f32);
    let mut tris: Vec<Tri> = Vec::new();
    let mut normals: Vec<[Vec3; 3]> = Vec::new();
    let mut tex: Vec<[Vec2; 3]> = Vec::new();
    for k in 0..surface.triangles.len() {
        let (p, n, t) = (surface.triangles[k], surface.normals[k], surface.tex[k]);
        let det = (t[1] - t[0]).perp_dot(t[2] - t[0]);
        let geometric = (p[1] - p[0]).cross(p[2] - p[0]);
        if det.abs() < 1e-12 || geometric.length_squared() == 0.0 {
            tris.push(p);
            normals.push(n);
            tex.push(t);
            continue;
        }
        let side = det.signum();
        let (lo, hi) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
        let extent = ((hi.x - lo.x) * tw).max((hi.y - lo.y) * th);
        let mut level = 0;
        while extent / (1u64 << level) as f32 > MAX_CELLS && level + 1 < mips.levels.len() {
            level += 1;
        }
        let (lw, lh) = (mips.levels[level].0 as f32, mips.levels[level].1 as f32);
        let grid = |i: i64, j: i64| Vec2::new((i as f32 + 0.5) / lw, (j as f32 + 0.5) / lh);
        let value = |i: i64, j: i64| mips.at(level, i, j) - cutoff;
        // A convex polygon in texture space: clipped to the triangle, fanned,
        // and mapped back.
        let mut emit = |poly: &[Vec2]| {
            let mut q = poly.to_vec();
            for e in 0..3 {
                q = clip(&q, t[e], t[(e + 1) % 3], side);
                if q.len() < 3 {
                    return;
                }
            }
            let map = |x: Vec2| {
                let l1 = (x - t[0]).perp_dot(t[2] - t[0]) / det;
                let l2 = (t[1] - t[0]).perp_dot(x - t[0]) / det;
                let l0 = 1.0 - l1 - l2;
                (
                    p[0] * l0 + p[1] * l1 + p[2] * l2,
                    (n[0] * l0 + n[1] * l1 + n[2] * l2).normalize_or(n[0]),
                )
            };
            for m in 1..q.len() - 1 {
                let corners = [q[0], q[m], q[m + 1]];
                let mapped = corners.map(map);
                let mut pos = mapped.map(|c| c.0);
                let mut nor = mapped.map(|c| c.1);
                let mut uv = corners;
                let cross = (pos[1] - pos[0]).cross(pos[2] - pos[0]);
                if cross.length_squared() <= 1e-20 * geometric.length_squared() {
                    continue;
                }
                if cross.dot(geometric) < 0.0 {
                    pos.swap(1, 2);
                    nor.swap(1, 2);
                    uv.swap(1, 2);
                }
                tris.push(pos);
                normals.push(nor);
                tex.push(uv);
            }
        };
        let (i0, i1) = (
            (lo.x * lw - 0.5).floor() as i64,
            (hi.x * lw - 0.5).floor() as i64,
        );
        let (j0, j1) = (
            (lo.y * lh - 0.5).floor() as i64,
            (hi.y * lh - 0.5).floor() as i64,
        );
        for j in j0..=j1 {
            let mut run: Option<i64> = None;
            for i in i0..=i1 + 1 {
                let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
                let values = corners.map(|(a, b)| value(a, b));
                let inside = values.map(|v| v >= 0.0);
                if i <= i1 && inside.iter().all(|&x| x) {
                    run.get_or_insert(i);
                    continue;
                }
                if let Some(start) = run.take() {
                    let (a, b) = (grid(start, j), grid(i, j + 1));
                    emit(&[a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y)]);
                }
                if i > i1 || !inside.iter().any(|&x| x) {
                    continue;
                }
                // Each cell edge's crossing, computed from its lower corner
                // first so that neighbouring cells agree exactly.
                let crossing = |a: usize, b: usize| {
                    let (a, b) = if (corners[a].1, corners[a].0) < (corners[b].1, corners[b].0) {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    let (pa, pb) = (
                        grid(corners[a].0, corners[a].1),
                        grid(corners[b].0, corners[b].1),
                    );
                    pa + (pb - pa) * (values[a] / (values[a] - values[b]))
                };
                let corner = |q: usize| grid(corners[q].0, corners[q].1);
                let saddle =
                    inside[0] == inside[2] && inside[1] == inside[3] && inside[0] != inside[1];
                if saddle && values.iter().sum::<f32>() < 0.0 {
                    // Two separate corners.
                    for q in (0..4).filter(|&q| inside[q]) {
                        let (before, after) = ((q + 3) % 4, (q + 1) % 4);
                        emit(&[crossing(before, q), corner(q), crossing(q, after)]);
                    }
                    continue;
                }
                let mut poly = Vec::with_capacity(8);
                for a in 0..4 {
                    let b = (a + 1) % 4;
                    if inside[a] {
                        poly.push(corner(a));
                    }
                    if inside[a] != inside[b] {
                        poly.push(crossing(a, b));
                    }
                }
                if saddle {
                    // Joined through the cell's center: fanned from it, as
                    // the two cuts need not leave a convex hexagon.
                    let center = (corner(0) + corner(2)) * 0.5;
                    for m in 0..poly.len() {
                        emit(&[center, poly[m], poly[(m + 1) % poly.len()]]);
                    }
                } else {
                    emit(&poly);
                }
            }
        }
    }
    Surface::from_mesh(tris, Some(normals), Some(tex))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 × 1 quad with texture coordinates over the whole texture.
    fn quad() -> Surface {
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let uv = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        Surface::from_mesh(
            vec![[p[0], p[1], p[2]], [p[0], p[2], p[3]]],
            None,
            Some(vec![[uv[0], uv[1], uv[2]], [uv[0], uv[2], uv[3]]]),
        )
    }

    fn area(s: &Surface) -> f32 {
        s.triangles
            .iter()
            .map(|t| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length())
            .sum()
    }

    fn texture(size: u32, alpha: impl Fn(u32, u32) -> u8) -> Texture {
        let mut rgba = Vec::new();
        for y in 0..size {
            for x in 0..size {
                rgba.extend([255, 255, 255, alpha(x, y)]);
            }
        }
        Texture {
            width: size,
            height: size,
            rgba,
        }
    }

    #[test]
    fn opaque_and_clear_textures_keep_all_or_nothing() {
        let q = quad();
        assert!((area(&cut_out(&q, &texture(16, |_, _| 255), 0.5)) - 1.0).abs() < 1e-5);
        assert_eq!(cut_out(&q, &texture(16, |_, _| 0), 0.5).triangles.len(), 0);
    }

    #[test]
    fn a_half_clear_texture_keeps_half_facing_the_same_way() {
        // Left half opaque: the contour runs midway between the last opaque
        // and first clear texel centers, at u = 0.5 (repeating, the seam at
        // u = 0 is also an edge, half a texel from each side).
        let q = quad();
        let cut = cut_out(&q, &texture(16, |x, _| if x < 8 { 255 } else { 0 }), 0.5);
        let a = area(&cut);
        assert!((a - 0.5).abs() < 1e-4, "area {a}");
        for t in &cut.triangles {
            assert!(
                (t[1] - t[0]).cross(t[2] - t[0]).z > 0.0,
                "a triangle flipped"
            );
            assert!(t.iter().all(|p| p.x <= 0.5 + 1e-5 || p.x >= 1.0 - 1e-5));
        }
    }

    #[test]
    fn a_disc_keeps_its_area() {
        // A disc of radius 0.3 (in texture units) at the center.
        let size = 128;
        let q = quad();
        let cut = cut_out(
            &q,
            &texture(size, |x, y| {
                let (u, v) = (
                    (x as f32 + 0.5) / size as f32 - 0.5,
                    (y as f32 + 0.5) / size as f32 - 0.5,
                );
                if (u * u + v * v).sqrt() < 0.3 { 255 } else { 0 }
            }),
            0.5,
        );
        let a = area(&cut);
        let disc = std::f32::consts::PI * 0.09;
        assert!((a - disc).abs() < 0.01 * disc, "area {a} against {disc}");
    }
}
