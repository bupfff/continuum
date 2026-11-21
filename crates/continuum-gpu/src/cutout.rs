//! Alpha cutouts (M10 gate 6): a masked triangle becomes the opaque part of
//! its texture, as geometry, so every path (camera rays, light and sky
//! cones, the cache, bounce light) sees the cut with no change.
//!
//! The opaque part is where the bilinear alpha field reaches the cutoff:
//! marching squares on texel centers (crossings interpolated linearly along
//! cell edges), at the mip where the triangle spans at most [`MAX_CELLS`]
//! texels, clipped to the triangle in texture space and mapped back through
//! its barycentric coordinates.
//!
//! The contours are traced into closed rings, simplified to within
//! [`TOLERANCE`] of a cell (Douglas–Peucker) and filled as trapezoids
//! (even–odd) within the triangle by a sweep into monotone polygons, so a
//! triangle costs about one triangle per contour vertex left on it. Cell by cell
//! (each boundary cell's polygon, runs of opaque cells as rectangles) cost
//! Sponza's ivy 345 triangles for each one (24.9 million in all).

use continuum::glam::{Vec2, Vec3};
use continuum::mesh::{Surface, Tri};

use crate::{Texture, Wrap};

/// Texels a triangle spans at most along each texture axis at the mip its
/// cut is made at.
pub const MAX_CELLS: f32 = 128.0;

/// How far, in cells, a simplified contour may stray from the traced one.
pub const TOLERANCE: f32 = 0.2;

/// An alpha mip pyramid: box-filtered, read with the texture's wrap modes.
struct AlphaMips {
    levels: Vec<(usize, usize, Vec<f32>)>,
    wrap: [Wrap; 2],
}

impl AlphaMips {
    fn new(t: &Texture, wrap: [Wrap; 2]) -> AlphaMips {
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
        AlphaMips { levels, wrap }
    }

    fn at(&self, level: usize, x: i64, y: i64) -> f32 {
        let (w, h, a) = &self.levels[level];
        let (x, y) = (
            self.wrap[0].texel(x, *w as i64),
            self.wrap[1].texel(y, *h as i64),
        );
        a[y as usize * w + x as usize]
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

/// Marching squares over a triangle's cells, traced into closed rings.
#[derive(Default)]
struct Contours {
    /// Corner values over the padded grid, row by row.
    values: Vec<f32>,
    /// Per cell edge (two per corner: along x, along y), the edge the
    /// contour leaves for, or `u32::MAX`.
    next: Vec<u32>,
    /// The edges with a crossing.
    crossed: Vec<u32>,
}

impl Contours {
    /// The contours over cells `i0..=i1` × `j0..=j1` (corners `value(i, j)`,
    /// inside where at least 0), padded by a cell of outside all round so
    /// every ring closes: rings in local cell coordinates (corner (i0 - 1,
    /// j0 - 1) at the origin), the inside on the left. Saddles join or
    /// separate by the cell's mean, as cell by cell.
    fn trace(
        &mut self,
        i0: i64,
        i1: i64,
        j0: i64,
        j1: i64,
        value: &impl Fn(i64, i64) -> f32,
    ) -> Vec<Vec<Vec2>> {
        let (w, h) = ((i1 - i0 + 4) as usize, (j1 - j0 + 4) as usize);
        self.values.clear();
        for cj in 0..h {
            for ci in 0..w {
                let real = ci >= 1 && ci + 1 < w && cj >= 1 && cj + 1 < h;
                self.values.push(if real {
                    value(ci as i64 + i0 - 1, cj as i64 + j0 - 1)
                } else {
                    -1.0
                });
            }
        }
        if self.next.len() < 2 * w * h {
            self.next.resize(2 * w * h, u32::MAX);
        }
        let v = |ci: usize, cj: usize| self.values[cj * w + ci];
        let hor = |ci: usize, cj: usize| (2 * (cj * w + ci)) as u32;
        let ver = |ci: usize, cj: usize| (2 * (cj * w + ci) + 1) as u32;
        for cj in 0..h - 1 {
            for ci in 0..w - 1 {
                let values = [v(ci, cj), v(ci + 1, cj), v(ci + 1, cj + 1), v(ci, cj + 1)];
                let inside = values.map(|x| x >= 0.0);
                if inside.iter().all(|&x| x) || !inside.iter().any(|&x| x) {
                    continue;
                }
                let edges = [hor(ci, cj), ver(ci + 1, cj), hor(ci, cj + 1), ver(ci, cj)];
                let saddle =
                    inside[0] == inside[2] && inside[1] == inside[3] && inside[0] != inside[1];
                let separate = values.iter().sum::<f32>() < 0.0;
                for k in 0..4 {
                    if !(inside[k] && !inside[(k + 1) % 4]) {
                        continue;
                    }
                    let m = if saddle {
                        if separate { (k + 3) % 4 } else { (k + 1) % 4 }
                    } else {
                        (0..4)
                            .find(|&m| !inside[m] && inside[(m + 1) % 4])
                            .expect("a crossing back in")
                    };
                    self.next[edges[k] as usize] = edges[m];
                    self.crossed.push(edges[k]);
                }
            }
        }
        // Each edge's crossing, from its lower corner.
        let point = |e: u32| {
            let c = (e / 2) as usize;
            let (ci, cj) = (c % w, c / w);
            let (a, b) = if e.is_multiple_of(2) {
                ((ci, cj), (ci + 1, cj))
            } else {
                ((ci, cj), (ci, cj + 1))
            };
            let (va, vb) = (v(a.0, a.1), v(b.0, b.1));
            let (pa, pb) = (
                Vec2::new(a.0 as f32, a.1 as f32),
                Vec2::new(b.0 as f32, b.1 as f32),
            );
            pa + (pb - pa) * (va / (va - vb))
        };
        let mut rings = Vec::new();
        for k in 0..self.crossed.len() {
            let start = self.crossed[k];
            if self.next[start as usize] == u32::MAX {
                continue;
            }
            let mut ring = Vec::new();
            let mut e = start;
            loop {
                ring.push(point(e));
                let n = std::mem::replace(&mut self.next[e as usize], u32::MAX);
                if n == u32::MAX || n == start {
                    break;
                }
                e = n;
            }
            rings.push(ring);
        }
        for &e in &self.crossed {
            self.next[e as usize] = u32::MAX;
        }
        self.crossed.clear();
        rings
    }
}

/// `ring` (closed) simplified by Douglas–Peucker to within `tolerance`.
fn simplify(ring: &[Vec2], tolerance: f32) -> Vec<Vec2> {
    let n = ring.len();
    if n <= 3 {
        return ring.to_vec();
    }
    let far = (1..n)
        .max_by(|&a, &b| {
            (ring[a] - ring[0])
                .length_squared()
                .total_cmp(&(ring[b] - ring[0]).length_squared())
        })
        .unwrap_or(1);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    let mut stack = vec![(0, far), (far, n)];
    while let Some((a, b)) = stack.pop() {
        if b - a < 2 {
            continue;
        }
        let (pa, pb) = (ring[a], ring[b % n]);
        let d = pb - pa;
        let len2 = d.length_squared();
        let mut worst = (0.0, a);
        for (k, &p) in ring.iter().enumerate().take(b).skip(a + 1) {
            let t = if len2 > 0.0 {
                ((p - pa).dot(d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let dist = (p - (pa + d * t)).length_squared();
            if dist > worst.0 {
                worst = (dist, k);
            }
        }
        if worst.0 > tolerance * tolerance {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    ring.iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| *p)
        .collect()
}

/// The even–odd fill of `rings` inside the triangle `clip`, as triangles:
/// a sweep up through bands between vertex heights (and the heights where
/// a contour crosses the triangle's edges), each run of bands bounded on
/// both sides by connected chains made one monotone polygon and
/// triangulated, about one triangle per vertex.
fn fill(rings: &[Vec<Vec2>], clip: [Vec2; 3], mut out: impl FnMut([Vec2; 3])) {
    // Edges, lower end first, and whether each bounds the triangle.
    let mut edges: Vec<(Vec2, Vec2, bool)> = Vec::new();
    let mut push = |a: Vec2, b: Vec2, bounds: bool| {
        if a.y != b.y {
            edges.push(if a.y < b.y {
                (a, b, bounds)
            } else {
                (b, a, bounds)
            });
        }
    };
    for e in 0..3 {
        push(clip[e], clip[(e + 1) % 3], true);
    }
    for r in rings {
        for k in 0..r.len() {
            push(r[k], r[(k + 1) % r.len()], false);
        }
    }
    let (bottom, top) = (
        clip[0].y.min(clip[1].y).min(clip[2].y),
        clip[0].y.max(clip[1].y).max(clip[2].y),
    );
    if bottom >= top {
        return;
    }
    let mut ys: Vec<f32> = vec![bottom, top];
    for (a, b, bounds) in &edges {
        for y in [a.y, b.y] {
            if y > bottom && y < top {
                ys.push(y);
            }
        }
        if *bounds {
            continue;
        }
        // Where the contour crosses the triangle's edges.
        for e in 0..3 {
            let (c, d) = (clip[e], clip[(e + 1) % 3]);
            let (r, s) = (*b - *a, d - c);
            let den = r.perp_dot(s);
            if den == 0.0 {
                continue;
            }
            let t = (c - *a).perp_dot(s) / den;
            let u = (c - *a).perp_dot(r) / den;
            if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
                let y = a.y + r.y * t;
                if y > bottom && y < top {
                    ys.push(y);
                }
            }
        }
    }
    ys.sort_by(f32::total_cmp);
    ys.dedup();
    let x_at = |e: usize, y: f32| {
        let (a, b, _) = edges[e];
        a.x + (b.x - a.x) * ((y - a.y) / (b.y - a.y))
    };
    struct Piece {
        left: usize,
        right: usize,
        chains: [Vec<Vec2>; 2],
        matched: bool,
    }
    let mut pieces: Vec<Piece> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| edges[a].0.y.total_cmp(&edges[b].0.y));
    let mut added = 0;
    let same = |a: f32, b: f32| (a - b).abs() <= 1e-4 * (1.0 + a.abs());
    for k in 0..ys.len() - 1 {
        let (y0, y1) = (ys[k], ys[k + 1]);
        let mid = 0.5 * (y0 + y1);
        active.retain(|&e| edges[e].1.y > y0);
        while added < order.len() && edges[order[added]].0.y <= y0 {
            let e = order[added];
            if edges[e].1.y > y0 {
                active.push(e);
            }
            added += 1;
        }
        active.sort_by(|&a, &b| x_at(a, mid).total_cmp(&x_at(b, mid)));
        for p in &mut pieces {
            p.matched = false;
        }
        let mut fresh: Vec<Piece> = Vec::new();
        let (mut odd, mut within) = (false, false);
        let mut left = None;
        for &e in &active {
            let was = odd && within;
            if edges[e].2 {
                within = !within;
            } else {
                odd = !odd;
            }
            match (was, odd && within) {
                (false, true) => left = Some(e),
                (true, false) => {
                    let (l, r) = (left.take().expect("a span's left"), e);
                    let (xl, xr) = (x_at(l, y0), x_at(r, y0));
                    let continues = pieces.iter_mut().find(|p| {
                        !p.matched
                            && (p.left == l || same(x_at(p.left, y0), xl))
                            && (p.right == r || same(x_at(p.right, y0), xr))
                    });
                    if let Some(p) = continues {
                        p.matched = true;
                        for (side, edge, x) in [(0, l, xl), (1, r, xr)] {
                            let old = if side == 0 { &mut p.left } else { &mut p.right };
                            if *old != edge {
                                *old = edge;
                                p.chains[side].push(Vec2::new(x, y0));
                            }
                        }
                    } else {
                        fresh.push(Piece {
                            left: l,
                            right: r,
                            chains: [vec![Vec2::new(xl, y0)], vec![Vec2::new(xr, y0)]],
                            matched: true,
                        });
                    }
                }
                _ => {}
            }
        }
        for mut p in pieces.drain(..) {
            if p.matched {
                fresh.push(p);
            } else {
                p.chains[0].push(Vec2::new(x_at(p.left, y0), y0));
                p.chains[1].push(Vec2::new(x_at(p.right, y0), y0));
                monotone(&p.chains, &mut out);
            }
        }
        pieces = fresh;
    }
    for mut p in pieces {
        p.chains[0].push(Vec2::new(x_at(p.left, top), top));
        p.chains[1].push(Vec2::new(x_at(p.right, top), top));
        monotone(&p.chains, &mut out);
    }
}

/// Triangulates the polygon bounded on the left by `chains[0]` and on the
/// right by `chains[1]` (each from bottom to top, the two joined by their
/// ends), monotone in y (de Berg et al., Computational Geometry, 3.3).
fn monotone(chains: &[Vec<Vec2>; 2], out: &mut impl FnMut([Vec2; 3])) {
    // Merged bottom to top, ties by x (so the bottom left corner comes
    // first and the top right last), tagged with their chain.
    let mut v: Vec<(Vec2, usize)> = chains[0]
        .iter()
        .map(|&p| (p, 0))
        .chain(chains[1].iter().map(|&p| (p, 1)))
        .collect();
    v.sort_by(|a, b| a.0.y.total_cmp(&b.0.y).then(a.0.x.total_cmp(&b.0.x)));
    let n = v.len();
    if n < 3 {
        return;
    }
    let mut stack = vec![v[0], v[1]];
    for &(p, chain) in &v[2..n - 1] {
        let &(_, top_chain) = stack.last().expect("two");
        if chain != top_chain {
            for w in stack.windows(2) {
                out([p, w[0].0, w[1].0]);
            }
            let last = *stack.last().expect("two");
            stack.clear();
            stack.push(last);
            stack.push((p, chain));
        } else {
            let mut last = stack.pop().expect("two");
            while let Some(&below) = stack.last() {
                let turn = (last.0 - below.0).perp_dot(p - last.0);
                let convex = if chain == 0 { turn < 0.0 } else { turn > 0.0 };
                if !convex {
                    break;
                }
                out([below.0, last.0, p]);
                last = stack.pop().expect("one");
            }
            stack.push(last);
            stack.push((p, chain));
        }
    }
    let p = v[n - 1].0;
    for w in stack.windows(2) {
        out([p, w[0].0, w[1].0]);
    }
}

/// `surface` with each triangle replaced by the part of it where `texture`'s
/// alpha (bilinear, repeating, through the triangle's texture coordinates)
/// is at least `cutoff`. Triangles without texture coordinates, or with
/// degenerate ones, are kept whole.
pub fn cut_out(surface: &Surface, texture: &Texture, cutoff: f32) -> Surface {
    cut_out_colored(surface, None, texture, [Wrap::Repeat; 2], cutoff).0
}

/// [`cut_out`], with the surface's vertex colors (per triangle corner, M10
/// gate 17) interpolated into the triangles it makes, and the texture read
/// with wrap modes `wrap` (M10 gate 20).
pub(crate) fn cut_out_colored(
    surface: &Surface,
    colors: Option<&[[Vec3; 3]]>,
    texture: &Texture,
    wrap: [Wrap; 2],
    cutoff: f32,
) -> (Surface, Option<Vec<[Vec3; 3]>>) {
    cut_out_with(surface, colors, texture, wrap, cutoff, Some(TOLERANCE))
}

/// [`cut_out_colored`], its contours simplified to within `tolerance`
/// cells, or emitted cell by cell (`None`: the exact contour, many more
/// triangles).
fn cut_out_with(
    surface: &Surface,
    colors: Option<&[[Vec3; 3]]>,
    texture: &Texture,
    wrap: [Wrap; 2],
    cutoff: f32,
    tolerance: Option<f32>,
) -> (Surface, Option<Vec<[Vec3; 3]>>) {
    let colors = colors.filter(|c| c.len() == surface.triangles.len());
    if surface.tex.len() != surface.triangles.len() {
        return (surface.clone(), colors.map(|c| c.to_vec()));
    }
    let mips = AlphaMips::new(texture, wrap);
    // A texture opaque everywhere cuts nothing (files often mark opaque
    // materials as masked: Sponza's walls went from 55,592 triangles to 19
    // million).
    if mips.levels[0].2.iter().all(|&a| a >= cutoff) {
        return (surface.clone(), colors.map(|c| c.to_vec()));
    }
    let (tw, th) = (texture.width.max(1) as f32, texture.height.max(1) as f32);
    // Pieces keep their triangle's chart, at chart coordinates interpolated
    // from its (so a cut adds no charts: Sponza's ivy, re-charted, went from
    // 18,899 charts to 666,277 and its frames to 120 ms).
    let charted = surface.chart.len() == surface.triangles.len()
        && surface.uv.len() == surface.triangles.len();
    let cut = |range: std::ops::Range<usize>| -> Parts {
        let mut parts = Parts::default();
        let Parts {
            tris,
            normals,
            tex,
            colors: out_colors,
            chart,
            uv: chart_uv,
        } = &mut parts;
        let mut grid_scratch = Contours::default();
        for k in range {
            let (p, n, t) = (surface.triangles[k], surface.normals[k], surface.tex[k]);
            let c = colors.map_or([Vec3::ONE; 3], |c| c[k]);
            let (ck, cu) = if charted {
                (surface.chart[k], surface.uv[k])
            } else {
                (0, [Vec2::ZERO; 3])
            };
            let det = (t[1] - t[0]).perp_dot(t[2] - t[0]);
            let geometric = (p[1] - p[0]).cross(p[2] - p[0]);
            if det.abs() < 1e-12 || geometric.length_squared() == 0.0 {
                tris.push(p);
                normals.push(n);
                tex.push(t);
                out_colors.push(c);
                chart.push(ck);
                chart_uv.push(cu);
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
                        c[0] * l0 + c[1] * l1 + c[2] * l2,
                        cu[0] * l0 + cu[1] * l1 + cu[2] * l2,
                    )
                };
                for m in 1..q.len() - 1 {
                    let corners = [q[0], q[m], q[m + 1]];
                    let mapped = corners.map(map);
                    let mut pos = mapped.map(|c| c.0);
                    let mut nor = mapped.map(|c| c.1);
                    let mut col = mapped.map(|c| c.2);
                    let mut cuv = mapped.map(|c| c.3);
                    let mut uv = corners;
                    let cross = (pos[1] - pos[0]).cross(pos[2] - pos[0]);
                    if cross.length_squared() <= 1e-20 * geometric.length_squared() {
                        continue;
                    }
                    if cross.dot(geometric) < 0.0 {
                        pos.swap(1, 2);
                        nor.swap(1, 2);
                        col.swap(1, 2);
                        cuv.swap(1, 2);
                        uv.swap(1, 2);
                    }
                    chart.push(ck);
                    chart_uv.push(cuv);
                    tris.push(pos);
                    normals.push(nor);
                    tex.push(uv);
                    out_colors.push(col);
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
            // The grid's corners all on one side: the whole triangle, kept as
            // it is (the cells' rectangles clipped to it would be just it), or
            // none of it.
            let mut any_in = false;
            let mut any_out = false;
            'corners: for j in j0..=j1 + 1 {
                for i in i0..=i1 + 1 {
                    if value(i, j) >= 0.0 {
                        any_in = true;
                    } else {
                        any_out = true;
                    }
                    if any_in && any_out {
                        break 'corners;
                    }
                }
            }
            if !any_out {
                tris.push(p);
                normals.push(n);
                tex.push(t);
                out_colors.push(c);
                chart.push(ck);
                chart_uv.push(cu);
                continue;
            }
            if !any_in {
                continue;
            }
            if let Some(tolerance) = tolerance {
                let rings = grid_scratch.trace(i0, i1, j0, j1, &value);
                let rings: Vec<Vec<Vec2>> = rings
                    .iter()
                    .map(|r| simplify(r, tolerance))
                    .filter(|r| r.len() >= 3)
                    .collect();
                // Local cell coordinates (the padding cell at 0) to texture
                // coordinates.
                let to_uv = |x: Vec2| {
                    Vec2::new(
                        (x.x + (i0 - 1) as f32 + 0.5) / lw,
                        (x.y + (j0 - 1) as f32 + 0.5) / lh,
                    )
                };
                let clip = t.map(|x| {
                    Vec2::new(
                        x.x * lw - 0.5 - (i0 - 1) as f32,
                        x.y * lh - 0.5 - (j0 - 1) as f32,
                    )
                });
                fill(&rings, clip, |q| emit(&q.map(to_uv)));
                continue;
            }
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
                        let (a, b) = if (corners[a].1, corners[a].0) < (corners[b].1, corners[b].0)
                        {
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
        parts
    };
    // In parallel over runs of triangles, joined in order.
    let count = surface.triangles.len();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(count.div_ceil(64))
        .max(1);
    let chunk = count.div_ceil(threads);
    let parts: Vec<Parts> = std::thread::scope(|scope| {
        let cut = &cut;
        let handles: Vec<_> = (0..threads)
            .map(|c| scope.spawn(move || cut(c * chunk..((c + 1) * chunk).min(count))))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a cut"))
            .collect()
    });
    let mut all = Parts::default();
    for p in parts {
        all.tris.extend(p.tris);
        all.normals.extend(p.normals);
        all.tex.extend(p.tex);
        all.colors.extend(p.colors);
        all.chart.extend(p.chart);
        all.uv.extend(p.uv);
    }
    if !charted {
        let out_colors = colors.map(|_| {
            all.tris
                .iter()
                .zip(all.colors)
                .filter(|(t, _)| !continuum::mesh::degenerate(t))
                .map(|(_, c)| c)
                .collect()
        });
        return (
            Surface::from_mesh(all.tris, Some(all.normals), Some(all.tex)),
            out_colors,
        );
    }
    let keep: Vec<usize> = (0..all.tris.len())
        .filter(|&k| !continuum::mesh::degenerate(&all.tris[k]))
        .collect();
    let mut out = Surface {
        triangles: keep.iter().map(|&k| all.tris[k]).collect(),
        normals: keep.iter().map(|&k| all.normals[k]).collect(),
        chart: keep.iter().map(|&k| all.chart[k]).collect(),
        uv: keep.iter().map(|&k| all.uv[k]).collect(),
        charts: surface.charts.clone(),
        tex: keep.iter().map(|&k| all.tex[k]).collect(),
        chart_grid: Vec::new(),
    };
    // Each chart's grid for its pieces, as `Surface::from_mesh` sizes it
    // (about two triangles per cell).
    let mut members = vec![0usize; out.charts.len()];
    for &c in &out.chart {
        members[c as usize] += 1;
    }
    out.chart_grid = out
        .charts
        .iter()
        .zip(&members)
        .map(|(size, &m)| {
            let cells = (m as f32 / 2.0).max(1.0);
            let aspect = size.x / size.y.max(1e-12);
            let g = |x: f32| x.sqrt().round().clamp(1.0, 512.0) as u32;
            [g(cells * aspect), g(cells / aspect)]
        })
        .collect();
    let out_colors = colors.map(|_| keep.iter().map(|&k| all.colors[k]).collect());
    (out, out_colors)
}

/// Triangles a run of the cut makes, with what each carries.
#[derive(Default)]
struct Parts {
    tris: Vec<Tri>,
    normals: Vec<[Vec3; 3]>,
    tex: Vec<[Vec2; 3]>,
    colors: Vec<[Vec3; 3]>,
    /// The chart and chart coordinates of the triangle cut.
    chart: Vec<u32>,
    uv: Vec<[Vec2; 3]>,
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

    /// Blobs, thin strands and holes: a sum of waves, thresholded.
    fn foliage(size: u32) -> Texture {
        texture(size, |x, y| {
            let (u, v) = (x as f32 / size as f32, y as f32 / size as f32);
            let f = (u * 23.0).sin() * (v * 17.0 + u * 5.0).cos()
                + 0.6 * (u * 61.0 + v * 7.0).sin()
                + 0.4 * (v * 97.0).cos();
            (f.clamp(-1.0, 1.0) * 127.5 + 127.5) as u8
        })
    }

    /// Triangles at odd angles over the texture, some beyond [0, 1]
    /// (repeating) and some spanning more texels than [`MAX_CELLS`].
    fn scattered() -> Surface {
        let mut tris = Vec::new();
        let mut uvs = Vec::new();
        let mut seed = 7u32;
        let mut r = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..60 {
            let c = Vec2::new(r() * 2.0 - 0.5, r() * 2.0 - 0.5);
            let s = 0.02 + r() * 0.6;
            let uv: [Vec2; 3] = std::array::from_fn(|_| c + Vec2::new(r() - 0.5, r() - 0.5) * s);
            let p = uv.map(|t| Vec3::new(t.x * 3.0, t.y * 2.0 + t.x, 0.5 * t.y));
            tris.push(p);
            uvs.push(uv);
        }
        Surface::from_mesh(tris, None, Some(uvs))
    }

    #[test]
    fn simplified_contours_keep_the_cut() {
        let (s, t) = (scattered(), foliage(256));
        let exact = cut_out_with(&s, None, &t, [Wrap::Repeat; 2], 0.5, None).0;
        let traced = cut_out_with(&s, None, &t, [Wrap::Repeat; 2], 0.5, Some(0.0)).0;
        let simple = cut_out(&s, &t, 0.5);
        let (a, b, c) = (area(&exact), area(&traced), area(&simple));
        assert!(
            (b - a).abs() < 1e-4 * a,
            "traced {b} against cell by cell {a}"
        );
        assert!((c - a).abs() < 0.01 * a, "simplified {c} against {a}");
        assert!(
            simple.triangles.len() * 3 < exact.triangles.len(),
            "{} triangles against {}",
            simple.triangles.len(),
            exact.triangles.len()
        );
        // Each piece faces its triangle's way.
        for t in &simple.triangles {
            let n = (t[1] - t[0]).cross(t[2] - t[0]);
            assert!(n.length() > 0.0);
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
