//! Coarser levels of a mesh (DESIGN §3.8 item 2.3): half-edge collapses
//! ordered by quadric error (Garland & Heckbert 1997), for meshes that come
//! without levels (imported ones).
//!
//! A half-edge collapse moves a vertex onto a neighbour, so a coarser level
//! keeps a subset of the finest level's vertices, with their shading normals
//! (the finest's normals at the coarse vertices, as M14 gate 13's rock
//! levels carry), chart coordinates and texture coordinates. Attribute
//! seams (a change of chart, chart or texture coordinates, or shading
//! normal across an edge) and open borders stay in place: a vertex on one
//! moves only along it, onto a neighbour on the same seam, and vertices
//! where seams meet never move. Collapses that would fold a triangle over
//! (in space, or in its chart's coordinates) or break the surface's
//! manifoldness are skipped.
//!
//! Each level states two sizes and the error level choice uses:
//! - [`Level::distance`]: a bound on the largest distance between it and
//!   the finest, both ways. Points on a grid over each triangle are measured
//!   to the other surface's nearest triangle, and each triangle adds the
//!   farthest any of its points lies from the grid (distance to a surface
//!   changes no faster than the point moves), up to f32 rounding.
//! - [`Level::spacing`]: its mean edge length.
//! - [`Level::error`]: the larger of the distance and half the spacing.
//!   Collapses ordered by quadric error keep the shape with few triangles
//!   (on the rocks, a third of the analytic levels' distance at the same
//!   count), so a level chosen by distance alone shows triangles several
//!   pixels across, and its vertices' normals miss the finest's shading
//!   between them. With half the spacing, a level shown where its error is
//!   half a pixel has triangles about a pixel across on average.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use glam::{DMat4, DVec3, DVec4, Vec2, Vec3};

use crate::mesh::{Surface, Tri};

/// A coarser level, its distance to the finest and its error.
#[derive(Clone, Debug)]
pub struct Level {
    /// The level, in the finest's charts (its `charts` and `chart_grid`
    /// copied).
    pub surface: Surface,
    /// A bound on the largest distance between this level and the finest,
    /// both ways, in the mesh's own units.
    pub distance: f32,
    /// Its mean edge length.
    pub spacing: f32,
    /// The larger of `distance` and half of `spacing`: what level choice
    /// compares with a pixel's size.
    pub error: f32,
}

/// Coarser levels of `finest` with at most each of `targets` triangles (in
/// decreasing order), each simplified from the one before; a level stops
/// early where no collapse remains allowed.
pub fn levels(finest: &Surface, targets: &[usize]) -> Vec<Level> {
    let mut mesh = Mesh::new(finest);
    let reference = Nearest::new(&finest.triangles);
    let mut out = Vec::new();
    for &target in targets {
        mesh.simplify(target);
        let surface = mesh.surface(finest);
        let distance = distance(&surface.triangles, &reference, &finest.triangles);
        let spacing = mean_edge(&surface.triangles);
        out.push(Level {
            surface,
            distance,
            spacing,
            error: distance.max(0.5 * spacing),
        });
    }
    out
}

/// The attributes a triangle corner carries (one "wedge" of a vertex).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Wedge {
    normal: Vec3,
    chart: u32,
    uv: Vec2,
    tex: Vec2,
}

impl Wedge {
    fn key(&self) -> [u32; 8] {
        [
            self.normal.x.to_bits(),
            self.normal.y.to_bits(),
            self.normal.z.to_bits(),
            self.chart,
            self.uv.x.to_bits(),
            self.uv.y.to_bits(),
            self.tex.x.to_bits(),
            self.tex.y.to_bits(),
        ]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Inside one attribute region: may move onto any neighbour.
    Interior,
    /// On one seam or border (exactly two such edges): may move along it.
    Seam,
    /// Never moves.
    Locked,
}

struct Mesh {
    positions: Vec<DVec3>,
    /// Per triangle, its vertices and corner wedges (indices into `wedges`);
    /// `alive` false once collapsed away.
    tris: Vec<[u32; 3]>,
    corners: Vec<[u32; 3]>,
    alive: Vec<bool>,
    wedges: Vec<Wedge>,
    /// Per vertex, the triangles around it (some may be dead).
    around: Vec<Vec<u32>>,
    quadric: Vec<DMat4>,
    /// Bumped whenever a vertex's neighbourhood changes, to invalidate
    /// queued collapses.
    version: Vec<u32>,
    removed: Vec<bool>,
    live: usize,
    has_tex: bool,
}

#[derive(PartialEq)]
struct Candidate {
    cost: f64,
    from: u32,
    to: u32,
    from_version: u32,
    to_version: u32,
}

impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    /// The cheapest first (a max-heap reversed).
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then(other.from.cmp(&self.from))
            .then(other.to.cmp(&self.to))
    }
}

/// The quadric of plane `n·x + d = 0`, weighted by `w`.
fn plane_quadric(n: DVec3, d: f64, w: f64) -> DMat4 {
    let p = DVec4::new(n.x, n.y, n.z, d);
    DMat4::from_cols(p * p.x, p * p.y, p * p.z, p * p.w) * w
}

fn quadric_cost(q: &DMat4, p: DVec3) -> f64 {
    let v = DVec4::new(p.x, p.y, p.z, 1.0);
    v.dot(*q * v).max(0.0)
}

impl Mesh {
    fn new(s: &Surface) -> Mesh {
        let mut vertex_of: HashMap<[u32; 3], u32> = HashMap::new();
        let mut wedge_of: HashMap<(u32, [u32; 8]), u32> = HashMap::new();
        let mut positions = Vec::new();
        let mut wedges = Vec::new();
        let mut tris = Vec::with_capacity(s.triangles.len());
        let mut corners = Vec::with_capacity(s.triangles.len());
        let has_tex = s.tex.len() == s.triangles.len();
        for (t, tri) in s.triangles.iter().enumerate() {
            let mut vs = [0u32; 3];
            let mut ws = [0u32; 3];
            for k in 0..3 {
                let p = tri[k];
                let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
                let v = *vertex_of.entry(key).or_insert_with(|| {
                    positions.push(p.as_dvec3());
                    positions.len() as u32 - 1
                });
                let wedge = Wedge {
                    normal: s.normals[t][k],
                    chart: s.chart[t],
                    uv: s.uv[t][k],
                    tex: if has_tex { s.tex[t][k] } else { Vec2::ZERO },
                };
                let w = *wedge_of.entry((v, wedge.key())).or_insert_with(|| {
                    wedges.push(wedge);
                    wedges.len() as u32 - 1
                });
                vs[k] = v;
                ws[k] = w;
            }
            tris.push(vs);
            corners.push(ws);
        }
        let n = positions.len();
        let mut around = vec![Vec::new(); n];
        for (t, vs) in tris.iter().enumerate() {
            for &v in vs {
                around[v as usize].push(t as u32);
            }
        }
        let mut mesh = Mesh {
            positions,
            alive: vec![true; tris.len()],
            live: tris.len(),
            tris,
            corners,
            wedges,
            around,
            quadric: vec![DMat4::ZERO; n],
            version: vec![0; n],
            removed: vec![false; n],
            has_tex,
        };
        mesh.init_quadrics();
        mesh
    }

    fn tri_normal(&self, t: usize) -> DVec3 {
        let [a, b, c] = self.tris[t].map(|v| self.positions[v as usize]);
        (b - a).cross(c - a)
    }

    fn init_quadrics(&mut self) {
        for t in 0..self.tris.len() {
            let n2 = self.tri_normal(t);
            let area2 = n2.length();
            if area2 <= 0.0 {
                continue;
            }
            let n = n2 / area2;
            let p = self.positions[self.tris[t][0] as usize];
            let q = plane_quadric(n, -n.dot(p), area2 * 0.5);
            for &v in &self.tris[t] {
                self.quadric[v as usize] += q;
            }
        }
        // Seams and borders keep their place: a plane through each special
        // edge, perpendicular to its triangle, heavily weighted.
        for t in 0..self.tris.len() {
            let n = self.tri_normal(t).normalize_or_zero();
            for k in 0..3 {
                let (a, b) = (self.tris[t][k], self.tris[t][(k + 1) % 3]);
                if !self.special_edge(a, b) {
                    continue;
                }
                let (pa, pb) = (self.positions[a as usize], self.positions[b as usize]);
                let e = pb - pa;
                let m = e.cross(n).normalize_or_zero();
                if m == DVec3::ZERO {
                    continue;
                }
                let q = plane_quadric(m, -m.dot(pa), 100.0 * e.length_squared());
                self.quadric[a as usize] += q;
                self.quadric[b as usize] += q;
            }
        }
    }

    /// The live triangles holding both `a` and `b`.
    fn shared(&self, a: u32, b: u32) -> Vec<u32> {
        self.around[a as usize]
            .iter()
            .copied()
            .filter(|&t| self.alive[t as usize] && self.tris[t as usize].contains(&b))
            .collect()
    }

    /// The wedge of vertex `v` in triangle `t`.
    fn wedge_in(&self, t: u32, v: u32) -> u32 {
        let k = self.tris[t as usize].iter().position(|&x| x == v).unwrap();
        self.corners[t as usize][k]
    }

    /// Whether edge `a–b` is a border, a seam, or non-manifold.
    fn special_edge(&self, a: u32, b: u32) -> bool {
        let s = self.shared(a, b);
        if s.len() != 2 {
            return true;
        }
        self.wedge_in(s[0], a) != self.wedge_in(s[1], a)
            || self.wedge_in(s[0], b) != self.wedge_in(s[1], b)
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &t in &self.around[v as usize] {
            if !self.alive[t as usize] {
                continue;
            }
            for &w in &self.tris[t as usize] {
                if w != v && !out.contains(&w) {
                    out.push(w);
                }
            }
        }
        out
    }

    /// The vertex's kind, and its special edges' other ends.
    fn kind(&self, v: u32) -> (Kind, Vec<u32>) {
        let mut special = Vec::new();
        for w in self.neighbours(v) {
            let s = self.shared(v, w).len();
            if s > 2 || s == 0 {
                return (Kind::Locked, special);
            }
            if self.special_edge(v, w) {
                special.push(w);
            }
        }
        match special.len() {
            0 => (Kind::Interior, special),
            2 => (Kind::Seam, special),
            _ => (Kind::Locked, special),
        }
    }

    fn push_candidates(&self, v: u32, heap: &mut BinaryHeap<Candidate>) {
        let (kind, special) = self.kind(v);
        let targets = match kind {
            Kind::Locked => return,
            Kind::Seam => special,
            Kind::Interior => self.neighbours(v),
        };
        for to in targets {
            let q = self.quadric[v as usize] + self.quadric[to as usize];
            heap.push(Candidate {
                cost: quadric_cost(&q, self.positions[to as usize]),
                from: v,
                to,
                from_version: self.version[v as usize],
                to_version: self.version[to as usize],
            });
        }
    }

    /// Whether moving `u` onto `v` keeps the surface manifold, unfolded in
    /// space and in chart coordinates; if so, the wedge each of `u`'s
    /// wedges becomes.
    fn check(&self, u: u32, v: u32) -> Option<Vec<(u32, u32)>> {
        let shared = self.shared(u, v);
        if shared.is_empty() || shared.len() > 2 {
            return None;
        }
        // Link condition: the neighbours common to u and v are exactly the
        // third vertices of the triangles on the edge.
        let nu = self.neighbours(u);
        let nv = self.neighbours(v);
        let common = nu.iter().filter(|w| nv.contains(w)).count();
        if common != shared.len() {
            return None;
        }
        // u's wedges map to v's wedge on the same side: read across each
        // triangle on the edge.
        let mut map: Vec<(u32, u32)> = Vec::new();
        for &t in &shared {
            let (wu, wv) = (self.wedge_in(t, u), self.wedge_in(t, v));
            if let Some(m) = map.iter().find(|m| m.0 == wu) {
                if m.1 != wv {
                    return None;
                }
            } else {
                map.push((wu, wv));
            }
        }
        let pv = self.positions[v as usize];
        for &t in &self.around[u as usize] {
            if !self.alive[t as usize] || shared.contains(&t) {
                continue;
            }
            let k = self.tris[t as usize].iter().position(|&x| x == u)?;
            let wu = self.corners[t as usize][k];
            // A wedge of u not on the edge: u is interior to it only if it
            // is u's one wedge.
            let wv = map.iter().find(|m| m.0 == wu)?.1;
            let mut p = self.tris[t as usize].map(|x| self.positions[x as usize]);
            let before = (p[1] - p[0]).cross(p[2] - p[0]);
            p[k] = pv;
            let after = (p[1] - p[0]).cross(p[2] - p[0]);
            let (lb, la) = (before.length(), after.length());
            if la <= 1e-12 * lb.max(1e-30) || before.dot(after) <= 0.25 * lb * la {
                return None;
            }
            // In chart coordinates too (the cache looks texels up through
            // them).
            let mut w = self.corners[t as usize].map(|c| self.wedges[c as usize].uv);
            let area = |w: &[Vec2; 3]| (w[1] - w[0]).perp_dot(w[2] - w[0]);
            let ab = area(&w);
            w[k] = self.wedges[wv as usize].uv;
            let aa = area(&w);
            if ab * aa <= 0.0 || aa.abs() < 1e-12 {
                return None;
            }
        }
        Some(map)
    }

    fn collapse(&mut self, u: u32, v: u32, map: &[(u32, u32)]) {
        let around_u = std::mem::take(&mut self.around[u as usize]);
        for &t in &around_u {
            let ti = t as usize;
            if !self.alive[ti] {
                continue;
            }
            if self.tris[ti].contains(&v) {
                self.alive[ti] = false;
                self.live -= 1;
                continue;
            }
            let k = self.tris[ti].iter().position(|&x| x == u).unwrap();
            let wu = self.corners[ti][k];
            self.tris[ti][k] = v;
            self.corners[ti][k] = map.iter().find(|m| m.0 == wu).map_or(wu, |m| m.1);
            self.around[v as usize].push(t);
        }
        self.around[v as usize].retain(|&t| self.alive[t as usize]);
        self.removed[u as usize] = true;
        let q = self.quadric[u as usize];
        self.quadric[v as usize] += q;
        self.version[v as usize] += 1;
        for w in self.neighbours(v) {
            self.version[w as usize] += 1;
        }
    }

    fn simplify(&mut self, target: usize) {
        let mut heap = BinaryHeap::new();
        for v in 0..self.positions.len() as u32 {
            if !self.removed[v as usize] {
                self.push_candidates(v, &mut heap);
            }
        }
        while self.live > target {
            let Some(c) = heap.pop() else { break };
            let (u, v) = (c.from, c.to);
            if self.removed[u as usize]
                || self.removed[v as usize]
                || self.version[u as usize] != c.from_version
                || self.version[v as usize] != c.to_version
            {
                continue;
            }
            let Some(map) = self.check(u, v) else {
                continue;
            };
            self.collapse(u, v, &map);
            let mut touched = self.neighbours(v);
            touched.push(v);
            for w in touched {
                self.push_candidates(w, &mut heap);
            }
        }
    }

    /// The live triangles as a surface in `finest`'s charts.
    fn surface(&self, finest: &Surface) -> Surface {
        let mut s = Surface {
            charts: finest.charts.clone(),
            chart_grid: finest.chart_grid.clone(),
            ..Default::default()
        };
        for t in 0..self.tris.len() {
            if !self.alive[t] {
                continue;
            }
            let p = self.tris[t].map(|v| self.positions[v as usize].as_vec3());
            let w = self.corners[t].map(|c| self.wedges[c as usize]);
            s.triangles.push(p);
            s.normals.push(w.map(|x| x.normal));
            s.chart.push(w[0].chart);
            s.uv.push(w.map(|x| x.uv));
            if self.has_tex {
                s.tex.push(w.map(|x| x.tex));
            }
        }
        s
    }
}

/// Nearest-point queries over triangles: a hierarchy of boxes.
struct Nearest {
    tris: Vec<Tri>,
    nodes: Vec<(Vec3, Vec3, u32, u32)>,
    order: Vec<u32>,
}

const NEAREST_LEAF: usize = 4;

impl Nearest {
    fn new(tris: &[Tri]) -> Nearest {
        let mut n = Nearest {
            tris: tris.to_vec(),
            nodes: Vec::new(),
            order: (0..tris.len() as u32).collect(),
        };
        if !tris.is_empty() {
            n.build(0, tris.len());
        }
        n
    }

    /// Builds the node over `order[start..end]` and returns its index: a
    /// leaf stores `(lo, hi, start, count)`, an inner node `(lo, hi, left,
    /// right | 1 << 31)`.
    fn build(&mut self, start: usize, end: usize) -> u32 {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &t in &self.order[start..end] {
            for p in self.tris[t as usize] {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        let index = self.nodes.len() as u32;
        self.nodes
            .push((lo, hi, start as u32, (end - start) as u32));
        if end - start <= NEAREST_LEAF {
            return index;
        }
        let e = hi - lo;
        let axis = if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        };
        let mid = (start + end) / 2;
        let tris = &self.tris;
        self.order[start..end].select_nth_unstable_by(mid - start, |a, b| {
            let ca = tris[*a as usize].iter().map(|p| p[axis]).sum::<f32>();
            let cb = tris[*b as usize].iter().map(|p| p[axis]).sum::<f32>();
            ca.total_cmp(&cb)
        });
        let left = self.build(start, mid);
        let right = self.build(mid, end);
        self.nodes[index as usize].2 = left;
        self.nodes[index as usize].3 = right | 0x8000_0000;
        index
    }

    /// The distance from `p` to the nearest triangle.
    fn distance(&self, p: Vec3) -> f32 {
        if self.nodes.is_empty() {
            return f32::MAX;
        }
        let mut best = f32::MAX;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        let box_distance = |n: u32| {
            let (lo, hi, _, _) = self.nodes[n as usize];
            (lo - p).max(p - hi).max(Vec3::ZERO).length_squared()
        };
        while sp > 0 {
            sp -= 1;
            let n = stack[sp];
            let (_, _, a, b) = self.nodes[n as usize];
            if box_distance(n) >= best * best {
                continue;
            }
            if b & 0x8000_0000 != 0 {
                // The nearer child last, so it is visited first.
                let (l, r) = (a, b & 0x7fff_ffff);
                let (near, far) = if box_distance(l) <= box_distance(r) {
                    (l, r)
                } else {
                    (r, l)
                };
                if sp + 2 <= stack.len() {
                    stack[sp] = far;
                    stack[sp + 1] = near;
                    sp += 2;
                }
            } else {
                for &t in &self.order[a as usize..(a + b) as usize] {
                    best = best.min(point_triangle(p, &self.tris[t as usize]));
                }
            }
        }
        best
    }
}

/// The distance from `p` to triangle `t` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn point_triangle(p: Vec3, t: &Tri) -> f32 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).length();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    (p - (a + ab * v + ac * w)).length()
}

fn mean_edge(tris: &[Tri]) -> f32 {
    let sum: f64 = tris
        .iter()
        .map(|t| ((t[1] - t[0]).length() + (t[2] - t[1]).length() + (t[0] - t[2]).length()) as f64)
        .sum();
    (sum / (3 * tris.len()).max(1) as f64) as f32
}

/// A bound on the largest distance from triangles `from` to the surface
/// `to`: per triangle, the farthest of its grid's points (at most `spacing`
/// apart along its longest edge, at least one division) plus the farthest
/// any of its points lies from the grid. A grid of `n` divisions an edge
/// leaves every point within the circumradius of its sub-triangles, at most
/// their longest edge over √3 for acute ones and half of it for obtuse
/// ones.
fn bound_to(from: &[Tri], to: &Nearest, spacing: f32) -> f32 {
    let mut worst = 0.0f32;
    for t in from {
        let e = (t[1] - t[0])
            .length()
            .max((t[2] - t[1]).length())
            .max((t[0] - t[2]).length());
        let n = ((e / spacing.max(1e-12)).ceil() as usize).clamp(1, 64);
        let mut far = 0.0f32;
        for i in 0..=n {
            for j in 0..=n - i {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                let p = t[0] + (t[1] - t[0]) * u + (t[2] - t[0]) * v;
                far = far.max(to.distance(p));
            }
        }
        worst = worst.max(far + e / (n as f32 * 3f32.sqrt()));
    }
    worst
}

/// A bound on the largest distance between `coarse` and `finest` (whose
/// nearest-point structure is `reference`), both ways: from the coarse
/// triangles on grids a quarter of their mean edge apart, from the finest
/// on grids half of theirs. It need only hold, not be tight: the error
/// takes half the spacing where that is larger.
fn distance(coarse: &[Tri], reference: &Nearest, finest: &[Tri]) -> f32 {
    let to_finest = bound_to(coarse, reference, 0.25 * mean_edge(coarse));
    let back = Nearest::new(coarse);
    let to_coarse = bound_to(finest, &back, 0.5 * mean_edge(finest));
    to_finest.max(to_coarse)
}

/// The largest distance between triangles `a` and `b`, both ways, from
/// `per_triangle` pseudo-random points on each triangle (uniform over it,
/// from seed `seed`) and every vertex: a denser check of [`Level::error`].
pub fn sampled_distance(a: &[Tri], b: &[Tri], per_triangle: usize, seed: u64) -> f32 {
    let one_way = |from: &[Tri], to: &[Tri], seed: u64| {
        let near = Nearest::new(to);
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32
        };
        let mut worst = 0.0f32;
        for t in from {
            for p in t {
                worst = worst.max(near.distance(*p));
            }
            for _ in 0..per_triangle {
                let (mut u, mut v) = (next(), next());
                if u + v > 1.0 {
                    (u, v) = (1.0 - u, 1.0 - v);
                }
                let p = t[0] + (t[1] - t[0]) * u + (t[2] - t[0]) * v;
                worst = worst.max(near.distance(p));
            }
        }
        worst
    };
    one_way(a, b, seed).max(one_way(b, a, seed ^ 0x9e37_79b9))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere of `n × n/2` quads, one chart per hemisphere side.
    fn sphere(n: usize) -> Surface {
        let mut tris = Vec::new();
        let p = |i: usize, j: usize| {
            let (th, ph) = (
                std::f32::consts::TAU * i as f32 / n as f32,
                std::f32::consts::PI * j as f32 / (n / 2) as f32,
            );
            Vec3::new(ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos())
        };
        for i in 0..n {
            for j in 0..n / 2 {
                let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
                if j > 0 {
                    tris.push([a, c, b]);
                }
                if j + 1 < n / 2 {
                    tris.push([a, d, c]);
                }
            }
        }
        Surface::from_mesh(tris, None, None)
    }

    #[test]
    fn levels_keep_vertices_and_bound_distance() {
        let s = sphere(64);
        let n = s.triangles.len();
        let out = levels(&s, &[n / 2, n / 8]);
        assert_eq!(out.len(), 2);
        let finest: std::collections::HashSet<[u32; 3]> = s
            .triangles
            .iter()
            .flatten()
            .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
            .collect();
        for l in &out {
            assert!(l.surface.triangles.len() < n);
            // Every coarse vertex is a finest vertex.
            for p in l.surface.triangles.iter().flatten() {
                assert!(finest.contains(&[p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]));
            }
            assert!(l.error > 0.0 && l.error < 0.2, "error {}", l.error);
        }
        assert!(out[1].error >= out[0].error);
        assert!(out[1].surface.triangles.len() <= n / 8 + n / 16);
    }
}
