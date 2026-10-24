//! 3D cones: rectangular pyramids with an 8×8 grid of direction bins.
//!
//! A cone has an apex and an orthonormal frame (`right`, `up`, `forward`). Its
//! cross-section is a window `[u0, u1] × [v0, v1]` on the plane one unit along
//! `forward`, split into 8×8 bins (bin `row * 8 + col`). Triangles are clipped to
//! the slab between a near distance and a far plane, projected onto the unit
//! plane, and the exact area of the projection inside each bin becomes that bin's
//! coverage in a [`CoverageMask`].
//!
//! Coverage within one mesh is summed over its front-facing triangles, which is
//! exact for convex meshes (their front faces tile the silhouette without
//! overlap) and conservative otherwise; across meshes it is combined by max, as
//! in 2D. Summing matters: with max alone, every internal edge of a mesh that
//! crosses a bin would leak light.
//!
//! Each bin is weighted by its projected solid angle for the receiver's normal
//! (Lambert's polygon formula, exact), so visibility sums directly to irradiance
//! per unit radiance.

use glam::{Vec2, Vec3};

use crate::CoverageMask;

/// Bins per side; a cone has `GRID * GRID` = 64 bins, one [`CoverageMask`] worth.
pub const GRID: usize = 8;
const BINS: usize = GRID * GRID;

/// `normal · x <= offset` is the kept side.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    /// Positive on the kept side.
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.offset - self.normal.dot(p)
    }
}

#[derive(Clone, Debug)]
pub struct PyramidCone {
    pub apex: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
    /// `[u0, v0, u1, v1]` on the plane one unit along `forward`.
    pub window: [f32; 4],
    pub near: f32,
    /// Geometry beyond this plane (for a light cone, the light itself) does not occlude.
    pub far: Plane,
    /// Projected solid angle of each bin for the receiver's normal.
    pub weights: [f32; BINS],
}

/// Projected solid angle ∫cos θ dω of a polygon, given as directions from the
/// receiver, for a receiver with unit normal `n` (Lambert's formula). Exact for
/// polygons above the receiver's horizon.
pub fn projected_solid_angle(n: Vec3, verts: &[Vec3]) -> f32 {
    let mut sum = 0.0;
    for i in 0..verts.len() {
        let a = verts[i].normalize();
        let b = verts[(i + 1) % verts.len()].normalize();
        let g = a.cross(b);
        let len = g.length();
        if len > 1e-12 {
            sum += a.dot(b).clamp(-1.0, 1.0).acos() * n.dot(g / len);
        }
    }
    (0.5 * sum).abs()
}

/// A convex polygon with fixed capacity (clipping a triangle by 6 planes adds at
/// most 6 vertices), so clipping never allocates.
#[derive(Clone, Copy)]
struct Poly<P: Copy> {
    v: [P; 10],
    n: usize,
}

impl<
    P: Copy
        + Default
        + std::ops::Add<Output = P>
        + std::ops::Sub<Output = P>
        + std::ops::Mul<f32, Output = P>,
> Poly<P>
{
    fn from(points: &[P]) -> Self {
        let mut v = [P::default(); 10];
        v[..points.len()].copy_from_slice(points);
        Poly { v, n: points.len() }
    }

    fn points(&self) -> &[P] {
        &self.v[..self.n]
    }

    /// Keeps the part where `d(p) >= 0` (Sutherland–Hodgman).
    fn clip(&self, d: impl Fn(P) -> f32) -> Self {
        let mut out = Poly {
            v: [P::default(); 10],
            n: 0,
        };
        for i in 0..self.n {
            let (a, b) = (self.v[i], self.v[(i + 1) % self.n]);
            let (da, db) = (d(a), d(b));
            if da >= 0.0 {
                out.v[out.n] = a;
                out.n += 1;
            }
            if (da >= 0.0) != (db >= 0.0) {
                out.v[out.n] = a + (b - a) * (da / (da - db));
                out.n += 1;
            }
        }
        out
    }
}

/// Accumulation buffer for exact-area coverage on the bin grid; the extra cells
/// absorb writes one past the end of the last row.
type Accumulator = [f32; BINS + 4];

/// Adds an edge's signed swept area to the cells it crosses (the accumulation
/// rasterizer used by font renderers such as font-rs). After every edge of a
/// closed polygon is drawn, a running sum over the buffer gives each cell's
/// covered area exactly. Points must lie in `[0, GRID]²`.
fn draw_edge(buf: &mut Accumulator, p0: Vec2, p1: Vec2) {
    if (p0.y - p1.y).abs() <= f32::EPSILON {
        return;
    }
    let (dir, p0, p1) = if p0.y < p1.y {
        (1.0, p0, p1)
    } else {
        (-1.0, p1, p0)
    };
    let dxdy = (p1.x - p0.x) / (p1.y - p0.y);
    let mut x = p0.x;
    let y_end = (p1.y.ceil() as usize).min(GRID);
    for y in (p0.y as usize)..y_end {
        let row = y * GRID;
        let dy = ((y + 1) as f32).min(p1.y) - (y as f32).max(p0.y);
        let x_next = x + dxdy * dy;
        let d = dy * dir;
        let (x0, x1) = if x < x_next { (x, x_next) } else { (x_next, x) };
        let x0_floor = x0.floor();
        let x0i = x0_floor as usize;
        let x1_ceil = x1.ceil();
        let x1i = x1_ceil as usize;
        if x1i <= x0i + 1 {
            let xmf = 0.5 * (x + x_next) - x0_floor;
            buf[row + x0i] += d - d * xmf;
            buf[row + x0i + 1] += d * xmf;
        } else {
            let s = (x1 - x0).recip();
            let x0f = x0 - x0_floor;
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
            let x1f = x1 - x1_ceil + 1.0;
            let am = 0.5 * s * x1f * x1f;
            buf[row + x0i] += d * a0;
            if x1i == x0i + 2 {
                buf[row + x0i + 1] += d * (1.0 - a0 - am);
            } else {
                let a1 = s * (1.5 - x0f);
                buf[row + x0i + 1] += d * (a1 - a0);
                for xi in x0i + 2..x1i - 1 {
                    buf[row + xi] += d * s;
                }
                let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                buf[row + x1i - 1] += d * (1.0 - a2 - am);
            }
            buf[row + x1i] += d * am;
        }
        x = x_next;
    }
}

impl PyramidCone {
    /// The cone from `apex` through a rectangular light (`corner` plus two
    /// perpendicular edges), with its window exactly the light's projection:
    /// `forward` is the light's plane normal, so the light projects to an
    /// axis-aligned rectangle. `None` if the apex is on or behind the light's plane.
    pub fn toward_rect_light(
        apex: Vec3,
        corner: Vec3,
        edge_u: Vec3,
        edge_v: Vec3,
        receiver_normal: Vec3,
    ) -> Option<Self> {
        let mut forward = edge_u.cross(edge_v).normalize();
        if forward.dot(corner - apex) < 0.0 {
            forward = -forward;
        }
        let dist = forward.dot(corner - apex);
        if dist <= 1e-6 {
            return None;
        }
        let right = edge_u.normalize();
        let up = forward.cross(right);
        let rel = corner - apex;
        let corners = [rel, rel + edge_u, rel + edge_v, rel + edge_u + edge_v];
        let (mut u0, mut v0, mut u1, mut v1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for c in corners {
            let (u, v) = (right.dot(c) / dist, up.dot(c) / dist);
            (u0, v0, u1, v1) = (u0.min(u), v0.min(v), u1.max(u), v1.max(v));
        }
        let mut cone = PyramidCone {
            apex,
            right,
            up,
            forward,
            window: [u0, v0, u1, v1],
            near: 1e-4,
            // Slightly in front of the light so the light's own surface never occludes.
            far: Plane {
                normal: forward,
                offset: forward.dot(apex) + dist * (1.0 - 1e-5),
            },
            weights: [0.0; BINS],
        };
        cone.weights = std::array::from_fn(|bin| {
            let (col, row) = ((bin % GRID) as f32, (bin / GRID) as f32);
            let corner_dir = |c: f32, r: f32| cone.direction(c / GRID as f32, r / GRID as f32);
            let quad = [
                corner_dir(col, row),
                corner_dir(col + 1.0, row),
                corner_dir(col + 1.0, row + 1.0),
                corner_dir(col, row + 1.0),
            ];
            projected_solid_angle(receiver_normal, &quad)
        });
        Some(cone)
    }

    /// Direction (not normalized) at normalized window position `(s, t)` in `[0, 1]²`.
    pub fn direction(&self, s: f32, t: f32) -> Vec3 {
        let [u0, v0, u1, v1] = self.window;
        self.forward + self.right * (u0 + s * (u1 - u0)) + self.up * (v0 + t * (v1 - v0))
    }

    pub fn total_weight(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// Unblocked projected solid angle: irradiance per unit radiance.
    pub fn visible(&self, mask: &CoverageMask) -> f32 {
        mask.visible_weighted(&self.weights)
    }

    /// Whether any part of a triangle could fall inside this cone's slab and window
    /// (a cheap conservative test before [`Self::add_mesh`]).
    pub fn may_touch(&self, tri: &[Vec3; 3]) -> bool {
        let local: Vec<Vec3> = tri.iter().map(|&p| self.to_local(p)).collect();
        if local.iter().all(|p| p.z < self.near) {
            return false;
        }
        let [u0, v0, u1, v1] = self.window;
        // Outside one of the four side planes of the pyramid for all vertices?
        let outside = |f: &dyn Fn(Vec3) -> f32| local.iter().all(|&p| f(p) < 0.0);
        !(outside(&|p| p.x - u0 * p.z)
            || outside(&|p| u1 * p.z - p.x)
            || outside(&|p| p.y - v0 * p.z)
            || outside(&|p| v1 * p.z - p.y))
    }

    fn to_local(&self, p: Vec3) -> Vec3 {
        let d = p - self.apex;
        Vec3::new(self.right.dot(d), self.up.dot(d), self.forward.dot(d))
    }

    /// Adds one mesh's triangles to `mask`. Triangles are wound counter-clockwise
    /// seen from outside; only those facing the apex count. Culling triangles that
    /// cannot touch the cone ([`Self::may_touch`]) is left to the caller.
    pub fn add_mesh<'a>(
        &self,
        mask: &mut CoverageMask,
        triangles: impl IntoIterator<Item = &'a [Vec3; 3]>,
    ) {
        let mut buf: Accumulator = [0.0; BINS + 4];
        let mut touched = false;
        for tri in triangles {
            let normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
            if normal.dot(self.apex - tri[0]) <= 0.0 {
                continue;
            }
            touched |= self.rasterize(&mut buf, tri);
        }
        if touched {
            let mut running = 0.0f32;
            for (bin, &delta) in buf[..BINS].iter().enumerate() {
                running += delta;
                let covered = running.abs();
                if covered > 1e-6 {
                    mask.add_bin(bin, covered.min(1.0));
                }
            }
        }
    }

    /// Clips a triangle to the cone's slab and window and draws its projection
    /// into the accumulation buffer (front-facing triangles of one mesh all
    /// project with the same winding, so their areas add).
    fn rasterize(&self, buf: &mut Accumulator, tri: &[Vec3; 3]) -> bool {
        let mut local = Poly::from(tri).clip(|p: Vec3| self.far.signed_distance(p));
        for p in &mut local.v[..local.n] {
            *p = self.to_local(*p);
        }
        let local = local.clip(|p: Vec3| p.z - self.near);
        if local.n < 3 {
            return false;
        }
        let [u0, v0, u1, v1] = self.window;
        let (su, sv) = (GRID as f32 / (u1 - u0), GRID as f32 / (v1 - v0));
        let mut projected = [Vec2::ZERO; 10];
        for (q, p) in projected.iter_mut().zip(local.points()) {
            *q = Vec2::new((p.x / p.z - u0) * su, (p.y / p.z - v0) * sv);
        }
        let g = GRID as f32;
        let poly = Poly {
            v: projected,
            n: local.n,
        }
        .clip(|p: Vec2| p.x)
        .clip(|p: Vec2| g - p.x)
        .clip(|p: Vec2| p.y)
        .clip(|p: Vec2| g - p.y);
        if poly.n < 3 {
            return false;
        }
        let pts = poly.points();
        let at = |i: usize| pts[i % pts.len()].clamp(Vec2::ZERO, Vec2::splat(g));
        for i in 0..pts.len() {
            draw_edge(buf, at(i), at(i + 1));
        }
        true
    }
}

/// Distance along `dir` from `origin` to triangle `tri`, if hit (Möller–Trumbore).
pub fn ray_triangle(origin: Vec3, dir: Vec3, tri: &[Vec3; 3]) -> Option<f32> {
    let (e1, e2) = (tri[1] - tri[0], tri[2] - tri[0]);
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - tri[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

/// An axis-aligned box as 12 outward-facing triangles.
pub fn box_mesh(min: Vec3, max: Vec3) -> Vec<[Vec3; 3]> {
    let c = |x: bool, y: bool, z: bool| {
        Vec3::new(
            if x { max.x } else { min.x },
            if y { max.y } else { min.y },
            if z { max.z } else { min.z },
        )
    };
    // Each face as a quad wound counter-clockwise seen from outside.
    let quads = [
        [
            c(false, false, false),
            c(false, true, false),
            c(true, true, false),
            c(true, false, false),
        ], // -z
        [
            c(false, false, true),
            c(true, false, true),
            c(true, true, true),
            c(false, true, true),
        ], // +z
        [
            c(false, false, false),
            c(true, false, false),
            c(true, false, true),
            c(false, false, true),
        ], // -y
        [
            c(false, true, false),
            c(false, true, true),
            c(true, true, true),
            c(true, true, false),
        ], // +y
        [
            c(false, false, false),
            c(false, false, true),
            c(false, true, true),
            c(false, true, false),
        ], // -x
        [
            c(true, false, false),
            c(true, true, false),
            c(true, true, true),
            c(true, false, true),
        ], // +x
    ];
    quads
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor_cone(p: Vec3) -> PyramidCone {
        // A 0.2 × 0.2 light at height 1, centered over (0.5, 0.5).
        PyramidCone::toward_rect_light(
            p,
            Vec3::new(0.4, 0.4, 1.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::Z,
        )
        .unwrap()
    }

    /// Reference: stratified area sampling of the light, cosine at both ends.
    fn sampled_visible(p: Vec3, meshes: &[Vec<[Vec3; 3]>], n: usize) -> f32 {
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                let q = Vec3::new(
                    0.4 + 0.2 * (i as f32 + 0.5) / n as f32,
                    0.4 + 0.2 * (j as f32 + 0.5) / n as f32,
                    1.0,
                );
                let d = q - p;
                let r2 = d.length_squared();
                let blocked = meshes
                    .iter()
                    .flatten()
                    .any(|t| ray_triangle(p, d, t).is_some_and(|t| t < 1.0));
                if !blocked {
                    let cos = d.z / r2.sqrt();
                    sum += cos * cos / r2 * (0.04 / (n * n) as f32);
                }
            }
        }
        sum
    }

    #[test]
    fn bin_weights_sum_to_the_lights_projected_solid_angle() {
        for p in [
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.1, 0.8, 0.0),
            Vec3::new(0.9, 0.3, 0.2),
        ] {
            let cone = floor_cone(p);
            let reference = sampled_visible(p, &[], 256);
            assert!(
                (cone.total_weight() - reference).abs() / reference < 1e-3,
                "{p}: {} vs {reference}",
                cone.total_weight()
            );
        }
    }

    #[test]
    fn a_box_blocking_everything_leaves_nothing() {
        let p = Vec3::new(0.5, 0.5, 0.0);
        let cone = floor_cone(p);
        let mut m = CoverageMask::EMPTY;
        cone.add_mesh(
            &mut m,
            &box_mesh(Vec3::new(0.2, 0.2, 0.4), Vec3::new(0.8, 0.8, 0.5)),
        );
        assert!(cone.visible(&m) < 1e-6);
    }

    #[test]
    fn mask_cone_matches_sampled_visibility() {
        let meshes = vec![
            box_mesh(Vec3::new(0.35, 0.3, 0.4), Vec3::new(0.5, 0.7, 0.45)),
            box_mesh(Vec3::new(0.55, 0.45, 0.6), Vec3::new(0.62, 0.52, 0.7)),
        ];
        for p in [
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.42, 0.55, 0.0),
            Vec3::new(0.7, 0.4, 0.0),
        ] {
            let cone = floor_cone(p);
            let mut m = CoverageMask::EMPTY;
            for mesh in &meshes {
                cone.add_mesh(&mut m, mesh);
            }
            let (masked, reference) = (cone.visible(&m), sampled_visible(p, &meshes, 128));
            assert!(
                (masked - reference).abs() < 0.02 * cone.total_weight(),
                "{p}: mask {masked} vs rays {reference}"
            );
        }
    }

    #[test]
    fn internal_mesh_edges_do_not_leak() {
        // A box whose silhouette covers half the window: its two front triangles
        // split many bins between them.
        let p = Vec3::new(0.5, 0.5, 0.0);
        let cone = floor_cone(p);
        let mut m = CoverageMask::EMPTY;
        cone.add_mesh(
            &mut m,
            &box_mesh(Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.5, 1.0, 0.55)),
        );
        let left_half: f32 = (0..BINS)
            .filter(|b| b % GRID < GRID / 2)
            .map(|b| cone.weights[b])
            .sum();
        assert!(
            (cone.visible(&m) - (cone.total_weight() - left_half)).abs()
                < 1e-4 * cone.total_weight()
        );
    }

    #[test]
    fn geometry_beyond_the_light_does_not_occlude() {
        let p = Vec3::new(0.5, 0.5, 0.0);
        let cone = floor_cone(p);
        let mut m = CoverageMask::EMPTY;
        cone.add_mesh(
            &mut m,
            &box_mesh(Vec3::new(0.2, 0.2, 1.1), Vec3::new(0.8, 0.8, 1.2)),
        );
        assert!((cone.visible(&m) - cone.total_weight()).abs() < 1e-6);
    }
}
