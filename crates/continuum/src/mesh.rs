//! Triangle meshes with what lighting caches need: shading normals and a chart
//! layout.
//!
//! A chart is a rectangle in a 2D parameter space. Every triangle lies in one
//! chart, and each of its vertices has coordinates in that chart in `[0, 1]²`.
//! Lighting caches lay a texel grid over each chart and place texels on the
//! surface through these coordinates.

use std::collections::HashMap;

use glam::{Mat3, Vec2, Vec3};

use crate::cone3::box_mesh;

pub type Tri = [Vec3; 3];

/// A triangle mesh with shading normals and a chart layout.
#[derive(Clone, Debug, Default)]
pub struct Surface {
    pub triangles: Vec<Tri>,
    /// Shading normal at each vertex of each triangle (unit length).
    pub normals: Vec<[Vec3; 3]>,
    /// The chart each triangle lies in.
    pub chart: Vec<u32>,
    /// Chart coordinates of each triangle's vertices, in `[0, 1]²`.
    pub uv: Vec<[Vec2; 3]>,
    /// World-space size of each chart along u and v, which sets texel counts.
    pub charts: Vec<Vec2>,
}

fn flat_normal(t: &Tri) -> Vec3 {
    (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero()
}

impl Surface {
    /// Triangles with flat normals. A consecutive pair `[a, b, c]`, `[a, c, d]`
    /// that forms a parallelogram (as [`box_mesh`] emits each face) shares one
    /// chart; any other triangle gets a chart of its own.
    pub fn from_triangles(triangles: Vec<Tri>) -> Surface {
        let mut s = Surface::default();
        let mut i = 0;
        while i < triangles.len() {
            let t = triangles[i];
            let n = flat_normal(&t);
            let chart = s.charts.len() as u32;
            if let Some(&u) = triangles.get(i + 1)
                && u[0] == t[0]
                && u[1] == t[2]
                && {
                    let (a, b, c, d) = (t[0], t[1], t[2], u[2]);
                    let scale = (c - a).length().max(1e-12);
                    ((a + c) - (b + d)).length() <= 1e-5 * scale
                }
            {
                let (a, b, d) = (t[0], t[1], u[2]);
                s.charts.push(Vec2::new((b - a).length(), (d - a).length()));
                for (tri, uv) in [
                    (t, [Vec2::ZERO, Vec2::X, Vec2::ONE]),
                    (u, [Vec2::ZERO, Vec2::ONE, Vec2::Y]),
                ] {
                    s.triangles.push(tri);
                    s.normals.push([flat_normal(&tri); 3]);
                    s.chart.push(chart);
                    s.uv.push(uv);
                }
                i += 2;
            } else {
                s.charts
                    .push(Vec2::new((t[1] - t[0]).length(), (t[2] - t[0]).length()));
                s.triangles.push(t);
                s.normals.push([n; 3]);
                s.chart.push(chart);
                s.uv.push([Vec2::ZERO, Vec2::X, Vec2::Y]);
                i += 1;
            }
        }
        s
    }

    /// An axis-aligned box: 12 outward triangles, one chart per face.
    pub fn cuboid(min: Vec3, max: Vec3) -> Surface {
        Surface::from_triangles(box_mesh(min, max))
    }

    /// A smooth parametric surface `f(u, v)` over `[0, 1]²`, as an `nu × nv`
    /// grid of quads in one chart. Triangles face along `∂f/∂u × ∂f/∂v`;
    /// degenerate ones (at poles) are dropped. Shading normals average the face
    /// normals around each vertex, merged by position, so seams and poles are
    /// smooth.
    pub fn parametric(nu: usize, nv: usize, f: impl Fn(f32, f32) -> Vec3) -> Surface {
        let p = |i: usize, j: usize| f(i as f32 / nu as f32, j as f32 / nv as f32);
        let grid: Vec<Vec<Vec3>> = (0..=nu)
            .map(|i| (0..=nv).map(|j| p(i, j)).collect())
            .collect();
        let along_u = (0..=nv)
            .map(|j| {
                (0..nu)
                    .map(|i| (grid[i + 1][j] - grid[i][j]).length())
                    .sum::<f32>()
            })
            .fold(0.0, f32::max);
        let along_v = (0..=nu)
            .map(|i| {
                (0..nv)
                    .map(|j| (grid[i][j + 1] - grid[i][j]).length())
                    .sum::<f32>()
            })
            .fold(0.0, f32::max);
        let mut s = Surface {
            charts: vec![Vec2::new(along_u, along_v)],
            ..Default::default()
        };
        let uv = |i: usize, j: usize| Vec2::new(i as f32 / nu as f32, j as f32 / nv as f32);
        for i in 0..nu {
            for j in 0..nv {
                let (a, b, c, d) = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1));
                for [x, y, z] in [[a, b, c], [a, c, d]] {
                    let tri = [grid[x.0][x.1], grid[y.0][y.1], grid[z.0][z.1]];
                    let scale = (tri[1] - tri[0]).length().max((tri[2] - tri[0]).length());
                    if (tri[1] - tri[0]).cross(tri[2] - tri[0]).length() <= 1e-6 * scale * scale {
                        continue;
                    }
                    s.triangles.push(tri);
                    s.uv.push([uv(x.0, x.1), uv(y.0, y.1), uv(z.0, z.1)]);
                    s.chart.push(0);
                }
            }
        }
        s.normals = smooth_normals(&s.triangles);
        s
    }

    /// This surface moved rigidly: rotated by `rotation` about the origin, then
    /// translated.
    pub fn transformed(&self, rotation: Mat3, translation: Vec3) -> Surface {
        Surface {
            triangles: self
                .triangles
                .iter()
                .map(|t| t.map(|p| rotation * p + translation))
                .collect(),
            normals: self
                .normals
                .iter()
                .map(|n| n.map(|v| (rotation * v).normalize()))
                .collect(),
            ..self.clone()
        }
    }

    /// The same surface facing the other way: windings and normals reversed.
    pub fn flipped(&self) -> Surface {
        Surface {
            triangles: self.triangles.iter().map(|t| [t[0], t[2], t[1]]).collect(),
            normals: self.normals.iter().map(|n| [-n[0], -n[2], -n[1]]).collect(),
            uv: self.uv.iter().map(|uv| [uv[0], uv[2], uv[1]]).collect(),
            ..self.clone()
        }
    }

    /// Axis-aligned bounds of the triangles.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.triangles.iter().flatten().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), &p| (lo.min(p), hi.max(p)),
        )
    }
}

/// Per-vertex normals: area-weighted face normals summed over every triangle
/// sharing the vertex's position.
fn smooth_normals(triangles: &[Tri]) -> Vec<[Vec3; 3]> {
    let key = |p: Vec3| {
        let q = (p * 1e5).round();
        (q.x as i64, q.y as i64, q.z as i64)
    };
    let mut sums: HashMap<(i64, i64, i64), Vec3> = HashMap::new();
    for t in triangles {
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        for &p in t {
            *sums.entry(key(p)).or_default() += n;
        }
    }
    triangles
        .iter()
        .map(|t| t.map(|p| sums[&key(p)].normalize_or_zero()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_faces_pair_into_quad_charts() {
        let s = Surface::cuboid(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(s.triangles.len(), 12);
        assert_eq!(s.charts.len(), 6);
        for (k, t) in s.triangles.iter().enumerate() {
            let size = s.charts[s.chart[k] as usize];
            // Chart coordinates map to positions by an affine map with the
            // chart's size: check edge lengths along u and v.
            let [u0, u1, u2] = s.uv[k];
            for (a, b) in [(0, 1), (1, 2), (0, 2)] {
                let (pa, pb) = (t[a], t[b]);
                let d = [u0, u1, u2][b] - [u0, u1, u2][a];
                let expected = (d * size).length();
                assert!(((pb - pa).length() - expected).abs() < 1e-5);
            }
            assert!(s.normals[k].iter().all(|n| (n.length() - 1.0).abs() < 1e-6));
        }
    }

    #[test]
    fn lone_triangles_get_their_own_charts() {
        let t = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let s = Surface::from_triangles(vec![t, [Vec3::ZERO, Vec3::Y, Vec3::Z]]);
        assert_eq!(s.charts.len(), 2);
        assert_eq!(s.uv[0], [Vec2::ZERO, Vec2::X, Vec2::Y]);
    }

    #[test]
    fn parametric_sphere_is_smooth_and_outward() {
        use std::f32::consts::{PI, TAU};
        let c = Vec3::new(1.0, 2.0, 3.0);
        let s = Surface::parametric(32, 16, |u, v| {
            let (theta, phi) = (TAU * u, PI * v);
            // u around, v from the south pole up, so ∂u × ∂v points outward.
            c + Vec3::new(
                (phi).sin() * theta.cos(),
                (phi).sin() * theta.sin(),
                -(phi).cos(),
            ) * 0.5
        });
        // 32 × 16 quads, minus one degenerate triangle per quad at each pole.
        assert_eq!(s.triangles.len(), 2 * 32 * 16 - 2 * 32);
        for (t, n) in s.triangles.iter().zip(&s.normals) {
            assert!((t[1] - t[0]).cross(t[2] - t[0]).dot(t[0] - c) > 0.0);
            for (p, n) in t.iter().zip(n) {
                assert!(n.dot((*p - c).normalize()) > 0.99, "{n} at {p}");
            }
        }
        assert!((s.charts[0].x - TAU * 0.5).abs() < 0.01);
        assert!((s.charts[0].y - PI * 0.5).abs() < 0.01);
    }
}
