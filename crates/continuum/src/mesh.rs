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

/// A triangle's three vertices.
pub type Tri = [Vec3; 3];

/// A triangle mesh with shading normals and a chart layout.
#[derive(Clone, Debug, Default)]
pub struct Surface {
    /// The triangles, in the mesh's own coordinates.
    pub triangles: Vec<Tri>,
    /// Shading normal at each vertex of each triangle (unit length).
    pub normals: Vec<[Vec3; 3]>,
    /// The chart each triangle lies in.
    pub chart: Vec<u32>,
    /// Chart coordinates of each triangle's vertices, in `[0, 1]²`.
    pub uv: Vec<[Vec2; 3]>,
    /// World-space size of each chart along u and v, which sets texel counts.
    pub charts: Vec<Vec2>,
    /// Texture coordinates of each triangle's vertices (glTF's TEXCOORD_0),
    /// or empty without a texture.
    pub tex: Vec<[Vec2; 3]>,
    /// Each chart's grid of cells: chart coordinates split into `x × y` equal
    /// cells, for finding the triangle under a chart coordinate. A renderer
    /// lists in each cell the triangles whose chart coordinates' bounds meet
    /// it: two for a parametric quad, a few for automatic charts.
    pub chart_grid: Vec<[u32; 2]>,
}

/// Charts of [`Surface::from_mesh`] hold triangles whose normals lie within
/// this angle of their first triangle's.
pub const CHART_ANGLE_DEGREES: f32 = 45.0;

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
                s.chart_grid.push([1, 1]);
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
                s.chart_grid.push([1, 1]);
                s.triangles.push(t);
                s.normals.push([n; 3]);
                s.chart.push(chart);
                s.uv.push([Vec2::ZERO, Vec2::X, Vec2::Y]);
                i += 1;
            }
        }
        s
    }

    /// A general triangle mesh (imported assets, M6) with automatic charts.
    ///
    /// - Charts grow from a seed triangle across shared edges, taking
    ///   triangles whose normals lie within [`CHART_ANGLE_DEGREES`] of the
    ///   seed's.
    /// - Each chart is projected onto the plane facing the seed's normal, so
    ///   no triangle flips, and chart coordinates stretch distances by at
    ///   most 1 / cos 45°.
    /// - Shading normals are `normals` where given (per triangle corner), or
    ///   smooth normals merged by position over the triangles there that
    ///   face the same side (back-to-back faces stay apart).
    ///
    /// Degenerate triangles are dropped.
    pub fn from_mesh(
        triangles: Vec<Tri>,
        normals: Option<Vec<[Vec3; 3]>>,
        tex: Option<Vec<[Vec2; 3]>>,
    ) -> Surface {
        let keep: Vec<usize> = (0..triangles.len())
            .filter(|&k| {
                let t = triangles[k];
                let scale = (t[1] - t[0]).length().max((t[2] - t[0]).length());
                (t[1] - t[0]).cross(t[2] - t[0]).length() > 1e-6 * scale * scale
            })
            .collect();
        let tris: Vec<Tri> = keep.iter().map(|&k| triangles[k]).collect();
        let face: Vec<Vec3> = tris.iter().map(flat_normal).collect();
        // Shared edges, by quantized position.
        let key = |p: Vec3| {
            let q = (p * 1e5).round();
            (q.x as i64, q.y as i64, q.z as i64)
        };
        type Edge = ((i64, i64, i64), (i64, i64, i64));
        let mut edges: HashMap<Edge, Vec<u32>> = HashMap::new();
        for (k, t) in tris.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (key(t[e]), key(t[(e + 1) % 3]));
                let edge = if a < b { (a, b) } else { (b, a) };
                edges.entry(edge).or_default().push(k as u32);
            }
        }
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); tris.len()];
        for list in edges.values() {
            for &a in list {
                for &b in list {
                    if a != b {
                        neighbors[a as usize].push(b);
                    }
                }
            }
        }
        let limit = CHART_ANGLE_DEGREES.to_radians().cos();
        let mut chart_of = vec![u32::MAX; tris.len()];
        let mut members: Vec<Vec<u32>> = Vec::new();
        let mut axes: Vec<Vec3> = Vec::new();
        for seed in 0..tris.len() {
            if chart_of[seed] != u32::MAX {
                continue;
            }
            let c = members.len() as u32;
            let axis = face[seed];
            chart_of[seed] = c;
            let mut list = vec![seed as u32];
            let mut next = 0;
            while next < list.len() {
                let t = list[next] as usize;
                next += 1;
                for &u in &neighbors[t] {
                    let u = u as usize;
                    if chart_of[u] == u32::MAX && face[u].dot(axis) >= limit {
                        chart_of[u] = c;
                        list.push(u as u32);
                    }
                }
            }
            members.push(list);
            axes.push(axis);
        }
        let mut s = Surface {
            triangles: tris.clone(),
            chart: chart_of.clone(),
            uv: vec![[Vec2::ZERO; 3]; tris.len()],
            ..Default::default()
        };
        for (list, axis) in members.iter().zip(&axes) {
            // A basis of the plane facing the axis, right-handed with it, so
            // front faces keep their winding in chart coordinates.
            let helper = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let t = axis.cross(helper).normalize();
            let b = axis.cross(t);
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            for &k in list {
                for p in tris[k as usize] {
                    let q = Vec2::new(p.dot(t), p.dot(b));
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
            }
            let size = (hi - lo).max(Vec2::splat(1e-6));
            for &k in list {
                s.uv[k as usize] =
                    tris[k as usize].map(|p| (Vec2::new(p.dot(t), p.dot(b)) - lo) / size);
            }
            s.charts.push(size);
            // About two triangles per cell.
            let cells = (list.len() as f32 / 2.0).max(1.0);
            let aspect = size.x / size.y;
            let gx = (cells * aspect).sqrt().round().clamp(1.0, 512.0) as u32;
            let gy = (cells / aspect).sqrt().round().clamp(1.0, 512.0) as u32;
            s.chart_grid.push([gx, gy]);
        }
        s.normals = match normals {
            Some(n) => keep
                .iter()
                .map(|&k| n[k].map(|v| v.normalize_or_zero()))
                .collect(),
            None => smooth_normals(&s.triangles),
        };
        if let Some(t) = tex {
            s.tex = keep.iter().map(|&k| t[k]).collect();
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
            chart_grid: vec![[nu as u32, nv as u32]],
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
            tex: self.tex.iter().map(|t| [t[0], t[2], t[1]]).collect(),
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

/// Per-vertex normals: area-weighted face normals summed over the triangles
/// sharing the vertex's position that face the same side as the corner's
/// own (a positive dot of face normals). Back-to-back faces (a double-sided
/// card's front and back) stay apart instead of cancelling to zero (M10
/// gate 8).
fn smooth_normals(triangles: &[Tri]) -> Vec<[Vec3; 3]> {
    let key = |p: Vec3| {
        let q = (p * 1e5).round();
        (q.x as i64, q.y as i64, q.z as i64)
    };
    let face: Vec<Vec3> = triangles
        .iter()
        .map(|t| (t[1] - t[0]).cross(t[2] - t[0]))
        .collect();
    let mut at: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (k, t) in triangles.iter().enumerate() {
        for &p in t {
            at.entry(key(p)).or_default().push(k as u32);
        }
    }
    triangles
        .iter()
        .zip(&face)
        .map(|(t, n)| {
            t.map(|p| {
                at[&key(p)]
                    .iter()
                    .map(|&u| face[u as usize])
                    .filter(|f| f.dot(*n) > 0.0)
                    .sum::<Vec3>()
                    .normalize_or_zero()
            })
        })
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
    fn automatic_charts_cover_a_sphere_without_flips() {
        use std::f32::consts::{PI, TAU};
        let sphere = Surface::parametric(48, 24, |u, v| {
            let (theta, phi) = (TAU * u, PI * v);
            Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), -phi.cos())
        });
        let s = Surface::from_mesh(sphere.triangles.clone(), None, None);
        assert_eq!(s.triangles.len(), sphere.triangles.len());
        let limit = CHART_ANGLE_DEGREES.to_radians().cos();
        // Charts on a sphere: a handful, each within the angle of its seed.
        assert!(
            s.charts.len() > 4 && s.charts.len() < 64,
            "{} charts",
            s.charts.len()
        );
        for (k, t) in s.triangles.iter().enumerate() {
            let [a, b, c] = s.uv[k];
            // Chart coordinates in [0, 1]², counter-clockwise like the
            // triangle seen from outside (no flips).
            for q in [a, b, c] {
                assert!(q.min_element() >= -1e-5 && q.max_element() <= 1.0 + 1e-5);
            }
            assert!((b - a).perp_dot(c - a) > 0.0, "triangle {k} flipped");
            // Within the chart angle of the chart's first triangle.
            let first = s.chart.iter().position(|&x| x == s.chart[k]).unwrap();
            assert!(flat_normal(t).dot(flat_normal(&s.triangles[first])) >= limit - 1e-5);
        }
        // Every chart has a cell grid.
        assert_eq!(s.chart_grid.len(), s.charts.len());
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
