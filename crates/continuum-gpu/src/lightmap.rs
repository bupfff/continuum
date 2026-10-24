//! The world-space lighting cache: texels and indirect-light probes placed on
//! static surfaces through their charts (see `continuum::mesh::Surface`).

use bytemuck::{Pod, Zeroable};
use continuum::glam::{Vec2, Vec3};
use continuum::mesh::Surface;

use crate::{RectLight, Tri};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ChartGpu {
    /// `[first texel, nu, nv, blocks per row]`
    pub info: [u32; 4],
    /// `[first probe, pu, pv, mesh]` (`pu = 0`: no probes)
    pub probes: [u32; 4],
}

/// A chart's triangles with their shading normals and chart coordinates.
struct ChartSource {
    tris: Vec<(Tri, [Vec3; 3], [Vec2; 3])>,
    size: Vec2,
}

/// A world-space lighting cache over the static meshes of a scene. Every chart
/// of a static [`Surface`] gets a grid of texels about `texel_size` apart, each
/// caching direct irradiance at its point on the surface; dynamic meshes have
/// none and are shaded per pixel.
///
/// Texels are stored in 8×8 blocks of 64 consecutive entries (charts are padded
/// to whole blocks; texels that land on no triangle have a zero normal and stay
/// dark), so a block can be re-baked as a unit when something moves near it.
pub struct Lightmap {
    pub(crate) charts: Vec<ChartGpu>,
    sources: Vec<ChartSource>,
    /// Chart of each triangle (global index), or `u32::MAX` for dynamic geometry.
    pub(crate) tri_chart: Vec<u32>,
    /// Chart coordinates of each triangle's vertices, three per triangle (zero
    /// for dynamic geometry).
    pub(crate) tri_uv: Vec<[f32; 2]>,
    /// Shading normals of each triangle's vertices, three per triangle.
    pub(crate) tri_normals: Vec<[f32; 4]>,
    /// Texel points nudged off the surface along the shading normal, with that
    /// normal: the receivers to bake.
    pub receivers: Vec<(Vec3, Vec3)>,
    /// Cached irradiance per texel (empty until baked).
    pub irradiance: Vec<f32>,
    /// Bounds of each block's texels (their receiver points).
    pub blocks: Vec<(Vec3, Vec3)>,
    /// Indirect-light probes `(point, normal)`, row-major per chart (see
    /// [`Self::with_probes`]).
    pub probes: Vec<(Vec3, Vec3)>,
    /// Indirect irradiance per probe per unit light radiance (RGB, then 1 if
    /// valid), for renders outside [`crate::LiveRenderer`]; empty for none.
    pub indirect: Vec<[f32; 4]>,
}

/// Texels per block side; a block is `BLOCK × BLOCK` consecutive texels.
pub const BLOCK: u32 = 8;

/// Planes per block in [`Lightmap::block_planes`].
pub const BLOCK_PLANES: usize = 16;

impl Lightmap {
    /// Texels about `texel_size` apart on every chart of the surfaces
    /// `is_static` selects. Every surface's triangles, in order, are the scene's
    /// meshes.
    pub fn new(
        surfaces: &[Surface],
        texel_size: f32,
        is_static: impl Fn(usize) -> bool,
    ) -> Lightmap {
        let mut lm = Lightmap {
            charts: Vec::new(),
            sources: Vec::new(),
            tri_chart: Vec::new(),
            tri_uv: Vec::new(),
            tri_normals: Vec::new(),
            receivers: Vec::new(),
            irradiance: Vec::new(),
            blocks: Vec::new(),
            probes: Vec::new(),
            indirect: Vec::new(),
        };
        for (m, s) in surfaces.iter().enumerate() {
            lm.tri_normals
                .extend(s.normals.iter().flatten().map(|n| [n.x, n.y, n.z, 0.0]));
            if !is_static(m) {
                lm.tri_chart
                    .extend(std::iter::repeat_n(u32::MAX, s.triangles.len()));
                lm.tri_uv
                    .extend(std::iter::repeat_n([0.0; 2], 3 * s.triangles.len()));
                continue;
            }
            let base = lm.charts.len();
            for &size in &s.charts {
                lm.charts.push(ChartGpu {
                    info: [0; 4],
                    probes: [0, 0, 0, m as u32],
                });
                lm.sources.push(ChartSource {
                    tris: Vec::new(),
                    size,
                });
            }
            for k in 0..s.triangles.len() {
                let chart = base + s.chart[k] as usize;
                lm.sources[chart]
                    .tris
                    .push((s.triangles[k], s.normals[k], s.uv[k]));
                lm.tri_chart.push(chart as u32);
                lm.tri_uv.extend(s.uv[k].map(|uv| uv.to_array()));
            }
        }
        for (c, src) in lm.sources.iter().enumerate() {
            let nu = ((src.size.x / texel_size).ceil() as u32).clamp(1, 1024);
            let nv = ((src.size.y / texel_size).ceil() as u32).clamp(1, 1024);
            let (nbu, nbv) = (nu.div_ceil(BLOCK), nv.div_ceil(BLOCK));
            // Bilinear lookups read one texel beyond a triangle, so texels within
            // 1.5 texels of one get its nearest point.
            let cells = place(src, nu, nv, 1.5);
            lm.charts[c].info = [lm.receivers.len() as u32, nu, nv, nbu];
            let pad = src.tris[0].0[0];
            for bj in 0..nbv {
                for bi in 0..nbu {
                    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                    for k in 0..BLOCK * BLOCK {
                        let (i, j) = (bi * BLOCK + k % BLOCK, bj * BLOCK + k / BLOCK);
                        match (i < nu && j < nv)
                            .then(|| cells[(j * nu + i) as usize])
                            .flatten()
                        {
                            Some((p, n)) => {
                                let p = p + n * 1e-4;
                                (lo, hi) = (lo.min(p), hi.max(p));
                                lm.receivers.push((p, n));
                            }
                            None => lm.receivers.push((pad, Vec3::ZERO)),
                        }
                    }
                    lm.blocks.push((lo, hi));
                }
            }
        }
        lm
    }

    /// Flat-shaded meshes ([`Surface::from_triangles`]: a chart per planar quad
    /// or per triangle), for example box meshes.
    pub fn from_meshes(
        meshes: &[Vec<Tri>],
        texel_size: f32,
        is_static: impl Fn(usize) -> bool,
    ) -> Lightmap {
        let surfaces: Vec<Surface> = meshes
            .iter()
            .map(|m| Surface::from_triangles(m.clone()))
            .collect();
        Self::new(&surfaces, texel_size, is_static)
    }

    pub fn texels(&self) -> usize {
        self.receivers.len()
    }

    /// Adds indirect-light probes about `spacing` apart on every chart, at grid
    /// cell centers placed on the surface like texels (a cell that lands on no
    /// triangle takes the nearest point on one).
    pub fn with_probes(mut self, spacing: f32) -> Self {
        for (c, src) in self.sources.iter().enumerate() {
            let pu = ((src.size.x / spacing).ceil() as u32).clamp(1, 1024);
            let pv = ((src.size.y / spacing).ceil() as u32).clamp(1, 1024);
            let mut cells = place(src, pu, pv, 2.0);
            if cells.iter().any(Option::is_none) {
                let all = place(src, pu, pv, f32::INFINITY);
                for (cell, fallback) in cells.iter_mut().zip(all) {
                    cell.get_or_insert(fallback.expect("every cell has a nearest triangle"));
                }
            }
            let mesh = self.charts[c].probes[3];
            self.charts[c].probes = [self.probes.len() as u32, pu, pv, mesh];
            self.probes.extend(
                cells
                    .into_iter()
                    .map(|cell| cell.map(|(p, n)| (p + n * 1e-4, n)).unwrap()),
            );
        }
        self
    }

    /// For each block, [`BLOCK_PLANES`] planes `[n.x, n.y, n.z, w]` (inside:
    /// `n · p <= w`) of a convex volume enclosing every one of its texels' cones
    /// toward `light`: the hull of the block's texels and the light, pushed out by
    /// a small margin. Unused planes are zero (they exclude nothing). A block
    /// with no texel that can receive direct light (all facing away, behind a
    /// one-sided light, or unused) gets a plane that excludes everything.
    pub fn block_planes(&self, light: RectLight) -> Vec<[f32; 4]> {
        const MARGIN: f32 = 1e-3;
        let center = light.corner + (light.edge_u + light.edge_v) * 0.5;
        let emits = light.edge_u.cross(light.edge_v).normalize();
        let light_corners = [
            light.corner,
            light.corner + light.edge_u,
            light.corner + light.edge_v,
            light.corner + light.edge_u + light.edge_v,
        ];
        let mut out = Vec::with_capacity(self.blocks.len() * BLOCK_PLANES);
        for (b, &(lo, hi)) in self.blocks.iter().enumerate() {
            let texels = &self.receivers[b * 64..(b + 1) * 64];
            let mut planes: Vec<[f32; 4]> = Vec::new();
            let lit = texels.iter().any(|&(p, n)| {
                n != Vec3::ZERO
                    && n.dot(center - p) > -1e-3
                    && (!light.one_sided || emits.dot(p - light.corner) > -1e-3)
            });
            if !lit {
                planes.push([0.0, 0.0, 0.0, -1.0]);
            } else {
                let mut points: Vec<Vec3> = Vec::new();
                for k in 0..8 {
                    let p = Vec3::new(
                        if k & 1 == 0 { lo.x } else { hi.x },
                        if k & 2 == 0 { lo.y } else { hi.y },
                        if k & 4 == 0 { lo.z } else { hi.z },
                    );
                    if !points.contains(&p) {
                        points.push(p);
                    }
                }
                points.extend(light_corners);
                // Every plane through three points with all points on one side is
                // a face plane of the hull.
                for i in 0..points.len() {
                    for j in i + 1..points.len() {
                        for k in j + 1..points.len() {
                            let n = (points[j] - points[i]).cross(points[k] - points[i]);
                            if n.length_squared() < 1e-14 {
                                continue;
                            }
                            let n = n.normalize();
                            let d = n.dot(points[i]);
                            let (mut below, mut above) = (0.0f32, 0.0f32);
                            for &p in &points {
                                (below, above) = (below.min(n.dot(p) - d), above.max(n.dot(p) - d));
                            }
                            let (n, d) = if above <= 1e-6 {
                                (n, d)
                            } else if below >= -1e-6 {
                                (-n, -d)
                            } else {
                                continue;
                            };
                            if planes.iter().any(|q| {
                                Vec3::new(q[0], q[1], q[2]).dot(n) > 1.0 - 1e-6
                                    && (q[3] - MARGIN - d).abs() < 1e-6
                            }) {
                                continue;
                            }
                            planes.push([n.x, n.y, n.z, d + MARGIN]);
                        }
                    }
                }
            }
            // Fewer planes enclose more, so dropping extras stays conservative.
            planes.resize(BLOCK_PLANES, [0.0; 4]);
            out.extend(planes);
        }
        out
    }
}

/// Centers of an `nu × nv` grid of cells over a chart, placed on its triangles
/// (row-major): a cell whose center lies in a triangle (in chart coordinates)
/// gets that point and the interpolated shading normal; one within `reach`
/// cells of a triangle gets the nearest point on the nearest triangle; the
/// rest get `None`.
fn place(src: &ChartSource, nu: u32, nv: u32, reach: f32) -> Vec<Option<(Vec3, Vec3)>> {
    let scale = Vec2::new(nu as f32, nv as f32);
    let mut best: Vec<(f32, Option<(Vec3, Vec3)>)> =
        vec![(f32::INFINITY, None); (nu * nv) as usize];
    for (tri, normals, uv) in &src.tris {
        let q = uv.map(|t| t * scale);
        if (q[1] - q[0]).perp_dot(q[2] - q[0]).abs() < 1e-12 {
            continue;
        }
        let lo = q[0].min(q[1]).min(q[2]) - Vec2::splat(reach) - Vec2::splat(0.5);
        let hi = q[0].max(q[1]).max(q[2]) + Vec2::splat(reach) - Vec2::splat(0.5);
        let (i0, j0) = (lo.x.ceil().max(0.0) as u32, lo.y.ceil().max(0.0) as u32);
        let i1 = hi.x.floor().min(nu as f32 - 1.0);
        let j1 = hi.y.floor().min(nv as f32 - 1.0);
        if i1 < 0.0 || j1 < 0.0 {
            continue;
        }
        let geometric = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
        for j in j0..=j1 as u32 {
            for i in i0..=i1 as u32 {
                let (d, b) = closest_barycentric(Vec2::new(i as f32 + 0.5, j as f32 + 0.5), q);
                let cell = &mut best[(j * nu + i) as usize];
                if d <= reach && d < cell.0 {
                    let p = tri[0] * b.x + tri[1] * b.y + tri[2] * b.z;
                    let mut n = (normals[0] * b.x + normals[1] * b.y + normals[2] * b.z)
                        .normalize_or_zero();
                    if n == Vec3::ZERO {
                        n = geometric;
                    }
                    *cell = (d, Some((p, n)));
                }
            }
        }
    }
    best.into_iter().map(|(_, cell)| cell).collect()
}

/// Distance from `c` to triangle `q` (zero inside) and the barycentric
/// coordinates of the nearest point.
fn closest_barycentric(c: Vec2, q: [Vec2; 3]) -> (f32, Vec3) {
    let (e1, e2, d) = (q[1] - q[0], q[2] - q[0], c - q[0]);
    let area = e1.perp_dot(e2);
    let s = d.perp_dot(e2) / area;
    let t = e1.perp_dot(d) / area;
    if s >= -1e-6 && t >= -1e-6 && s + t <= 1.0 + 1e-6 {
        return (0.0, Vec3::new(1.0 - s - t, s, t));
    }
    let mut best = (f32::INFINITY, Vec3::ZERO);
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        let edge = q[b] - q[a];
        let u = ((c - q[a]).dot(edge) / edge.length_squared()).clamp(0.0, 1.0);
        let dist = (q[a] + edge * u).distance(c);
        if dist < best.0 {
            let mut w = Vec3::ZERO;
            w[a] = 1.0 - u;
            w[b] = u;
            best = (dist, w);
        }
    }
    best
}

/// Bins of a hemicube around a probe's normal with `r × r` bins on the top face
/// and `r × r/2` on each side face, in the kernel's order: the projected solid
/// angle of each (they sum to π).
pub fn hemicube_weights(r: u32) -> Vec<f32> {
    assert!(r.is_multiple_of(2), "hemicube resolution must be even");
    let g = r as f32;
    let at = |i: u32, j: u32, di: f32, dj: f32| {
        (
            -1.0 + 2.0 * (i as f32 + di) / g,
            -1.0 + 2.0 * (j as f32 + dj) / g,
            2.0 * (j as f32 + dj) / g,
        )
    };
    let mut weights = Vec::with_capacity(3 * (r * r) as usize);
    for bin in 0..r * r {
        let (i, j) = (bin % r, bin / r);
        let quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(di, dj)| {
            let (x, y, _) = at(i, j, di, dj);
            Vec3::new(x, y, 1.0)
        });
        weights.push(continuum::cone3::projected_solid_angle(Vec3::Z, &quad));
    }
    for side in 0..4 {
        for k in 0..r * r / 2 {
            let (i, j) = (k % r, k / r);
            let quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(di, dj)| {
                let (a, _, z) = at(i, j, di, dj);
                match side {
                    0 => Vec3::new(1.0, a, z),
                    1 => Vec3::new(-1.0, a, z),
                    2 => Vec3::new(a, 1.0, z),
                    _ => Vec3::new(a, -1.0, z),
                }
            });
            weights.push(continuum::cone3::projected_solid_angle(Vec3::Z, &quad));
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hemicube_weights_cover_the_hemisphere() {
        for r in [2, 8, 16, 64] {
            let w = hemicube_weights(r);
            assert_eq!(w.len(), 3 * (r * r) as usize);
            let total: f32 = w.iter().sum();
            assert!(
                (total - std::f32::consts::PI).abs() < 1e-4,
                "r = {r}: {total}"
            );
        }
    }

    #[test]
    fn box_texels_tile_each_face() {
        let lm = Lightmap::from_meshes(
            &[continuum::cone3::box_mesh(
                Vec3::ZERO,
                Vec3::new(0.1, 0.05, 0.02),
            )],
            0.01,
            |_| true,
        );
        // 6 faces; e.g. the 0.1 × 0.05 faces have 10 × 5 texels in one block.
        assert_eq!(lm.charts.len(), 6);
        let real = lm.receivers.iter().filter(|r| r.1 != Vec3::ZERO).count();
        assert_eq!(real, 2 * (10 * 5 + 10 * 2 + 5 * 2));
        // Every real texel sits just off the box surface, on the outside.
        for &(p, n) in lm.receivers.iter().filter(|r| r.1 != Vec3::ZERO) {
            let inside = p.cmpge(Vec3::ZERO).all() && p.cmple(Vec3::new(0.1, 0.05, 0.02)).all();
            assert!(!inside, "{p} with normal {n}");
            assert!((p - n * 1e-4).cmpge(Vec3::splat(-1e-6)).all());
        }
    }

    #[test]
    fn lone_triangle_texels_stay_on_the_triangle() {
        let t = [
            Vec3::ZERO,
            Vec3::new(0.08, 0.0, 0.0),
            Vec3::new(0.0, 0.08, 0.0),
        ];
        let lm = Lightmap::from_meshes(&[vec![t]], 0.01, |_| true);
        for &(p, n) in lm.receivers.iter().filter(|r| r.1 != Vec3::ZERO) {
            let q = p - n * 1e-4;
            assert!(
                q.x >= -1e-6 && q.y >= -1e-6 && q.x + q.y <= 0.08 + 1e-5,
                "{q}"
            );
        }
    }
}
