//! Shared pieces for the 3D experiments: scenes of box meshes over a floor of
//! receivers, lit by a rectangular area light.

use continuum::bvh::Bvh;
use continuum::cone3::{PyramidCone, box_mesh, ray_triangle, sphere_mesh};
use continuum::glam::Vec3;
use flatland::rng::Pcg32;

pub type Tri = [Vec3; 3];

pub struct RectLight {
    pub corner: Vec3,
    pub edge_u: Vec3,
    pub edge_v: Vec3,
}

impl RectLight {
    pub fn area(&self) -> f32 {
        self.edge_u.cross(self.edge_v).length()
    }

    pub fn point(&self, s: f32, t: f32) -> Vec3 {
        self.corner + self.edge_u * s + self.edge_v * t
    }

    pub fn normal(&self) -> Vec3 {
        self.edge_u.cross(self.edge_v).normalize()
    }
}

pub struct Room {
    pub meshes: Vec<Vec<Tri>>,
    pub light: RectLight,
}

/// Receivers on the floor (z = 0) over the unit square, row-major, y up.
pub fn floor_point(px: usize, size: usize) -> Vec3 {
    let (i, j) = (px % size, px / size);
    Vec3::new(
        (i as f32 + 0.5) / size as f32,
        1.0 - (j as f32 + 0.5) / size as f32,
        0.0,
    )
}

/// Gate 1 in 3D: a 0.2 × 0.2 ceiling light at height 1 over the middle of the
/// floor; two double fences of thin slats (one with aligned gaps, one with offset
/// gaps, to stress occlusion correlation); a table top; a floating cube.
pub fn gate1() -> Room {
    let mut meshes = Vec::new();
    let mut slats = |y0: f32, y1: f32, z: f32, shift: f32| {
        for i in 0..8 {
            let x = 0.30 + i as f32 * 0.04 + shift;
            meshes.push(box_mesh(
                Vec3::new(x, y0, z),
                Vec3::new(x + 0.02, y1, z + 0.01),
            ));
        }
    };
    slats(0.10, 0.35, 0.55, 0.0);
    slats(0.10, 0.35, 0.62, 0.0);
    slats(0.65, 0.90, 0.55, 0.0);
    slats(0.65, 0.90, 0.62, 0.02);
    meshes.push(box_mesh(
        Vec3::new(0.62, 0.40, 0.30),
        Vec3::new(0.85, 0.60, 0.33),
    ));
    meshes.push(box_mesh(
        Vec3::new(0.15, 0.45, 0.20),
        Vec3::new(0.25, 0.55, 0.30),
    ));
    Room {
        meshes,
        light: RectLight {
            corner: Vec3::new(0.4, 0.4, 1.0),
            edge_u: Vec3::new(0.2, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.2, 0.0),
        },
    }
}

/// Gate 1 plus a finely tessellated sphere (about 40,000 triangles).
pub fn with_sphere() -> Room {
    let mut room = gate1();
    room.meshes
        .push(sphere_mesh(Vec3::new(0.3, 0.72, 0.45), 0.08, 200, 100));
    room
}

/// Gate 1 plus 3,000 small boxes scattered between the floor and the fences.
pub fn with_rubble() -> Room {
    let mut room = gate1();
    let mut rng = Pcg32::new(42, 7);
    for _ in 0..3000 {
        let p = Vec3::new(
            0.05 + 0.9 * rng.next_f32(),
            0.05 + 0.9 * rng.next_f32(),
            0.35 + 0.15 * rng.next_f32(),
        );
        let s = Vec3::new(
            0.01 + 0.01 * rng.next_f32(),
            0.01 + 0.01 * rng.next_f32(),
            0.01 + 0.01 * rng.next_f32(),
        );
        room.meshes.push(box_mesh(p, p + s));
    }
    room
}

/// A convex volume enclosing every receiver cone of a floor tile: the
/// prismatoid between the tile (a rectangle on the floor) and the light (a
/// rectangle parallel to it). Its planes are kept as `normal · p <= offset`.
pub struct TileFrustum {
    planes: Vec<(Vec3, f32)>,
}

impl TileFrustum {
    /// `tile` is `[x0, y0, x1, y1]` on the floor; the light must be horizontal and
    /// axis-aligned, as in these scenes.
    pub fn new(tile: [f32; 4], light: &RectLight) -> TileFrustum {
        let [tx0, ty0, tx1, ty1] = tile;
        let (a, b) = (light.corner, light.corner + light.edge_u + light.edge_v);
        let (lx0, ly0, lx1, ly1, h) = (a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y), a.z);
        // Each side face joins a tile edge at z = 0 to the light's matching edge at z = h.
        let side = |t: f32, l: f32, axis: Vec3, sign: f32| {
            // sign · (axis·p − (t + (l − t) z / h)) <= 0
            let n = (axis - Vec3::Z * ((l - t) / h)) * sign;
            (n, sign * t)
        };
        let planes = vec![
            side(tx0, lx0, Vec3::X, -1.0),
            side(tx1, lx1, Vec3::X, 1.0),
            side(ty0, ly0, Vec3::Y, -1.0),
            side(ty1, ly1, Vec3::Y, 1.0),
            (-Vec3::Z, 0.0),
            (Vec3::Z, h),
        ];
        TileFrustum { planes }
    }

    fn outside_one_plane(&self, points: &[Vec3]) -> bool {
        self.planes
            .iter()
            .any(|&(n, o)| points.iter().all(|&p| n.dot(p) > o))
    }

    pub fn may_touch_box(&self, min: Vec3, max: Vec3) -> bool {
        let corners: [Vec3; 8] = std::array::from_fn(|i| {
            Vec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            )
        });
        !self.outside_one_plane(&corners)
    }

    pub fn may_touch(&self, tri: &Tri) -> bool {
        !self.outside_one_plane(tri)
    }
}

/// A receiver's cone toward the light and the triangles that may touch it,
/// grouped by mesh (the culling a hierarchy would do, shared by every method).
pub struct Receiver {
    pub point: Vec3,
    pub cone: PyramidCone,
    pub candidates: Vec<(usize, Vec<Tri>)>,
}

impl Receiver {
    pub fn new(room: &Room, point: Vec3) -> Option<Receiver> {
        let l = &room.light;
        let cone = PyramidCone::toward_rect_light(point, l.corner, l.edge_u, l.edge_v, Vec3::Z)?;
        let candidates = room
            .meshes
            .iter()
            .enumerate()
            .filter_map(|(m, mesh)| {
                let tris: Vec<Tri> = mesh.iter().filter(|t| cone.may_touch(t)).copied().collect();
                (!tris.is_empty()).then_some((m, tris))
            })
            .collect();
        Some(Receiver {
            point,
            cone,
            candidates,
        })
    }

    /// Like [`Receiver::new`], but culling with a hierarchy instead of testing
    /// every triangle. Returns the receiver and the number of nodes visited.
    pub fn with_bvh(room: &Room, bvh: &Bvh, point: Vec3) -> Option<(Receiver, usize)> {
        let l = &room.light;
        let cone = PyramidCone::toward_rect_light(point, l.corner, l.edge_u, l.edge_v, Vec3::Z)?;
        let (candidates, visited) = bvh.candidates(&cone);
        Some((
            Receiver {
                point,
                cone,
                candidates,
            },
            visited,
        ))
    }

    /// Like [`Receiver::with_bvh`], but filtering a tile's shared candidates
    /// instead of walking the hierarchy.
    pub fn from_tile(room: &Room, tile_candidates: &[(u32, Tri)], point: Vec3) -> Option<Receiver> {
        let l = &room.light;
        let cone = PyramidCone::toward_rect_light(point, l.corner, l.edge_u, l.edge_v, Vec3::Z)?;
        let mine: Vec<(u32, Tri)> = tile_candidates
            .iter()
            .filter(|(_, t)| cone.may_touch(t))
            .copied()
            .collect();
        Some(Receiver {
            point,
            cone,
            candidates: continuum::bvh::group_by_mesh(mine),
        })
    }

    /// Irradiance per unit radiance through the mask cone.
    pub fn masked(&self) -> f32 {
        let mut m = continuum::CoverageMask::EMPTY;
        for (_, tris) in &self.candidates {
            self.cone.add_mesh(&mut m, tris);
            if m.is_full() {
                break;
            }
        }
        self.cone.visible(&m)
    }

    /// Irradiance per unit radiance from area samples at `(s, t)` on the light,
    /// cosine-weighted at both ends.
    pub fn sampled(&self, light: &RectLight, samples: impl Iterator<Item = (f32, f32)>) -> f32 {
        let (mut sum, mut n) = (0.0, 0);
        let light_normal = light.normal();
        for (s, t) in samples {
            n += 1;
            let q = light.point(s, t);
            let d = q - self.point;
            let blocked = self
                .candidates
                .iter()
                .flat_map(|(_, tris)| tris)
                .any(|tri| ray_triangle(self.point, d, tri).is_some_and(|h| h < 1.0));
            if !blocked {
                let r2 = d.length_squared();
                let r = r2.sqrt();
                sum += (d.z / r) * (light_normal.dot(-d).abs() / r) / r2;
            }
        }
        sum * light.area() / n as f32
    }
}

/// Stratified samples on the unit square, `k × k` cells, jittered by `jitter`
/// (cell-relative offsets in `[0, 1)`) or centered.
pub fn stratified(
    k: usize,
    mut jitter: impl FnMut() -> (f32, f32),
) -> impl Iterator<Item = (f32, f32)> {
    let cells: Vec<(f32, f32)> = (0..k * k)
        .map(|c| {
            let (dx, dy) = jitter();
            (
                ((c % k) as f32 + dx) / k as f32,
                ((c / k) as f32 + dy) / k as f32,
            )
        })
        .collect();
    cells.into_iter()
}

/// The render scene: the gate 1 room plus a floor and two walls, an albedo per
/// mesh, and the camera used for `render` and `render_gpu`.
pub struct RenderScene {
    pub room: Room,
    pub albedo: Vec<Vec3>,
    pub eye: Vec3,
    pub target: Vec3,
    pub vertical_fov_degrees: f32,
    pub radiance: f32,
}

pub fn render_scene() -> RenderScene {
    let mut room = gate1();
    let props = room.meshes.len();
    // Floor and two walls for context (they never sit between a surface and the light).
    room.meshes.push(box_mesh(
        Vec3::new(-0.5, -0.5, -0.02),
        Vec3::new(1.5, 1.5, 0.0),
    ));
    room.meshes.push(box_mesh(
        Vec3::new(-0.5, 1.0, 0.0),
        Vec3::new(1.5, 1.02, 1.2),
    ));
    room.meshes.push(box_mesh(
        Vec3::new(-0.02, -0.5, 0.0),
        Vec3::new(0.0, 1.5, 1.2),
    ));
    let albedo = (0..room.meshes.len())
        .map(|m| match m {
            m if m < 32 => Vec3::new(0.55, 0.42, 0.30), // slats: wood
            32 => Vec3::new(0.62, 0.48, 0.34),          // table top
            33 => Vec3::new(0.25, 0.42, 0.70),          // cube
            m if m == props => Vec3::new(0.72, 0.72, 0.70), // floor
            _ => Vec3::new(0.78, 0.74, 0.68),           // walls
        })
        .collect();
    RenderScene {
        room,
        albedo,
        eye: Vec3::new(1.05, -0.55, 0.95),
        target: Vec3::new(0.45, 0.55, 0.18),
        vertical_fov_degrees: 50.0,
        radiance: 9.0,
    }
}

/// Three boxes moving between the floor and the light, as in 3D gate 4: one
/// sliding under the fences, one orbiting near the light, one bobbing. `frame`
/// counts 60 Hz frames; returns box meshes.
pub fn movers(frame: f32) -> Vec<Vec<Tri>> {
    let t = frame;
    let cube = |c: Vec3, half: f32| box_mesh(c - Vec3::splat(half), c + Vec3::splat(half));
    vec![
        cube(
            Vec3::new(0.2 + 0.6 * (0.5 + 0.5 * (t * 0.03).sin()), 0.5, 0.45),
            0.03,
        ),
        cube(
            Vec3::new(
                0.5 + 0.25 * (t * 0.02).cos(),
                0.5 + 0.25 * (t * 0.02).sin(),
                0.8,
            ),
            0.02,
        ),
        cube(Vec3::new(0.75, 0.25, 0.35 + 0.15 * (t * 0.05).sin()), 0.025),
    ]
}

/// Albedo of the three [`movers`].
pub const MOVER_ALBEDO: [Vec3; 3] = [
    Vec3::new(0.80, 0.25, 0.20),
    Vec3::new(0.85, 0.75, 0.25),
    Vec3::new(0.30, 0.65, 0.35),
];

impl RenderScene {
    /// Appends the [`movers`] at `frame`; returns the index of the first.
    pub fn add_movers(&mut self, frame: f32) -> usize {
        let first = self.room.meshes.len();
        self.room.meshes.extend(movers(frame));
        self.albedo.extend(MOVER_ALBEDO);
        first
    }
}

/// A camera ray's nearest surface hit, nudged off the surface.
pub struct Hit {
    pub point: Vec3,
    pub normal: Vec3,
    pub albedo: Vec3,
}

impl RenderScene {
    /// Nearest hit for every pixel of a `width × height` view (brute force).
    pub fn primary_hits(&self, width: usize, height: usize, threads: usize) -> Vec<Option<Hit>> {
        let tris: Vec<(usize, Tri)> = self
            .room
            .meshes
            .iter()
            .enumerate()
            .flat_map(|(m, mesh)| mesh.iter().map(move |&t| (m, t)))
            .collect();
        let forward = (self.target - self.eye).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward);
        let half = (self.vertical_fov_degrees.to_radians() * 0.5).tan();
        flatland::incremental::par_map(width * height, threads, 256, |px| {
            let (x, y) = (px % width, px / width);
            let sx =
                ((x as f32 + 0.5) / width as f32 * 2.0 - 1.0) * half * width as f32 / height as f32;
            let sy = (1.0 - (y as f32 + 0.5) / height as f32 * 2.0) * half;
            let dir = (forward + right * sx + up * sy).normalize();
            let (mut best, mut hit) = (f32::MAX, None);
            for (m, t) in &tris {
                if let Some(d) = ray_triangle(self.eye, dir, t)
                    && d < best
                {
                    best = d;
                    hit = Some((*m, *t));
                }
            }
            let (m, t) = hit?;
            let mut normal = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
            if normal.dot(dir) > 0.0 {
                normal = -normal;
            }
            Some(Hit {
                point: self.eye + dir * best + normal * 1e-4,
                normal,
                albedo: self.albedo[m],
            })
        })
    }

    /// The cone from a hit toward the light, if the surface faces the light.
    pub fn light_cone(&self, h: &Hit) -> Option<PyramidCone> {
        let l = &self.room.light;
        let to_light = l.corner + (l.edge_u + l.edge_v) * 0.5 - h.point;
        if h.normal.dot(to_light) <= 0.0 {
            return None;
        }
        PyramidCone::toward_rect_light(h.point, l.corner, l.edge_u, l.edge_v, h.normal)
    }

    /// Diffuse reflected radiance for a given irradiance per unit light radiance.
    pub fn shade(&self, h: &Hit, irradiance: f32) -> Vec3 {
        h.albedo * (self.radiance * irradiance / std::f32::consts::PI)
    }

    /// Every pixel shaded with a mask cone toward the light.
    pub fn masked(&self, bvh: &Bvh, hits: &[Option<Hit>], threads: usize) -> Vec<Vec3> {
        flatland::incremental::par_map(hits.len(), threads, 256, |px| {
            let Some(h) = &hits[px] else {
                return Vec3::ZERO;
            };
            let Some(cone) = self.light_cone(h) else {
                return Vec3::ZERO;
            };
            let mut m = continuum::CoverageMask::EMPTY;
            for (_, tris) in bvh.candidates(&cone).0 {
                cone.add_mesh(&mut m, &tris);
            }
            self.shade(h, cone.visible(&m))
        })
    }
}

/// Reinhard tone curve plus display gamma.
pub fn to_rgb(c: Vec3) -> [u8; 3] {
    c.to_array()
        .map(|v| ((v / (v + 0.6)).powf(1.0 / 2.2) * 255.0).round() as u8)
}
