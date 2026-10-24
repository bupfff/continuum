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
