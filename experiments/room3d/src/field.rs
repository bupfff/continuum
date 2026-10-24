//! A field of instanced boulders for scale tests: a ground plane about 2 km
//! across, up to a million copies of one detailed procedural boulder on a
//! jittered 2 m grid, and a sun-sized light 3 km away. Positions are f64 so the
//! whole field can sit anywhere in a large world.

use std::f32::consts::{PI, TAU};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Instance, InstancedScene, RectLight};
use flatland::rng::Pcg32;

/// Grid spacing of the boulders (meters) and the field's half width.
const SPACING: f64 = 2.0;
const HALF: usize = 500;

pub struct Field {
    pub scene: InstancedScene,
    /// World positions (f64).
    pub eye: DVec3,
    pub target: DVec3,
    pub vertical_fov_degrees: f32,
    /// The sun panel: center, and its two edges.
    pub sun_center: DVec3,
    pub sun_u: Vec3,
    pub sun_v: Vec3,
    pub radiance: f32,
}

/// A lumpy boulder about 1 m across: a sphere with low-frequency bumps,
/// squashed vertically (100 × 50 segments, 9,800 triangles).
pub fn boulder() -> Surface {
    let s = Surface::parametric(100, 50, |u, v| {
        let (theta, phi) = (TAU * u, PI * v);
        let r = 1.0
            + 0.22 * (3.0 * theta + 1.0).sin() * (2.0 * phi).sin()
            + 0.12 * (5.0 * theta).cos() * (3.0 * phi + 0.5).sin()
            + 0.06 * (9.0 * theta + 2.0 * phi).sin();
        Vec3::new(
            phi.sin() * theta.cos() * r * 0.5,
            phi.sin() * theta.sin() * r * 0.5,
            -phi.cos() * r * 0.33,
        )
    });
    // Outward, judged at the first triangle.
    let t = s.triangles[0];
    if (t[1] - t[0]).cross(t[2] - t[0]).dot(t[0]) < 0.0 {
        s.flipped()
    } else {
        s
    }
}

impl Field {
    /// The field with the `count` boulders nearest the camera (up to 1,000,000),
    /// everything shifted by `offset` in the world.
    pub fn new(count: usize, offset: DVec3) -> Field {
        let eye = DVec3::new(-990.0, -990.0, 1.7) + offset;
        let target = DVec3::new(0.0, 0.0, 1.0) + offset;
        let mut rng = Pcg32::new(7, 11);
        let mut boulders = Vec::with_capacity((2 * HALF) * (2 * HALF));
        for j in 0..2 * HALF {
            for i in 0..2 * HALF {
                let jitter = |rng: &mut Pcg32| (rng.next_f32() as f64 - 0.5) * 1.2;
                let x = (i as f64 - HALF as f64 + 0.5) * SPACING + jitter(&mut rng);
                let y = (j as f64 - HALF as f64 + 0.5) * SPACING + jitter(&mut rng);
                let angle = rng.next_f32() * TAU;
                let scale = 0.6 + 0.8 * rng.next_f32();
                let shade = 0.8 + 0.4 * rng.next_f32();
                let position = DVec3::new(x, y, 0.2 * scale as f64) + offset;
                // Keep the camera's own spot clear.
                if (position - eye).truncate().length() < 4.0 {
                    continue;
                }
                boulders.push(Instance {
                    geometry: 1,
                    rotation: Mat3::from_rotation_z(angle),
                    scale,
                    position,
                    albedo: Vec3::new(0.50, 0.47, 0.42) * shade,
                });
            }
        }
        boulders.sort_by(|a, b| {
            (a.position - eye)
                .length_squared()
                .total_cmp(&(b.position - eye).length_squared())
        });
        boulders.truncate(count);
        let ground = {
            let h = 1050.0f32;
            let (a, u, v) = (
                Vec3::new(-h, -h, 0.0),
                Vec3::new(2.0 * h, 0.0, 0.0),
                Vec3::new(0.0, 2.0 * h, 0.0),
            );
            Surface::from_triangles(vec![[a, a + u, a + u + v], [a, a + u + v, a + v]])
        };
        let mut instances = vec![Instance {
            geometry: 0,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: offset,
            albedo: Vec3::new(0.38, 0.36, 0.30),
        }];
        instances.extend(boulders);
        // The sun: 30° up, behind the camera and to its left, 3 km from the
        // field's center, 0.5° across, emitting toward the field.
        let (elevation, azimuth) = (30f64.to_radians(), 200f64.to_radians());
        let toward_sun = DVec3::new(
            elevation.cos() * azimuth.cos(),
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
        );
        let distance = 3000.0;
        let size = (2.0 * distance * 0.25f64.to_radians().tan()) as f32;
        let w = (-toward_sun).as_vec3();
        let a = Vec3::Z.cross(w).normalize();
        let b = w.cross(a);
        Field {
            scene: InstancedScene {
                geometries: vec![ground, boulder()],
                instances,
            },
            eye,
            target,
            vertical_fov_degrees: 50.0,
            sun_center: offset + toward_sun * distance,
            sun_u: a * size,
            sun_v: b * size,
            radiance: 15000.0,
        }
    }

    /// Boulders in the field (instances other than the ground).
    pub fn boulders(&self) -> usize {
        self.scene.instances.len() - 1
    }

    pub fn triangles(&self) -> u64 {
        self.scene
            .instances
            .iter()
            .map(|i| self.scene.geometries[i.geometry].triangles.len() as u64)
            .sum()
    }

    /// A render origin near the camera: its position snapped to a 256 m grid,
    /// so it changes rarely as the camera moves.
    pub fn origin_near_camera(&self) -> DVec3 {
        (self.eye / 256.0).floor() * 256.0
    }

    /// The camera and the sun relative to `origin`.
    pub fn relative(&self, origin: DVec3) -> (Camera, RectLight) {
        let camera = Camera {
            eye: (self.eye - origin).as_vec3(),
            target: (self.target - origin).as_vec3(),
            vertical_fov_degrees: self.vertical_fov_degrees,
        };
        let corner = self.sun_center - origin - (self.sun_u + self.sun_v).as_dvec3() * 0.5;
        let light = RectLight {
            corner: corner.as_vec3(),
            edge_u: self.sun_u,
            edge_v: self.sun_v,
            one_sided: true,
        };
        (camera, light)
    }
}
