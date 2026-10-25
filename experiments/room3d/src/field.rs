//! A field of instanced boulders for scale tests: a ground plane about 2 km
//! across, up to a million copies of one detailed procedural boulder on a
//! jittered 2 m grid, and a sun-sized light 3 km away. Positions are f64 so the
//! whole field can sit anywhere in a large world.

use std::f32::consts::{PI, TAU};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Instance, InstancedScene, Light, RectLight, Spot};
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
    boulder_with(100, 50)
}

/// [`boulder`] with `nu × nv` segments.
pub fn boulder_with(nu: usize, nv: usize) -> Surface {
    let s = Surface::parametric(nu, nv, |u, v| {
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
                ..Default::default()
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

    /// The same field with the sun `distance` meters from the field's
    /// center, keeping its direction, angular size and radiance (Cycles loses
    /// 1.9% of a sun this small 3 km away, none at 1 km: M5 gate 1).
    pub fn with_sun_at(mut self, distance: f64) -> Field {
        let center = self.scene.instances[0].position;
        let toward = (self.sun_center - center).normalize();
        let old = (self.sun_center - center).length();
        self.sun_center = center + toward * distance;
        self.sun_u *= (distance / old) as f32;
        self.sun_v *= (distance / old) as f32;
        self
    }

    /// Uniform sky radiance (linear RGB, blue-white) whose unoccluded
    /// irradiance on the ground is a quarter of the sun's, as in
    /// `room3d::outdoor`.
    pub fn sky(&self) -> Vec3 {
        let center = self.scene.instances[0].position;
        let distance = (self.sun_center - center).length() as f32;
        let toward = ((self.sun_center - center) / distance as f64).as_vec3();
        let solid_angle = self.sun_u.length() * self.sun_v.length() / (distance * distance);
        let sun_irradiance = self.radiance * solid_angle * toward.z;
        let tint = Vec3::new(0.6, 0.75, 1.0);
        tint / (tint.element_sum() / 3.0) * (sun_irradiance / 4.0 / std::f32::consts::PI)
    }

    /// Three colored lights relative to `origin` (M5 gate 6), placed around
    /// the camera so its surroundings face them all: the sun, warm, in its
    /// direction but `distance` from the camera, 0.5° across; a cool fill at
    /// the opposite azimuth, 20° up, `distance` away, 4° across, giving a
    /// quarter of the sun's irradiance; and an orange lamp (1.5 m square, 6 m
    /// up, facing down) 25 m ahead of the camera and 8 m to its left.
    /// (`with_sun_at` measures from the field's center, 1.4 km from the
    /// camera, which leaves the camera's surroundings behind a one-sided sun.)
    pub fn colored_lights(&self, origin: DVec3, distance: f64) -> [Light; 3] {
        let (camera, _) = self.relative(origin);
        let eye = self.eye;
        let center = self.scene.instances[0].position;
        let toward_sun = (self.sun_center - center).normalize();
        // A one-sided square `size` across, `distance` from the eye toward
        // `toward`, facing the eye.
        let panel = |toward: DVec3, size: f64| {
            let w = (-toward).as_vec3();
            let a = Vec3::Z.cross(w).normalize() * size as f32;
            let b = w.cross(a).normalize() * size as f32;
            let middle = eye + toward * distance - origin;
            RectLight {
                corner: (middle - (a + b).as_dvec3() * 0.5).as_vec3(),
                edge_u: a,
                edge_v: b,
                one_sided: true,
            }
        };
        let tinted = |rgb: Vec3, radiance: f64| rgb / (rgb.element_sum() / 3.0) * radiance as f32;
        let sun_size = 2.0 * distance * 0.25f64.to_radians().tan();
        let warm = Light::rect(
            panel(toward_sun, sun_size),
            tinted(Vec3::new(1.0, 0.9, 0.75), self.radiance as f64),
        );
        // The fill: azimuth opposite the sun's, elevation 20°, a quarter of
        // the sun's irradiance on the ground.
        let azimuth = toward_sun.y.atan2(toward_sun.x) + std::f64::consts::PI;
        let elevation = 20f64.to_radians();
        let toward = DVec3::new(
            elevation.cos() * azimuth.cos(),
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
        );
        let fill_size = 2.0 * distance * 2f64.to_radians().tan();
        let solid = |size: f64| (size / distance) * (size / distance);
        let sun_e = self.radiance as f64 * solid(sun_size) * toward_sun.z;
        let fill_radiance = 0.25 * sun_e / (solid(fill_size) * toward.z);
        let fill = Light::rect(
            panel(toward, fill_size),
            tinted(Vec3::new(0.55, 0.7, 1.0), fill_radiance),
        );
        // The lamp, facing down (edge_u × edge_v = -z).
        let forward = (camera.target - camera.eye).normalize();
        let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
        let left = Vec3::Z.cross(flat);
        let lamp_center = camera.eye + flat * 25.0 + left * 8.0 + Vec3::Z * (6.0 - camera.eye.z);
        let (u, v) = (flat * 1.5, left * 1.5);
        let (u, v) = if u.cross(v).z < 0.0 { (u, v) } else { (v, u) };
        let lamp = Light::rect(
            RectLight {
                corner: lamp_center - (u + v) * 0.5,
                edge_u: u,
                edge_v: v,
                one_sided: true,
            },
            Vec3::new(1.0, 0.6, 0.3) * 150.0,
        );
        [warm, fill, lamp]
    }

    /// Two sphere lights relative to `origin` (M5 gate 8), ahead of the
    /// camera: a warm point light 10 cm across, 3 m up, 15 m ahead and 3 m to
    /// the left; and a cool spot 5 cm across, 8 m up, 30 m ahead, aimed 45°
    /// down along the camera's line, 25° inner and 35° outer half-angles.
    /// Each gives the sun's ground irradiance: the point light straight below
    /// it, the spot where its axis meets the ground.
    pub fn sphere_lights(&self, origin: DVec3) -> [Light; 2] {
        let (camera, _) = self.relative(origin);
        let forward = (camera.target - camera.eye).normalize();
        let flat = Vec3::new(forward.x, forward.y, 0.0).normalize();
        let left = Vec3::Z.cross(flat);
        let ground = Vec3::new(camera.eye.x, camera.eye.y, 0.0);
        let center = self.scene.instances[0].position;
        let distance = (self.sun_center - center).length() as f32;
        let toward = ((self.sun_center - center) / distance as f64).as_vec3();
        let solid_angle = self.sun_u.length() * self.sun_v.length() / (distance * distance);
        let sun_irradiance = self.radiance * solid_angle * toward.z;
        let point = Light::point(
            ground + flat * 15.0 + left * 3.0 + Vec3::Z * 3.0,
            0.05,
            Vec3::new(1.0, 0.75, 0.5) * sun_irradiance * 9.0,
        );
        let spot_at = ground + flat * 30.0 + Vec3::Z * 8.0;
        let axis = (flat - Vec3::Z).normalize();
        // On-axis irradiance on the ground: I cos 45° / (8 √2)².
        let spot = Light::spot(
            spot_at,
            0.025,
            Vec3::new(0.6, 0.8, 1.0) * sun_irradiance * 128.0 / std::f32::consts::FRAC_1_SQRT_2,
            Spot {
                axis,
                inner: 25f32.to_radians(),
                outer: 35f32.to_radians(),
            },
        );
        [point, spot]
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
