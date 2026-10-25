//! An outdoor test scene for sky light (M5): a 400 m ground, 60 boulders of
//! the field's shape near the camera (some touching), a wall and a pillar, the
//! field's sun, and a uniform blue-white sky whose unoccluded irradiance on
//! the ground is about a quarter of the sun's.

use std::f32::consts::TAU;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Instance, InstancedScene, RectLight};
use flatland::rng::Pcg32;

use crate::field::boulder_with;

pub struct Outdoor {
    pub scene: InstancedScene,
    pub camera: Camera,
    /// The sun: one-sided, 0.5° wide, 30° up, behind the camera's left.
    pub light: RectLight,
    pub radiance: f32,
    /// Uniform sky radiance (linear RGB).
    pub sky: Vec3,
}

impl Default for Outdoor {
    fn default() -> Self {
        Self::new()
    }
}

impl Outdoor {
    /// The scene with the sun 3 km away (M5 gate 1).
    pub fn new() -> Outdoor {
        Self::with_sun_at(3000.0)
    }

    /// The scene with the sun `distance` meters from the origin, keeping its
    /// angular size and radiance (Cycles loses 1.9% of a sun this small at 3
    /// km, but none at 1 km: M5 gate 1).
    pub fn with_sun_at(distance: f64) -> Outdoor {
        let ground = {
            let h = 200.0f32;
            let (a, u, v) = (
                Vec3::new(-h, -h, 0.0),
                Vec3::new(2.0 * h, 0.0, 0.0),
                Vec3::new(0.0, 2.0 * h, 0.0),
            );
            Surface::from_triangles(vec![[a, a + u, a + u + v], [a, a + u + v, a + v]])
        };
        let wall = Surface::cuboid(Vec3::new(14.0, -5.0, 0.0), Vec3::new(14.6, 3.0, 3.2));
        let pillar = Surface::cuboid(Vec3::new(8.0, 3.5, 0.0), Vec3::new(8.6, 4.1, 4.5));
        let mut instances = vec![
            Instance {
                geometry: 0,
                rotation: Mat3::IDENTITY,
                scale: 1.0,
                position: DVec3::ZERO,
                albedo: Vec3::new(0.38, 0.36, 0.30),
            },
            Instance {
                geometry: 2,
                rotation: Mat3::IDENTITY,
                scale: 1.0,
                position: DVec3::ZERO,
                albedo: Vec3::new(0.70, 0.66, 0.60),
            },
            Instance {
                geometry: 3,
                rotation: Mat3::IDENTITY,
                scale: 1.0,
                position: DVec3::ZERO,
                albedo: Vec3::new(0.60, 0.60, 0.62),
            },
        ];
        let mut rng = Pcg32::new(29, 3);
        let boulder = |position: DVec3, scale: f32, rng: &mut Pcg32| Instance {
            geometry: 1,
            rotation: Mat3::from_rotation_z(rng.next_f32() * TAU),
            scale,
            position: DVec3::new(position.x, position.y, 0.2 * scale as f64),
            albedo: Vec3::new(0.50, 0.47, 0.42) * (0.8 + 0.4 * rng.next_f32()),
        };
        // 40 scattered in front of the camera, then 10 touching pairs.
        for _ in 0..40 {
            let x = 3.0 + 27.0 * rng.next_f32() as f64;
            let y = (rng.next_f32() as f64 - 0.5) * 2.0 * (2.0 + 0.6 * x);
            let scale = 0.6 + 1.4 * rng.next_f32();
            instances.push(boulder(DVec3::new(x, y, 0.0), scale, &mut rng));
        }
        for _ in 0..10 {
            let x = 4.0 + 20.0 * rng.next_f32() as f64;
            let y = (rng.next_f32() as f64 - 0.5) * 2.0 * (1.5 + 0.5 * x);
            let scale = 0.8 + 1.2 * rng.next_f32();
            let angle = rng.next_f32() * TAU;
            let offset = DVec3::new(angle.cos() as f64, angle.sin() as f64, 0.0) * 0.45 * scale as f64;
            instances.push(boulder(DVec3::new(x, y, 0.0) - offset, scale, &mut rng));
            instances.push(boulder(DVec3::new(x, y, 0.0) + offset, scale * 0.8, &mut rng));
        }
        // The field's sun, 3 km away.
        let (elevation, azimuth) = (30f64.to_radians(), 200f64.to_radians());
        let toward_sun = DVec3::new(
            elevation.cos() * azimuth.cos(),
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
        );
        let size = (2.0 * distance * 0.25f64.to_radians().tan()) as f32;
        let w = (-toward_sun).as_vec3();
        let a = Vec3::Z.cross(w).normalize();
        let b = w.cross(a);
        let center = (toward_sun * distance).as_vec3();
        let light = RectLight {
            corner: center - (a + b) * size * 0.5,
            edge_u: a * size,
            edge_v: b * size,
            one_sided: true,
        };
        let radiance = 15000.0;
        // A quarter of the sun's irradiance on the ground: π · mean(sky) =
        // radiance · Ω_sun · sin(30°) / 4.
        let sun_irradiance = radiance * (size * size) / (distance * distance) as f32 * 0.5;
        let tint = Vec3::new(0.6, 0.75, 1.0);
        let sky = tint / (tint.element_sum() / 3.0) * (sun_irradiance / 4.0 / std::f32::consts::PI);
        Outdoor {
            scene: InstancedScene {
                geometries: vec![ground, boulder_with(60, 30), wall, pillar],
                instances,
            },
            camera: Camera {
                eye: Vec3::new(0.0, 0.0, 1.7),
                target: Vec3::new(20.0, 2.0, 0.3),
                vertical_fov_degrees: 50.0,
            },
            light,
            radiance,
            sky,
        }
    }

    /// Every instance as a world-space surface, with its albedo (for Cycles).
    pub fn flattened(&self) -> (Vec<Surface>, Vec<Vec3>) {
        self.scene
            .instances
            .iter()
            .map(|i| {
                let g = &self.scene.geometries[i.geometry];
                (
                    g.transformed(i.rotation * i.scale, i.position.as_vec3()),
                    i.albedo,
                )
            })
            .unzip()
    }
}
