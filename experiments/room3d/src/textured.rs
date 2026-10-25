//! A small textured outdoor scene for glTF import tests (M6): a tiled ground,
//! boulders with banded lichen, and a brick block, every texture generated
//! here (1024 × 1024, so resampling into the renderer's texture layers is
//! exact), lit by a sun and a sky.

use std::f32::consts::TAU;

use continuum::glam::{DVec3, Mat3, Vec2, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Instance, InstancedScene, Light, RectLight, Texture};
use flatland::rng::Pcg32;

use crate::field::boulder;

const SIZE: u32 = 1024;

/// A smooth value noise in [0, 1], periodic over `period` cells.
fn noise(x: f32, y: f32, period: u32, seed: u32) -> f32 {
    let hash = |i: i32, j: i32| {
        let (i, j) = (
            i.rem_euclid(period as i32) as u32,
            j.rem_euclid(period as i32) as u32,
        );
        let mut h = i.wrapping_mul(0x8da6_b343)
            ^ j.wrapping_mul(0xd816_3841)
            ^ seed.wrapping_mul(0xcb1a_b31f);
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1_e995);
        h ^= h >> 15;
        (h & 0xffff) as f32 / 65535.0
    };
    let (i, j) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let a = hash(i, j) + (hash(i + 1, j) - hash(i, j)) * sx;
    let b = hash(i, j + 1) + (hash(i + 1, j + 1) - hash(i, j + 1)) * sx;
    a + (b - a) * sy
}

/// A texture from a function of (u, v) in [0, 1)² giving linear RGB.
fn texture(f: impl Fn(f32, f32) -> Vec3) -> Texture {
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let c = f(
                (x as f32 + 0.5) / SIZE as f32,
                (y as f32 + 0.5) / SIZE as f32,
            );
            rgba.extend([encode(c.x), encode(c.y), encode(c.z), 255]);
        }
    }
    Texture {
        width: SIZE,
        height: SIZE,
        rgba,
    }
}

pub struct Textured {
    pub scene: InstancedScene,
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub sky: Vec3,
    pub origin: DVec3,
}

impl Textured {
    pub fn new() -> Textured {
        // Stone tiles: 8 × 8 per texture, mortar lines, noisy tone.
        let tiles = texture(|u, v| {
            let (tu, tv) = (u * 8.0, v * 8.0);
            let mortar = (tu - tu.round()).abs().min((tv - tv.round()).abs()) < 0.04;
            let tile = noise(tu.floor() + 0.5, tv.floor() + 0.5, 8, 1);
            let grain = noise(u * 64.0, v * 64.0, 64, 2);
            if mortar {
                Vec3::splat(0.12)
            } else {
                Vec3::new(0.42, 0.40, 0.36) * (0.7 + 0.35 * tile) * (0.85 + 0.3 * grain)
            }
        });
        // Lichen: pale bands around the boulder over grey rock.
        let lichen = texture(|u, v| {
            let band = (0.5 + 0.5 * (v * TAU * 3.0).sin()) * noise(u * 16.0, v * 16.0, 16, 3);
            let rock = Vec3::new(0.45, 0.43, 0.40) * (0.8 + 0.3 * noise(u * 48.0, v * 48.0, 48, 4));
            rock.lerp(
                Vec3::new(0.62, 0.66, 0.38),
                (band * 1.6 - 0.5).clamp(0.0, 1.0),
            )
        });
        // Bricks: 4 × 8 per texture, staggered.
        let bricks = texture(|u, v| {
            let row = (v * 8.0).floor();
            let shift = if row as i32 % 2 == 0 { 0.0 } else { 0.5 };
            let (bu, bv) = (u * 4.0 + shift, v * 8.0);
            let mortar = (bu - bu.round()).abs() < 0.03 || (bv - bv.round()).abs() < 0.06;
            let tone = noise(bu.floor() + 0.5, bv.floor() + 0.5, 8, 5);
            if mortar {
                Vec3::splat(0.55)
            } else {
                Vec3::new(0.55, 0.22, 0.14) * (0.75 + 0.4 * tone)
            }
        });

        // The ground: 60 m square, tiles repeating every 4 m.
        let h = 30.0f32;
        let corners = [
            Vec3::new(-h, -h, 0.0),
            Vec3::new(h, -h, 0.0),
            Vec3::new(h, h, 0.0),
            Vec3::new(-h, h, 0.0),
        ];
        let mut ground = Surface::from_triangles(vec![
            [corners[0], corners[1], corners[2]],
            [corners[0], corners[2], corners[3]],
        ]);
        ground.tex = ground
            .triangles
            .iter()
            .map(|t| t.map(|p| Vec2::new(p.x, p.y) / 4.0))
            .collect();
        // Boulders: texture coordinates from their parametric chart, the
        // lichen repeating twice around and once from pole to pole.
        let mut rock = boulder();
        rock.tex = rock
            .uv
            .iter()
            .map(|t| t.map(|q| q * Vec2::new(2.0, 1.0)))
            .collect();
        // The block: each face's chart coordinates, bricks once per metre.
        let mut block = Surface::cuboid(Vec3::new(-1.5, -0.5, 0.0), Vec3::new(1.5, 0.5, 2.0));
        block.tex = (0..block.triangles.len())
            .map(|k| {
                let size = block.charts[block.chart[k] as usize];
                block.uv[k].map(|q| q * size)
            })
            .collect();

        let mut instances = vec![Instance {
            geometry: 0,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: DVec3::ZERO,
            albedo: Vec3::ONE,
        }];
        let mut rng = Pcg32::new(17, 3);
        for _ in 0..30 {
            let x = (rng.next_f32() - 0.5) * 16.0;
            let y = 4.0 + rng.next_f32() * 16.0;
            let scale = 0.6 + 0.8 * rng.next_f32();
            instances.push(Instance {
                geometry: 1,
                rotation: Mat3::from_rotation_z(rng.next_f32() * TAU),
                scale,
                position: DVec3::new(x as f64, y as f64, (0.2 * scale) as f64),
                albedo: Vec3::splat(0.85 + 0.3 * rng.next_f32()),
            });
        }
        instances.push(Instance {
            geometry: 2,
            rotation: Mat3::from_rotation_z(0.4),
            scale: 1.0,
            position: DVec3::new(-3.0, 9.0, 0.0),
            albedo: Vec3::ONE,
        });
        let scene = InstancedScene {
            geometries: vec![ground, rock, block],
            instances,
            textures: vec![tiles, lichen, bricks],
            geometry_textures: vec![Some(0), Some(1), Some(2)],
        };
        // The camera stands at 1.5 m looking along +y, slightly down; the sun
        // is 30° up behind and to the left, 1 km away, 0.5° across.
        let camera = Camera {
            eye: Vec3::new(1.0, 0.5, 1.5),
            target: Vec3::new(-1.0, 10.0, 0.5),
            vertical_fov_degrees: 45.0,
        };
        let (az, el) = (230f64.to_radians(), 30f64.to_radians());
        let toward = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
        let distance = 1000.0;
        let size = 2.0 * distance * 0.25f64.to_radians().tan();
        let w = (-toward).as_vec3();
        let a = Vec3::Z.cross(w).normalize() * size as f32;
        let b = w.cross(a).normalize() * size as f32;
        let middle = camera.eye.as_dvec3() + toward * distance;
        let sun = RectLight {
            corner: (middle - (a + b).as_dvec3() * 0.5).as_vec3(),
            edge_u: a,
            edge_v: b,
            one_sided: true,
        };
        let solid = (size / distance) * (size / distance);
        // Irradiance 3 on the ground from the sun; a sky giving a quarter.
        let radiance = (3.0 / (solid * el.sin())) as f32;
        let lights = vec![Light {
            rect: sun,
            radiance: Vec3::new(1.0, 0.95, 0.85) * radiance,
        }];
        let sky = Vec3::new(0.6, 0.75, 1.0) / (2.35 / 3.0) * (0.75 / std::f32::consts::PI);
        Textured {
            scene,
            camera,
            lights,
            sky,
            origin: DVec3::ZERO,
        }
    }
}

impl Default for Textured {
    fn default() -> Self {
        Self::new()
    }
}
