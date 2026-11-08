//! A small textured outdoor scene for glTF import tests (M6): a tiled ground,
//! boulders with banded lichen, and a brick block, every texture generated
//! here (1024 × 1024, so resampling into the renderer's texture layers is
//! exact), lit by a sun and a sky.

use std::f32::consts::TAU;

use continuum::glam::{DVec3, Mat3, Vec2, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, Instance, InstancedScene, Light, RectLight, Spot, Texture};
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

/// A texture from a function of (u, v) in [0, 1)² giving a tangent-space
/// normal, encoded linearly (M10 gate 5).
fn normal_texture(f: impl Fn(f32, f32) -> Vec3) -> Texture {
    let encode = |c: f32| ((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let n = f(
                (x as f32 + 0.5) / SIZE as f32,
                (y as f32 + 0.5) / SIZE as f32,
            );
            rgba.extend([encode(n.x), encode(n.y), encode(n.z), 255]);
        }
    }
    Texture {
        width: SIZE,
        height: SIZE,
        rgba,
    }
}

/// A texture from a function of (u, v) in [0, 1)² giving linear RGB and
/// alpha (M10 gate 6).
fn texture_alpha(f: impl Fn(f32, f32) -> (Vec3, f32)) -> Texture {
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
            let (c, a) = f(
                (x as f32 + 0.5) / SIZE as f32,
                (y as f32 + 0.5) / SIZE as f32,
            );
            rgba.extend([
                encode(c.x),
                encode(c.y),
                encode(c.z),
                (a.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
        }
    }
    Texture {
        width: SIZE,
        height: SIZE,
        rgba,
    }
}

/// Vertical double-sided cards through the origin, each `width` wide and
/// `height` tall, at the given angles about the vertical (the back faces
/// `back` metres behind the front ones; exactly back-to-back since M10
/// gates 8–9, 1 mm in gates 6–7);
/// texture coordinates over the whole texture times `repeat`, v down from
/// the top (M10 gate 6).
fn cards(angles: &[f32], width: f32, height: f32, repeat: Vec2, back: f32) -> Surface {
    let mut tris = Vec::new();
    let mut tex = Vec::new();
    for &a in angles {
        let d = Vec3::new(a.cos(), a.sin(), 0.0) * (width * 0.5);
        let p = [-d, d, d + Vec3::Z * height, -d + Vec3::Z * height];
        let uv = [
            Vec2::new(0.0, repeat.y),
            Vec2::new(repeat.x, repeat.y),
            Vec2::new(repeat.x, 0.0),
            Vec2::new(0.0, 0.0),
        ];
        let windings = [(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)];
        let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
        for (k, &(a, b, c)) in windings.iter().enumerate() {
            let shift = if k >= 2 { -normal * back } else { Vec3::ZERO };
            tris.push([p[a] + shift, p[b] + shift, p[c] + shift]);
            tex.push([uv[a], uv[b], uv[c]]);
        }
    }
    Surface::from_mesh(tris, None, Some(tex))
}

/// A beveled tile's tangent-space normal at (fu, fv) in [0, 1)² within it
/// (v down the image): flat in the mortar (half-widths `mu`, `mv` from the
/// edges) and inside; within `bevel` of the mortar, tilted 30° toward the
/// nearest edge (green toward the image's top).
fn bevel(fu: f32, fv: f32, mu: f32, mv: f32, bevel: f32) -> Vec3 {
    let (s, c) = 30f32.to_radians().sin_cos();
    let edges = [
        (fu - mu, Vec3::new(-s, 0.0, c)),
        (1.0 - fu - mu, Vec3::new(s, 0.0, c)),
        (fv - mv, Vec3::new(0.0, s, c)),
        (1.0 - fv - mv, Vec3::new(0.0, -s, c)),
    ];
    let (d, n) = edges
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    if d < 0.0 || d >= bevel { Vec3::Z } else { n }
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
            geometry_materials: Vec::new(),
            geometry_emission: Vec::new(),
            geometry_normal_maps: Vec::new(),
            geometry_metallic_roughness: Vec::new(),
            geometry_alpha_cutoff: Vec::new(),
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
        let lights = vec![Light::rect(sun, Vec3::new(1.0, 0.95, 0.85) * radiance)];
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

impl Textured {
    /// M10 gate 2's materials: the ground dielectric (roughness 0.6), the
    /// boulders metallic (roughness 0.3), the brick block dielectric
    /// (roughness 0.4).
    pub fn with_materials(mut self) -> Textured {
        use continuum_gpu::Material;
        self.scene.geometry_materials = vec![
            Some(Material {
                metallic: 0.0,
                roughness: 0.6,
            }),
            Some(Material {
                metallic: 1.0,
                roughness: 0.3,
            }),
            Some(Material {
                metallic: 0.0,
                roughness: 0.4,
            }),
        ];
        self
    }

    /// M10 gate 6's foliage: three shrubs of crossed double-sided leaf
    /// cards and a lattice fence panel, alpha-masked at 0.5.
    pub fn with_foliage(self) -> Textured {
        self.with_foliage_back(0.001)
    }

    /// `with_foliage` with the cards' back faces `back` metres behind the
    /// front ones (0: exactly back-to-back).
    pub fn with_foliage_back(mut self, back: f32) -> Textured {
        // Leaves: pointed ellipses in a cluster, on a transparent ground of
        // the same green (so filtering at the cut adds no dark fringe).
        let mut rng = Pcg32::new(41, 7);
        let leaves: Vec<(Vec2, f32, Vec2)> = (0..60)
            .map(|_| {
                let r = 0.38 * rng.next_f32().sqrt();
                let a = rng.next_f32() * TAU;
                let center = Vec2::new(0.5 + r * a.cos(), 0.55 + 0.9 * r * a.sin());
                let size = Vec2::new(0.05 + 0.03 * rng.next_f32(), 0.018 + 0.01 * rng.next_f32());
                (center, rng.next_f32() * TAU, size)
            })
            .collect();
        let leaf_texture = texture_alpha(|u, v| {
            let q = Vec2::new(u, v);
            let mut inside = false;
            for (c, angle, size) in &leaves {
                let d = q - *c;
                let (s, co) = angle.sin_cos();
                let local = Vec2::new(d.x * co + d.y * s, -d.x * s + d.y * co);
                let x = local.x / size.x;
                // Narrowing toward both tips.
                let half = size.y * (1.0 - x * x).max(0.0).sqrt() * (1.0 - 0.3 * x.abs());
                if x.abs() < 1.0 && local.y.abs() < half {
                    inside = true;
                    break;
                }
            }
            let tone = 0.75 + 0.5 * noise(u * 32.0, v * 32.0, 32, 6);
            (
                Vec3::new(0.16, 0.34, 0.09) * tone,
                if inside { 1.0 } else { 0.0 },
            )
        });
        // A diagonal lattice, four diamonds a tile.
        let lattice = texture_alpha(|u, v| {
            let (a, b) = (((u + v) * 4.0).fract(), ((u - v) * 4.0).rem_euclid(1.0));
            let bar = a < 0.12 || a > 0.88 || b < 0.12 || b > 0.88;
            (Vec3::new(0.45, 0.32, 0.2), if bar { 1.0 } else { 0.0 })
        });
        let first = self.scene.textures.len() as u32;
        self.scene.textures.extend([leaf_texture, lattice]);
        let shrub = self.scene.geometries.len();
        self.scene.geometries.push(cards(
            &[0.0, TAU / 6.0, TAU / 3.0],
            1.4,
            1.2,
            Vec2::ONE,
            back,
        ));
        self.scene
            .geometries
            .push(cards(&[0.0], 4.0, 1.5, Vec2::new(2.0, 0.75), back));
        let n = self.scene.geometries.len();
        let mut textures = self.scene.geometry_textures.clone();
        textures.resize(n, None);
        textures[shrub] = Some(first);
        textures[shrub + 1] = Some(first + 1);
        self.scene.geometry_textures = textures;
        let mut cutoff = vec![None; n];
        cutoff[shrub] = Some(0.5);
        cutoff[shrub + 1] = Some(0.5);
        self.scene.geometry_alpha_cutoff = cutoff;
        for (x, y, angle, scale) in [
            (2.5, 5.0, 0.3, 1.0),
            (-0.5, 6.5, 1.1, 1.2),
            (4.0, 8.5, 2.0, 0.9),
        ] {
            self.scene.instances.push(Instance {
                geometry: shrub,
                rotation: Mat3::from_rotation_z(angle),
                scale,
                position: DVec3::new(x, y, 0.0),
                albedo: Vec3::ONE,
            });
        }
        self.scene.instances.push(Instance {
            geometry: shrub + 1,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: DVec3::new(0.5, 12.0, 0.0),
            albedo: Vec3::ONE,
        });
        self
    }

    /// M10 gate 5's normal maps: the ground's tiles and the block's bricks
    /// beveled toward their mortar (bevels 12% of a tile, sloped 30°).
    pub fn with_normal_maps(mut self) -> Textured {
        let tiles = normal_texture(|u, v| {
            let (tu, tv) = (u * 8.0, v * 8.0);
            bevel(tu.fract(), tv.fract(), 0.04, 0.04, 0.12)
        });
        let bricks = normal_texture(|u, v| {
            let row = (v * 8.0).floor();
            let shift = if row as i32 % 2 == 0 { 0.0 } else { 0.5 };
            let (bu, bv) = (u * 4.0 + shift, v * 8.0);
            bevel(bu.fract(), bv.fract(), 0.03, 0.06, 0.12)
        });
        let first = self.scene.textures.len() as u32;
        self.scene.textures.extend([tiles, bricks]);
        self.scene.geometry_normal_maps = vec![Some(first), None, Some(first + 1)];
        self
    }

    /// M10 gate 11's scene: the scene at night (M5 gate 8's point and spot
    /// lights, the sky at 2% of the day's) with a glowing panel standing on
    /// the ground 5 m ahead, 1.2 m wide and 0.9 m tall, emitting warm light
    /// (radiance (3.0, 1.86, 0.9) from every face).
    pub fn with_glow(mut self) -> Textured {
        self.lights = Self::night_lights().to_vec();
        self.sky *= 0.02;
        let panel = self.scene.geometries.len();
        self.scene.geometries.push(Surface::cuboid(
            Vec3::new(-0.6, -0.04, 0.0),
            Vec3::new(0.6, 0.04, 0.9),
        ));
        let n = self.scene.geometries.len();
        self.scene.geometry_textures.resize(n, None);
        let mut emission = self.scene.geometry_emission.clone();
        emission.resize(n, Vec3::ZERO);
        emission[panel] = Vec3::new(1.0, 0.62, 0.3) * 3.0;
        self.scene.geometry_emission = emission;
        self.scene.instances.push(Instance {
            geometry: panel,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: DVec3::new(0.3, 5.0, 0.0),
            albedo: Vec3::splat(0.8),
        });
        self
    }

    /// Night lights for the scene (M5 gate 8): a warm point light 10 cm
    /// across, 2 m up near the brick block, and a cool spot 5 cm across, 4 m
    /// up, aimed down the scene with 20° inner and 30° outer half-angles.
    pub fn night_lights() -> [Light; 2] {
        let point = Light::point(
            Vec3::new(-1.0, 7.0, 2.0),
            0.05,
            Vec3::new(1.0, 0.8, 0.6) * 20.0,
        );
        let spot = Light::spot(
            Vec3::new(2.0, 4.0, 4.0),
            0.025,
            Vec3::new(0.6, 0.8, 1.0) * 120.0,
            Spot {
                axis: Vec3::new(-0.3, 1.0, -1.2).normalize(),
                inner: 20f32.to_radians(),
                outer: 30f32.to_radians(),
            },
        );
        [point, spot]
    }
}

impl Default for Textured {
    fn default() -> Self {
        Self::new()
    }
}
