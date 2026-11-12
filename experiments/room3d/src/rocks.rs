//! A field of unique procedural rocks for streaming tests (M14): every rock
//! is its own geometry, with finer camera levels.
//!
//! A rock is a squashed sphere displaced by octaves of 3D sinusoids on the
//! unit sphere (frequencies 2, 4, 8 and 16), each octave with its own random
//! directions and phases. Its levels are band-limited: the lighting level
//! (32 × 16 segments) keeps the first octave, and each finer level doubles
//! the segments and adds an octave, up to 256 × 128 segments (65,536
//! triangles) with all four. All levels share one parametric chart, so the
//! lighting cache's pages serve every level. Each level's error is its
//! largest distance from the finest at the finest grid's vertices.

use std::f32::consts::{PI, TAU};

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use std::sync::Arc;

use continuum_gpu::{
    Camera, Instance, InstancedScene, Level, LevelSource, Levels, Light, RectLight,
};
use flatland::rng::Pcg32;

/// Segments around (u) of the lighting level; each finer level doubles them.
const BASE_SEGMENTS: usize = 32;
/// Octaves of displacement; the lighting level keeps the first, each finer
/// level one more.
const OCTAVES: usize = 4;
/// Random directions per octave.
const WAVES: usize = 3;

/// One rock's shape: its radius, vertical squash, and per octave its
/// waves' directions and phases.
#[derive(Clone, Debug)]
pub struct Shape {
    pub radius: f32,
    pub squash: f32,
    waves: Vec<[(Vec3, f32); WAVES]>,
    amplitude: f32,
    falloff: f32,
}

/// The rocks' displacement: the first octave's amplitude (a share of the
/// radius) and how fast finer octaves fall (amplitude ∝ frequency^-falloff).
#[derive(Clone, Copy, Debug)]
pub struct Detail {
    pub amplitude: f32,
    pub falloff: f32,
}

impl Default for Detail {
    fn default() -> Self {
        Detail {
            amplitude: 0.16,
            falloff: 1.8,
        }
    }
}

impl Shape {
    /// A random rock from `rng`.
    pub fn random(rng: &mut Pcg32, detail: Detail) -> Shape {
        let radius = 0.4 + 0.85 * rng.next_f32();
        let squash = 0.55 + 0.25 * rng.next_f32();
        let waves = (0..OCTAVES)
            .map(|_| {
                std::array::from_fn(|_| {
                    // A uniform direction on the sphere.
                    let z = 2.0 * rng.next_f32() - 1.0;
                    let a = TAU * rng.next_f32();
                    let s = (1.0 - z * z).sqrt();
                    (Vec3::new(s * a.cos(), s * a.sin(), z), TAU * rng.next_f32())
                })
            })
            .collect();
        Shape {
            radius,
            squash,
            waves,
            amplitude: detail.amplitude,
            falloff: detail.falloff,
        }
    }

    /// The surface point at chart coordinates `(u, v)` with the first
    /// `octaves` octaves.
    pub fn point(&self, u: f32, v: f32, octaves: usize) -> Vec3 {
        let (theta, phi) = (TAU * u, PI * v);
        let dir = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), -phi.cos());
        let mut r = 1.0;
        for (o, waves) in self.waves.iter().take(octaves).enumerate() {
            let f = (2u32 << o) as f32;
            let a = self.amplitude * (2.0 / f).powf(self.falloff) / (WAVES as f32).sqrt();
            for (d, phase) in waves {
                r += a * (PI * f * d.dot(dir) + phase).sin();
            }
        }
        Vec3::new(dir.x * r, dir.y * r, dir.z * r * self.squash) * self.radius
    }

    /// Level `level` (0: the lighting level; up to `OCTAVES - 1`, the
    /// finest) as a surface.
    pub fn surface(&self, level: usize) -> Surface {
        let nu = BASE_SEGMENTS << level;
        let s = Surface::parametric(nu, nu / 2, |u, v| self.point(u, v, level + 1));
        // Outward, judged at the first triangle (as `field::boulder`).
        let t = s.triangles[0];
        if (t[1] - t[0]).cross(t[2] - t[0]).dot(t[0]) < 0.0 {
            s.flipped()
        } else {
            s
        }
    }

    /// At least as many triangles as level `level` has (`2 × nu × nv`;
    /// degenerate ones at the poles are dropped).
    pub fn triangles_at_most(level: usize) -> u32 {
        let nu = (BASE_SEGMENTS << level) as u32;
        nu * nu
    }

    /// The largest distance from level `level`'s triangles to the finest
    /// level's, at the finest grid's vertices (both meshes interpolated
    /// along their own triangles at the same chart coordinates).
    pub fn error(&self, level: usize) -> f32 {
        let finest = OCTAVES - 1;
        if level >= finest {
            return 0.0;
        }
        let (nu, nv) = (BASE_SEGMENTS << finest, (BASE_SEGMENTS << finest) / 2);
        let step = 1 << (finest - level);
        let (lu, lv) = (nu / step, nv / step);
        let grid =
            |i: usize, j: usize| self.point(i as f32 / lu as f32, j as f32 / lv as f32, level + 1);
        let mut worst = 0.0f32;
        for i in 0..=nu {
            for j in 0..=nv {
                let p = self.point(i as f32 / nu as f32, j as f32 / nv as f32, OCTAVES);
                // The level's quad and the position in it (its split as
                // `Surface::parametric`: triangles a-b-c and a-c-d).
                let (ci, cj) = ((i / step).min(lu - 1), (j / step).min(lv - 1));
                let s = (i - ci * step) as f32 / step as f32;
                let t = (j - cj * step) as f32 / step as f32;
                let (a, b, c, d) = (
                    grid(ci, cj),
                    grid(ci + 1, cj),
                    grid(ci + 1, cj + 1),
                    grid(ci, cj + 1),
                );
                let q = if s >= t {
                    a + (b - a) * s + (c - b) * t
                } else {
                    a + (d - a) * t + (c - d) * s
                };
                worst = worst.max((p - q).length());
            }
        }
        worst
    }
}

/// The rock field: the scene (geometry 0 the ground, then one geometry per
/// rock, each with its camera levels), its sun, and each rock's position.
pub struct Rocks {
    pub scene: InstancedScene,
    pub sun: Light,
    /// Rock centres on the ground.
    pub positions: Vec<DVec3>,
    /// Each rock's bounding radius (meters, about its centre).
    pub radii: Vec<f32>,
    /// The field's half width (meters).
    pub half: f64,
}

impl Rocks {
    /// `side × side` unique rocks on a jittered grid `spacing` meters apart,
    /// centred on the origin, on a 2 km ground; the sun 35° up, 1 km away,
    /// 0.5° across.
    pub fn new(side: usize, spacing: f64) -> Rocks {
        Rocks::with_detail(side, spacing, Detail::default())
    }

    /// [`Rocks::new`] with the rocks' displacement given.
    pub fn with_detail(side: usize, spacing: f64, detail: Detail) -> Rocks {
        Rocks::build(side, spacing, detail, false)
    }

    /// [`Rocks::new`] with the camera levels produced on demand (M14 gate
    /// 3): only the lighting levels are built now; the scene's
    /// `level_source` generates the rest from each rock's shape.
    pub fn on_demand(side: usize, spacing: f64) -> Rocks {
        Rocks::build(side, spacing, Detail::default(), true)
    }

    fn build(side: usize, spacing: f64, detail: Detail, on_demand: bool) -> Rocks {
        let mut rng = Pcg32::new(14, 1);
        let ground = {
            let h = 1000.0f32;
            let (a, u, v) = (
                Vec3::new(-h, -h, 0.0),
                Vec3::new(2.0 * h, 0.0, 0.0),
                Vec3::new(0.0, 2.0 * h, 0.0),
            );
            Surface::from_triangles(vec![[a, a + u, a + u + v], [a, a + u + v, a + v]])
        };
        let mut geometries = vec![ground];
        let mut levels = vec![None];
        let mut instances = vec![Instance {
            geometry: 0,
            rotation: Mat3::IDENTITY,
            scale: 1.0,
            position: DVec3::ZERO,
            albedo: Vec3::new(0.38, 0.36, 0.30),
        }];
        let half = side as f64 * spacing * 0.5;
        let mut positions = Vec::with_capacity(side * side);
        let shapes: Vec<(Shape, DVec3, f32, f32)> = (0..side * side)
            .map(|k| {
                let (i, j) = (k % side, k / side);
                let jitter = |rng: &mut Pcg32| (rng.next_f32() as f64 - 0.5) * 0.5 * spacing;
                let x = (i as f64 + 0.5) * spacing - half + jitter(&mut rng);
                let y = (j as f64 + 0.5) * spacing - half + jitter(&mut rng);
                let shape = Shape::random(&mut rng, detail);
                let angle = TAU * rng.next_f32();
                let shade = 0.8 + 0.4 * rng.next_f32();
                let sink = 0.25 * shape.radius * shape.squash;
                (shape, DVec3::new(x, y, -sink as f64), angle, shade)
            })
            .collect();
        // Building levels is the slow part: one thread per chunk of rocks.
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = shapes.len().div_ceil(threads).max(1);
        let built: Vec<(Surface, Levels)> = std::thread::scope(|scope| {
            let handles: Vec<_> = shapes
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|(shape, _, _, _)| {
                                let finer = (1..OCTAVES)
                                    .map(|l| Level {
                                        surface: if on_demand {
                                            Surface::default()
                                        } else {
                                            shape.surface(l)
                                        },
                                        error: shape.error(l),
                                        triangles: Shape::triangles_at_most(l),
                                    })
                                    .collect();
                                (
                                    shape.surface(0),
                                    Levels {
                                        base_error: shape.error(0),
                                        finer,
                                    },
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("a rock thread"))
                .collect()
        });
        let mut radii = Vec::with_capacity(shapes.len());
        for ((surface, l), (_, position, angle, shade)) in built.into_iter().zip(&shapes) {
            let (lo, hi) = surface.bounds();
            radii.push(1.05 * lo.abs().max(hi.abs()).length());
            instances.push(Instance {
                geometry: geometries.len(),
                rotation: Mat3::from_rotation_z(*angle),
                scale: 1.0,
                position: *position,
                albedo: Vec3::new(0.50, 0.47, 0.42) * *shade,
            });
            positions.push(*position);
            geometries.push(surface);
            levels.push(Some(l));
        }
        let (elevation, azimuth) = (35f32.to_radians(), 200f32.to_radians());
        let toward = Vec3::new(
            elevation.cos() * azimuth.cos(),
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
        );
        let distance = 1000.0;
        let size = 2.0 * distance * 0.25f32.to_radians().tan();
        let w = -toward;
        let a = Vec3::Z.cross(w).normalize() * size;
        let b = w.cross(a).normalize() * size;
        let sun = Light::rect(
            RectLight {
                corner: toward * distance - (a + b) * 0.5,
                edge_u: a,
                edge_v: b,
                one_sided: true,
            },
            Vec3::new(1.0, 0.95, 0.88) * 15000.0,
        );
        Rocks {
            scene: InstancedScene {
                geometries,
                instances,
                geometry_levels: levels,
                level_source: on_demand.then(|| {
                    Arc::new(RockSource {
                        shapes: shapes.iter().map(|(s, _, _, _)| s.clone()).collect(),
                    }) as Arc<dyn LevelSource>
                }),
                ..Default::default()
            },
            sun,
            positions,
            radii,
            half,
        }
    }

    /// Triangles of every rock's finest level.
    pub fn finest_triangles(&self) -> u64 {
        self.scene
            .geometry_levels
            .iter()
            .flatten()
            .map(|l| {
                l.finer
                    .last()
                    .map_or(0, |f| f.surface.triangles.len() as u64)
            })
            .sum()
    }
}

/// Flight `k`'s camera (of three, M14 gate 1) at `t` in [0, 1] over a field
/// `half` meters wide, with a vertical field of view `fov` (degrees).
/// Each crosses the field turning half a circle, so it starts looking along
/// its path and ends looking back across the field.
pub fn flight(k: usize, t: f32, half: f32, fov: f32) -> Camera {
    let look = |eye: Vec3, yaw: f32, down: f32| {
        let (y, d) = (yaw.to_radians(), down.to_radians());
        Camera {
            eye,
            target: eye + Vec3::new(y.cos() * d.cos(), y.sin() * d.cos(), -d.sin()),
            vertical_fov_degrees: fov,
        }
    };
    let span = 2.0 * half + 10.0;
    match k {
        // Low along x, 2 m up.
        0 => look(
            Vec3::new(-half - 5.0 + span * t, 0.3 * half, 2.0),
            180.0 * t,
            6.0,
        ),
        // Diagonal, 6 m up, looking down 15°.
        1 => look(
            Vec3::new(-half - 4.0, -half - 4.0, 6.0)
                + Vec3::new(2.0 * half + 8.0, 2.0 * half + 8.0, 0.0) * t,
            45.0 - 180.0 * t,
            15.0,
        ),
        // Weaving along y near the ground, swaying as it turns.
        _ => {
            let a = std::f32::consts::TAU * t;
            look(
                Vec3::new(
                    -0.2 * half + 0.25 * half * (1.5 * a).sin(),
                    -half - 5.0 + span * t,
                    1.7 + 1.3 * (0.5 - 0.5 * a.cos()),
                ),
                90.0 + 180.0 * t + 20.0 * (1.5 * a).cos(),
                4.0,
            )
        }
    }
}

/// Generates the rocks' camera levels on demand (geometry `k + 1` is rock
/// `k`; level `l` of `Levels::finer` is the shape's level `l + 1`).
#[derive(Debug)]
pub struct RockSource {
    shapes: Vec<Shape>,
}

impl LevelSource for RockSource {
    fn level(&self, geometry: usize, level: usize) -> Surface {
        self.shapes[geometry - 1].surface(level + 1)
    }
}
