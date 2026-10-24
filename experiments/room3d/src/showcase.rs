//! The showcase: a gallery room lit by one skylight panel, with a colonnade, an
//! arch, a torus-knot sculpture on a pedestal, two spheres, and two moving
//! objects (a bouncing ball and a spinning cube orbiting the sculpture).
//!
//! Units are meters, `+Z` up. Every surface is procedural, so the scene needs no
//! assets and can be exported for other renderers.

use std::f32::consts::{PI, TAU};

use continuum::glam::{Mat3, Vec3};
use continuum::mesh::{Surface, Tri};
use continuum_gpu::{Camera, RectLight};

/// Lightmap texel spacing for the showcase (meters).
pub const TEXEL: f32 = 0.0125;
/// Indirect-light probe spacing for the showcase (meters).
pub const PROBES: f32 = 0.1;
/// Length of the camera path in 60 Hz frames.
pub const PATH_FRAMES: u32 = 300;

pub struct Showcase {
    /// Static surfaces first, then the movers (from `first_mover` on).
    pub surfaces: Vec<Surface>,
    pub albedo: Vec<Vec3>,
    pub light: RectLight,
    /// Radiance of the light.
    pub radiance: f32,
    pub first_mover: usize,
    pub vertical_fov_degrees: f32,
}

/// A planar quad `a, a + u, a + u + v, a + v`, facing along `u × v`.
fn quad(a: Vec3, u: Vec3, v: Vec3) -> Surface {
    Surface::from_triangles(vec![[a, a + u, a + u + v], [a, a + u + v, a + v]])
}

/// `s`, flipped if needed so its first triangle faces along `outward(point)`.
fn orient(s: Surface, outward: impl Fn(Vec3) -> Vec3) -> Surface {
    let t = s.triangles[0];
    let n = (t[1] - t[0]).cross(t[2] - t[0]);
    if n.dot(outward((t[0] + t[1] + t[2]) / 3.0)) < 0.0 {
        s.flipped()
    } else {
        s
    }
}

fn cylinder(center: Vec3, radius: f32, height: f32, segments: usize) -> Surface {
    let s = Surface::parametric(segments, 4, |u, v| {
        center
            + Vec3::new(
                radius * (TAU * u).cos(),
                radius * (TAU * u).sin(),
                height * v,
            )
    });
    orient(s, |p| (p - center) * Vec3::new(1.0, 1.0, 0.0))
}

fn sphere(center: Vec3, radius: f32, segments: usize, rings: usize) -> Surface {
    let s = Surface::parametric(segments, rings, |u, v| {
        let (theta, phi) = (TAU * u, PI * v);
        center + Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), -phi.cos()) * radius
    });
    orient(s, |p| p - center)
}

/// A (2, 3) torus knot around `center` with the given radius and tube radius.
fn torus_knot(center: Vec3, radius: f32, tube: f32, segments: usize, sides: usize) -> Surface {
    let s = radius / 3.0;
    let curve = |t: f32| {
        let r = 2.0 + (3.0 * t).cos();
        center
            + Vec3::new(
                r * (2.0 * t).cos() * s,
                r * (2.0 * t).sin() * s,
                (3.0 * t).sin() * s * 1.6,
            )
    };
    let core =
        |t: f32| center + Vec3::new(2.0 * (2.0 * t).cos() * s, 2.0 * (2.0 * t).sin() * s, 0.0);
    // A periodic frame: the normal points from the torus core to the curve.
    let frame = |t: f32| {
        let tangent = (curve(t + 1e-3) - curve(t - 1e-3)).normalize();
        let d = curve(t) - core(t);
        let n = (d - tangent * d.dot(tangent)).normalize();
        (n, tangent.cross(n))
    };
    let surface = Surface::parametric(sides, segments, |a, b| {
        let t = TAU * b;
        let (n, binormal) = frame(t);
        curve(t) + (n * (TAU * a).cos() + binormal * (TAU * a).sin()) * tube
    });
    // Outward: away from the nearest point of the curve, judged at the first triangle.
    let c0 = curve(0.0);
    orient(surface, |p| p - c0)
}

/// A semicircular arch over the y axis at `(x0..x1, cy, z0)`, inner radius
/// `r`, outer radius `big_r`: outer and inner curved faces and two flat sides.
fn arch(x0: f32, x1: f32, cy: f32, z0: f32, r: f32, big_r: f32, segments: usize) -> Vec<Surface> {
    let at = |x: f32, radius: f32, a: f32| {
        Vec3::new(
            x,
            cy + radius * (PI * a).cos(),
            z0 + radius * (PI * a).sin(),
        )
    };
    let axis = |p: Vec3| Vec3::new(0.0, p.y - cy, p.z - z0);
    vec![
        orient(
            Surface::parametric(segments, 2, |a, b| at(x0 + (x1 - x0) * b, big_r, a)),
            axis,
        ),
        orient(
            Surface::parametric(segments, 2, |a, b| at(x0 + (x1 - x0) * b, r, a)),
            |p| -axis(p),
        ),
        orient(
            Surface::parametric(segments, 2, |a, b| at(x0, r + (big_r - r) * b, a)),
            |_| -Vec3::X,
        ),
        orient(
            Surface::parametric(segments, 2, |a, b| at(x1, r + (big_r - r) * b, a)),
            |_| Vec3::X,
        ),
    ]
}

const BALL_RADIUS: f32 = 0.16;
const CUBE_HALF: f32 = 0.15;
const KNOT_CENTER: Vec3 = Vec3::new(2.5, 1.9, 1.06);

/// The moving surfaces at `frame` (60 Hz): a ball bouncing across the floor in
/// front of the sculpture, and a cube spinning as it orbits the sculpture.
pub fn movers(frame: f32) -> Vec<Surface> {
    let t = frame / 60.0;
    let sweep = 0.5 - 0.5 * (TAU * t / 6.0).cos();
    let ball = Vec3::new(
        1.1 + 2.8 * sweep,
        0.95 + 0.25 * (TAU * t / 3.0).sin(),
        BALL_RADIUS + 0.55 * (PI * t / 0.75).sin().abs(),
    );
    let orbit = TAU * t / 8.0;
    let cube_at = KNOT_CENTER + Vec3::new(0.9 * orbit.cos(), 0.9 * orbit.sin(), 0.75);
    let spin = Mat3::from_rotation_z(TAU * t / 4.0) * Mat3::from_rotation_x(0.6);
    vec![
        sphere(Vec3::ZERO, BALL_RADIUS, 24, 12).transformed(Mat3::IDENTITY, ball),
        Surface::cuboid(Vec3::splat(-CUBE_HALF), Vec3::splat(CUBE_HALF)).transformed(spin, cube_at),
    ]
}

impl Showcase {
    /// The scene with its movers at frame 0.
    pub fn new() -> Showcase {
        let (w, d, h) = (5.0, 4.0, 3.0);
        let mut surfaces = Vec::new();
        let mut albedo = Vec::new();
        let mut add = |s: Surface, a: Vec3| {
            surfaces.push(s);
            albedo.push(a);
        };
        let wall = Vec3::new(0.78, 0.76, 0.72);
        let stone = Vec3::new(0.82, 0.80, 0.74);
        let sandstone = Vec3::new(0.70, 0.64, 0.54);
        // Room shell: single quads facing inward.
        add(
            quad(Vec3::ZERO, Vec3::X * w, Vec3::Y * d),
            Vec3::new(0.62, 0.60, 0.56),
        );
        add(
            quad(Vec3::Z * h, Vec3::Y * d, Vec3::X * w),
            Vec3::new(0.80, 0.80, 0.78),
        );
        add(
            quad(Vec3::Y * d, Vec3::X * w, Vec3::Z * h),
            Vec3::new(0.72, 0.36, 0.24),
        );
        add(quad(Vec3::ZERO, Vec3::Z * h, Vec3::X * w), wall);
        add(
            quad(Vec3::ZERO, Vec3::Y * d, Vec3::Z * h),
            Vec3::new(0.28, 0.52, 0.55),
        );
        add(quad(Vec3::X * w, Vec3::Z * h, Vec3::Y * d), wall);
        // Colonnade with a beam.
        for x in [1.0, 2.0, 3.0, 4.0] {
            let c = Vec3::new(x, 3.0, 0.0);
            add(
                Surface::cuboid(
                    c + Vec3::new(-0.2, -0.2, 0.0),
                    c + Vec3::new(0.2, 0.2, 0.15),
                ),
                stone,
            );
            add(cylinder(c + Vec3::Z * 0.15, 0.13, 2.5, 32), stone);
            add(
                Surface::cuboid(
                    c + Vec3::new(-0.18, -0.18, 2.65),
                    c + Vec3::new(0.18, 0.18, 2.77),
                ),
                stone,
            );
        }
        add(
            Surface::cuboid(Vec3::new(0.7, 2.8, 2.77), Vec3::new(4.3, 3.2, 2.95)),
            stone,
        );
        // Sculpture on a pedestal.
        add(
            Surface::cuboid(Vec3::new(2.2, 1.6, 0.0), Vec3::new(2.8, 2.2, 0.8)),
            Vec3::new(0.42, 0.42, 0.46),
        );
        add(
            torus_knot(KNOT_CENTER, 0.36, 0.065, 160, 12),
            Vec3::new(0.80, 0.58, 0.22),
        );
        // Spheres resting on the floor.
        add(
            sphere(Vec3::new(4.0, 1.0, 0.35), 0.35, 32, 16),
            Vec3::new(0.22, 0.38, 0.78),
        );
        add(
            sphere(Vec3::new(0.85, 1.2, 0.2), 0.2, 32, 16),
            Vec3::new(0.85, 0.85, 0.82),
        );
        // An arch on two pillars by the right wall, spanning along y.
        add(
            Surface::cuboid(
                Vec3::new(4.55, 0.75 + 0.9, 0.0),
                Vec3::new(4.85, 0.9 + 0.9, 1.4),
            ),
            sandstone,
        );
        add(
            Surface::cuboid(
                Vec3::new(4.55, 1.9 + 0.9, 0.0),
                Vec3::new(4.85, 2.05 + 0.9, 1.4),
            ),
            sandstone,
        );
        for s in arch(4.55, 4.85, 2.3, 1.4, 0.5, 0.65, 24) {
            add(s, sandstone);
        }
        let first_mover = surfaces.len();
        surfaces.extend(movers(0.0));
        albedo.extend([Vec3::new(0.85, 0.22, 0.12), Vec3::new(0.20, 0.62, 0.50)]);
        Showcase {
            surfaces,
            albedo,
            // A skylight panel just under the ceiling, emitting downward.
            light: RectLight {
                corner: Vec3::new(2.2, 0.85, 2.99),
                edge_u: Vec3::new(0.0, 0.45, 0.0),
                edge_v: Vec3::new(0.6, 0.0, 0.0),
                one_sided: true,
            },
            radiance: 70.0,
            first_mover,
            vertical_fov_degrees: 55.0,
        }
    }

    /// The scene without its movers (static geometry only).
    pub fn without_movers(mut self) -> Showcase {
        self.surfaces.truncate(self.first_mover);
        self.albedo.truncate(self.first_mover);
        self
    }

    /// Every surface's triangles, in order: the meshes the renderer traces.
    pub fn meshes(&self) -> Vec<Vec<Tri>> {
        self.surfaces.iter().map(|s| s.triangles.clone()).collect()
    }

    /// Moves the movers to `frame`.
    pub fn set_frame(&mut self, frame: f32) {
        if self.surfaces.len() > self.first_mover {
            for (k, s) in movers(frame).into_iter().enumerate() {
                self.surfaces[self.first_mover + k] = s;
            }
        }
    }

    /// The camera at `frame` along the path (held at its end after
    /// [`PATH_FRAMES`]): a slow pan across the room past the sculpture.
    pub fn camera(&self, frame: f32) -> Camera {
        let keys = [
            (Vec3::new(0.45, 0.35, 1.55), Vec3::new(2.9, 2.6, 1.15)),
            (Vec3::new(2.4, 0.3, 1.75), Vec3::new(2.5, 2.9, 1.0)),
            (Vec3::new(4.5, 0.45, 1.5), Vec3::new(2.0, 2.8, 1.1)),
        ];
        let x = (frame / PATH_FRAMES as f32).clamp(0.0, 1.0) * (keys.len() - 1) as f32;
        let k = (x.floor() as usize).min(keys.len() - 2);
        let f = x - k as f32;
        let s = f * f * (3.0 - 2.0 * f);
        Camera {
            eye: keys[k].0.lerp(keys[k + 1].0, s),
            target: keys[k].1.lerp(keys[k + 1].1, s),
            vertical_fov_degrees: self.vertical_fov_degrees,
        }
    }
}

impl Default for Showcase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every surface faces outward (away from its solid) or, for the room shell,
    /// inward toward the room's center.
    #[test]
    fn surfaces_face_the_right_way() {
        let scene = Showcase::new();
        let room_center = Vec3::new(2.5, 2.0, 1.5);
        for (m, s) in scene.surfaces.iter().enumerate() {
            let (lo, hi) = s.bounds();
            let mut facing = 0.0;
            for t in &s.triangles {
                let n = (t[1] - t[0]).cross(t[2] - t[0]);
                let c = (t[0] + t[1] + t[2]) / 3.0;
                // Shell quads face the room center, the arch's inner face (third
                // from last static surface) faces its axis, and solids face away
                // from their own center.
                let reference = if m < 6 {
                    room_center - c
                } else if m == scene.first_mover - 3 {
                    Vec3::new(0.0, 2.3 - c.y, 1.4 - c.z)
                } else {
                    c - (lo + hi) * 0.5
                };
                facing += n.dot(reference).signum();
            }
            assert!(facing > 0.0, "surface {m} faces the wrong way ({facing})");
        }
    }

    #[test]
    fn knot_faces_away_from_its_curve() {
        let s = torus_knot(KNOT_CENTER, 0.36, 0.065, 160, 12);
        for (t, n) in s.triangles.iter().zip(&s.normals) {
            let geometric = (t[1] - t[0]).cross(t[2] - t[0]);
            for &v in n {
                assert!(v.dot(geometric) > 0.0);
            }
        }
    }
}
