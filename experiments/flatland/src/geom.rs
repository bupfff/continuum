//! 2D vectors, segments and the light cone.

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

pub const fn v2(x: f32, y: f32) -> Vec2 {
    Vec2 { x, y }
}

impl Vec2 {
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    /// z-component of the 3D cross product; positive when `o` is counter-clockwise from `self`.
    pub fn cross(self, o: Vec2) -> f32 {
        self.x * o.y - self.y * o.x
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Vec2 {
        self * (1.0 / self.length())
    }

    /// Rotated 90° counter-clockwise.
    pub fn perp(self) -> Vec2 {
        v2(-self.y, self.x)
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        v2(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        v2(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f32) -> Vec2 {
        v2(self.x * s, self.y * s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: Vec2,
    pub b: Vec2,
}

pub const fn seg(ax: f32, ay: f32, bx: f32, by: f32) -> Segment {
    Segment {
        a: v2(ax, ay),
        b: v2(bx, by),
    }
}

/// Keeps the part of the parameter range `[t0, t1]` where a linear function with
/// endpoint values `fa` (t = 0) and `fb` (t = 1) is non-negative.
fn clip_half_plane(t0: &mut f32, t1: &mut f32, fa: f32, fb: f32) -> bool {
    match (fa >= 0.0, fb >= 0.0) {
        (true, true) => {}
        (false, false) => return false,
        (inside_a, _) => {
            let t = fa / (fa - fb);
            if inside_a {
                *t1 = t1.min(t);
            } else {
                *t0 = t0.max(t);
            }
        }
    }
    *t0 < *t1
}

/// The cone from a receiver point to a segment light: the triangle (apex, l0, l1).
#[derive(Clone, Copy, Debug)]
pub struct Cone {
    pub apex: Vec2,
    /// Apex to the first light endpoint; the cone sweeps counter-clockwise from here.
    pub d0: Vec2,
    pub d1: Vec2,
    pub l0: Vec2,
    pub l1: Vec2,
    /// Angle subtended by the light, in radians, in `(0, π)`.
    pub angle: f32,
    /// Which side of the light's line the apex is on (+1 or -1).
    side: f32,
}

impl Cone {
    /// `None` when the apex lies on the light's line, where the cone is degenerate.
    pub fn toward(apex: Vec2, light: Segment) -> Option<Cone> {
        let (mut l0, mut l1) = (light.a, light.b);
        let (mut d0, mut d1) = (l0 - apex, l1 - apex);
        let c = d0.cross(d1);
        if c.abs() < 1e-9 {
            return None;
        }
        if c < 0.0 {
            std::mem::swap(&mut l0, &mut l1);
            std::mem::swap(&mut d0, &mut d1);
        }
        let angle = d0.cross(d1).atan2(d0.dot(d1));
        let side = (l1 - l0).cross(apex - l0).signum();
        Some(Cone {
            apex,
            d0,
            d1,
            l0,
            l1,
            angle,
            side,
        })
    }

    /// Normalized angular position of `q` inside the cone (0 at `d0`, 1 at `d1`).
    pub fn position_of(&self, q: Vec2) -> f32 {
        let v = q - self.apex;
        let theta = self.d0.cross(v).atan2(self.d0.dot(v));
        (theta / self.angle).clamp(0.0, 1.0)
    }

    /// The angular interval `[ta, tb]` of the part of `s` that lies inside the cone
    /// and in front of the light, i.e. exactly the directions `s` blocks.
    pub fn project(&self, s: &Segment) -> Option<(f32, f32)> {
        let (mut t0, mut t1) = (0.0f32, 1.0f32);
        let in_wedge_0 = |q: Vec2| self.d0.cross(q - self.apex);
        let in_wedge_1 = |q: Vec2| (q - self.apex).cross(self.d1);
        let before_light = |q: Vec2| self.side * (self.l1 - self.l0).cross(q - self.l0);
        if !clip_half_plane(&mut t0, &mut t1, in_wedge_0(s.a), in_wedge_0(s.b))
            || !clip_half_plane(&mut t0, &mut t1, in_wedge_1(s.a), in_wedge_1(s.b))
            || !clip_half_plane(&mut t0, &mut t1, before_light(s.a), before_light(s.b))
        {
            return None;
        }
        let e = s.b - s.a;
        let ta = self.position_of(s.a + e * t0);
        let tb = self.position_of(s.a + e * t1);
        Some(if ta <= tb { (ta, tb) } else { (tb, ta) })
    }

    /// Unit direction at normalized angular position `t`.
    pub fn direction(&self, t: f32) -> Vec2 {
        let (sin, cos) = (t * self.angle).sin_cos();
        let d = self.d0.normalized();
        d * cos + d.perp() * sin
    }

    /// Distance from the apex to the light along unit direction `dir`.
    pub fn light_distance(&self, dir: Vec2) -> f32 {
        let e = self.l1 - self.l0;
        (self.l0 - self.apex).cross(e) / dir.cross(e)
    }
}

/// Does the ray `origin + t·dir` hit `s` with `0 < t < max_t`?
pub fn ray_hits(origin: Vec2, dir: Vec2, s: &Segment, max_t: f32) -> bool {
    let e = s.b - s.a;
    let denom = dir.cross(e);
    if denom.abs() < 1e-12 {
        return false;
    }
    let w = s.a - origin;
    let t = w.cross(e) / denom;
    let u = w.cross(dir) / denom;
    t > 0.0 && t < max_t && (0.0..=1.0).contains(&u)
}
