//! Animated scenarios for the ledger experiments (gates 2–5). Frames are at
//! 60 fps in a unit square that stands in for a room about 10 m across.

use std::f32::consts::TAU;
use std::ops::Range;

use crate::geom::{Segment, Vec2, seg, v2};
use crate::scene;

pub struct Light {
    pub segment: Segment,
    pub radiance: Box<dyn Fn(u32) -> f32>,
}

/// Uniform light arriving from every direction, gathered through `sectors`
/// equal wide cones per receiver.
pub struct Sky {
    pub sectors: usize,
    pub radiance: Box<dyn Fn(u32) -> f32>,
}

/// Distance to the far edge of sky cones: beyond everything in the unit square.
const SKY_DISTANCE: f32 = 4.0;

/// Where a family of cones ends for a receiver.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// A light: the same segment for every receiver.
    Fixed(Segment),
    /// A sky sector: a far segment carried along with the receiver.
    Relative(Segment),
}

impl Target {
    pub fn at(self, p: Vec2) -> Segment {
        match self {
            Target::Fixed(s) => s,
            Target::Relative(s) => Segment {
                a: p + s.a,
                b: p + s.b,
            },
        }
    }
}

pub struct Scenario {
    pub name: &'static str,
    /// File-name-safe identifier.
    pub slug: &'static str,
    pub frames: u32,
    /// Occluders that never move (declared, like an engine's static mobility flag).
    pub statics: Vec<Segment>,
    /// Occluders that may move, by frame. Same count and order every frame.
    pub movers: Box<dyn Fn(u32) -> Vec<Segment>>,
    pub lights: Vec<Light>,
    pub sky: Option<Sky>,
    /// Named frame ranges reported separately.
    pub phases: Vec<(&'static str, Range<u32>)>,
    /// Frames shown in the image sheet.
    pub key_frames: [u32; 3],
    /// Fluence scale applied before tone mapping images.
    pub exposure: f32,
}

impl Scenario {
    /// One target per cone family: lights first, then sky sectors.
    pub fn targets(&self) -> Vec<Target> {
        let mut targets: Vec<Target> = self
            .lights
            .iter()
            .map(|l| Target::Fixed(l.segment))
            .collect();
        if let Some(sky) = &self.sky {
            targets.extend(sky_targets(sky.sectors));
        }
        targets
    }

    /// Radiance of each cone family at frame `f`, in the order of [`Self::targets`].
    pub fn weights(&self, f: u32) -> Vec<f32> {
        let mut w: Vec<f32> = self.lights.iter().map(|l| (l.radiance)(f)).collect();
        if let Some(sky) = &self.sky {
            w.extend(std::iter::repeat_n((sky.radiance)(f), sky.sectors));
        }
        w
    }
}

/// The full circle of directions split into `sectors` equal wide cones.
pub fn sky_targets(sectors: usize) -> Vec<Target> {
    (0..sectors)
        .map(|k| {
            let angle = |i: usize| i as f32 * TAU / sectors as f32;
            let dir = |a: f32| v2(a.cos(), a.sin()) * SKY_DISTANCE;
            Target::Relative(Segment {
                a: dir(angle(k)),
                b: dir(angle(k + 1)),
            })
        })
        .collect()
}

const LIGHT_B_ON: u32 = 60;
const FLICKER_FROM: u32 = 150;

fn rotated(center: Vec2, half_length: f32, angle: f32) -> Segment {
    let d = v2(angle.cos(), angle.sin()) * half_length;
    Segment {
        a: center - d,
        b: center + d,
    }
}

/// A swinging door, a person walking, and a spinning fan.
fn room_movers(f: u32) -> Vec<Segment> {
    let t = f as f32 / 60.0;
    let hinge = v2(0.40, 0.40);
    let door_angle = (40.0 + 40.0 * (TAU * t / 3.0).sin()).to_radians();
    let door = Segment {
        a: hinge,
        b: hinge + v2(door_angle.cos(), door_angle.sin()) * 0.08,
    };
    // Walks back and forth at about 1.5 m/s (0.0025 of the room per frame).
    let walk = (f as f32 * 0.0025) % 1.4;
    let x = 0.15 + if walk < 0.7 { walk } else { 1.4 - walk };
    let walker = seg(x - 0.015, 0.28, x + 0.015, 0.28);
    let fan = rotated(v2(0.50, 0.55), 0.025, (f as f32 * 2.0).to_radians());
    vec![door, walker, fan]
}

/// A mostly static room, which is what most game frames look like: the gate 1
/// geometry plus the room movers. A second light switches on at frame 60 and
/// flickers from frame 150.
pub fn typical() -> Scenario {
    Scenario {
        name: "Typical: mostly static room",
        slug: "typical",
        frames: 240,
        statics: scene::gate1().occluders,
        movers: Box::new(room_movers),
        lights: vec![
            Light {
                segment: seg(0.38, 0.95, 0.62, 0.95),
                radiance: Box::new(|_| 1.0),
            },
            Light {
                segment: seg(0.03, 0.10, 0.03, 0.30),
                radiance: Box::new(|f| match f {
                    f if f < LIGHT_B_ON => 0.0,
                    f if f < FLICKER_FROM => 0.8,
                    f => 0.8 * (0.85 + 0.15 * (f as f32 * 0.7).sin()),
                }),
            },
        ],
        sky: None,
        phases: vec![
            ("Movers only", 1..LIGHT_B_ON),
            ("Light B switches on", LIGHT_B_ON..LIGHT_B_ON + 1),
            ("Movers, two lights", LIGHT_B_ON + 1..FLICKER_FROM),
            ("Movers + flicker", FLICKER_FROM..240),
        ],
        key_frames: [30, LIGHT_B_ON, 200],
        exposure: 1.0,
    }
}

/// Worst case: every occluder drifts every frame, so nothing is static.
pub fn everything_moves() -> Scenario {
    let base = scene::gate1().occluders;
    let movers = move |f: u32| {
        let t = f as f32;
        base.iter()
            .enumerate()
            .map(|(i, s)| {
                let k = i as f32;
                let offset = v2((t * 0.3 + k).sin(), (t * 0.37 + 2.0 * k).cos()) * 0.004;
                Segment {
                    a: s.a + offset,
                    b: s.b + offset,
                }
            })
            .collect()
    };
    Scenario {
        name: "Worst case: everything moves",
        slug: "worst_case",
        frames: 240,
        statics: Vec::new(),
        movers: Box::new(movers),
        lights: vec![
            Light {
                segment: seg(0.38, 0.95, 0.62, 0.95),
                radiance: Box::new(|_| 1.0),
            },
            Light {
                segment: seg(0.03, 0.10, 0.03, 0.30),
                radiance: Box::new(|_| 0.8),
            },
        ],
        sky: None,
        phases: vec![("All occluders move", 1..240)],
        key_frames: [30, 120, 200],
        exposure: 1.0,
    }
}

/// Wide cones: the same room and movers, lit only by a sky gathered through 8
/// cones of 45° per receiver. Every mover is a sliver of a wide cone, and the sky
/// brightens and dims slowly like passing clouds.
pub fn sky() -> Scenario {
    Scenario {
        name: "Sky light through 8 wide cones",
        slug: "sky",
        frames: 180,
        statics: scene::gate1().occluders,
        movers: Box::new(room_movers),
        lights: Vec::new(),
        sky: Some(Sky {
            sectors: 8,
            radiance: Box::new(|f| 1.0 + 0.15 * (TAU * f as f32 / 180.0).sin()),
        }),
        phases: vec![("Movers under a changing sky", 1..180)],
        key_frames: [30, 90, 150],
        exposure: 0.12,
    }
}

/// The room movers with the walker raised to y = 0.33 and the door 0.003 off
/// its hinge (about 3 cm at room scale), so no mover ever touches a static
/// segment. The gather's tight charges assume movers pass in front of or behind
/// surfaces, never through them; crossings are detected and charged
/// conservatively, but this keeps the scene typical.
fn bounce_movers(f: u32) -> Vec<Segment> {
    let mut movers = room_movers(f);
    // A small gap at the hinge, so the door never touches the wall it swings from.
    let door = &mut movers[0];
    door.a = door.a + (door.b - door.a).normalized() * 0.003;
    let walker = &mut movers[1];
    walker.a.y = 0.33;
    walker.b.y = 0.33;
    movers
}

/// One-bounce indirect light: static segments reflect (albedo 0.7, both sides),
/// movers are black. Lights as in [`typical`]: B switches on at frame 60 and
/// flickers from frame 150. Receivers gather indirect light through 8 sky-sector
/// cones (see `sky_targets`); the lights here only light the surfaces.
pub fn bounce() -> Scenario {
    let mut s = typical();
    s.name = "One-bounce indirect light";
    s.slug = "bounce";
    s.movers = Box::new(bounce_movers);
    s.exposure = 2.0;
    s
}
