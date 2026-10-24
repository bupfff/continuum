//! Animated scenarios for the ledger experiments (gates 2–4). Frames are at
//! 60 fps in a unit square that stands in for a room about 10 m across.

use std::f32::consts::TAU;
use std::ops::Range;

use crate::geom::{Segment, Vec2, seg, v2};
use crate::scene;

pub struct Light {
    pub segment: Segment,
    pub radiance: Box<dyn Fn(u32) -> f32>,
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
    /// Named frame ranges reported separately.
    pub phases: Vec<(&'static str, Range<u32>)>,
    /// Frames shown in the image sheet.
    pub key_frames: [u32; 3],
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

/// A mostly static room, which is what most game frames look like: the gate 1
/// geometry plus a swinging door, a person walking, and a spinning fan. A second
/// light switches on at frame 60 and flickers from frame 150.
pub fn typical() -> Scenario {
    let movers = |f: u32| {
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
    };
    Scenario {
        name: "Typical: mostly static room",
        slug: "typical",
        frames: 240,
        statics: scene::gate1().occluders,
        movers: Box::new(movers),
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
        phases: vec![
            ("Movers only", 1..LIGHT_B_ON),
            ("Light B switches on", LIGHT_B_ON..LIGHT_B_ON + 1),
            ("Movers, two lights", LIGHT_B_ON + 1..FLICKER_FROM),
            ("Movers + flicker", FLICKER_FROM..240),
        ],
        key_frames: [30, LIGHT_B_ON, 200],
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
        phases: vec![("All occluders move", 1..240)],
        key_frames: [30, 120, 200],
    }
}
