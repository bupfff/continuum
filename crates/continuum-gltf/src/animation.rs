//! glTF node animations, played through movable instances: each animated
//! instance's rotation and position at a time, from the file's translation
//! and rotation channels (step, linear and cubic-spline interpolation),
//! through the node hierarchy.

use continuum::glam::{DMat3, DMat4, DQuat, DVec3, DVec4, Mat3};

/// How a channel interpolates between keyframes (glTF's sampler modes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

/// A channel's keyframe values (for cubic splines, three per keyframe: the
/// in-tangent, the value and the out-tangent), in glTF's axes.
#[derive(Clone, Debug)]
pub(crate) enum Values {
    Translation(Vec<DVec3>),
    Rotation(Vec<DQuat>),
}

/// One animated property of one node.
#[derive(Clone, Debug)]
pub(crate) struct Channel {
    pub(crate) node: usize,
    pub(crate) interpolation: Interpolation,
    pub(crate) times: Vec<f32>,
    pub(crate) values: Values,
}

/// One of the file's animations.
#[derive(Clone, Debug)]
pub struct Animation {
    /// Its name in the file, if any.
    pub name: Option<String>,
    /// Its last keyframe's time, in seconds.
    pub duration: f32,
    pub(crate) channels: Vec<Channel>,
}

/// A node at rest: its parent, and its local translation, rotation and
/// scale (glTF's axes).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub(crate) parent: Option<usize>,
    pub(crate) translation: DVec3,
    pub(crate) rotation: DQuat,
    pub(crate) scale: DVec3,
}

/// The file's animations, and what they move.
///
/// Instances move by rotation and translation: a mover keeps its scale
/// (`Renderer::set_transform`), so scale channels and morph weights are
/// skipped (`ImportStats::animation_channels_skipped`), and so are
/// instances baked into their own geometry (non-uniform scale, shear or a
/// mirror).
#[derive(Clone, Debug, Default)]
pub struct Animations {
    /// The animations, in the file's order.
    pub list: Vec<Animation>,
    pub(crate) nodes: Vec<Node>,
    /// Per instance, its node where the instance can move (not baked).
    pub(crate) instance_nodes: Vec<Option<usize>>,
}

impl Animations {
    /// Whether the file has no animations.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// The instances animation `animation` moves: those whose node, or one
    /// of its ancestors, it animates. List them in
    /// `RendererOptions::movable`.
    pub fn instances(&self, animation: usize) -> Vec<usize> {
        let animated: Vec<usize> = self.list[animation]
            .channels
            .iter()
            .map(|c| c.node)
            .collect();
        let moves = |mut node: usize| loop {
            if animated.contains(&node) {
                return true;
            }
            match self.nodes[node].parent {
                Some(p) => node = p,
                None => return false,
            }
        };
        self.instance_nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.filter(|&n| moves(n)).map(|_| i))
            .collect()
    }

    /// Each instance animation `animation` moves, with its rotation and
    /// world position (Continuum's Z-up axes) at `time` seconds (held at
    /// the first and last keyframes outside them; loop by taking `time`
    /// modulo [`Animation::duration`]).
    pub fn pose(&self, animation: usize, time: f32) -> Vec<(usize, Mat3, DVec3)> {
        let mut local: Vec<(DVec3, DQuat, DVec3)> = self
            .nodes
            .iter()
            .map(|n| (n.translation, n.rotation, n.scale))
            .collect();
        for c in &self.list[animation].channels {
            match &c.values {
                Values::Translation(v) => {
                    local[c.node].0 =
                        sample(&c.times, v, c.interpolation, time, |a, b, s| a.lerp(b, s))
                }
                Values::Rotation(v) => {
                    local[c.node].1 =
                        sample(&c.times, v, c.interpolation, time, |a, b, s| a.slerp(b, s))
                            .normalize()
                }
            }
        }
        let mut world: Vec<Option<DMat4>> = vec![None; self.nodes.len()];
        let (c, c_inv) = crate::axes();
        self.instances(animation)
            .into_iter()
            .map(|i| {
                let node = self.instance_nodes[i].expect("a movable instance");
                let m = c * world_of(node, &self.nodes, &local, &mut world) * c_inv;
                let a = DMat3::from_mat4(m);
                let scales = DVec3::new(a.x_axis.length(), a.y_axis.length(), a.z_axis.length());
                let rotation = DMat3::from_cols(
                    a.x_axis / scales.x,
                    a.y_axis / scales.y,
                    a.z_axis / scales.z,
                );
                (i, rotation.as_mat3(), m.w_axis.truncate())
            })
            .collect()
    }
}

/// Node `node`'s world matrix from the local transforms, memoized.
fn world_of(
    node: usize,
    nodes: &[Node],
    local: &[(DVec3, DQuat, DVec3)],
    world: &mut [Option<DMat4>],
) -> DMat4 {
    if let Some(m) = world[node] {
        return m;
    }
    let (t, r, s) = local[node];
    let own = DMat4::from_scale_rotation_translation(s, r, t);
    let m = match nodes[node].parent {
        Some(p) => world_of(p, nodes, local, world) * own,
        None => own,
    };
    world[node] = Some(m);
    m
}

/// Values that cubic splines interpolate component-wise.
trait Components: Copy {
    fn to4(self) -> DVec4;
    fn from4(v: DVec4) -> Self;
}

impl Components for DVec3 {
    fn to4(self) -> DVec4 {
        self.extend(0.0)
    }
    fn from4(v: DVec4) -> Self {
        v.truncate()
    }
}

impl Components for DQuat {
    fn to4(self) -> DVec4 {
        DVec4::from(self)
    }
    fn from4(v: DVec4) -> Self {
        DQuat::from_vec4(v)
    }
}

/// A channel's value at `time`, as glTF defines its interpolation (`lerp`
/// for linear: slerp for rotations).
fn sample<T: Components>(
    times: &[f32],
    values: &[T],
    interpolation: Interpolation,
    time: f32,
    lerp: impl Fn(T, T, f64) -> T,
) -> T {
    let cubic = interpolation == Interpolation::CubicSpline;
    let value = |k: usize| if cubic { values[3 * k + 1] } else { values[k] };
    let n = times.len();
    if n == 0 || time <= times[0] {
        return value(0);
    }
    if time >= times[n - 1] {
        return value(n - 1);
    }
    let k = times.partition_point(|&t| t <= time) - 1;
    let dt = (times[k + 1] - times[k]) as f64;
    let s = (time - times[k]) as f64 / dt;
    match interpolation {
        Interpolation::Step => value(k),
        Interpolation::Linear => lerp(value(k), value(k + 1), s),
        Interpolation::CubicSpline => {
            let (v0, b0) = (values[3 * k + 1].to4(), values[3 * k + 2].to4());
            let (a1, v1) = (values[3 * k + 3].to4(), values[3 * k + 4].to4());
            let (s2, s3) = (s * s, s * s * s);
            T::from4(
                v0 * (2.0 * s3 - 3.0 * s2 + 1.0)
                    + b0 * (dt * (s3 - 2.0 * s2 + s))
                    + v1 * (-2.0 * s3 + 3.0 * s2)
                    + a1 * (dt * (s3 - s2)),
            )
        }
    }
}
