//! glTF node animations, played through movable instances: each animated
//! instance's rotation and position at a time, from the file's translation
//! and rotation channels (step, linear and cubic-spline interpolation),
//! through the node hierarchy. And skins and morph targets (M12 gate 3):
//! deforming geometry's triangles at a time, from its joints' world
//! matrices and its morph weights.

use continuum::glam::{DMat3, DMat4, DQuat, DVec3, DVec4, Mat3, Vec3};

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
    Scale(Vec<DVec3>),
    /// Morph weights: a weight per target per keyframe (or per cubic
    /// triple's element), flattened.
    Weights(Vec<f32>),
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

/// A skin: its joints' nodes and their inverse bind matrices (glTF's axes).
#[derive(Clone, Debug)]
pub(crate) struct Skin {
    pub(crate) joints: Vec<usize>,
    pub(crate) inverse_bind: Vec<DMat4>,
}

/// A deforming geometry (M12 gate 3): a skinned or morphing primitive placed
/// by a node, per triangle corner (in its geometry's order, glTF's axes,
/// mesh space): its rest position and normal (the normal's sign: -1 on a
/// double-sided primitive's back faces), its joints and weights, and its
/// morph targets' deltas; the mesh's default weights. A skinned geometry
/// lies in world space (its instance at the origin); a morphing one in its
/// node's space (its instance where its node is).
#[derive(Clone, Debug)]
pub(crate) struct Deformer {
    pub(crate) geometry: usize,
    pub(crate) node: usize,
    pub(crate) skin: Option<usize>,
    pub(crate) rest: Vec<DVec3>,
    pub(crate) normals: Option<Vec<DVec3>>,
    pub(crate) signs: Vec<f64>,
    pub(crate) joints: Vec<[u16; 4]>,
    pub(crate) weights: Vec<[f32; 4]>,
    pub(crate) targets: Vec<(Vec<DVec3>, Option<Vec<DVec3>>)>,
    pub(crate) default_weights: Vec<f32>,
}

/// A deforming geometry's pose at a time ([`Animations::deform`]): its
/// triangles and each corner's normal, in Continuum's Z-up axes, for
/// `Renderer::set_vertices`.
#[derive(Clone, Debug)]
pub struct Pose {
    /// The geometry, one of [`Animations::deforming`].
    pub geometry: usize,
    /// Its triangles, as many as its geometry's and in their order.
    pub triangles: Vec<[Vec3; 3]>,
    /// Each triangle corner's shading normal.
    pub normals: Vec<[Vec3; 3]>,
}

/// The file's animations, and what they move.
///
/// Instances move by rotation and translation: a mover keeps its scale
/// (`Renderer::set_transform`), so instances baked into their own geometry
/// (non-uniform scale, shear or a mirror) do not move. Skinned and
/// morphing meshes deform instead ([`Animations::deform`], M12 gate 3).
#[derive(Clone, Debug, Default)]
pub struct Animations {
    /// The animations, in the file's order.
    pub list: Vec<Animation>,
    pub(crate) nodes: Vec<Node>,
    /// Per instance, its node where the instance can move (not baked, not
    /// deforming).
    pub(crate) instance_nodes: Vec<Option<usize>>,
    pub(crate) skins: Vec<Skin>,
    pub(crate) deformers: Vec<Deformer>,
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

    /// The nodes' local transforms and morph weights (where animated) at
    /// `time` under `animations`' channels.
    #[allow(clippy::type_complexity)]
    fn locals(
        &self,
        animations: &[usize],
        time: f32,
    ) -> (Vec<(DVec3, DQuat, DVec3)>, Vec<Option<Vec<f32>>>) {
        let mut local: Vec<(DVec3, DQuat, DVec3)> = self
            .nodes
            .iter()
            .map(|n| (n.translation, n.rotation, n.scale))
            .collect();
        let mut weights: Vec<Option<Vec<f32>>> = vec![None; self.nodes.len()];
        for &a in animations {
            for c in &self.list[a].channels {
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
                    Values::Scale(v) => {
                        local[c.node].2 =
                            sample(&c.times, v, c.interpolation, time, |a, b, s| a.lerp(b, s))
                    }
                    Values::Weights(v) => {
                        weights[c.node] = Some(sample_weights(&c.times, v, c.interpolation, time))
                    }
                }
            }
        }
        (local, weights)
    }

    /// The animations to play together: the first, and each other that
    /// animates none of the nodes those before it do (a file that animates
    /// a rig and a mesh's morph weights apart plays both; one with
    /// alternative clips of the same rig, as the Fox's Survey, Walk and
    /// Run, plays the first alone).
    pub fn together(&self) -> Vec<usize> {
        let mut taken: Vec<usize> = Vec::new();
        let mut out = Vec::new();
        for (k, a) in self.list.iter().enumerate() {
            let nodes: Vec<usize> = a.channels.iter().map(|c| c.node).collect();
            if k == 0 || nodes.iter().all(|n| !taken.contains(n)) {
                taken.extend(nodes);
                out.push(k);
            }
        }
        out
    }

    /// The geometries that deform (skinned or morphing meshes, M12 gate
    /// 3): list them in `RendererOptions::deforming`.
    pub fn deforming(&self) -> Vec<usize> {
        self.deformers.iter().map(|d| d.geometry).collect()
    }

    /// Each deforming geometry's pose at `time` seconds under the channels
    /// of `animations` (several play together: a file often animates a rig
    /// and a mesh's morph weights apart): the morph targets' deltas by the
    /// weights, then linear blend skinning by the joints' world matrices
    /// times their inverse bind matrices. Pass each to
    /// `Renderer::set_vertices`.
    pub fn deform(&self, animations: &[usize], time: f32) -> Vec<Pose> {
        let (local, weights) = self.locals(animations, time);
        let mut world: Vec<Option<DMat4>> = vec![None; self.nodes.len()];
        let (c, _) = crate::axes();
        let c3 = DMat3::from_mat4(c);
        self.deformers
            .iter()
            .map(|d| {
                let w = weights[d.node].as_deref().unwrap_or(&d.default_weights);
                let skin: Option<Vec<DMat4>> = d.skin.map(|k| {
                    let s = &self.skins[k];
                    s.joints
                        .iter()
                        .zip(&s.inverse_bind)
                        .map(|(&j, ib)| world_of(j, &self.nodes, &local, &mut world) * *ib)
                        .collect()
                });
                let n = d.rest.len();
                let mut tris = Vec::with_capacity(n / 3);
                let mut normals = Vec::with_capacity(n / 3);
                let corner = |k: usize| {
                    let mut p = d.rest[k];
                    let mut nr = d.normals.as_ref().map(|ns| ns[k]);
                    for (t, (dp, dn)) in d.targets.iter().enumerate() {
                        let wt = w.get(t).copied().unwrap_or(0.0) as f64;
                        if wt != 0.0 {
                            p += dp[k] * wt;
                            if let (Some(nr), Some(dn)) = (nr.as_mut(), dn) {
                                *nr += dn[k] * wt;
                            }
                        }
                    }
                    if let Some(joints) = &skin {
                        let m = (0..4).fold(DMat4::ZERO, |acc, i| {
                            let wt = d.weights[k][i] as f64;
                            if wt == 0.0 {
                                acc
                            } else {
                                acc + joints[d.joints[k][i] as usize] * wt
                            }
                        });
                        p = m.transform_point3(p);
                        nr = nr.map(|nr| m.transform_vector3(nr));
                    }
                    let to = |v: DVec3| (c3 * v).as_vec3();
                    (
                        to(p),
                        nr.map(|nr| (to(nr) * d.signs[k] as f32).normalize_or_zero()),
                    )
                };
                for t in 0..n / 3 {
                    let v = [corner(3 * t), corner(3 * t + 1), corner(3 * t + 2)];
                    let tri = [v[0].0, v[1].0, v[2].0];
                    let face = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
                    tris.push(tri);
                    normals.push([0, 1, 2].map(|i| v[i].1.unwrap_or(face)));
                }
                Pose {
                    geometry: d.geometry,
                    triangles: tris,
                    normals,
                }
            })
            .collect()
    }

    /// Each instance animation `animation` moves, with its rotation and
    /// world position (Continuum's Z-up axes) at `time` seconds (held at
    /// the first and last keyframes outside them; loop by taking `time`
    /// modulo [`Animation::duration`]).
    pub fn pose(&self, animation: usize, time: f32) -> Vec<(usize, Mat3, DVec3)> {
        let (local, _) = self.locals(&[animation], time);
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

/// A morph-weights channel's weights at `time` (`values` holds the targets'
/// weights per keyframe, three sets a keyframe for cubic splines).
fn sample_weights(
    times: &[f32],
    values: &[f32],
    interpolation: Interpolation,
    time: f32,
) -> Vec<f32> {
    let keys = times.len().max(1);
    let per = if interpolation == Interpolation::CubicSpline {
        3
    } else {
        1
    };
    let n = values.len() / (keys * per);
    (0..n)
        .map(|t| {
            let series: Vec<DVec3> = (0..keys * per)
                .map(|k| DVec3::new(values[k * n + t] as f64, 0.0, 0.0))
                .collect();
            sample(times, &series, interpolation, time, |a, b, s| a.lerp(b, s)).x as f32
        })
        .collect()
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
