//! Bounding volume hierarchies for the GPU: one over meshes, and one over each
//! mesh's triangles.

use bytemuck::{Pod, Zeroable};
use continuum::glam::{Mat3, Vec3};

use crate::Tri;

/// Meshes flattened for the GPU. Bindings may not be empty, so empty inputs
/// get unread padding.
pub(crate) struct Flat {
    /// Triangles as `vec4` vertices, three per triangle.
    pub tris: Vec<[f32; 4]>,
    /// Per object: `[first triangle, count, triangle hierarchy root (or NONE),
    /// transform (or NONE: the triangles are in world space)]`.
    pub ranges: Vec<[u32; 4]>,
    /// Every mesh's triangle hierarchy: its root first, then its subtree.
    pub blas: Vec<NodeGpu>,
    /// Triangle indices referenced by `blas` leaves; mesh `m`'s are at the same
    /// offsets as its triangles.
    pub blas_items: Vec<u32>,
    /// [`COVERAGE_SAMPLES`] per `blas` node: the projected area of its
    /// front-facing triangles seen from each of the 26 cube directions.
    pub blas_cov: Vec<f32>,
}

pub(crate) const NONE: u32 = u32::MAX;

pub(crate) fn flatten(meshes: &[Vec<Tri>]) -> Flat {
    let mut flat = Flat {
        tris: Vec::new(),
        ranges: Vec::new(),
        blas: Vec::new(),
        blas_items: Vec::new(),
        blas_cov: Vec::new(),
    };
    for mesh in meshes {
        let first = (flat.tris.len() / 3) as u32;
        let root = if mesh.is_empty() {
            NONE
        } else {
            let root = flat.blas.len() as u32;
            let (nodes, items, cov) = triangle_hierarchy(mesh, root, first);
            flat.blas.extend(nodes);
            flat.blas_items.extend(items);
            flat.blas_cov.extend(cov);
            root
        };
        flat.ranges.push([first, mesh.len() as u32, root, NONE]);
        for t in mesh {
            flat.tris.extend(t.iter().map(|v| [v.x, v.y, v.z, 0.0]));
        }
    }
    if flat.tris.is_empty() {
        flat.tris.extend([[0.0; 4]; 3]);
    }
    if flat.ranges.is_empty() {
        flat.ranges.push([0, 0, NONE, NONE]);
    }
    if flat.blas.is_empty() {
        flat.blas.push(NodeGpu::zeroed());
        flat.blas_items.push(0);
        flat.blas_cov.extend([0.0; COVERAGE_SAMPLES]);
    }
    flat
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct NodeGpu {
    lo: [f32; 4],
    hi: [f32; 4],
    /// Leaf: `[first item, item count, 0, 0]`; interior: `[left child, 0, 0, 0]`
    /// with the right child at `left + 1`.
    info: [u32; 4],
}

impl NodeGpu {
    /// The node with its children's index moved by `nodes` (interior) or its
    /// first item's by `items` (leaf), for moving a hierarchy (M14 gate 4).
    pub(crate) fn rebased(&self, nodes: i64, items: i64) -> NodeGpu {
        let mut n = *self;
        let by = if n.info[1] > 0 { items } else { nodes };
        n.info[0] = (n.info[0] as i64 + by) as u32;
        n
    }

    /// Whether it is a leaf, and an interior node's left child (the right
    /// one follows it).
    pub(crate) fn is_leaf(&self) -> bool {
        self.info[1] > 0
    }

    /// See `is_leaf`.
    pub(crate) fn left(&self) -> u32 {
        self.info[0]
    }

    /// This interior node with its children at `left` and `left + 1`.
    pub(crate) fn with_left(&self, left: u32) -> NodeGpu {
        let mut n = *self;
        n.info = [left, 0, 0, 0];
        n
    }

    /// This node's box, marked as the frontier of a lighting level's
    /// resident stub (M14 gate 8: `info.w` 1): what lies below is absent.
    pub(crate) fn absent(&self) -> NodeGpu {
        let mut n = *self;
        n.info = [0, 0, 0, 1];
        n
    }

    /// The node's box.
    pub(crate) fn bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(self.lo[0], self.lo[1], self.lo[2]),
            Vec3::new(self.hi[0], self.hi[1], self.hi[2]),
        )
    }
}

/// Median splits on the longest centroid axis over `items` (indices into
/// `bounds`), writing `nodes[node]` and its subtree. Leaves hold at most `leaf`
/// items, recorded as `[item_base + start, count]`; children as `node_base +
/// index`. The tree's shape depends only on the item count.
#[allow(clippy::too_many_arguments)]
fn split(
    nodes: &mut Vec<NodeGpu>,
    node: usize,
    items: &mut [u32],
    start: usize,
    bounds: &[(Vec3, Vec3)],
    leaf: usize,
    node_base: u32,
    item_base: u32,
) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let (mut clo, mut chi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for &m in items.iter() {
        let (a, b) = bounds[m as usize];
        (lo, hi) = (lo.min(a), hi.max(b));
        let c = (a + b) * 0.5;
        (clo, chi) = (clo.min(c), chi.max(c));
    }
    nodes[node].lo = [lo.x, lo.y, lo.z, 0.0];
    nodes[node].hi = [hi.x, hi.y, hi.z, 0.0];
    if items.len() <= leaf {
        nodes[node].info = [item_base + start as u32, items.len() as u32, 0, 0];
        return;
    }
    let e = chi - clo;
    let axis = if e.x >= e.y && e.x >= e.z {
        0
    } else if e.y >= e.z {
        1
    } else {
        2
    };
    let mid = items.len() / 2;
    let key = |m: &u32| bounds[*m as usize].0[axis] + bounds[*m as usize].1[axis];
    items.select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)));
    let left = nodes.len();
    nodes.push(NodeGpu::zeroed());
    nodes.push(NodeGpu::zeroed());
    nodes[node].info = [node_base + left as u32, 0, 0, 0];
    let (l, r) = items.split_at_mut(mid);
    split(nodes, left, l, start, bounds, leaf, node_base, item_base);
    split(
        nodes,
        left + 1,
        r,
        start + mid,
        bounds,
        leaf,
        node_base,
        item_base,
    );
}

/// A bounding volume hierarchy over meshes (at most two meshes per leaf),
/// flattened for the GPU: nodes and the mesh indices their leaves refer to.
pub(crate) fn mesh_hierarchy(meshes: &[Vec<Tri>]) -> (Vec<NodeGpu>, Vec<u32>) {
    mesh_hierarchy_of(meshes, |_| true)
}

/// [`mesh_hierarchy`] over only the meshes `keep` selects (indices stay global).
pub(crate) fn mesh_hierarchy_of(
    meshes: &[Vec<Tri>],
    keep: impl Fn(usize) -> bool,
) -> (Vec<NodeGpu>, Vec<u32>) {
    let bounds: Vec<(Vec3, Vec3)> = meshes.iter().map(|m| padded(mesh_bounds(m))).collect();
    object_hierarchy(&bounds, keep)
}

/// A hierarchy over objects with the given (padded) world bounds, at most two
/// per leaf, over the objects `keep` selects (indices stay global).
pub(crate) fn object_hierarchy(
    bounds: &[(Vec3, Vec3)],
    keep: impl Fn(usize) -> bool,
) -> (Vec<NodeGpu>, Vec<u32>) {
    let mut items: Vec<u32> = (0..bounds.len() as u32)
        .filter(|&m| keep(m as usize))
        .collect();
    let mut nodes = vec![NodeGpu::zeroed()];
    if items.is_empty() {
        // A leaf holding mesh 0, which must then have no triangles: true when
        // there are no meshes (`flatten` pads an empty range). Callers never
        // traverse a hierarchy that filtered every mesh out.
        items.push(0);
        nodes[0].info = [0, 1, 0, 0];
        return (nodes, items);
    }
    // Objects far larger than most (a 2 km ground among 1 m boulders) get a
    // subtree of their own under the root: otherwise every node above them
    // takes their bounds, and every cone descends that chain too (M5 gate 7).
    let diagonal = |m: u32| {
        let (lo, hi) = bounds[m as usize];
        (hi - lo).length()
    };
    let mut sizes: Vec<f32> = items.iter().map(|&m| diagonal(m)).collect();
    let mid = sizes.len() / 2;
    let median = *sizes.select_nth_unstable_by(mid, f32::total_cmp).1;
    let (big, small): (Vec<u32>, Vec<u32>) = items
        .iter()
        .partition(|&&m| diagonal(m) > OVERSIZED * median);
    if big.is_empty() || small.is_empty() {
        split(&mut nodes, 0, &mut items, 0, bounds, 2, 0, 0);
        return (nodes, items);
    }
    let big_count = big.len();
    items = big.into_iter().chain(small).collect();
    nodes.push(NodeGpu::zeroed());
    nodes.push(NodeGpu::zeroed());
    let (head, tail) = items.split_at_mut(big_count);
    split(&mut nodes, 1, head, 0, bounds, 2, 0, 0);
    split(&mut nodes, 2, tail, big_count, bounds, 2, 0, 0);
    let (a, b) = (nodes[1], nodes[2]);
    nodes[0].lo = [
        a.lo[0].min(b.lo[0]),
        a.lo[1].min(b.lo[1]),
        a.lo[2].min(b.lo[2]),
        0.0,
    ];
    nodes[0].hi = [
        a.hi[0].max(b.hi[0]),
        a.hi[1].max(b.hi[1]),
        a.hi[2].max(b.hi[2]),
        0.0,
    ];
    nodes[0].info = [1, 0, 0, 0];
    (nodes, items)
}

/// Objects whose bounds' diagonal exceeds this many times the median get a
/// subtree of their own under the root.
const OVERSIZED: f32 = 64.0;

/// A copy of a geometry placed in the world: `world = scale · rotation · p +
/// translation`, with `rotation` orthonormal and `translation` relative to the
/// render origin.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    pub geometry: usize,
    pub rotation: Mat3,
    pub scale: f32,
    pub translation: Vec3,
    /// One less the object's opacity if it is alpha-blended (M10 gate 16;
    /// 0: opaque): shadows from it keep this share of the light.
    pub clear: f32,
}

impl Placement {
    /// The transform as the shader reads it (`xforms`): rows 0–2 of the
    /// rotation with the translation in `w`, then `[scale, 1 / scale, clear,
    /// 0]`.
    pub(crate) fn rows(&self) -> [[f32; 4]; 4] {
        let r = self.rotation.transpose();
        let t = self.translation;
        [
            [r.x_axis.x, r.x_axis.y, r.x_axis.z, t.x],
            [r.y_axis.x, r.y_axis.y, r.y_axis.z, t.y],
            [r.z_axis.x, r.z_axis.y, r.z_axis.z, t.z],
            [self.scale, 1.0 / self.scale, self.clear, 0.0],
        ]
    }

    /// The padded world bounds of an object-space box `(lo, hi)` placed here.
    pub(crate) fn world_bounds(&self, (lo, hi): (Vec3, Vec3)) -> (Vec3, Vec3) {
        let (mut wlo, mut whi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for c in 0..8 {
            let p = Vec3::new(
                if c & 1 == 0 { lo.x } else { hi.x },
                if c & 2 == 0 { lo.y } else { hi.y },
                if c & 4 == 0 { lo.z } else { hi.z },
            );
            let w = self.rotation * p * self.scale + self.translation;
            (wlo, whi) = (wlo.min(w), whi.max(w));
        }
        padded((wlo, whi))
    }
}

/// `info.z` of object-hierarchy nodes whose coverage is not prefiltered:
/// cones always descend them (the movers' subtree, whose boxes change).
const NO_SPLAT: u32 = 1;

/// Where the movable objects' subtree lives in a hierarchy from
/// [`object_hierarchy_with_movers`]: its root is node 2 (the root's second
/// child; node 1 roots the static objects), its other nodes are the
/// `nodes` from `first_node`, its items the `objects.len()` from
/// `first_item`.
#[derive(Clone, Debug)]
pub(crate) struct MoverTree {
    /// The static objects' root (node 1), for the root's bounds.
    pub fixed_root: NodeGpu,
    pub first_node: u32,
    pub nodes: u32,
    pub first_item: u32,
    /// The movable objects.
    pub objects: Vec<u32>,
}

/// An object hierarchy whose root's first child roots the static objects
/// (as [`object_hierarchy`] builds them) and whose second child roots the
/// `movable` ones, a subtree rebuilt with [`mover_subtree`] as they move.
/// Neither the root nor the movers' nodes carry prefiltered coverage.
pub(crate) fn object_hierarchy_with_movers(
    bounds: &[(Vec3, Vec3)],
    movable: &[u32],
) -> (Vec<NodeGpu>, Vec<u32>, MoverTree) {
    let mut is_movable = vec![false; bounds.len()];
    for &m in movable {
        is_movable[m as usize] = true;
    }
    assert!(
        is_movable.iter().any(|&m| !m),
        "a scene with movers needs a static object"
    );
    let (fixed, mut items) = object_hierarchy(bounds, |m| !is_movable[m]);
    // Static node i ≥ 1 goes to i + 2; node 0 to 1.
    let place = |i: u32| if i == 0 { 1 } else { i + 2 };
    let mut tree = MoverTree {
        fixed_root: NodeGpu::zeroed(),
        first_node: fixed.len() as u32 + 2,
        nodes: 0,
        first_item: items.len() as u32,
        objects: movable.to_vec(),
    };
    let mover_bounds: Vec<(Vec3, Vec3)> = movable.iter().map(|&m| bounds[m as usize]).collect();
    let (root, region, mover_items) = mover_subtree(&tree, &mover_bounds);
    tree.nodes = region.len() as u32;
    let mut nodes = vec![NodeGpu::zeroed(); tree.first_node as usize];
    for (i, n) in fixed.iter().enumerate() {
        let mut n = *n;
        if n.info[1] == 0 {
            n.info[0] = place(n.info[0]);
        }
        nodes[place(i as u32) as usize] = n;
    }
    nodes[2] = root;
    nodes.extend(region);
    nodes[0] = union_root(nodes[1], nodes[2]);
    tree.fixed_root = nodes[1];
    items.extend(mover_items);
    (nodes, items, tree)
}

/// The root over the static subtree (node 1) and the movers' (node 2).
pub(crate) fn union_root(a: NodeGpu, b: NodeGpu) -> NodeGpu {
    let mut root = NodeGpu::zeroed();
    for k in 0..3 {
        root.lo[k] = a.lo[k].min(b.lo[k]);
        root.hi[k] = a.hi[k].max(b.hi[k]);
    }
    root.info = [1, 0, NO_SPLAT, 0];
    root
}

/// The movers' subtree over their current (padded) world bounds, in
/// `tree.objects` order: its root (node 2), its other nodes (from
/// `tree.first_node`) and its items (from `tree.first_item`). Its shape
/// depends only on the movers' count, so it always fits its place.
pub(crate) fn mover_subtree(
    tree: &MoverTree,
    bounds: &[(Vec3, Vec3)],
) -> (NodeGpu, Vec<NodeGpu>, Vec<u32>) {
    let mut local = vec![NodeGpu::zeroed()];
    let mut order: Vec<u32> = (0..bounds.len() as u32).collect();
    split(
        &mut local,
        0,
        &mut order,
        0,
        bounds,
        2,
        tree.first_node - 1,
        tree.first_item,
    );
    for n in &mut local {
        n.info[2] = NO_SPLAT;
    }
    let items = order.iter().map(|&k| tree.objects[k as usize]).collect();
    let root = local[0];
    (root, local.split_off(1), items)
}

/// A box around the region where a cone from a point to the light box
/// `l` can meet the box `m` (M9): the points `m + t (m − l)`, `t ≥ 0`
/// (the region behind `m` as seen from `l`), clipped to `s`, which holds
/// every point that matters. `None` if the region misses `s`. For each `t`
/// those points fill a box whose bounds are linear in `t`, so the region's
/// box spans the boxes at the ends of the range of `t` meeting `s`.
pub(crate) fn shadow_region(
    m: (Vec3, Vec3),
    l: (Vec3, Vec3),
    s: (Vec3, Vec3),
) -> Option<(Vec3, Vec3)> {
    let d_lo = m.0 - l.1;
    let d_hi = m.1 - l.0;
    let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
    for a in 0..3 {
        // m.lo + t d_lo <= s.hi
        let (b, c) = (d_lo[a], s.1[a] - m.0[a]);
        if b > 0.0 {
            t1 = t1.min(c / b);
        } else if b < 0.0 {
            t0 = t0.max(c / b);
        } else if c < 0.0 {
            return None;
        }
        // m.hi + t d_hi >= s.lo
        let (b, c) = (d_hi[a], s.0[a] - m.1[a]);
        if b > 0.0 {
            t0 = t0.max(c / b);
        } else if b < 0.0 {
            t1 = t1.min(c / b);
        } else if c > 0.0 {
            return None;
        }
    }
    if t0 > t1 * (1.0 + 1e-6) + 1e-6 {
        return None;
    }
    if !t1.is_finite() {
        return Some(s);
    }
    let lo = (m.0 + d_lo * t0).min(m.0 + d_lo * t1).max(s.0);
    let hi = (m.1 + d_hi * t0).max(m.1 + d_hi * t1).min(s.1);
    // A margin for rounding: the exact test follows on the GPU.
    let pad = 1e-4 * lo.abs().max(hi.abs()).max_element().max(1.0);
    Some((lo - pad, hi + pad))
}

/// Objects placing shared geometry, flattened: each geometry's triangles and
/// hierarchy once, and per object a range pointing at them with a transform
/// (`[row 0, row 1, row 2]` of the rotation with the translation in `w`, then
/// `[scale, 1 / scale, 0, 0]`). Also returns each object's padded world bounds.
#[allow(clippy::type_complexity)]
pub(crate) fn flatten_instanced(
    geometries: &[Vec<Tri>],
    objects: &[Placement],
) -> (Flat, Vec<[f32; 4]>, Vec<(Vec3, Vec3)>, Vec<[u32; 4]>) {
    let mut flat = flatten(geometries);
    // A blended geometry's coverage samples scale by its opacity (M10 gate
    // 16): its nodes are its root's and those after it, up to the next
    // geometry's root.
    let mut roots: Vec<(usize, u32)> = flat
        .ranges
        .iter()
        .enumerate()
        .filter(|(_, r)| r[2] != NONE)
        .map(|(g, r)| (g, r[2]))
        .collect();
    roots.sort_by_key(|&(_, root)| root);
    for (k, &(g, root)) in roots.iter().enumerate() {
        let Some(o) = objects.iter().find(|o| o.geometry == g) else {
            continue;
        };
        if o.clear <= 0.0 {
            continue;
        }
        let end = roots.get(k + 1).map_or(flat.blas.len() as u32, |r| r.1);
        let opacity = 1.0 - o.clear;
        for c in
            &mut flat.blas_cov[root as usize * COVERAGE_SAMPLES..end as usize * COVERAGE_SAMPLES]
        {
            *c *= opacity;
        }
    }
    let ranges = std::mem::take(&mut flat.ranges);
    let geometry_ranges = ranges.clone();
    let local: Vec<(Vec3, Vec3)> = geometries.iter().map(|g| mesh_bounds(g)).collect();
    let mut transforms = Vec::with_capacity(4 * objects.len().max(1));
    let mut bounds = Vec::with_capacity(objects.len());
    for (k, o) in objects.iter().enumerate() {
        let [first, count, root, _] = ranges[o.geometry];
        flat.ranges.push([first, count, root, k as u32]);
        transforms.extend(o.rows());
        bounds.push(o.world_bounds(local[o.geometry]));
    }
    if flat.ranges.is_empty() {
        flat.ranges.push([0, 0, NONE, NONE]);
    }
    if transforms.is_empty() {
        transforms.extend([[0.0; 4]; 4]);
    }
    (flat, transforms, bounds, geometry_ranges)
}

/// One mesh's triangle hierarchy (at most [`LEAF_TRIANGLES`] per leaf): nodes
/// numbered from `root`, triangle indices numbered from `first` (stored at item
/// positions from `first`), and each node's coverage samples.
pub(crate) fn triangle_hierarchy(
    mesh: &[Tri],
    root: u32,
    first: u32,
) -> (Vec<NodeGpu>, Vec<u32>, Vec<f32>) {
    let bounds: Vec<(Vec3, Vec3)> = mesh.iter().map(|t| padded(mesh_bounds(&[*t]))).collect();
    let mut items: Vec<u32> = (0..mesh.len() as u32).collect();
    let mut nodes = vec![NodeGpu::zeroed()];
    split(
        &mut nodes,
        0,
        &mut items,
        0,
        &bounds,
        LEAF_TRIANGLES,
        root,
        first,
    );
    // Children always follow their parent, so a reverse sweep sees children first.
    let dirs = coverage_directions();
    let mut cov = vec![0.0f32; nodes.len() * COVERAGE_SAMPLES];
    for n in (0..nodes.len()).rev() {
        let [x, count, _, _] = nodes[n].info;
        let (head, tail) = cov.split_at_mut((n + 1) * COVERAGE_SAMPLES);
        let out = &mut head[n * COVERAGE_SAMPLES..];
        if count > 0 {
            let start = (x - first) as usize;
            for &t in &items[start..start + count as usize] {
                let t = mesh[t as usize];
                let area_normal = (t[1] - t[0]).cross(t[2] - t[0]) * 0.5;
                for (o, d) in out.iter_mut().zip(&dirs) {
                    *o += area_normal.dot(*d).max(0.0);
                }
            }
        } else {
            let left = (x - root) as usize;
            // Both children come after `n`, in `tail`.
            let base = (left - n - 1) * COVERAGE_SAMPLES;
            for k in 0..COVERAGE_SAMPLES {
                out[k] = tail[base + k] + tail[base + COVERAGE_SAMPLES + k];
            }
        }
    }
    (nodes, items.iter().map(|&t| first + t).collect(), cov)
}

/// The projected front area of a node with coverage samples `cov` (one
/// node's [`COVERAGE_SAMPLES`]) seen from direction `v` (unit, toward the
/// viewer): bilinear on the face of the cube of sample directions `v` points
/// through, as the shader's `node_front_area`.
fn front_area(cov: &[f32], v: Vec3) -> f32 {
    let a = v.abs();
    let axis = if a.y > a.x && a.y >= a.z {
        1
    } else if a.z > a.x && a.z > a.y {
        2
    } else {
        0
    };
    let (main, s, t) = match axis {
        0 => (v.x, v.y / a.x, v.z / a.x),
        1 => (v.y, v.x / a.y, v.z / a.y),
        _ => (v.z, v.x / a.z, v.y / a.z),
    };
    let m = if main >= 0.0 { 1 } else { -1 };
    let s0 = if s < 0.0 { -1 } else { 0 };
    let t0 = if t < 0.0 { -1 } else { 0 };
    let (fs, ft) = (s - s0 as f32, t - t0 as f32);
    let sample = |i: i32, j: i32, k: i32| cov[(9 * (i + 1) + 3 * (j + 1) + (k + 1)) as usize];
    let f: [f32; 4] = std::array::from_fn(|c| {
        let gs = s0 + (c as i32 & 1);
        let gt = t0 + (c as i32 >> 1);
        match axis {
            0 => sample(m, gs, gt),
            1 => sample(gs, m, gt),
            _ => sample(gs, gt, m),
        }
    });
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(lerp(f[0], f[1], fs), lerp(f[2], f[3], fs), ft)
}

/// Directional coverage of the object hierarchy (M5 gate 3): per node,
/// [`COVERAGE_SAMPLES`] summing its objects' projected front areas, each from
/// its geometry's root coverage (`blas_cov` at `roots[object]`) seen from the
/// sample direction rotated into the object and scaled by its scale squared.
pub(crate) fn object_coverage(
    nodes: &[NodeGpu],
    items: &[u32],
    blas_cov: &[f32],
    roots: &[u32],
    objects: &[Placement],
) -> Vec<f32> {
    let dirs = coverage_directions();
    let per_object: Vec<[f32; COVERAGE_SAMPLES]> = objects
        .iter()
        .zip(roots)
        .map(|(o, &root)| {
            let mut out = [0.0f32; COVERAGE_SAMPLES];
            if root == NONE {
                return out;
            }
            let cov = &blas_cov[root as usize * COVERAGE_SAMPLES..][..COVERAGE_SAMPLES];
            let to_object = o.rotation.transpose();
            for (n, d) in dirs.iter().enumerate() {
                if *d != Vec3::ZERO {
                    out[n] = o.scale * o.scale * front_area(cov, to_object * *d);
                }
            }
            out
        })
        .collect();
    // Children always follow their parent: a reverse sweep sees them first.
    let mut cov = vec![0.0f32; nodes.len() * COVERAGE_SAMPLES];
    for n in (0..nodes.len()).rev() {
        let [x, count, _, _] = nodes[n].info;
        let (head, tail) = cov.split_at_mut((n + 1) * COVERAGE_SAMPLES);
        let out = &mut head[n * COVERAGE_SAMPLES..];
        if count > 0 {
            for &m in &items[x as usize..(x + count) as usize] {
                for (o, v) in out.iter_mut().zip(&per_object[m as usize]) {
                    *o += v;
                }
            }
        } else {
            let base = (x as usize - n - 1) * COVERAGE_SAMPLES;
            for k in 0..COVERAGE_SAMPLES {
                out[k] = tail[base + k] + tail[base + COVERAGE_SAMPLES + k];
            }
        }
    }
    cov
}

/// Coverage samples per triangle-hierarchy node: directions `(i, j, k)` with
/// each of `i, j, k` in `{-1, 0, 1}`, at index `9(i+1) + 3(j+1) + (k+1)`
/// (index 13, the zero vector, is unused).
pub(crate) const COVERAGE_SAMPLES: usize = 27;

fn coverage_directions() -> [Vec3; COVERAGE_SAMPLES] {
    std::array::from_fn(|n| {
        let d = Vec3::new(
            (n / 9) as f32 - 1.0,
            (n / 3 % 3) as f32 - 1.0,
            (n % 3) as f32 - 1.0,
        );
        d.normalize_or_zero()
    })
}

/// Triangles per leaf of a mesh's triangle hierarchy. A box test costs about
/// three triangle tests, so small leaves only pay off for large meshes.
const LEAF_TRIANGLES: usize = 8;

/// How far rays and cones descend before testing triangles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hierarchy {
    /// Test every triangle of every mesh.
    None = 0,
    /// Cull meshes through the mesh hierarchy, then test each touched mesh's
    /// triangles one by one.
    Meshes = 1,
    /// Also cull triangles through each mesh's triangle hierarchy.
    Triangles = 2,
}

impl From<bool> for Hierarchy {
    fn from(on: bool) -> Self {
        if on {
            Hierarchy::Triangles
        } else {
            Hierarchy::None
        }
    }
}

pub(crate) fn mesh_bounds(mesh: &[Tri]) -> (Vec3, Vec3) {
    mesh.iter().flatten().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), &p| (lo.min(p), hi.max(p)),
    )
}

/// Bounds grown by a margin well above float rounding, so box tests that cull
/// rays or cones never reject a triangle the exact test would hit. (Box meshes lie
/// exactly on their bounds, so a ray grazing a face can otherwise pass the
/// triangle test and fail the box test, depending on how boxes are grouped.)
pub(crate) fn padded((lo, hi): (Vec3, Vec3)) -> (Vec3, Vec3) {
    let pad = 1e-5 * lo.abs().max(hi.abs()).max_element().max(1.0);
    (lo - pad, hi + pad)
}
