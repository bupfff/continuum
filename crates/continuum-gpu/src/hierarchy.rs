//! Bounding volume hierarchies for the GPU: one over meshes, and one over each
//! mesh's triangles.

use bytemuck::{Pod, Zeroable};
use continuum::glam::Vec3;

use crate::Tri;

/// Meshes flattened for the GPU. Bindings may not be empty, so empty inputs
/// get unread padding.
pub(crate) struct Flat {
    /// Triangles as `vec4` vertices, three per triangle.
    pub tris: Vec<[f32; 4]>,
    /// Per mesh: `[first triangle, count, triangle hierarchy root (or NONE), 0]`.
    pub ranges: Vec<[u32; 4]>,
    /// Every mesh's triangle hierarchy: its root first, then its subtree.
    pub blas: Vec<NodeGpu>,
    /// Triangle indices referenced by `blas` leaves; mesh `m`'s are at the same
    /// offsets as its triangles.
    pub blas_items: Vec<u32>,
}

pub(crate) const NONE: u32 = u32::MAX;

pub(crate) fn flatten(meshes: &[Vec<Tri>]) -> Flat {
    let mut flat = Flat {
        tris: Vec::new(),
        ranges: Vec::new(),
        blas: Vec::new(),
        blas_items: Vec::new(),
    };
    for mesh in meshes {
        let first = (flat.tris.len() / 3) as u32;
        let root = if mesh.is_empty() {
            NONE
        } else {
            let root = flat.blas.len() as u32;
            let (nodes, items) = triangle_hierarchy(mesh, root, first);
            flat.blas.extend(nodes);
            flat.blas_items.extend(items);
            root
        };
        flat.ranges.push([first, mesh.len() as u32, root, 0]);
        for t in mesh {
            flat.tris.extend(t.iter().map(|v| [v.x, v.y, v.z, 0.0]));
        }
    }
    if flat.tris.is_empty() {
        flat.tris.extend([[0.0; 4]; 3]);
    }
    if flat.ranges.is_empty() {
        flat.ranges.push([0, 0, NONE, 0]);
    }
    if flat.blas.is_empty() {
        flat.blas.push(NodeGpu::zeroed());
        flat.blas_items.push(0);
    }
    flat
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct NodeGpu {
    lo: [f32; 4],
    hi: [f32; 4],
    /// Leaf: `[first item, item count, 0, 0]`; interior: `[left child, 0, 0, 0]`
    /// with the right child at `left + 1`.
    info: [u32; 4],
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
    let mut items: Vec<u32> = (0..meshes.len() as u32)
        .filter(|&m| keep(m as usize))
        .collect();
    let mut nodes = vec![NodeGpu::zeroed()];
    if items.is_empty() {
        // A leaf holding mesh 0, which must then have no triangles: true when
        // there are no meshes (`flatten` pads an empty range). Callers never
        // traverse a hierarchy that filtered every mesh out.
        items.push(0);
        nodes[0].info = [0, 1, 0, 0];
    } else {
        split(&mut nodes, 0, &mut items, 0, &bounds, 2, 0, 0);
    }
    (nodes, items)
}

/// One mesh's triangle hierarchy (at most [`LEAF_TRIANGLES`] per leaf): nodes
/// numbered from `root`, and triangle indices numbered from `first`, stored at
/// item positions from `first`.
pub(crate) fn triangle_hierarchy(mesh: &[Tri], root: u32, first: u32) -> (Vec<NodeGpu>, Vec<u32>) {
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
    (nodes, items.iter().map(|&t| first + t).collect())
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
