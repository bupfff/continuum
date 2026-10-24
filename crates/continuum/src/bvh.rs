//! A bounding volume hierarchy over triangles, traversed by cones.
//!
//! Built once over static triangles (median splits on the longest centroid
//! axis). A cone descends it and collects only the triangles whose boxes it may
//! touch, grouped by mesh for [`PyramidCone::add_mesh`].

use glam::Vec3;

use crate::cone3::PyramidCone;

type Tri = [Vec3; 3];
const LEAF_SIZE: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: first triangle. Interior: index of the left child (right is +1).
    start: u32,
    /// Triangles in a leaf; 0 for interior nodes.
    count: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    tris: Vec<Tri>,
    mesh_of: Vec<u32>,
}

impl Bvh {
    /// Builds a hierarchy over every triangle of every mesh.
    pub fn build(meshes: &[Vec<Tri>]) -> Bvh {
        let mut items: Vec<(Tri, u32)> = meshes
            .iter()
            .enumerate()
            .flat_map(|(m, mesh)| mesh.iter().map(move |&t| (t, m as u32)))
            .collect();
        let mut nodes = vec![Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            start: 0,
            count: 0,
        }];
        let len = items.len();
        Self::split(&mut nodes, 0, &mut items, 0, len);
        Bvh {
            nodes,
            tris: items.iter().map(|i| i.0).collect(),
            mesh_of: items.iter().map(|i| i.1).collect(),
        }
    }

    fn split(
        nodes: &mut Vec<Node>,
        node: usize,
        items: &mut [(Tri, u32)],
        start: usize,
        end: usize,
    ) {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let (mut cmin, mut cmax) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for (t, _) in &items[start..end] {
            for p in t {
                (min, max) = (min.min(*p), max.max(*p));
            }
            let c = (t[0] + t[1] + t[2]) / 3.0;
            (cmin, cmax) = (cmin.min(c), cmax.max(c));
        }
        nodes[node].min = min;
        nodes[node].max = max;
        if end - start <= LEAF_SIZE {
            nodes[node].start = start as u32;
            nodes[node].count = (end - start) as u32;
            return;
        }
        let extent = cmax - cmin;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let mid = (start + end) / 2;
        let key = |t: &Tri| t[0][axis] + t[1][axis] + t[2][axis];
        items[start..end]
            .select_nth_unstable_by(mid - start, |a, b| key(&a.0).total_cmp(&key(&b.0)));
        let left = nodes.len();
        nodes.push(Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            start: 0,
            count: 0,
        });
        nodes.push(Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            start: 0,
            count: 0,
        });
        nodes[node].start = left as u32;
        Self::split(nodes, left, items, start, mid);
        Self::split(nodes, left + 1, items, mid, end);
    }

    pub fn triangles(&self) -> usize {
        self.tris.len()
    }

    /// Triangles that may touch `cone`, grouped by mesh (sorted by mesh id), as
    /// `(mesh, triangles)` pairs. Also returns how many nodes were visited.
    pub fn candidates(&self, cone: &PyramidCone) -> (Vec<(usize, Vec<Tri>)>, usize) {
        let (found, visited) = self.collect(
            |min, max| cone.may_touch_box(min, max),
            |t| cone.may_touch(t),
        );
        (group_by_mesh(found), visited)
    }

    /// Every `(mesh, triangle)` whose node boxes pass `node_test` and which passes
    /// `tri_test`, plus how many nodes were visited. For custom culling volumes,
    /// such as a frustum enclosing a whole tile of cones.
    pub fn collect(
        &self,
        node_test: impl Fn(Vec3, Vec3) -> bool,
        tri_test: impl Fn(&Tri) -> bool,
    ) -> (Vec<(u32, Tri)>, usize) {
        let mut found = Vec::new();
        let mut stack = vec![0usize];
        let mut visited = 0;
        while let Some(i) = stack.pop() {
            visited += 1;
            let n = self.nodes[i];
            if !node_test(n.min, n.max) {
                continue;
            }
            if n.count > 0 {
                for t in n.start as usize..(n.start + n.count) as usize {
                    if tri_test(&self.tris[t]) {
                        found.push((self.mesh_of[t], self.tris[t]));
                    }
                }
            } else {
                stack.push(n.start as usize);
                stack.push(n.start as usize + 1);
            }
        }
        (found, visited)
    }
}

/// Groups `(mesh, triangle)` pairs by mesh, sorted by mesh id.
pub fn group_by_mesh(mut found: Vec<(u32, Tri)>) -> Vec<(usize, Vec<Tri>)> {
    found.sort_by_key(|f| f.0);
    let mut grouped: Vec<(usize, Vec<Tri>)> = Vec::new();
    for (m, t) in found {
        match grouped.last_mut() {
            Some((last, tris)) if *last == m as usize => tris.push(t),
            _ => grouped.push((m as usize, vec![t])),
        }
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cone3::{box_mesh, sphere_mesh};

    #[test]
    fn candidates_match_brute_force() {
        let mut meshes: Vec<Vec<Tri>> = (0..20)
            .map(|i| {
                let x = 0.05 * i as f32;
                box_mesh(
                    Vec3::new(x, 0.3, 0.4 + 0.01 * i as f32),
                    Vec3::new(x + 0.03, 0.7, 0.42 + 0.01 * i as f32),
                )
            })
            .collect();
        meshes.push(sphere_mesh(Vec3::new(0.5, 0.5, 0.6), 0.1, 24, 12));
        let bvh = Bvh::build(&meshes);
        for p in [
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.1, 0.9, 0.0),
            Vec3::new(0.8, 0.2, 0.0),
        ] {
            let cone = PyramidCone::toward_rect_light(
                p,
                Vec3::new(0.4, 0.4, 1.0),
                Vec3::X * 0.2,
                Vec3::Y * 0.2,
                Vec3::Z,
            )
            .unwrap();
            let (from_bvh, _) = bvh.candidates(&cone);
            let brute: Vec<(usize, Vec<Tri>)> = meshes
                .iter()
                .enumerate()
                .filter_map(|(m, mesh)| {
                    let tris: Vec<Tri> =
                        mesh.iter().filter(|t| cone.may_touch(t)).copied().collect();
                    (!tris.is_empty()).then_some((m, tris))
                })
                .collect();
            assert_eq!(from_bvh.len(), brute.len());
            for ((ma, ta), (mb, tb)) in from_bvh.iter().zip(&brute) {
                assert_eq!(ma, mb);
                assert_eq!(ta.len(), tb.len());
            }
        }
    }
}
