//! Hardware ray tracing for camera rays, through wgpu's experimental ray
//! queries: acceleration structures built from the same geometry and
//! placements as the compute hierarchies.

use continuum::glam::{DVec3, Vec3};

use crate::hierarchy::Placement;
use crate::{InstancedScene, Tri};

/// Objects per cluster of proxies (M14 gate 6).
pub(crate) const CLUSTER: usize = 32;
/// Marks a top-level instance's custom data as a cluster of proxies: the
/// rest is the word, in the objects' ranges' buffer, where its members'
/// object indices start (one per geometry of its structure).
pub(crate) const CLUSTER_FLAG: u32 = 0x80_0000;
const NONE: u32 = u32::MAX;

/// Whether geometry `g` has a proxy level (M14 gate 4): camera rays never
/// see the geometry itself.
pub(crate) fn proxied(scene: &InstancedScene, g: usize) -> bool {
    scene
        .geometry_levels
        .get(g)
        .and_then(|l| l.as_ref())
        .is_some_and(|l| l.proxy.is_some())
}

/// The clusters of proxied objects (M14 gate 6): the objects whose geometry
/// has a proxy level, in Morton order of their position, `CLUSTER` at a
/// time. Camera rays see an object showing its proxy through its cluster's
/// one structure, so the top level holds a cluster per `CLUSTER` such
/// objects, not one instance each.
pub(crate) fn proxy_clusters(scene: &InstancedScene, origin: DVec3) -> Vec<Vec<u32>> {
    let objects: Vec<(u32, Vec3)> = scene
        .instances
        .iter()
        .enumerate()
        .filter(|(_, i)| proxied(scene, i.geometry))
        .map(|(k, i)| (k as u32, (i.position - origin).as_vec3()))
        .collect();
    if objects.is_empty() {
        return Vec::new();
    }
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for (_, p) in &objects {
        (lo, hi) = (lo.min(*p), hi.max(*p));
    }
    // One scale for all three axes, so a flat field's small spread in
    // height does not decide the order.
    let span = Vec3::splat((hi - lo).max_element().max(1e-6));
    let spread = |x: u32| {
        let mut x = x & 0x3ff;
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        (x | (x << 2)) & 0x0924_9249
    };
    let mut keyed: Vec<(u32, u32)> = objects
        .iter()
        .map(|(k, p)| {
            let q = ((*p - lo) / span * 1023.0).as_uvec3();
            (spread(q.x) | spread(q.y) << 1 | spread(q.z) << 2, *k)
        })
        .collect();
    keyed.sort_unstable();
    // Diagnostics: `CONTINUUM_CLUSTER` sets the cluster size (1: an
    // instance per proxy).
    let size = std::env::var("CONTINUUM_CLUSTER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CLUSTER)
        .max(1);
    keyed
        .chunks(size)
        .map(|c| c.iter().map(|&(_, k)| k).collect())
        .collect()
}

/// Clusters of proxies (M14 gate 6): per cluster each member's first vertex in `vertices` (the proxies placed in the world)
/// and size, which members show a finer level (their geometry then reads
/// degenerate triangles), and its structure; per object its cluster and
/// member (objects shown through a cluster) and its slot among the top
/// level's slots for finer levels.
struct Clusters {
    /// The first cluster's top-level slot, and the first slot for finer
    /// levels.
    base: u32,
    dynamic: u32,
    first: Vec<Vec<u32>>,
    sizes: Vec<Vec<wgpu::BlasTriangleGeometrySizeDescriptor>>,
    detailed: Vec<Vec<bool>>,
    dirty: Vec<u32>,
    blas: Vec<wgpu::Blas>,
    words: Vec<u32>,
    vertices: wgpu::Buffer,
    /// Indices 0, 1, 2, … (a member's proxy) and all 0 (a member showing a
    /// finer level: its triangles collapsed onto its first vertex, so its
    /// cluster's bounds stay its own).
    identity: wgpu::Buffer,
    zeros: wgpu::Buffer,
    /// An empty structure for unused slots (mask 0, so never hit).
    null_blas: wgpu::Blas,
    of_object: Vec<(u32, u32)>,
    slot: Vec<u32>,
    free: Vec<u32>,
    /// Replaced structures (with their cluster's layout), kept until
    /// frames in flight no longer trace them, then reused by clusters of
    /// the same layout (creating a structure costs more than building it).
    graveyard: Vec<(u64, usize, wgpu::Blas)>,
    spare: Vec<Vec<wgpu::Blas>>,
    /// Per cluster, its layout: clusters whose members' proxies have the
    /// same sizes share one.
    layout: Vec<usize>,
}

/// Acceleration structures for a scene: one bottom-level structure per
/// geometry, and a top-level instance per object carrying its transform and,
/// as custom data, its index.
pub(crate) struct HwScene {
    pub tlas: wgpu::Tlas,
    /// The movable objects' own top-level structure (they are left out of
    /// `tlas`), rebuilt as they move, and those objects.
    pub movers: Option<(wgpu::Tlas, Vec<u32>)>,
    blas: Vec<Option<wgpu::Blas>>,
    _vertices: wgpu::Buffer,
    /// Each object's top-level transform, kept for objects whose geometry
    /// has no structure of its own (M14 gate 4: camera rays see its proxy),
    /// so an instance can be made when a level is pointed at.
    transforms: Vec<[f32; 12]>,
    /// Per object, its fixed top-level slot (NONE: shown through a cluster,
    /// or movable).
    slots: Vec<u32>,
    clusters: Option<Clusters>,
}

impl HwScene {
    /// Points object `object` (of geometry `geometry`) at `blas`, or at its
    /// geometry's own structure (None), from the next top-level build (M14:
    /// camera levels).
    pub(crate) fn point_at(&mut self, object: usize, geometry: usize, blas: Option<&wgpu::Blas>) {
        if let Some(c) = &mut self.clusters
            && c.of_object[object].0 != NONE
        {
            // Shown through its cluster (None: its proxy) or, with a finer
            // level, through a slot of its own (M14 gate 6).
            let (cluster, member) = c.of_object[object];
            let detailed = blas.is_some();
            if c.detailed[cluster as usize][member as usize] != detailed {
                c.detailed[cluster as usize][member as usize] = detailed;
                c.dirty.push(cluster);
            }
            match blas {
                Some(b) => {
                    if c.slot[object] == NONE {
                        let Some(k) = c.free.pop() else {
                            // No slot left (the capacity is sized from the
                            // budget): it stays shown as its proxy.
                            c.detailed[cluster as usize][member as usize] = false;
                            return;
                        };
                        c.slot[object] = k;
                    }
                    self.tlas[c.slot[object] as usize] = Some(wgpu::TlasInstance::new(
                        b,
                        self.transforms[object],
                        object as u32,
                        0xff,
                    ));
                }
                None => {
                    let k = std::mem::replace(&mut c.slot[object], NONE);
                    if k != NONE {
                        self.tlas[k as usize] = Some(null_instance(&c.null_blas));
                        c.free.push(k);
                    }
                }
            }
            return;
        }
        let slot = self.slots[object] as usize;
        let Some(b) = blas.or(self.blas[geometry].as_ref()) else {
            self.tlas[slot] = None;
            return;
        };
        if let Some(inst) = &mut self.tlas[slot] {
            inst.set_blas(b);
        } else {
            self.tlas[slot] = Some(wgpu::TlasInstance::new(
                b,
                self.transforms[object],
                object as u32,
                0xff,
            ));
        }
    }

    /// Rebuilds, in `encoder`, the structures of the clusters whose members
    /// changed between their proxy and a finer level (M14 gate 6), before
    /// the top-level build that reads them; `frame` dates the replaced ones.
    pub(crate) fn rebuild_clusters(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: u64,
    ) {
        let Some(c) = &mut self.clusters else { return };
        let mut kept = Vec::with_capacity(c.graveyard.len());
        for (f, layout, blas) in std::mem::take(&mut c.graveyard) {
            if f + 4 > frame {
                kept.push((f, layout, blas));
            } else {
                c.spare[layout].push(blas);
            }
        }
        c.graveyard = kept;
        if c.dirty.is_empty() {
            return;
        }
        let mut dirty = std::mem::take(&mut c.dirty);
        dirty.sort_unstable();
        dirty.dedup();
        let fresh: Vec<wgpu::Blas> = dirty
            .iter()
            .map(|&k| {
                let layout = c.layout[k as usize];
                c.spare[layout]
                    .pop()
                    .unwrap_or_else(|| cluster_blas(device, &c.sizes[k as usize]))
            })
            .collect();
        {
            let entries: Vec<wgpu::BlasBuildEntry> = dirty
                .iter()
                .zip(&fresh)
                .map(|(&k, blas)| cluster_entry(c, k as usize, blas))
                .collect();
            encoder.build_acceleration_structures(entries.iter(), std::iter::empty());
        }
        for (&k, blas) in dirty.iter().zip(fresh) {
            let slot = (c.base + k) as usize;
            if let Some(inst) = &mut self.tlas[slot] {
                inst.set_blas(&blas);
            }
            let old = std::mem::replace(&mut c.blas[k as usize], blas);
            c.graveyard.push((frame, c.layout[k as usize], old));
        }
    }

    /// Rebuilds the movers' structure with their placements now (one per
    /// mover, in the order given at build), in `encoder`.
    pub(crate) fn update_movers(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        placements: &[Placement],
    ) {
        let Some((tlas, objects)) = &mut self.movers else {
            return;
        };
        for (k, (o, &object)) in placements.iter().zip(objects.iter()).enumerate() {
            if let Some(b) = &self.blas[o.geometry] {
                tlas[k] = Some(wgpu::TlasInstance::new(
                    b,
                    instance_transform(o),
                    object,
                    0xff,
                ));
            }
        }
        encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&*tlas));
    }
}

/// An unused top-level slot: an empty structure never hit (mask 0).
fn null_instance(blas: &wgpu::Blas) -> wgpu::TlasInstance {
    wgpu::TlasInstance::new(
        blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0,
    )
}

/// A cluster's structure, unbuilt (one geometry per member).
fn cluster_blas(
    device: &wgpu::Device,
    sizes: &[wgpu::BlasTriangleGeometrySizeDescriptor],
) -> wgpu::Blas {
    device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("cluster of proxies"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: sizes.to_vec(),
        },
    )
}

/// Cluster `k`'s build into `blas`: each member's proxy in the world, or
/// degenerate triangles for a member showing a finer level.
fn cluster_entry<'a>(c: &'a Clusters, k: usize, blas: &'a wgpu::Blas) -> wgpu::BlasBuildEntry<'a> {
    wgpu::BlasBuildEntry {
        blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(
            c.sizes[k]
                .iter()
                .enumerate()
                .map(|(m, size)| wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: &c.vertices,
                    first_vertex: c.first[k][m],
                    vertex_stride: 12,
                    index_buffer: Some(if c.detailed[k][m] {
                        &c.zeros
                    } else {
                        &c.identity
                    }),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                })
                .collect(),
        ),
    }
}

/// A placement as a top-level instance's 3×4 row-major transform.
fn instance_transform(o: &Placement) -> [f32; 12] {
    let m = o.rotation * o.scale;
    let row = |r: usize| Vec3::new(m.x_axis[r], m.y_axis[r], m.z_axis[r]);
    let (r0, r1, r2) = (row(0), row(1), row(2));
    let t = o.translation;
    [
        r0.x, r0.y, r0.z, t.x, r1.x, r1.y, r1.z, t.y, r2.x, r2.y, r2.z, t.z,
    ]
}

/// Builds and submits the acceleration structures. Objects are numbered as in
/// the compute hierarchy, so a hit's custom data is the object index (or, on
/// a cluster of proxies, names it through the cluster's member list).
///
/// M14 gate 6: objects in `clusters` (proxied, `skip`ped geometries; their
/// proxies' triangles per geometry in `proxies`) are shown through their
/// cluster's structure, with `dynamic` top-level slots for those showing a
/// finer level; the others keep a slot each, in order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    geometries: &[Vec<Tri>],
    objects: &[Placement],
    movable: &[usize],
    skip: &[bool],
    clusters: Vec<Vec<u32>>,
    proxies: &[Option<Vec<Tri>>],
    dynamic: u32,
) -> HwScene {
    use wgpu::util::DeviceExt;
    assert!(
        objects.len() < 1 << 24,
        "at most 2^24 objects with hardware rays"
    );
    // Geometries without a structure (M14 gate 4: proxied) upload nothing.
    let built = |k: usize| !skip.get(k).copied().unwrap_or(false);
    let vertices: Vec<[f32; 3]> = geometries
        .iter()
        .enumerate()
        .filter(|(k, _)| built(*k))
        .flat_map(|(_, g)| g.iter())
        .flat_map(|t| t.map(|v| v.to_array()))
        .chain(std::iter::once([0.0; 3]))
        .collect();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("blas vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let sizes: Vec<Option<wgpu::BlasTriangleGeometrySizeDescriptor>> = geometries
        .iter()
        .enumerate()
        .map(|(k, g)| {
            (!g.is_empty() && !skip.get(k).copied().unwrap_or(false)).then(|| {
                wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: 3 * g.len() as u32,
                    index_format: None,
                    index_count: None,
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                }
            })
        })
        .collect();
    let blas: Vec<Option<wgpu::Blas>> = sizes
        .iter()
        .map(|size| {
            size.as_ref().map(|size| {
                device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("geometry"),
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                )
            })
        })
        .collect();
    let mut first = Vec::with_capacity(geometries.len());
    let mut next = 0u32;
    for (k, g) in geometries.iter().enumerate() {
        first.push(next);
        if built(k) {
            next += 3 * g.len() as u32;
        }
    }
    // Slots: objects not in a cluster, in order; then the clusters; then
    // the slots for finer levels.
    let mut of_object = vec![(NONE, NONE); objects.len()];
    for (c, members) in clusters.iter().enumerate() {
        for (m, &k) in members.iter().enumerate() {
            of_object[k as usize] = (c as u32, m as u32);
        }
    }
    let mut slots = vec![NONE; objects.len()];
    let mut count = 0u32;
    for (k, slot) in slots.iter_mut().enumerate() {
        if of_object[k].0 == NONE {
            *slot = count;
            count += 1;
        }
    }
    let cluster_base = count;
    let dynamic = if clusters.is_empty() { 0 } else { dynamic };
    let total = count + clusters.len() as u32 + dynamic;
    let transforms: Vec<[f32; 12]> = objects.iter().map(instance_transform).collect();
    let cluster_set = (!clusters.is_empty()).then(|| {
        // Each member's proxy placed in the world, then degenerate
        // triangles (as many vertices as the largest proxy), then the empty
        // structure's one tiny triangle, far below.
        let mut world: Vec<[f32; 3]> = Vec::new();
        let mut firsts = Vec::with_capacity(clusters.len());
        let mut cluster_sizes = Vec::with_capacity(clusters.len());
        let mut largest = 3u32;
        for members in &clusters {
            let mut f = Vec::with_capacity(members.len());
            let mut z = Vec::with_capacity(members.len());
            for &k in members {
                let o = &objects[k as usize];
                let tris = proxies[o.geometry].as_deref().unwrap_or(&[]);
                let m = o.rotation * o.scale;
                f.push(world.len() as u32);
                for t in tris {
                    for v in t {
                        world.push((m * *v + o.translation).to_array());
                    }
                }
                let n = 3 * tris.len().max(1) as u32;
                if tris.is_empty() {
                    let p = o.translation.to_array();
                    world.extend([p; 3]);
                }
                largest = largest.max(n);
                z.push(wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: n,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(n),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                });
            }
            firsts.push(f);
            cluster_sizes.push(z);
        }
        let tiny = world.len() as u32;
        world.extend([[0.0, 0.0, -100.0], [1e-3, 0.0, -100.0], [0.0, 1e-3, -100.0]]);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster proxies"),
            contents: bytemuck::cast_slice(&world),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let indices = |f: &dyn Fn(u32) -> u32, label| {
            let list: Vec<u32> = (0..largest).map(f).collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&list),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            })
        };
        let identity = indices(&|i| i, "cluster indices");
        let zeros = indices(&|_| 0, "collapsed indices");
        let base_word = 4 * objects.len() as u32;
        let mut words = Vec::with_capacity(clusters.len());
        let mut start = 0u32;
        for members in &clusters {
            let word = base_word + start;
            assert!(
                word < CLUSTER_FLAG,
                "too many objects for clusters of proxies"
            );
            words.push(CLUSTER_FLAG | word);
            start += members.len() as u32;
        }
        let null_size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: 3,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let null_blas = cluster_blas(device, std::slice::from_ref(&null_size));
        // Layouts: clusters whose members' vertex counts match share one.
        let mut layouts: Vec<Vec<u32>> = Vec::new();
        let layout: Vec<usize> = cluster_sizes
            .iter()
            .map(|z| {
                let counts: Vec<u32> = z.iter().map(|d| d.vertex_count).collect();
                layouts
                    .iter()
                    .position(|l| *l == counts)
                    .unwrap_or_else(|| {
                        layouts.push(counts);
                        layouts.len() - 1
                    })
            })
            .collect();
        let c = Clusters {
            base: cluster_base,
            dynamic,
            blas: cluster_sizes
                .iter()
                .map(|z| cluster_blas(device, z))
                .collect(),
            detailed: clusters.iter().map(|m| vec![false; m.len()]).collect(),
            first: firsts,
            sizes: cluster_sizes,
            dirty: Vec::new(),
            words,
            vertices,
            identity,
            zeros,
            null_blas,
            of_object: of_object.clone(),
            slot: vec![NONE; objects.len()],
            free: (0..dynamic)
                .rev()
                .map(|k| cluster_base + clusters.len() as u32 + k)
                .collect(),
            graveyard: Vec::new(),
            spare: vec![Vec::new(); layouts.len()],
            layout,
        };
        (c, null_size, tiny)
    });
    let entries: Vec<wgpu::BlasBuildEntry> = blas
        .iter()
        .zip(&sizes)
        .zip(&first)
        .filter_map(|((b, size), &first_vertex)| {
            Some(wgpu::BlasBuildEntry {
                blas: b.as_ref()?,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: size.as_ref()?,
                        vertex_buffer: &vertex_buffer,
                        first_vertex,
                        vertex_stride: 12,
                        index_buffer: None,
                        first_index: None,
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            })
        })
        .chain(cluster_set.iter().flat_map(|(c, null_size, tiny)| {
            (0..c.blas.len())
                .map(|k| cluster_entry(c, k, &c.blas[k]))
                .chain(std::iter::once(wgpu::BlasBuildEntry {
                    blas: &c.null_blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: null_size,
                            vertex_buffer: &c.vertices,
                            first_vertex: *tiny,
                            vertex_stride: 12,
                            index_buffer: None,
                            first_index: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                }))
        }))
        .collect();
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("objects"),
        max_instances: total.max(1),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let mut is_movable = vec![false; objects.len()];
    for &m in movable {
        is_movable[m] = true;
    }
    for (k, o) in objects.iter().enumerate() {
        if slots[k] == NONE {
            continue;
        }
        let Some(b) = &blas[o.geometry] else { continue };
        if !is_movable[k] {
            tlas[slots[k] as usize] =
                Some(wgpu::TlasInstance::new(b, transforms[k], k as u32, 0xff));
        }
    }
    if let Some((c, _, _)) = &cluster_set {
        for (k, b) in c.blas.iter().enumerate() {
            tlas[(c.base + k as u32) as usize] = Some(wgpu::TlasInstance::new(
                b,
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                c.words[k],
                0xff,
            ));
        }
        for k in 0..c.dynamic {
            tlas[(c.base + c.blas.len() as u32 + k) as usize] = Some(null_instance(&c.null_blas));
        }
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(entries.iter(), std::iter::once(&tlas));
    drop(entries);
    let movers = (!movable.is_empty()).then(|| {
        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("movers"),
            max_instances: movable.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        (tlas, movable.iter().map(|&m| m as u32).collect())
    });
    let mut scene = HwScene {
        tlas,
        movers,
        blas,
        _vertices: vertex_buffer,
        transforms,
        slots,
        clusters: cluster_set.map(|(c, _, _)| c),
    };
    let placements: Vec<Placement> = movable.iter().map(|&m| objects[m]).collect();
    scene.update_movers(&mut encoder, &placements);
    queue.submit([encoder.finish()]);
    scene
}
