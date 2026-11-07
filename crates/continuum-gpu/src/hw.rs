//! Hardware ray tracing for camera rays, through wgpu's experimental ray
//! queries: acceleration structures built from the same geometry and
//! placements as the compute hierarchies.

use continuum::glam::Vec3;

use crate::Tri;
use crate::hierarchy::Placement;

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
}

impl HwScene {
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
/// the compute hierarchy, so a hit's custom data is the object index.
pub(crate) fn build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    geometries: &[Vec<Tri>],
    objects: &[Placement],
    movable: &[usize],
) -> HwScene {
    use wgpu::util::DeviceExt;
    assert!(
        objects.len() < 1 << 24,
        "at most 2^24 objects with hardware rays"
    );
    let vertices: Vec<[f32; 3]> = geometries
        .iter()
        .flatten()
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
        .map(|g| {
            (!g.is_empty()).then(|| wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: 3 * g.len() as u32,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
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
    for g in geometries {
        first.push(next);
        next += 3 * g.len() as u32;
    }
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
        .collect();
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("objects"),
        max_instances: objects.len().max(1) as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let mut is_movable = vec![false; objects.len()];
    for &m in movable {
        is_movable[m] = true;
    }
    for (k, o) in objects.iter().enumerate() {
        let Some(b) = &blas[o.geometry] else { continue };
        if !is_movable[k] {
            tlas[k] = Some(wgpu::TlasInstance::new(
                b,
                instance_transform(o),
                k as u32,
                0xff,
            ));
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
    };
    let placements: Vec<Placement> = movable.iter().map(|&m| objects[m]).collect();
    scene.update_movers(&mut encoder, &placements);
    queue.submit([encoder.finish()]);
    scene
}
