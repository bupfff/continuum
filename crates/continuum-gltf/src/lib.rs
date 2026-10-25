//! glTF import (milestone M6): a glTF 2.0 file (`.gltf` or `.glb`) as an
//! [`InstancedScene`].
//!
//! - Each mesh primitive (triangles) becomes a geometry, with automatic charts
//!   ([`Surface::from_mesh`]) and the file's normals where it has them.
//! - Each node holding a mesh places its primitives as instances. A node's
//!   world matrix is split into rotation, uniform scale and translation; a
//!   matrix with non-uniform scale, shear or a mirror is baked into a copy of
//!   the geometry instead.
//! - glTF is Y-up; Continuum is Z-up: (x, y, z) becomes (x, −z, y).
//! - Albedo is the material's base color factor (linear), times the mean of
//!   its base color texture where it has one: textures themselves are not
//!   sampled yet.
//! - Punctual lights (`KHR_lights_punctual`) are counted but not imported:
//!   Continuum's lights are rectangles.

use std::path::Path;

use continuum::glam::{DMat4, DVec3, Mat3, Vec3};
use continuum::mesh::{Surface, Tri};
use continuum_gpu::{Instance, InstancedScene};

/// A camera from the file, in Continuum's Z-up world.
#[derive(Clone, Copy, Debug)]
pub struct ImportedCamera {
    pub eye: DVec3,
    pub target: DVec3,
    pub vertical_fov_degrees: f32,
}

/// What [`load`] found.
#[derive(Clone, Debug, Default)]
pub struct ImportStats {
    pub meshes: usize,
    pub primitives: usize,
    /// Primitives that are not triangle lists, skipped.
    pub skipped_primitives: usize,
    pub triangles: u64,
    pub instances: usize,
    /// Instances whose transform was baked into a copy of their geometry.
    pub baked: usize,
    pub charts: usize,
    pub lights_skipped: usize,
}

/// An imported scene: geometry, instances (world positions), the file's first
/// camera if it has one, and the instances' world bounds.
pub struct Imported {
    pub scene: InstancedScene,
    pub camera: Option<ImportedCamera>,
    pub bounds: (DVec3, DVec3),
    pub stats: ImportStats,
}

/// Y-up to Z-up.
fn to_z_up(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], -v[2], v[1])
}

/// The same change of axes as a matrix, and its inverse.
fn axes() -> (DMat4, DMat4) {
    let c = DMat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, // x stays
        0.0, 0.0, 1.0, 0.0, // y goes to z
        0.0, -1.0, 0.0, 0.0, // z goes to -y
        0.0, 0.0, 0.0, 1.0,
    ]);
    (c, c.inverse())
}

/// One primitive's geometry before placement.
struct Primitive {
    triangles: Vec<Tri>,
    normals: Option<Vec<[Vec3; 3]>>,
    albedo: Vec3,
    /// Its geometry's index, once built.
    geometry: usize,
}

/// The mean color of a decoded image, linear (images are sRGB-encoded).
fn mean_color(image: &gltf::image::Data) -> Option<Vec3> {
    use gltf::image::Format;
    let channels = match image.format {
        Format::R8 => 1,
        Format::R8G8 => 2,
        Format::R8G8B8 => 3,
        Format::R8G8B8A8 => 4,
        _ => return None,
    };
    let linear = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for px in image.pixels.chunks_exact(channels) {
        let rgb = match channels {
            1 | 2 => Vec3::splat(linear(px[0])),
            _ => Vec3::new(linear(px[0]), linear(px[1]), linear(px[2])),
        };
        sum += rgb;
        count += 1;
    }
    (count > 0).then(|| sum / count as f32)
}

/// Loads a glTF 2.0 file as an instanced scene.
pub fn load(path: &Path) -> Result<Imported, gltf::Error> {
    let (document, buffers, images) = gltf::import(path)?;
    let mut stats = ImportStats {
        meshes: document.meshes().len(),
        lights_skipped: document.lights().map_or(0, |l| l.len()),
        ..Default::default()
    };
    // Primitives per mesh.
    let mut meshes: Vec<Vec<Primitive>> = Vec::new();
    for mesh in document.meshes() {
        let mut list = Vec::new();
        for primitive in mesh.primitives() {
            stats.primitives += 1;
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                stats.skipped_primitives += 1;
                continue;
            }
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let Some(positions) = reader.read_positions() else {
                stats.skipped_primitives += 1;
                continue;
            };
            let positions: Vec<Vec3> = positions.map(to_z_up).collect();
            let normals: Option<Vec<Vec3>> =
                reader.read_normals().map(|n| n.map(to_z_up).collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let mut triangles = Vec::with_capacity(indices.len() / 3);
            let mut corner_normals = Vec::with_capacity(indices.len() / 3);
            for t in indices.chunks_exact(3) {
                let k = [t[0] as usize, t[1] as usize, t[2] as usize];
                triangles.push(k.map(|i| positions[i]));
                if let Some(n) = &normals {
                    corner_normals.push(k.map(|i| n[i]));
                }
            }
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            let factor = pbr.base_color_factor();
            let mut albedo = Vec3::new(factor[0], factor[1], factor[2]);
            if let Some(info) = pbr.base_color_texture()
                && let Some(mean) = mean_color(&images[info.texture().source().index()])
            {
                albedo *= mean;
            }
            list.push(Primitive {
                triangles,
                normals: normals.is_some().then_some(corner_normals),
                albedo,
                geometry: usize::MAX,
            });
        }
        meshes.push(list);
    }
    let mut scene = InstancedScene {
        geometries: Vec::new(),
        instances: Vec::new(),
    };
    for list in &mut meshes {
        for p in list.iter_mut() {
            if p.triangles.is_empty() {
                continue;
            }
            p.geometry = scene.geometries.len();
            scene
                .geometries
                .push(Surface::from_mesh(p.triangles.clone(), p.normals.clone()));
        }
    }
    // Nodes, with world matrices.
    let (c, c_inv) = axes();
    let mut camera = None;
    let roots: Vec<gltf::Node> = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(s) => s.nodes().collect(),
        None => Vec::new(),
    };
    let mut stack: Vec<(gltf::Node, DMat4)> =
        roots.into_iter().map(|n| (n, DMat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let local =
            DMat4::from_cols_array_2d(&node.transform().matrix().map(|col| col.map(|x| x as f64)));
        let world = parent * local;
        // In Z-up coordinates.
        let m = c * world * c_inv;
        if let Some(cam) = node.camera()
            && camera.is_none()
            && let gltf::camera::Projection::Perspective(p) = cam.projection()
        {
            // glTF cameras look down their local -z (y up), in glTF axes.
            let eye = m.transform_point3(DVec3::ZERO);
            let ahead = (c * world).transform_vector3(DVec3::new(0.0, 0.0, -1.0));
            camera = Some(ImportedCamera {
                eye,
                target: eye + ahead.normalize(),
                vertical_fov_degrees: p.yfov().to_degrees(),
            });
        }
        if let Some(mesh) = node.mesh() {
            for p in &meshes[mesh.index()] {
                if p.geometry == usize::MAX {
                    continue;
                }
                place(&mut scene, p, m, &mut stats);
            }
        }
        for child in node.children() {
            stack.push((child, world));
        }
    }
    let mut bounds = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
    for i in &scene.instances {
        let (lo, hi) = scene.geometries[i.geometry].bounds();
        for k in 0..8 {
            let corner = Vec3::new(
                if k & 1 == 0 { lo.x } else { hi.x },
                if k & 2 == 0 { lo.y } else { hi.y },
                if k & 4 == 0 { lo.z } else { hi.z },
            );
            let p = i.position + (i.rotation * corner * i.scale).as_dvec3();
            bounds = (bounds.0.min(p), bounds.1.max(p));
        }
    }
    stats.instances = scene.instances.len();
    stats.charts = scene.geometries.iter().map(|g| g.charts.len()).sum();
    stats.triangles = scene
        .instances
        .iter()
        .map(|i| scene.geometries[i.geometry].triangles.len() as u64)
        .sum();
    Ok(Imported {
        scene,
        camera,
        bounds,
        stats,
    })
}

/// Places primitive `p` with world matrix `m` (Z-up): an instance of its
/// geometry if `m` is rotation, uniform scale and translation; otherwise a
/// copy of the geometry with `m` baked in.
fn place(scene: &mut InstancedScene, p: &Primitive, m: DMat4, stats: &mut ImportStats) {
    let a = continuum::glam::DMat3::from_mat4(m);
    let scales = DVec3::new(a.x_axis.length(), a.y_axis.length(), a.z_axis.length());
    let uniform = (scales.max_element() - scales.min_element()) <= 1e-4 * scales.max_element();
    let rotation = continuum::glam::DMat3::from_cols(
        a.x_axis / scales.x,
        a.y_axis / scales.y,
        a.z_axis / scales.z,
    );
    let orthonormal = (rotation.transpose() * rotation - continuum::glam::DMat3::IDENTITY)
        .to_cols_array()
        .iter()
        .all(|x| x.abs() < 1e-4);
    let position = m.w_axis.truncate();
    if uniform && orthonormal && rotation.determinant() > 0.0 && scales.x > 0.0 {
        scene.instances.push(Instance {
            geometry: p.geometry,
            rotation: rotation.as_mat3(),
            scale: scales.x as f32,
            position,
            albedo: p.albedo,
        });
        return;
    }
    // Baked: positions through the matrix (relative to its translation),
    // normals through its inverse transpose, windings kept outward.
    stats.baked += 1;
    let linear = a.as_mat3();
    let normal = linear.inverse().transpose();
    let mirrored = linear.determinant() < 0.0;
    let mut triangles: Vec<Tri> = p.triangles.iter().map(|t| t.map(|v| linear * v)).collect();
    let mut normals = p.normals.as_ref().map(|n| {
        n.iter()
            .map(|c| c.map(|v| (normal * v).normalize_or_zero()))
            .collect::<Vec<_>>()
    });
    if mirrored {
        for t in &mut triangles {
            t.swap(1, 2);
        }
        if let Some(n) = &mut normals {
            for c in n.iter_mut() {
                c.swap(1, 2);
            }
        }
    }
    let geometry = scene.geometries.len();
    scene
        .geometries
        .push(Surface::from_mesh(triangles, normals));
    scene.instances.push(Instance {
        geometry,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position,
        albedo: p.albedo,
    });
}
