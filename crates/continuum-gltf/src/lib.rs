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
//! - Albedo is the material's base color factor (linear). Its base color
//!   texture, where it has one and the primitive has the texture coordinates
//!   it names, is imported and sampled when shading
//!   (`InstancedScene::textures`).
//! - Metallic and roughness factors become the geometry's [`Material`]
//!   (`InstancedScene::geometry_materials`). A primitive without a material,
//!   or with metallic 0 and roughness 1 (what [`save`] writes for a
//!   Lambertian surface), stays Lambertian: the difference is a rough 4%
//!   specular layer. A metallic-roughness texture (through the base
//!   color's texture coordinates, or alone) becomes the geometry's
//!   (`InstancedScene::geometry_metallic_roughness`). Alpha is not imported
//!   yet.
//! - A normal texture, through the same texture coordinates as the base
//!   color (or alone), becomes the geometry's normal map
//!   (`InstancedScene::geometry_normal_maps`; its scale is ignored).
//! - The emissive factor, times `KHR_materials_emissive_strength`, becomes
//!   the geometry's emission (`InstancedScene::geometry_emission`; emissive
//!   textures are not imported).
//! - Punctual lights (`KHR_lights_punctual`) are imported
//!   ([`Imported::lights`]): point and spot lights as small spheres,
//!   directional lights as suns.

#![warn(missing_docs)]

use std::path::Path;

use continuum::glam::{DMat4, DVec3, Mat3, Vec2, Vec3};
use continuum::mesh::{Surface, Tri};
use continuum_gpu::{Instance, InstancedScene, Light, Material, Spot, Texture};

/// What kind of punctual light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImportedLightKind {
    /// Shines equally in every direction from its position.
    Point,
    /// A point light limited to a cone around its direction, fading between
    /// two half-angles (radians).
    Spot {
        /// Full intensity inside this half-angle.
        inner: f32,
        /// No light outside this half-angle.
        outer: f32,
    },
    /// Parallel light from infinitely far away, like the sun.
    Directional,
}

/// A punctual light from the file, in Continuum's Z-up world and glTF's
/// units: candela for point and spot lights, lux for directional ones.
#[derive(Clone, Copy, Debug)]
pub struct ImportedLight {
    /// Point, spot or directional.
    pub kind: ImportedLightKind,
    /// Where the light's node is (unused for directional lights).
    pub position: DVec3,
    /// Where it shines (the node's −Z): a spot's axis, a directional light's
    /// direction of travel.
    pub direction: DVec3,
    /// Linear RGB.
    pub color: Vec3,
    /// Candela (point and spot) or lux (directional), scaling `color`.
    pub intensity: f32,
}

/// A camera from the file, in Continuum's Z-up world.
#[derive(Clone, Copy, Debug)]
pub struct ImportedCamera {
    /// The camera's position.
    pub eye: DVec3,
    /// A point it looks at, one unit ahead.
    pub target: DVec3,
    /// The full vertical field of view.
    pub vertical_fov_degrees: f32,
}

/// What [`load`] found.
#[derive(Clone, Debug, Default)]
pub struct ImportStats {
    /// Meshes in the file.
    pub meshes: usize,
    /// Mesh primitives in the file.
    pub primitives: usize,
    /// Primitives that are not triangle lists, skipped.
    pub skipped_primitives: usize,
    /// Triangles placed in the scene, counted once per instance.
    pub triangles: u64,
    /// Instances placed.
    pub instances: usize,
    /// Instances whose transform was baked into a copy of their geometry.
    pub baked: usize,
    /// Charts made for the lighting cache, over all geometries.
    pub charts: usize,
    /// Punctual lights placed by nodes.
    pub lights: usize,
    /// Base color textures imported.
    pub textures: usize,
    /// Images skipped: pixel formats other than 8 bits per channel.
    pub textures_skipped: usize,
}

/// An imported scene: geometry, instances (world positions), the file's first
/// camera if it has one, and the instances' world bounds.
pub struct Imported {
    /// Geometry, instances at world positions, and textures, ready for
    /// `continuum_gpu::Renderer`.
    pub scene: InstancedScene,
    /// The first perspective camera a node places, if any (orthographic
    /// cameras are skipped).
    pub camera: Option<ImportedCamera>,
    /// The punctual lights nodes place, in the file's units;
    /// [`Imported::lights`] turns them into Continuum's lights.
    pub lights: Vec<ImportedLight>,
    /// The instances' world bounds (low and high corners).
    pub bounds: (DVec3, DVec3),
    /// What was imported and what was skipped.
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

/// The accessors a primitive reads (positions, normals, texture
/// coordinates, indices) and its texture: primitives alike share a geometry.
type GeometryKey = (
    usize,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<u32>,
    Option<(u32, u32)>,
    [u32; 3],
    Option<u32>,
    Option<u32>,
);

/// One primitive's geometry before placement.
struct Primitive {
    triangles: Vec<Tri>,
    normals: Option<Vec<[Vec3; 3]>>,
    tex: Option<Vec<[Vec2; 3]>>,
    /// Its base color texture's index in the scene.
    texture: Option<u32>,
    albedo: Vec3,
    material: Option<Material>,
    emission: Vec3,
    /// Its normal map's index in the scene's textures.
    normal_map: Option<u32>,
    /// Its metallic-roughness texture's index in the scene's textures.
    metallic_roughness: Option<u32>,
    key: GeometryKey,
    /// Its geometry's index, once built.
    geometry: usize,
}

/// A decoded image as RGBA8 (8-bit formats only).
fn to_rgba(image: &gltf::image::Data) -> Option<Texture> {
    use gltf::image::Format;
    let rgba: Vec<u8> = match image.format {
        Format::R8 => image.pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        Format::R8G8 => image
            .pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        Format::R8G8B8 => image
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        Format::R8G8B8A8 => image.pixels.clone(),
        _ => return None,
    };
    Some(Texture {
        width: image.width,
        height: image.height,
        rgba,
    })
}

/// Loads a glTF 2.0 file as an instanced scene.
pub fn load(path: &Path) -> Result<Imported, gltf::Error> {
    let (document, buffers, images) = gltf::import(path)?;
    let mut stats = ImportStats {
        meshes: document.meshes().len(),
        ..Default::default()
    };
    // Base color textures, imported once per image.
    let mut textures: Vec<Texture> = Vec::new();
    let mut texture_of_image: Vec<Option<Option<u32>>> = vec![None; images.len()];
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
            let accessor = |s: gltf::Semantic| primitive.get(&s).map(|a| a.index());
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
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            let factor = pbr.base_color_factor();
            let albedo = Vec3::new(factor[0], factor[1], factor[2]);
            let (metallic, roughness) = (pbr.metallic_factor(), pbr.roughness_factor());
            let surface = (material.index().is_some()
                && (metallic != 0.0
                    || roughness != 1.0
                    || pbr.metallic_roughness_texture().is_some()))
            .then_some(Material {
                metallic,
                roughness,
            });
            let emission = Vec3::from_array(material.emissive_factor())
                * material.emissive_strength().unwrap_or(1.0);
            // The base color texture, through the texture coordinate set it
            // names.
            let mut texture = None;
            let mut texcoords: Option<Vec<Vec2>> = None;
            if let Some(info) = pbr.base_color_texture()
                && let Some(t) = reader.read_tex_coords(info.tex_coord())
            {
                let image = info.texture().source().index();
                let slot = *texture_of_image[image].get_or_insert_with(|| {
                    to_rgba(&images[image]).map(|t| {
                        textures.push(t);
                        textures.len() as u32 - 1
                    })
                });
                if let Some(index) = slot {
                    texture = Some(index);
                    texcoords = Some(t.into_f32().map(Vec2::from_array).collect());
                }
            }
            // The normal texture, through the base color's coordinates or
            // alone.
            let mut normal_map = None;
            let mut tex_set = pbr.base_color_texture().map_or(0, |i| i.tex_coord());
            if let Some(info) = material.normal_texture()
                && (texcoords.is_none() || info.tex_coord() == tex_set)
            {
                if texcoords.is_none()
                    && let Some(t) = reader.read_tex_coords(info.tex_coord())
                {
                    texcoords = Some(t.into_f32().map(Vec2::from_array).collect());
                    tex_set = info.tex_coord();
                }
                if texcoords.is_some() {
                    let image = info.texture().source().index();
                    normal_map = *texture_of_image[image].get_or_insert_with(|| {
                        to_rgba(&images[image]).map(|t| {
                            textures.push(t);
                            textures.len() as u32 - 1
                        })
                    });
                }
            }
            // The metallic-roughness texture, likewise.
            let mut metallic_roughness = None;
            if let Some(info) = pbr.metallic_roughness_texture()
                && surface.is_some()
                && (texcoords.is_none() || info.tex_coord() == tex_set)
            {
                if texcoords.is_none()
                    && let Some(t) = reader.read_tex_coords(info.tex_coord())
                {
                    texcoords = Some(t.into_f32().map(Vec2::from_array).collect());
                    tex_set = info.tex_coord();
                }
                if texcoords.is_some() {
                    let image = info.texture().source().index();
                    metallic_roughness = *texture_of_image[image].get_or_insert_with(|| {
                        to_rgba(&images[image]).map(|t| {
                            textures.push(t);
                            textures.len() as u32 - 1
                        })
                    });
                }
            }
            let mut triangles = Vec::with_capacity(indices.len() / 3);
            let mut corner_normals = Vec::with_capacity(indices.len() / 3);
            let mut corner_tex = Vec::with_capacity(indices.len() / 3);
            for t in indices.as_chunks::<3>().0 {
                let k = [t[0] as usize, t[1] as usize, t[2] as usize];
                triangles.push(k.map(|i| positions[i]));
                if let Some(n) = &normals {
                    corner_normals.push(k.map(|i| n[i]));
                }
                if let Some(tc) = &texcoords {
                    corner_tex.push(k.map(|i| tc[i]));
                }
            }
            list.push(Primitive {
                triangles,
                normals: normals.is_some().then_some(corner_normals),
                tex: texcoords.is_some().then_some(corner_tex),
                texture,
                albedo,
                material: surface,
                emission,
                normal_map,
                metallic_roughness,
                key: (
                    accessor(gltf::Semantic::Positions).unwrap_or(usize::MAX),
                    accessor(gltf::Semantic::Normals),
                    texcoords
                        .as_ref()
                        .and(accessor(gltf::Semantic::TexCoords(tex_set))),
                    primitive.indices().map(|a| a.index()),
                    texture,
                    surface.map(|m| (m.metallic.to_bits(), m.roughness.to_bits())),
                    emission.to_array().map(f32::to_bits),
                    normal_map,
                    metallic_roughness,
                ),
                geometry: usize::MAX,
            });
        }
        meshes.push(list);
    }
    stats.textures = textures.len();
    stats.textures_skipped = texture_of_image
        .iter()
        .filter(|t| matches!(t, Some(None)))
        .count();
    let mut scene = InstancedScene {
        textures,
        ..Default::default()
    };
    let mut geometry_of: std::collections::HashMap<GeometryKey, usize> =
        std::collections::HashMap::new();
    for list in &mut meshes {
        for p in list.iter_mut() {
            if p.triangles.is_empty() {
                continue;
            }
            if let Some(&g) = geometry_of.get(&p.key) {
                p.geometry = g;
                continue;
            }
            p.geometry = scene.geometries.len();
            geometry_of.insert(p.key, p.geometry);
            scene.geometries.push(Surface::from_mesh(
                p.triangles.clone(),
                p.normals.clone(),
                p.tex.clone(),
            ));
            scene.geometry_textures.push(p.texture);
            scene.geometry_materials.push(p.material);
            scene.geometry_emission.push(p.emission);
            scene.geometry_normal_maps.push(p.normal_map);
            scene.geometry_metallic_roughness.push(p.metallic_roughness);
        }
    }
    // Nodes, with world matrices.
    let (c, c_inv) = axes();
    let mut camera = None;
    let mut lights = Vec::new();
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
        if let Some(light) = node.light() {
            use gltf::khr_lights_punctual::Kind;
            // Lights shine down their local -z, in glTF axes.
            let direction = (c * world)
                .transform_vector3(DVec3::new(0.0, 0.0, -1.0))
                .normalize_or(DVec3::NEG_Z);
            lights.push(ImportedLight {
                kind: match light.kind() {
                    Kind::Point => ImportedLightKind::Point,
                    Kind::Spot {
                        inner_cone_angle,
                        outer_cone_angle,
                    } => ImportedLightKind::Spot {
                        inner: inner_cone_angle,
                        outer: outer_cone_angle,
                    },
                    Kind::Directional => ImportedLightKind::Directional,
                },
                position: m.transform_point3(DVec3::ZERO),
                direction,
                color: Vec3::from_array(light.color()),
                intensity: light.intensity(),
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
    stats.lights = lights.len();
    Ok(Imported {
        scene,
        camera,
        lights,
        bounds,
        stats,
    })
}

/// Radiometric watts per photometric lumen, as Blender's glTF importer
/// converts (at 555 nm).
const WATTS_PER_LUMEN: f32 = 1.0 / 683.0;

impl Imported {
    /// The file's lights for rendering about `origin`, in watts at 683 lumens
    /// per watt (as Blender imports them). glTF's lights have no size: point
    /// and spot lights become spheres of `radius`, and directional lights
    /// suns 0.5° across, 50 scene radii beyond the scene's center.
    pub fn lights(&self, origin: DVec3, radius: f32) -> Vec<Light> {
        let (lo, hi) = self.bounds;
        let center = (lo + hi) * 0.5;
        let extent = ((hi - lo).length() * 0.5).max(1e-3);
        self.lights
            .iter()
            .map(|l| {
                let power = l.color * l.intensity * WATTS_PER_LUMEN;
                let at = (l.position - origin).as_vec3();
                match l.kind {
                    ImportedLightKind::Point => Light::point(at, radius, power),
                    ImportedLightKind::Spot { inner, outer } => Light::spot(
                        at,
                        radius,
                        power,
                        Spot {
                            axis: l.direction.as_vec3(),
                            inner,
                            outer,
                        },
                    ),
                    ImportedLightKind::Directional => Light::sun(
                        (center - origin).as_vec3(),
                        -l.direction,
                        extent * 50.0,
                        power,
                    ),
                }
            })
            .collect()
    }
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
    let mut tex = p.tex.clone();
    if mirrored {
        for t in &mut triangles {
            t.swap(1, 2);
        }
        if let Some(n) = &mut normals {
            for c in n.iter_mut() {
                c.swap(1, 2);
            }
        }
        if let Some(t) = &mut tex {
            for c in t.iter_mut() {
                c.swap(1, 2);
            }
        }
    }
    let geometry = scene.geometries.len();
    scene
        .geometries
        .push(Surface::from_mesh(triangles, normals, tex));
    scene.geometry_textures.push(p.texture);
    scene.geometry_materials.push(p.material);
    scene.geometry_emission.push(p.emission);
    scene.geometry_normal_maps.push(p.normal_map);
    scene.geometry_metallic_roughness.push(p.metallic_roughness);
    scene.instances.push(Instance {
        geometry,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position,
        albedo: p.albedo,
    });
}

/// Writes an instanced scene as binary glTF (`.glb`), the inverse of [`load`]
/// (for tests and for comparison in other renderers).
///
/// - Each geometry's triangles are written once, unindexed, with normals and
///   texture coordinates where it has them.
/// - Instances sharing a geometry and an albedo share a mesh; each instance
///   is a node with its matrix.
/// - Textures are embedded as PNG.
/// - Z-up becomes Y-up: (x, y, z) becomes (x, z, −y).
pub fn save(scene: &InstancedScene, path: &Path) -> std::io::Result<()> {
    use std::collections::HashMap;
    use std::fmt::Write as _;

    let mut bin: Vec<u8> = Vec::new();
    let mut views: Vec<String> = Vec::new();
    let mut accessors: Vec<String> = Vec::new();
    // Appends bytes as a buffer view (aligned to four), returning its index.
    let mut view = |bin: &mut Vec<u8>, bytes: &[u8], target: Option<u32>| {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        let target = target.map_or(String::new(), |t| format!(r#","target":{t}"#));
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{}{target}}}"#,
            bytes.len()
        ));
        views.len() - 1
    };
    let floats =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let y_up = |v: Vec3| [v.x, v.z, -v.y];
    // Per geometry: accessors for positions, normals and texture coordinates.
    let mut attributes: Vec<String> = Vec::new();
    for g in &scene.geometries {
        let positions: Vec<f32> = g
            .triangles
            .iter()
            .flatten()
            .flat_map(|&p| y_up(p))
            .collect();
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in positions.as_chunks::<3>().0 {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let count = 3 * g.triangles.len();
        let v = view(&mut bin, &floats(&positions), Some(34962));
        accessors.push(format!(
            r#"{{"bufferView":{v},"componentType":5126,"count":{count},"type":"VEC3","min":[{:?},{:?},{:?}],"max":[{:?},{:?},{:?}]}}"#,
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        ));
        let mut attr = format!(r#""POSITION":{}"#, accessors.len() - 1);
        let normals: Vec<f32> = g.normals.iter().flatten().flat_map(|&n| y_up(n)).collect();
        let v = view(&mut bin, &floats(&normals), Some(34962));
        accessors.push(format!(
            r#"{{"bufferView":{v},"componentType":5126,"count":{count},"type":"VEC3"}}"#
        ));
        write!(attr, r#","NORMAL":{}"#, accessors.len() - 1).unwrap();
        if g.tex.len() == g.triangles.len() {
            let tex: Vec<f32> = g.tex.iter().flatten().flat_map(|t| [t.x, t.y]).collect();
            let v = view(&mut bin, &floats(&tex), Some(34962));
            accessors.push(format!(
                r#"{{"bufferView":{v},"componentType":5126,"count":{count},"type":"VEC2"}}"#
            ));
            write!(attr, r#","TEXCOORD_0":{}"#, accessors.len() - 1).unwrap();
        }
        attributes.push(attr);
    }
    // Textures as PNG.
    let mut images: Vec<String> = Vec::new();
    for t in &scene.textures {
        let mut png_bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png_bytes, t.width, t.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
            writer
                .write_image_data(&t.rgba)
                .map_err(std::io::Error::other)?;
        }
        let v = view(&mut bin, &png_bytes, None);
        images.push(format!(r#"{{"bufferView":{v},"mimeType":"image/png"}}"#));
    }
    // Meshes per (geometry, albedo), materials per mesh, nodes per instance.
    let mut meshes: Vec<String> = Vec::new();
    let mut materials: Vec<String> = Vec::new();
    let mut emissive_strength = false;
    let mut mesh_of: HashMap<(usize, [u32; 3]), usize> = HashMap::new();
    let mut nodes: Vec<String> = Vec::new();
    let (c, c_inv) = axes();
    for i in &scene.instances {
        let key = (i.geometry, i.albedo.to_array().map(f32::to_bits));
        let mesh = *mesh_of.entry(key).or_insert_with(|| {
            let texture = scene
                .geometry_textures
                .get(i.geometry)
                .copied()
                .flatten()
                .filter(|_| scene.geometries[i.geometry].tex.len() == scene.geometries[i.geometry].triangles.len());
            let has_tex = scene.geometries[i.geometry].tex.len() == scene.geometries[i.geometry].triangles.len();
            let mut base = texture.map_or(String::new(), |t| {
                format!(r#","baseColorTexture":{{"index":{t}}}"#)
            });
            if let Some(t) = scene
                .geometry_metallic_roughness
                .get(i.geometry)
                .copied()
                .flatten()
                .filter(|_| has_tex && scene.geometry_materials.get(i.geometry).copied().flatten().is_some())
            {
                write!(base, r#","metallicRoughnessTexture":{{"index":{t}}}"#).unwrap();
            }
            let bumps = scene
                .geometry_normal_maps
                .get(i.geometry)
                .copied()
                .flatten()
                .filter(|_| scene.geometries[i.geometry].tex.len() == scene.geometries[i.geometry].triangles.len())
                .map_or(String::new(), |t| format!(r#","normalTexture":{{"index":{t}}}"#));
            let m = scene
                .geometry_materials
                .get(i.geometry)
                .copied()
                .flatten()
                .unwrap_or(Material {
                    metallic: 0.0,
                    roughness: 1.0,
                });
            // Emission as a factor of at most 1 times a strength.
            let e = scene
                .geometry_emission
                .get(i.geometry)
                .copied()
                .unwrap_or(Vec3::ZERO);
            let strength = e.max_element().max(1.0);
            let glow = if e.max_element() > 0.0 {
                emissive_strength |= strength > 1.0;
                let f = e / strength;
                let mut g = format!(r#","emissiveFactor":[{:?},{:?},{:?}]"#, f.x, f.y, f.z);
                if strength > 1.0 {
                    write!(
                        g,
                        r#","extensions":{{"KHR_materials_emissive_strength":{{"emissiveStrength":{strength:?}}}}}"#
                    )
                    .unwrap();
                }
                g
            } else {
                String::new()
            };
            materials.push(format!(
                r#"{{"pbrMetallicRoughness":{{"baseColorFactor":[{:?},{:?},{:?},1.0],"metallicFactor":{:?},"roughnessFactor":{:?}{base}}}{glow}{bumps}}}"#,
                i.albedo.x, i.albedo.y, i.albedo.z, m.metallic, m.roughness
            ));
            meshes.push(format!(
                r#"{{"primitives":[{{"attributes":{{{}}},"material":{}}}]}}"#,
                attributes[i.geometry],
                materials.len() - 1
            ));
            meshes.len() - 1
        });
        // From the rotation's columns, without a round trip through a
        // quaternion.
        let m = DMat4::from_cols(
            (i.rotation.x_axis.as_dvec3() * i.scale as f64).extend(0.0),
            (i.rotation.y_axis.as_dvec3() * i.scale as f64).extend(0.0),
            (i.rotation.z_axis.as_dvec3() * i.scale as f64).extend(0.0),
            i.position.extend(1.0),
        );
        let g = c_inv * m * c;
        let values: Vec<String> = g
            .to_cols_array()
            .iter()
            .map(|v| format!("{:?}", *v as f32))
            .collect();
        nodes.push(format!(
            r#"{{"mesh":{mesh},"matrix":[{}]}}"#,
            values.join(",")
        ));
    }
    let node_list: Vec<String> = (0..nodes.len()).map(|k| k.to_string()).collect();
    let textures: Vec<String> = (0..images.len())
        .map(|k| format!(r#"{{"source":{k},"sampler":0}}"#))
        .collect();
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let mut json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"Continuum"}},"scene":0,"scenes":[{{"nodes":[{}]}}],"nodes":[{}],"meshes":[{}],"materials":[{}],"accessors":[{}],"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]"#,
        node_list.join(","),
        nodes.join(","),
        meshes.join(","),
        materials.join(","),
        accessors.join(","),
        views.join(","),
        bin.len()
    );
    if emissive_strength {
        json.push_str(r#","extensionsUsed":["KHR_materials_emissive_strength"]"#);
    }
    if !images.is_empty() {
        write!(
            json,
            r#","images":[{}],"textures":[{}],"samplers":[{{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}}]"#,
            images.join(","),
            textures.join(",")
        )
        .unwrap();
    }
    json.push('}');
    let glb = gltf::binary::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: std::borrow::Cow::Owned(json.into_bytes()),
        bin: Some(std::borrow::Cow::Owned(bin)),
    };
    let file = std::fs::File::create(path)?;
    glb.to_writer(std::io::BufWriter::new(file))
        .map_err(std::io::Error::other)
}
