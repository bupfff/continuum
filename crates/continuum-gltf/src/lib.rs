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
//!   (`InstancedScene::textures`). `KHR_texture_transform` is applied to the
//!   texture coordinates at import, as Blender's importer applies it. A
//!   geometry has one set of texture coordinates: the first texture slot
//!   that loads (the base color's, where it has one) chooses the set and
//!   transform, and a slot with others is skipped
//!   ([`ImportStats::texture_slots_skipped`]).
//! - Metallic and roughness factors become the geometry's [`Material`]
//!   (`InstancedScene::geometry_materials`). A primitive without a material,
//!   or with metallic 0 and roughness 1 (what [`save`] writes for a
//!   Lambertian surface), stays Lambertian: the difference is a rough 4%
//!   specular layer. A metallic-roughness texture (through the base
//!   color's texture coordinates, or alone) becomes the geometry's
//!   (`InstancedScene::geometry_metallic_roughness`).
//! - Alpha mode MASK becomes the geometry's alpha cutoff
//!   (`InstancedScene::geometry_alpha_cutoff`: cut away as geometry before
//!   upload), divided by the base color factor's alpha. BLEND becomes
//!   alpha blending (`InstancedScene::geometry_blend`, M10 gate 16): the
//!   base color factor's alpha as the opacity, times the texture's alpha
//!   (where blending is not available the renderer cuts a textured BLEND at
//!   0.5, as before).
//! - A double-sided material's triangles are doubled, the back faces
//!   reversed on the same plane.
//! - A normal texture, through the same texture coordinates as the base
//!   color (or alone), becomes the geometry's normal map
//!   (`InstancedScene::geometry_normal_maps`; its scale is ignored).
//! - An emissive texture, through the same texture coordinates as the base
//!   color (or alone), becomes the geometry's emissive texture
//!   (`InstancedScene::geometry_emissive_textures`, M10 gate 14).
//! - The emissive factor, times `KHR_materials_emissive_strength`, becomes
//!   the geometry's emission (`InstancedScene::geometry_emission`).
//! - Vertex colors (`COLOR_0`'s RGB; alpha ignored) become the geometry's
//!   (`InstancedScene::geometry_colors`, M10 gate 17), multiplying its base
//!   color, and are saved.
//! - Punctual lights (`KHR_lights_punctual`) are imported
//!   ([`Imported::lights`]): point and spot lights as small spheres,
//!   directional lights as suns.
//! - Node animations (translation and rotation channels; step, linear and
//!   cubic-spline) are kept ([`Imported::animations`]):
//!   [`Animations::pose`] gives each animated instance's transform at a
//!   time, for `Renderer::set_transform` on movable instances.
//! - Environment images (an equirectangular `.hdr` or `.png` sky for
//!   `RendererOptions::environment`) load with [`load_environment`].

#![warn(missing_docs)]

mod animation;
mod environment;

pub use animation::{Animation, Animations};
pub use environment::{decode_environment, load_environment};

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
    /// Texture slots skipped because their coordinate set or
    /// `KHR_texture_transform` differs from the primitive's other slots
    /// (a geometry has one set of texture coordinates).
    pub texture_slots_skipped: usize,
    /// Animation channels skipped: scale and morph weights (movers keep
    /// their scale), or keyframes that do not match their times.
    pub animation_channels_skipped: usize,
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
    /// The file's node animations, played through movable instances.
    pub animations: Animations,
}

/// Y-up to Z-up.
fn to_z_up(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], -v[2], v[1])
}

/// The same change of axes as a matrix, and its inverse.
pub(crate) fn axes() -> (DMat4, DMat4) {
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
    Option<u32>,
    bool,
    // The emissive texture, the blend opacity's bits, the texture
    // transform's, and the vertex colors' accessor.
    (Option<u32>, Option<u32>, [u32; 5], Option<usize>),
);

/// A texture slot's coordinate set and `KHR_texture_transform` (the identity
/// without one).
#[derive(Clone, Copy, Debug, PartialEq)]
struct UvSlot {
    set: u32,
    offset: [f32; 2],
    rotation: f32,
    scale: [f32; 2],
}

impl Default for UvSlot {
    fn default() -> Self {
        UvSlot {
            set: 0,
            offset: [0.0; 2],
            rotation: 0.0,
            scale: [1.0; 2],
        }
    }
}

impl UvSlot {
    fn of(info: &gltf::texture::Info) -> UvSlot {
        match info.texture_transform() {
            Some(t) => UvSlot {
                set: t.tex_coord().unwrap_or(info.tex_coord()),
                offset: t.offset(),
                rotation: t.rotation(),
                scale: t.scale(),
            },
            None => UvSlot {
                set: info.tex_coord(),
                ..Default::default()
            },
        }
    }

    /// A normal texture's (the gltf crate leaves its transform as JSON).
    fn of_normal(info: &gltf::material::NormalTexture) -> UvSlot {
        let mut slot = UvSlot {
            set: info.tex_coord(),
            ..Default::default()
        };
        if let Some(t) = info.extension_value("KHR_texture_transform") {
            let pair = |key: &str, default: [f32; 2]| match t.get(key).and_then(|v| v.as_array()) {
                Some(a) if a.len() == 2 => [0, 1].map(|k| a[k].as_f64().unwrap_or(0.0) as f32),
                _ => default,
            };
            slot.offset = pair("offset", [0.0; 2]);
            slot.scale = pair("scale", [1.0; 2]);
            slot.rotation = t.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            if let Some(set) = t.get("texCoord").and_then(|v| v.as_u64()) {
                slot.set = set as u32;
            }
        }
        slot
    }

    /// `uv` through the transform: scaled, rotated, then offset. The
    /// rotation's sign is the one Blender's importer applies (checked in
    /// `tests/texture_transform.rs`); the extension's GLSL sample, read
    /// column-major, is its transpose.
    fn apply(&self, uv: [f32; 2]) -> Vec2 {
        let (sin, cos) = self.rotation.sin_cos();
        let (u, v) = (uv[0] * self.scale[0], uv[1] * self.scale[1]);
        Vec2::new(
            cos * u + sin * v + self.offset[0],
            -sin * u + cos * v + self.offset[1],
        )
    }

    /// The transform's bits, for deduplicating geometries.
    fn bits(&self) -> [u32; 5] {
        [
            self.offset[0],
            self.offset[1],
            self.rotation,
            self.scale[0],
            self.scale[1],
        ]
        .map(f32::to_bits)
    }
}

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
    /// Its emissive texture's index in the scene's textures (M10 gate 14).
    emissive_texture: Option<u32>,
    /// Its opacity if alpha-blended (M10 gate 16).
    blend: Option<f32>,
    /// Its alpha cutoff (over its base color texture's alpha).
    alpha_cutoff: Option<f32>,
    /// Its vertex colors per triangle corner (`COLOR_0`, M10 gate 17).
    colors: Option<Vec<[Vec3; 3]>>,
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
            // Vertex colors (M10 gate 17): COLOR_0's RGB, linear.
            let colors: Option<Vec<Vec3>> = reader
                .read_colors(0)
                .map(|c| c.into_rgb_f32().map(Vec3::from_array).collect());
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
            // Texture slots share one coordinate set and transform: the
            // first slot that loads chooses them (the base color's, where it
            // has one), and a slot with others is skipped.
            let mut texcoords: Option<Vec<Vec2>> = None;
            let mut chosen: Option<UvSlot> = None;
            let mut skipped = 0;
            let mut slot_texture = |slot: UvSlot, image: usize| -> Option<u32> {
                if chosen.is_some_and(|c| c != slot) {
                    skipped += 1;
                    return None;
                }
                let index = (*texture_of_image[image].get_or_insert_with(|| {
                    to_rgba(&images[image]).map(|t| {
                        textures.push(t);
                        textures.len() as u32 - 1
                    })
                }))?;
                if chosen.is_none() {
                    let t = reader.read_tex_coords(slot.set)?;
                    texcoords = Some(t.into_f32().map(|uv| slot.apply(uv)).collect());
                    chosen = Some(slot);
                }
                Some(index)
            };
            let texture = pbr
                .base_color_texture()
                .and_then(|i| slot_texture(UvSlot::of(&i), i.texture().source().index()));
            let normal_map = material
                .normal_texture()
                .and_then(|i| slot_texture(UvSlot::of_normal(&i), i.texture().source().index()));
            // The emissive texture (sRGB, M10 gate 14).
            let emissive_texture = material
                .emissive_texture()
                .and_then(|i| slot_texture(UvSlot::of(&i), i.texture().source().index()));
            let metallic_roughness = pbr
                .metallic_roughness_texture()
                .filter(|_| surface.is_some())
                .and_then(|i| slot_texture(UvSlot::of(&i), i.texture().source().index()));
            stats.texture_slots_skipped += skipped;
            let chosen = chosen.unwrap_or_default();
            let alpha_cutoff = match material.alpha_mode() {
                gltf::material::AlphaMode::Mask => Some(material.alpha_cutoff().unwrap_or(0.5)),
                _ => None,
            }
            .filter(|_| texture.is_some())
            .map(|c| c / factor[3].max(1e-6));
            let blend = (material.alpha_mode() == gltf::material::AlphaMode::Blend)
                .then_some(factor[3].clamp(0.0, 1.0));
            let double_sided = material.double_sided();
            let mut triangles = Vec::with_capacity(indices.len() / 3);
            let mut corner_normals = Vec::with_capacity(indices.len() / 3);
            let mut corner_tex = Vec::with_capacity(indices.len() / 3);
            let mut corner_colors = Vec::with_capacity(indices.len() / 3);
            for t in indices.as_chunks::<3>().0 {
                let k = [t[0] as usize, t[1] as usize, t[2] as usize];
                triangles.push(k.map(|i| positions[i]));
                if let Some(n) = &normals {
                    corner_normals.push(k.map(|i| n[i]));
                }
                if let Some(tc) = &texcoords {
                    corner_tex.push(k.map(|i| tc[i]));
                }
                if let Some(c) = &colors {
                    corner_colors.push(k.map(|i| c[i]));
                }
            }
            if double_sided {
                // Back faces: reversed, on the same plane.
                for f in 0..triangles.len() {
                    let t = triangles[f];
                    triangles.push([t[0], t[2], t[1]]);
                    if normals.is_some() {
                        let n = corner_normals[f];
                        corner_normals.push([-n[0], -n[2], -n[1]]);
                    }
                    if texcoords.is_some() {
                        let c = corner_tex[f];
                        corner_tex.push([c[0], c[2], c[1]]);
                    }
                    if colors.is_some() {
                        let c = corner_colors[f];
                        corner_colors.push([c[0], c[2], c[1]]);
                    }
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
                emissive_texture,
                blend,
                alpha_cutoff,
                colors: colors.is_some().then_some(corner_colors),
                key: (
                    accessor(gltf::Semantic::Positions).unwrap_or(usize::MAX),
                    accessor(gltf::Semantic::Normals),
                    texcoords
                        .as_ref()
                        .and(accessor(gltf::Semantic::TexCoords(chosen.set))),
                    primitive.indices().map(|a| a.index()),
                    texture,
                    surface.map(|m| (m.metallic.to_bits(), m.roughness.to_bits())),
                    emission.to_array().map(f32::to_bits),
                    normal_map,
                    metallic_roughness,
                    alpha_cutoff.map(f32::to_bits),
                    double_sided,
                    (
                        emissive_texture,
                        blend.map(f32::to_bits),
                        chosen.bits(),
                        accessor(gltf::Semantic::Colors(0)),
                    ),
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
            scene
                .geometry_colors
                .push(kept_colors(&p.triangles, p.colors.as_ref()));
            scene.geometry_textures.push(p.texture);
            scene.geometry_materials.push(p.material);
            scene.geometry_emission.push(p.emission);
            scene.geometry_normal_maps.push(p.normal_map);
            scene.geometry_metallic_roughness.push(p.metallic_roughness);
            scene.geometry_alpha_cutoff.push(p.alpha_cutoff);
            scene.geometry_emissive_textures.push(p.emissive_texture);
            scene.geometry_blend.push(p.blend);
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
    // Every node at rest, with its parent (for animations).
    let mut nodes: Vec<animation::Node> = document
        .nodes()
        .map(|n| {
            let (t, r, s) = n.transform().decomposed();
            animation::Node {
                parent: None,
                translation: DVec3::from_array(t.map(f64::from)),
                rotation: continuum::glam::DQuat::from_array(r.map(f64::from)),
                scale: DVec3::from_array(s.map(f64::from)),
            }
        })
        .collect();
    for n in document.nodes() {
        for child in n.children() {
            nodes[child.index()].parent = Some(n.index());
        }
    }
    let mut instance_nodes: Vec<Option<usize>> = Vec::new();
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
                let baked = stats.baked;
                place(&mut scene, p, m, &mut stats);
                // A baked instance cannot move.
                instance_nodes.push((stats.baked == baked).then_some(node.index()));
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
    let mut list = Vec::new();
    for anim in document.animations() {
        use gltf::animation::util::ReadOutputs;
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for ch in anim.channels() {
            let reader = ch.reader(|b| Some(&buffers[b.index()]));
            let interpolation = match ch.sampler().interpolation() {
                gltf::animation::Interpolation::Step => animation::Interpolation::Step,
                gltf::animation::Interpolation::Linear => animation::Interpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => {
                    animation::Interpolation::CubicSpline
                }
            };
            let times: Vec<f32> = match reader.read_inputs() {
                Some(t) => t.collect(),
                None => Vec::new(),
            };
            let f64s = |v: [f32; 3]| DVec3::from_array(v.map(f64::from));
            let values = match reader.read_outputs() {
                Some(ReadOutputs::Translations(t)) => {
                    animation::Values::Translation(t.map(f64s).collect())
                }
                Some(ReadOutputs::Rotations(r)) => animation::Values::Rotation(
                    r.into_f32()
                        .map(|q| continuum::glam::DQuat::from_array(q.map(f64::from)))
                        .collect(),
                ),
                _ => {
                    stats.animation_channels_skipped += 1;
                    continue;
                }
            };
            let count = match &values {
                animation::Values::Translation(v) => v.len(),
                animation::Values::Rotation(v) => v.len(),
            };
            let per = if interpolation == animation::Interpolation::CubicSpline {
                3
            } else {
                1
            };
            if times.is_empty() || count != per * times.len() {
                stats.animation_channels_skipped += 1;
                continue;
            }
            duration = duration.max(*times.last().unwrap());
            channels.push(animation::Channel {
                node: ch.target().node().index(),
                interpolation,
                times,
                values,
            });
        }
        list.push(Animation {
            name: anim.name().map(String::from),
            duration,
            channels,
        });
    }
    let animations = Animations {
        list,
        nodes,
        instance_nodes,
    };
    Ok(Imported {
        scene,
        camera,
        lights,
        bounds,
        stats,
        animations,
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
/// Per-corner vertex colors for `Surface::from_mesh(triangles, ..)`: those
/// of the triangles it keeps.
fn kept_colors(triangles: &[Tri], colors: Option<&Vec<[Vec3; 3]>>) -> Option<Vec<[Vec3; 3]>> {
    colors.map(|c| {
        triangles
            .iter()
            .zip(c)
            .filter(|(t, _)| !continuum::mesh::degenerate(t))
            .map(|(_, c)| *c)
            .collect()
    })
}

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
    let mut colors = p.colors.clone();
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
        if let Some(t) = &mut colors {
            for c in t.iter_mut() {
                c.swap(1, 2);
            }
        }
    }
    let geometry = scene.geometries.len();
    scene
        .geometry_colors
        .push(kept_colors(&triangles, colors.as_ref()));
    scene
        .geometries
        .push(Surface::from_mesh(triangles, normals, tex));
    scene.geometry_textures.push(p.texture);
    scene.geometry_materials.push(p.material);
    scene.geometry_emission.push(p.emission);
    scene.geometry_normal_maps.push(p.normal_map);
    scene.geometry_metallic_roughness.push(p.metallic_roughness);
    scene.geometry_alpha_cutoff.push(p.alpha_cutoff);
    scene.geometry_emissive_textures.push(p.emissive_texture);
    scene.geometry_blend.push(p.blend);
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
/// - Each geometry's triangles are written once, unindexed, with normals,
///   and texture coordinates and vertex colors where it has them.
/// - Materials: metallic and roughness, emission (with its emissive
///   texture), normal maps, metallic-roughness textures, an alpha cutoff as
///   MASK, or else blending as BLEND (the opacity as the base color factor's
///   alpha). The environment is not part of glTF.
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
    // Per geometry: accessors for positions, normals, texture coordinates
    // and vertex colors.
    let mut attributes: Vec<String> = Vec::new();
    for (k, g) in scene.geometries.iter().enumerate() {
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
        if let Some(Some(colors)) = scene.geometry_colors.get(k)
            && colors.len() == g.triangles.len()
        {
            let colors: Vec<f32> = colors.iter().flatten().flat_map(|c| c.to_array()).collect();
            let v = view(&mut bin, &floats(&colors), Some(34962));
            accessors.push(format!(
                r#"{{"bufferView":{v},"componentType":5126,"count":{count},"type":"VEC3"}}"#
            ));
            write!(attr, r#","COLOR_0":{}"#, accessors.len() - 1).unwrap();
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
            // An alpha cutoff as MASK; otherwise blending as BLEND, its
            // opacity the base color factor's alpha.
            let mut opacity = 1.0f32;
            let cut = match scene
                .geometry_alpha_cutoff
                .get(i.geometry)
                .copied()
                .flatten()
                .filter(|_| texture.is_some())
            {
                Some(c) => format!(r#","alphaMode":"MASK","alphaCutoff":{c:?}"#),
                None => match scene.geometry_blend.get(i.geometry).copied().flatten() {
                    Some(o) => {
                        opacity = o.clamp(0.0, 1.0);
                        r#","alphaMode":"BLEND""#.to_string()
                    }
                    None => String::new(),
                },
            };
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
                if let Some(t) = scene
                    .geometry_emissive_textures
                    .get(i.geometry)
                    .copied()
                    .flatten()
                    .filter(|_| has_tex)
                {
                    write!(g, r#","emissiveTexture":{{"index":{t}}}"#).unwrap();
                }
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
                r#"{{"pbrMetallicRoughness":{{"baseColorFactor":[{:?},{:?},{:?},{:?}],"metallicFactor":{:?},"roughnessFactor":{:?}{base}}}{glow}{bumps}{cut}}}"#,
                i.albedo.x, i.albedo.y, i.albedo.z, opacity, m.metallic, m.roughness
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
