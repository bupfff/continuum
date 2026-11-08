//! glTF `KHR_texture_transform`: imported texture coordinates are the file's
//! through each slot's transform, as Blender's importer applies it, a slot's own
//! `texCoord` picks its set, and a slot whose transform differs from the
//! others' is skipped.

use continuum::glam::{Vec2, Vec3};

/// A unit quad in the glTF's xy plane (z = 0), two coordinate sets, and a
/// 1×1 PNG, as base64.
type Uvs = Vec<[f32; 2]>;

/// A transform: offset, rotation, scale.
type Transform = [f32; 5];

fn buffer() -> (String, Uvs, Uvs, Vec<[f32; 3]>) {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let uv0 = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let uv1 = vec![[0.1, 0.2], [0.7, 0.3], [0.9, 0.8], [0.2, 0.6]];
    let mut bytes = Vec::new();
    positions
        .iter()
        .flatten()
        .chain(uv0.iter().flatten())
        .chain(uv1.iter().flatten())
        .for_each(|f: &f32| bytes.extend(f.to_le_bytes()));
    for i in [0u16, 1, 2, 0, 2, 3] {
        bytes.extend(i.to_le_bytes());
    }
    (base64(&bytes), uv0, uv1, positions)
}

fn base64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                out.push(A[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn png() -> String {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[200, 100, 50, 255]).unwrap();
    }
    base64(&bytes)
}

/// Transform `t` (offset, rotation, scale) as Blender's glTF importer
/// applies it, an independent route: `texture_transform_gltf_to_blender`
/// (io_scene_gltf2, Blender 5.2) and a Mapping node (scale, rotate
/// counter-clockwise, offset) in Blender's UV space, where v is flipped.
fn transform(t: [f32; 5], uv: [f32; 2]) -> Vec2 {
    let (offset, rotation, scale) = ([t[0], t[1]], t[2], [t[3], t[4]]);
    let (sin, cos) = rotation.sin_cos();
    let location = [offset[0] + scale[1] * sin, 1.0 - offset[1] - scale[1] * cos];
    let p = [uv[0] * scale[0], (1.0 - uv[1]) * scale[1]];
    let q = [
        cos * p[0] - sin * p[1] + location[0],
        sin * p[0] + cos * p[1] + location[1],
    ];
    Vec2::new(q[0], 1.0 - q[1])
}

#[test]
fn texture_coordinates_go_through_each_slots_transform() {
    let (data, uv0, uv1, positions) = buffer();
    // Offset, rotation, scale.
    let a = [0.25f32, 0.5, 0.3, 2.0, 3.0];
    let b = [-0.5f32, 0.125, -1.2, 0.5, 4.0];
    let ext = |t: [f32; 5], set: Option<u32>| {
        format!(
            r#"{{"KHR_texture_transform": {{"offset": [{}, {}], "rotation": {}, "scale": [{}, {}]{}}}}}"#,
            t[0],
            t[1],
            t[2],
            t[3],
            t[4],
            set.map_or(String::new(), |s| format!(r#", "texCoord": {s}"#))
        )
    };
    let json = format!(
        r#"{{
  "asset": {{"version": "2.0"}},
  "extensionsUsed": ["KHR_texture_transform"],
  "extensionsRequired": ["KHR_texture_transform"],
  "buffers": [{{"byteLength": 124, "uri": "data:application/octet-stream;base64,{data}"}}],
  "bufferViews": [{{"buffer": 0, "byteOffset": 0, "byteLength": 112}},
                  {{"buffer": 0, "byteOffset": 112, "byteLength": 12}}],
  "accessors": [
    {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 0, "byteOffset": 48, "componentType": 5126, "count": 4, "type": "VEC2"}},
    {{"bufferView": 0, "byteOffset": 80, "componentType": 5126, "count": 4, "type": "VEC2"}},
    {{"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}}
  ],
  "images": [{{"uri": "data:image/png;base64,{png}"}}],
  "textures": [{{"source": 0}}],
  "materials": [
    {{"pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0, "extensions": {a0}}}}},
      "normalTexture": {{"index": 0, "extensions": {a0}}},
      "emissiveTexture": {{"index": 0, "extensions": {b0}}},
      "emissiveFactor": [1, 1, 1]}},
    {{"pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0, "extensions": {b1}}}}}}},
    {{"pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0}}}},
      "normalTexture": {{"index": 0, "extensions": {a0}}}}}
  ],
  "meshes": [
    {{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}}, "indices": 3, "material": 0}}]}},
    {{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}}, "indices": 3, "material": 1}}]}},
    {{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}}, "indices": 3, "material": 2}}]}}
  ],
  "nodes": [{{"mesh": 0}}, {{"mesh": 1, "translation": [2, 0, 0]}}, {{"mesh": 2, "translation": [4, 0, 0]}}],
  "scenes": [{{"nodes": [0, 1, 2]}}],
  "scene": 0
}}"#,
        png = png(),
        a0 = ext(a, None),
        b0 = ext(b, None),
        b1 = ext(b, Some(1)),
    );
    let dir = std::env::temp_dir().join(format!(
        "continuum-texture-transform-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("quads.gltf");
    std::fs::write(&path, json).unwrap();
    let imported = continuum_gltf::load(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let scene = &imported.scene;
    assert_eq!(
        scene.geometries.len(),
        3,
        "three materials, three geometries"
    );
    // Material 0: base color and normal map through `a` on set 0; the
    // emissive texture's `b` differs, so it is skipped.
    // Material 1: base color through `b` on set 1 (the transform's
    // texCoord). Material 2: base color untransformed, so its normal map
    // through `a` is skipped.
    let expected: [(&Uvs, Option<Transform>); 3] = [(&uv0, Some(a)), (&uv1, Some(b)), (&uv0, None)];
    for (g, (uvs, t)) in expected.iter().enumerate() {
        let geometry = &scene.geometries[g];
        assert_eq!(geometry.triangles.len(), 2);
        assert_eq!(geometry.tex.len(), 2);
        for (tri, tex) in geometry.triangles.iter().zip(&geometry.tex) {
            for (p, uv) in tri.iter().zip(tex) {
                // glTF (x, y, z) is Continuum's (x, −z, y): find the vertex.
                let k = positions
                    .iter()
                    .position(|q| (Vec3::new(q[0], -q[2], q[1]) - *p).length() < 1e-6)
                    .unwrap();
                let want = t.map_or(Vec2::from_array(uvs[k]), |t| transform(t, uvs[k]));
                assert!((*uv - want).length() < 1e-5, "geometry {g}: {uv} vs {want}");
            }
        }
        assert!(scene.geometry_textures[g].is_some());
    }
    assert!(scene.geometry_normal_maps[0].is_some());
    assert!(scene.geometry_emissive_textures[0].is_none());
    assert!(scene.geometry_normal_maps[2].is_none());
    assert_eq!(imported.stats.texture_slots_skipped, 2);
}
