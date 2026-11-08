//! glTF node animations (`Imported::animations`): poses at times, through
//! the node hierarchy, for linear (slerp for rotations) and cubic-spline
//! channels, against values worked out by hand from the glTF spec.

use continuum::glam::{DVec3, Mat3};

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

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn animations_pose_instances_through_the_hierarchy() {
    let mut bin = floats(&[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0]);
    for i in [0u16, 1, 2, 0, 2, 3] {
        bin.extend(i.to_le_bytes());
    }
    let h = std::f32::consts::FRAC_1_SQRT_2;
    bin.extend(floats(&[0.0, 2.0])); // times
    bin.extend(floats(&[0.0, 0.0, 0.0, 2.0, 0.0, 4.0])); // translations
    bin.extend(floats(&[0.0, 0.0, 0.0, 1.0, 0.0, h, 0.0, h])); // rotations: 0° to 90° about y
    bin.extend(floats(&[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // in, value, out at 0
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, // in, value, out at 2
    ]));
    assert_eq!(bin.len(), 196);
    let json = format!(
        r#"{{
  "asset": {{"version": "2.0"}},
  "buffers": [{{"byteLength": 196, "uri": "data:application/octet-stream;base64,{}"}}],
  "bufferViews": [{{"buffer": 0, "byteOffset": 0, "byteLength": 48}},
                  {{"buffer": 0, "byteOffset": 48, "byteLength": 12}},
                  {{"buffer": 0, "byteOffset": 60, "byteLength": 136}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [0.1, 0.1, 0]}},
    {{"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}},
    {{"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [2]}},
    {{"bufferView": 2, "byteOffset": 8, "componentType": 5126, "count": 2, "type": "VEC3"}},
    {{"bufferView": 2, "byteOffset": 32, "componentType": 5126, "count": 2, "type": "VEC4"}},
    {{"bufferView": 2, "byteOffset": 64, "componentType": 5126, "count": 6, "type": "VEC3"}}
  ],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1}}]}}],
  "nodes": [
    {{"mesh": 0, "children": [1]}},
    {{"mesh": 0, "translation": [0, 1, 0]}},
    {{"mesh": 0, "translation": [5, 0, 0]}}
  ],
  "animations": [
    {{"name": "move", "samplers": [{{"input": 2, "output": 3}}, {{"input": 2, "output": 4}}],
      "channels": [{{"sampler": 0, "target": {{"node": 0, "path": "translation"}}}},
                   {{"sampler": 1, "target": {{"node": 1, "path": "rotation"}}}}]}},
    {{"samplers": [{{"input": 2, "output": 5, "interpolation": "CUBICSPLINE"}}],
      "channels": [{{"sampler": 0, "target": {{"node": 2, "path": "translation"}}}}]}}
  ],
  "scenes": [{{"nodes": [0, 2]}}],
  "scene": 0
}}"#,
        base64(&bin)
    );
    let dir = std::env::temp_dir().join(format!("continuum-animation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("animated.gltf");
    std::fs::write(&path, json).unwrap();
    let imported = continuum_gltf::load(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let a = &imported.animations;
    assert_eq!(a.list.len(), 2);
    assert_eq!(a.list[0].name.as_deref(), Some("move"));
    assert_eq!(a.list[0].duration, 2.0);
    // Instances by their rest position (glTF (x, y, z) is Continuum's
    // (x, −z, y)): the root at 0, its child 1 m up, the other node 5 m out.
    let at = |p: DVec3| {
        imported
            .scene
            .instances
            .iter()
            .position(|i| (i.position - p).length() < 1e-9)
            .unwrap()
    };
    let (root, child, other) = (at(DVec3::ZERO), at(DVec3::Z), at(DVec3::X * 5.0));
    let mut moved = a.instances(0);
    moved.sort();
    let mut want = vec![root, child];
    want.sort();
    assert_eq!(moved, want, "the root and, through it, its child");
    assert_eq!(a.instances(1), vec![other]);
    let pose = |anim: usize, t: f32, i: usize| {
        a.pose(anim, t)
            .into_iter()
            .find(|p| p.0 == i)
            .map(|p| (p.1, p.2))
            .unwrap()
    };
    let close = |(r, p): (Mat3, DVec3), (wr, wp): (Mat3, DVec3)| {
        assert!((p - wp).length() < 1e-6, "position {p} against {wp}");
        assert!(
            (r.to_cols_array().iter().zip(wr.to_cols_array())).all(|(x, y)| (x - y).abs() < 1e-6),
            "rotation {r} against {wr}"
        );
    };
    // Halfway: the root at glTF (1, 0, 2); the child 1 above it, turned 45°
    // about glTF's y (Continuum's z) by slerp.
    close(
        pose(0, 1.0, root),
        (Mat3::IDENTITY, DVec3::new(1.0, -2.0, 0.0)),
    );
    close(
        pose(0, 1.0, child),
        (
            Mat3::from_rotation_z(std::f32::consts::FRAC_PI_4),
            DVec3::new(1.0, -2.0, 1.0),
        ),
    );
    // Past the end: held at the last keyframe.
    close(
        pose(0, 3.0, root),
        (Mat3::IDENTITY, DVec3::new(2.0, -4.0, 0.0)),
    );
    close(
        pose(0, 3.0, child),
        (
            Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2),
            DVec3::new(2.0, -4.0, 1.0),
        ),
    );
    // Cubic spline at s = 1/2 over 2 s: v0/2 + (2/8)·out0 + v1/2 (the
    // in-tangent's term is zero): x = 0.25 + 1 = 1.25.
    close(
        pose(1, 1.0, other),
        (Mat3::IDENTITY, DVec3::new(1.25, 0.0, 0.0)),
    );
    assert_eq!(imported.stats.animation_channels_skipped, 0);
}
