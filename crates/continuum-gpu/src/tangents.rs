//! MikkTSpace tangents for normal maps (M10 gate 13), and their code in the
//! normals buffer's fourth component.

use bevy_mikktspace::{Geometry, TangentSpace, generate_tangents};
use continuum::mesh::Surface;

/// Per triangle corner, `surface`'s MikkTSpace tangent (xyz) and handedness
/// (w, ±1), on its texture coordinates with v flipped to Blender's
/// convention (the Cycles export writes them so), so that the bitangent,
/// w × normal × tangent, points toward the image's top, as glTF's normal
/// maps' green does. None if MikkTSpace fails; a corner it leaves out is
/// zero.
pub(crate) fn mikktspace(surface: &Surface) -> Option<Vec<[[f32; 4]; 3]>> {
    struct Mesh<'a> {
        surface: &'a Surface,
        out: Vec<[[f32; 4]; 3]>,
    }
    impl Geometry for Mesh<'_> {
        fn num_faces(&self) -> usize {
            self.surface.triangles.len()
        }
        fn num_vertices_of_face(&self, _face: usize) -> usize {
            3
        }
        fn position(&self, face: usize, vert: usize) -> [f32; 3] {
            self.surface.triangles[face][vert].to_array()
        }
        fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
            self.surface.normals[face][vert].to_array()
        }
        fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
            let t = self.surface.tex[face][vert];
            [t.x, 1.0 - t.y]
        }
        fn set_tangent(&mut self, tangent: Option<TangentSpace>, face: usize, vert: usize) {
            if let Some(t) = tangent {
                self.out[face][vert] = t.tangent_encoded();
            }
        }
    }
    let mut mesh = Mesh {
        surface,
        out: vec![[[0.0; 4]; 3]; surface.triangles.len()],
    };
    generate_tangents(&mut mesh).ok()?;
    Some(mesh.out)
}

/// A tangent and handedness as one f32 holding an exact integer: the unit
/// tangent's octahedral coordinates at 11 bits each, plus one, its sign the
/// handedness (0: none). Integers below 2^24 are exact in f32, so no bit
/// patterns pass through the shader's float loads.
pub(crate) fn encode(t: [f32; 4]) -> f32 {
    let (x, y, z) = (t[0], t[1], t[2]);
    let l1 = x.abs() + y.abs() + z.abs();
    if l1 < 1e-12 || t[3] == 0.0 {
        return 0.0;
    }
    let (mut u, mut v) = (x / l1, y / l1);
    if z < 0.0 {
        let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
        (u, v) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
    }
    let q = |a: f32| ((a.clamp(-1.0, 1.0) + 1.0) * 0.5 * 2047.0).round();
    let code = q(u) * 2048.0 + q(v) + 1.0;
    if t[3] < 0.0 { -code } else { code }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum::glam::Vec3;

    /// The shader's decoding (`decode_tangent`), for the test.
    fn decode(code: f32) -> [f32; 4] {
        let a = code.abs() - 1.0;
        let qu = (a / 2048.0).floor();
        let qv = a - qu * 2048.0;
        let (u, v) = (qu / 2047.0 * 2.0 - 1.0, qv / 2047.0 * 2.0 - 1.0);
        let mut d = Vec3::new(u, v, 1.0 - u.abs() - v.abs());
        if d.z < 0.0 {
            let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
            d = Vec3::new(
                (1.0 - d.y.abs()) * sign(d.x),
                (1.0 - d.x.abs()) * sign(d.y),
                d.z,
            );
        }
        let d = d.normalize();
        [d.x, d.y, d.z, code.signum()]
    }

    #[test]
    fn codes_round_trip_within_0_15_degrees() {
        let mut worst = 1.0f32;
        for i in 0..2000 {
            let a = i as f32 * 0.7853;
            let b = (i as f32 * 0.3121).sin();
            let t = Vec3::new(
                a.cos() * (1.0 - b * b).sqrt(),
                a.sin() * (1.0 - b * b).sqrt(),
                b,
            );
            let w = if i % 2 == 0 { 1.0 } else { -1.0 };
            let code = encode([t.x, t.y, t.z, w]);
            assert!(code != 0.0 && code.abs() < (1 << 24) as f32 && code.fract() == 0.0);
            let d = decode(code);
            assert_eq!(d[3], w);
            worst = worst.min(t.dot(Vec3::new(d[0], d[1], d[2])));
        }
        assert!(worst > 0.15f32.to_radians().cos(), "worst cosine {worst}");
    }
}
