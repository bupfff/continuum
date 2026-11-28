//! Development (not a gate): where a chart's single top-mip texel lies. For
//! each of the field's boulder geometries (or `CC_GLTF=file`'s), each
//! chart's center (0.5, 0.5) placed as `vc_surface` places a texel (the
//! chart cell's triangle that most nearly contains it, the point clamped
//! into it): the share of centers outside every triangle, and where the
//! placed points sit (height in the object's box, normal's z).
//!
//! `cargo run --release -p room3d --bin chart_centres`

use continuum::glam::{DVec3, Vec2, Vec3};
use room3d::field::Field;

fn bary(uv: [Vec2; 3], p: Vec2) -> Vec3 {
    let e1 = uv[1] - uv[0];
    let e2 = uv[2] - uv[0];
    let d = p - uv[0];
    let det = e1.x * e2.y - e1.y * e2.x;
    if det.abs() < 1e-30 {
        return Vec3::splat(-1.0);
    }
    let s = (d.x * e2.y - d.y * e2.x) / det;
    let u = (e1.x * d.y - e1.y * d.x) / det;
    Vec3::new(1.0 - s - u, s, u)
}

fn main() {
    let field = Field::new(10_000, DVec3::ZERO);
    let scene = &field.scene;
    for (g, s) in scene.geometries.iter().enumerate() {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in s.triangles.iter().flatten() {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let mut outside = 0;
        let mut rows = Vec::new();
        for c in 0..s.charts.len() {
            let mut best = (f32::MIN, usize::MAX, Vec3::ZERO);
            for (t, &k) in s.chart.iter().enumerate() {
                if k as usize != c {
                    continue;
                }
                let b = bary(s.uv[t], Vec2::splat(0.5));
                let score = b.x.min(b.y).min(b.z);
                if score > best.0 {
                    best = (score, t, b);
                }
            }
            if best.1 == usize::MAX {
                continue;
            }
            if best.0 < 0.0 {
                outside += 1;
            }
            let b = best.2.clamp(Vec3::ZERO, Vec3::ONE);
            let b = b / (b.x + b.y + b.z).max(1e-20);
            let t = s.triangles[best.1];
            let p = t[0] * b.x + t[1] * b.y + t[2] * b.z;
            let n = s.normals[best.1][0] * b.x
                + s.normals[best.1][1] * b.y
                + s.normals[best.1][2] * b.z;
            let tris = s.chart.iter().filter(|&&k| k as usize == c).count();
            rows.push((
                (p.z - lo.z) / (hi.z - lo.z).max(1e-6),
                n.normalize_or_zero().z,
                best.0,
                tris,
                s.charts[c],
            ));
        }
        if s.triangles.len() < 100 {
            println!(
                "geometry {g}: {} triangles, {} charts (ground or small)",
                s.triangles.len(),
                s.charts.len()
            );
            continue;
        }
        let low = rows.iter().filter(|r| r.0 < 0.25).count();
        let down = rows.iter().filter(|r| r.1 < 0.0).count();
        println!(
            "geometry {g}: {} triangles, {} charts; centers outside every triangle {}; placed in the lowest quarter {}, facing down {}",
            s.triangles.len(),
            s.charts.len(),
            outside,
            low,
            down
        );
        for r in rows.iter().take(12) {
            println!(
                "  height {:.2}, normal z {:+.2}, score {:+.3}, {} triangles, size {:.2?}",
                r.0, r.1, r.2, r.3, r.4
            );
        }
    }
}
