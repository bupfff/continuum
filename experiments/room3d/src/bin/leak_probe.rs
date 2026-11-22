//! Development (light leaks): for triangles inside a box (`LP_BOX`,
//! `x0,y0,z0,x1,y1,z1` in world coordinates) facing the sun, casts a CPU ray
//! from each centroid toward the sun through every triangle of a glTF scene
//! and counts how many reach it, how many are blocked only by triangles seen
//! from behind (which the mask cones skip on closed meshes), and which
//! geometries block. The sun is `gltf_dev`'s. `LP_ONLY=a,b,…`: only those
//! geometries' instances; `LP_NOSELF=1`: a triangle's own geometry does not
//! block it; `LP_GPU=1`: the same receivers through the mask cones
//! (`LP_SCALE=k`: everything k times larger). Not a gate.
//!
//! `cargo run --release -p room3d --bin leak_probe -- Sponza.gltf`

use std::path::Path;

use continuum::glam::{DVec3, Vec3};

fn main() {
    let path = std::env::args().nth(1).expect("a glTF file");
    let b: Vec<f32> = std::env::var("LP_BOX")
        .expect("LP_BOX")
        .split(',')
        .map(|x| x.trim().parse().expect("six numbers"))
        .collect();
    let (lo, hi) = (Vec3::new(b[0], b[1], b[2]), Vec3::new(b[3], b[4], b[5]));
    let sun = DVec3::new(-0.6, -0.3, 0.75).normalize().as_vec3();
    let imported = continuum_gltf::load(Path::new(&path)).expect("the file");
    let mut scene = imported.scene.clone();
    // `LP_ONLY=a,b,…`: only the instances of these geometries.
    if let Ok(v) = std::env::var("LP_ONLY") {
        let keep: Vec<usize> = v
            .split(',')
            .map(|x| x.trim().parse().expect("indices"))
            .collect();
        scene.instances.retain(|i| keep.contains(&i.geometry));
    }
    // Every triangle in world space, with its geometry.
    let mut world: Vec<([Vec3; 3], usize)> = Vec::new();
    for inst in &scene.instances {
        let p = inst.position.as_vec3();
        for t in &scene.geometries[inst.geometry].triangles {
            world.push((
                t.map(|v| inst.rotation * (v * inst.scale) + p),
                inst.geometry,
            ));
        }
    }
    println!(
        "{} triangles in {} geometries",
        world.len(),
        scene.geometries.len()
    );
    // `LP_LIST=1`: the instances, and the geometries whose bounds meet the box.
    if std::env::var_os("LP_LIST").is_some() {
        for inst in &scene.instances {
            println!(
                "instance of {}: scale {}, position {:?}",
                inst.geometry, inst.scale, inst.position
            );
        }
        let mut bounds =
            vec![(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN), 0usize); scene.geometries.len()];
        for (t, g) in &world {
            for v in t {
                bounds[*g].0 = bounds[*g].0.min(*v);
                bounds[*g].1 = bounds[*g].1.max(*v);
            }
            bounds[*g].2 += 1;
        }
        for (g, (a, b, k)) in bounds.iter().enumerate() {
            if a.cmplt(hi).all() && b.cmpgt(lo).all() {
                println!("geometry {g}: {k} triangles, {a:.2} to {b:.2}");
            }
        }
    }
    let open: Vec<bool> = scene
        .geometries
        .iter()
        .map(|g| is_open(&g.triangles))
        .collect();
    // `LP_NOSELF=1`: a triangle's own geometry does not block it.
    let no_self = std::env::var_os("LP_NOSELF").is_some();
    let mut points: Vec<(Vec3, Vec3)> = Vec::new();
    let mut receivers_geometry = std::collections::BTreeSet::<usize>::new();
    let (mut tested, mut reach, mut back_only) = (0usize, 0usize, 0usize);
    let mut blockers = std::collections::BTreeMap::<usize, usize>::new();
    let mut back_blockers = std::collections::BTreeMap::<usize, usize>::new();
    for (t, g) in &world {
        let c = (t[0] + t[1] + t[2]) / 3.0;
        if c.cmplt(lo).any() || c.cmpgt(hi).any() {
            continue;
        }
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        if n.length() < 1e-12 || n.dot(sun) <= 0.0 {
            continue;
        }
        let n = n.normalize();
        let o = c + n * 1e-3;
        tested += 1;
        points.push((o, n));
        receivers_geometry.insert(*g);
        let (mut front, mut back) = (None::<(f32, usize)>, None::<(f32, usize)>);
        for (u, h) in &world {
            if no_self && h == g {
                continue;
            }
            if let Some(d) = hit(o, sun, u) {
                let un = (u[1] - u[0]).cross(u[2] - u[0]);
                // Front facing the apex: the ray goes into it.
                let slot = if un.dot(sun) < 0.0 {
                    &mut front
                } else {
                    &mut back
                };
                if slot.is_none_or(|(e, _)| d < e) {
                    *slot = Some((d, *h));
                }
            }
        }
        match (front, back) {
            (None, None) => reach += 1,
            (None, Some((_, h))) => {
                back_only += 1;
                *back_blockers.entry(h).or_insert(0) += 1;
            }
            (Some((_, h)), _) => *blockers.entry(h).or_insert(0) += 1,
        }
    }
    // `LP_GPU=1`: the same receivers through the mask cones (every
    // triangle, then through the hierarchy), relative to the scene's center
    // as a renderer places them.
    if std::env::var_os("LP_GPU").is_some() {
        let origin = ((imported.bounds.0 + imported.bounds.1) * 0.5).as_vec3();
        let radius = ((imported.bounds.1 - imported.bounds.0).length() * 0.5) as f64;
        let mut meshes: Vec<Vec<[Vec3; 3]>> = vec![Vec::new(); scene.geometries.len()];
        for (u, h) in &world {
            if !(no_self && receivers_geometry.contains(h)) {
                meshes[*h].push(u.map(|v| v - origin));
            }
        }
        meshes.retain(|m| !m.is_empty());
        let light = continuum_gpu::Light::sun(
            Vec3::ZERO,
            DVec3::new(-0.6, -0.3, 0.75),
            radius * 50.0,
            Vec3::ONE,
        )
        .as_rect()
        .expect("a rectangle");
        let receivers: Vec<(Vec3, Vec3)> = points.iter().map(|&(p, n)| (p - origin, n)).collect();
        // `LP_SCALE=k`: everything k times larger (an object's own units).
        let k: f32 = std::env::var("LP_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let meshes: Vec<Vec<[Vec3; 3]>> = meshes
            .iter()
            .map(|m| m.iter().map(|t| t.map(|v| v * k)).collect())
            .collect();
        let receivers: Vec<(Vec3, Vec3)> = receivers.iter().map(|&(p, n)| (p * k, n)).collect();
        let light = continuum_gpu::RectLight {
            corner: light.corner * k,
            edge_u: light.edge_u * k,
            edge_v: light.edge_v * k,
            ..light
        };
        let kernel = continuum_gpu::MaskConeKernel::new().expect("GPU");
        for h in [
            continuum_gpu::Hierarchy::None,
            continuum_gpu::Hierarchy::Triangles,
        ] {
            let (v, _) = kernel.irradiance_with(&receivers, light, &meshes, h, 1);
            let lit = v.iter().filter(|&&x| x > 1e-9).count();
            let worst = v.iter().cloned().fold(0.0f32, f32::max);
            println!(
                "GPU ({h:?}): {lit} of {} receivers lit, largest {worst:.3e}",
                v.len()
            );
            if h == continuum_gpu::Hierarchy::None {
                for (k, x) in v.iter().enumerate().filter(|(_, x)| **x > 1e-9).take(5) {
                    println!(
                        "  receiver {:?} normal {:?}: {x:.3e}",
                        receivers[k].0, receivers[k].1
                    );
                }
            }
        }
    }
    println!(
        "{tested} triangles facing the sun: {reach} reach it, {back_only} blocked only by back faces"
    );
    for (h, k) in &back_blockers {
        println!(
            "  back-face blocker geometry {h} ({} triangles, open {}): {k}",
            scene.geometries[*h].triangles.len(),
            open[*h]
        );
    }
    for (h, k) in &blockers {
        println!(
            "  front-face blocker geometry {h} ({} triangles, open {}): {k}",
            scene.geometries[*h].triangles.len(),
            open[*h]
        );
    }
}

fn hit(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> Option<f32> {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let s = o - t[0];
    let u = s.dot(p) / det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) / det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) / det;
    (t > 1e-5).then_some(t)
}

fn is_open(mesh: &[[Vec3; 3]]) -> bool {
    let mut edges = std::collections::HashMap::<([i64; 3], [i64; 3]), i32>::new();
    let key = |p: Vec3| (p * 1e4).round().as_i64vec3().to_array();
    for t in mesh {
        for e in 0..3 {
            let (a, b) = (key(t[e]), key(t[(e + 1) % 3]));
            if a == b {
                continue;
            }
            let (k, d) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
            *edges.entry(k).or_insert(0) += d;
        }
    }
    edges.values().any(|&v| v != 0)
}
