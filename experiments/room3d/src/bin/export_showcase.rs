//! Exports the showcase for other renderers (for example a UE5 side-by-side):
//! `out/showcase/showcase.obj` (one object per mesh, Z up, meters, per-vertex
//! normals; movers at frame 0) and `out/showcase/showcase.json` (materials,
//! light, camera path, movers).

use std::fmt::Write as _;
use std::path::Path;

use continuum::glam::Vec3;
use room3d::cycles::write_obj;
use room3d::showcase::{PATH_FRAMES, Showcase, movers};

fn main() -> std::io::Result<()> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/showcase");
    std::fs::create_dir_all(&dir)?;
    let scene = Showcase::new();
    write_obj(&scene.surfaces, &dir.join("showcase.obj"))?;
    let v = |p: Vec3| format!("[{:.4}, {:.4}, {:.4}]", p.x, p.y, p.z);
    let mut json = String::from("{\n  \"units\": \"meters, Z up, right-handed\",\n");
    let materials: Vec<String> = scene
        .albedo
        .iter()
        .enumerate()
        .map(|(m, &a)| {
            format!(
                "    {{\"object\": \"mesh_{m}\", \"albedo_linear\": {}, \"moving\": {}}}",
                v(a),
                m >= scene.first_mover
            )
        })
        .collect();
    writeln!(json, "  \"materials\": [\n{}\n  ],", materials.join(",\n")).unwrap();
    let l = scene.light;
    writeln!(
        json,
        "  \"light\": {{\"type\": \"one-sided rectangle, emits toward edge_u x edge_v\", \"corner\": {}, \"edge_u\": {}, \"edge_v\": {}, \"radiance\": {}}},",
        v(l.corner),
        v(l.edge_u),
        v(l.edge_v),
        scene.radiance
    )
    .unwrap();
    let keys: Vec<String> = (0..=PATH_FRAMES)
        .step_by(30)
        .map(|f| {
            let c = scene.camera(f as f32);
            format!(
                "    {{\"frame\": {f}, \"eye\": {}, \"target\": {}}}",
                v(c.eye),
                v(c.target)
            )
        })
        .collect();
    writeln!(
        json,
        "  \"camera\": {{\"vertical_fov_degrees\": {}, \"frames_per_second\": 60, \"path\": [\n{}\n  ]}},",
        scene.vertical_fov_degrees,
        keys.join(",\n")
    )
    .unwrap();
    let samples: Vec<String> = (0..=PATH_FRAMES)
        .step_by(30)
        .map(|f| {
            let at = movers(f as f32);
            let centers: Vec<String> = at
                .iter()
                .map(|s| {
                    let (lo, hi) = s.bounds();
                    v((lo + hi) * 0.5)
                })
                .collect();
            format!(
                "    {{\"frame\": {f}, \"centers\": [{}]}}",
                centers.join(", ")
            )
        })
        .collect();
    writeln!(
        json,
        "  \"movers\": {{\"note\": \"ball (radius 0.16) bounces; cube (0.3) spins about z by 90 degrees/s while orbiting\", \"samples\": [\n{}\n  ]}}\n}}",
        samples.join(",\n")
    )
    .unwrap();
    std::fs::write(dir.join("showcase.json"), json)?;
    println!("Wrote {}/showcase.obj and showcase.json.", dir.display());
    Ok(())
}
