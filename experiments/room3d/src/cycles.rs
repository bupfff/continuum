//! Independent references from Blender's Cycles path tracer, and OBJ export.
//!
//! `render` writes the scene as an OBJ (one object per mesh, Z up, per-vertex
//! normals) plus a JSON job, runs `cycles/render_reference.py` in a background
//! Blender (`BLENDER`, or the default macOS install), and reads back linear RGB.
//! `render_instanced` keeps instances instanced (geometry nodes), for scenes
//! too large to flatten. Cycles integrates each pixel's area (a one-pixel box
//! filter at least, in Blender 5.2), so compare it with supersampled renders.

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use continuum::glam::{DVec3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::{Camera, InstancedScene, RectLight};

/// Writes the surfaces as an OBJ: object `mesh_<index>` per surface, with
/// per-vertex normals; coordinates as they are (Z up, meters).
pub fn write_obj(surfaces: &[Surface], path: &Path) -> io::Result<()> {
    let mut out = String::from("# Exported from Continuum. Z up, meters.\n");
    let mut next = 1;
    for (m, s) in surfaces.iter().enumerate() {
        writeln!(out, "o mesh_{m}").unwrap();
        for t in &s.triangles {
            for p in t {
                writeln!(out, "v {} {} {}", p.x, p.y, p.z).unwrap();
            }
        }
        for n in &s.normals {
            for v in n {
                writeln!(out, "vn {} {} {}", v.x, v.y, v.z).unwrap();
            }
        }
        for _ in &s.triangles {
            writeln!(
                out,
                "f {0}//{0} {1}//{1} {2}//{2}",
                next,
                next + 1,
                next + 2
            )
            .unwrap();
            next += 3;
        }
    }
    std::fs::write(path, out)
}

/// A Cycles reference render.
pub struct Reference {
    pub pixels: Vec<[f32; 3]>,
    /// Cycles' own render time.
    pub seconds: f64,
    pub device: String,
}

/// Renders the scene with Cycles: Lambertian albedos, the one-sided light,
/// direct light plus up to `bounces` diffuse bounces, pixel-center sampling.
/// Files go to `work`.
#[allow(clippy::too_many_arguments)]
pub fn render(
    surfaces: &[Surface],
    albedo: &[Vec3],
    light: RectLight,
    radiance: f32,
    camera: Camera,
    size: (u32, u32),
    samples: u32,
    bounces: u32,
    seed: u32,
    work: &Path,
) -> io::Result<Reference> {
    render_with_sky(
        surfaces,
        albedo,
        light,
        radiance,
        Vec3::ZERO,
        camera,
        size,
        samples,
        bounces,
        seed,
        work,
    )
}

/// [`render`] with a uniform sky: the world's color, `sky` (linear RGB), seen
/// by camera rays that miss and lighting every surface.
#[allow(clippy::too_many_arguments)]
pub fn render_with_sky(
    surfaces: &[Surface],
    albedo: &[Vec3],
    light: RectLight,
    radiance: f32,
    sky: Vec3,
    camera: Camera,
    (width, height): (u32, u32),
    samples: u32,
    bounces: u32,
    seed: u32,
    work: &Path,
) -> io::Result<Reference> {
    std::fs::create_dir_all(work)?;
    let obj = work.join("scene.obj");
    write_obj(surfaces, &obj)?;
    let tag = format!(
        "{width}x{height}_{samples}spp_{bounces}b_seed{seed}_r{radiance}_sky{}-{}-{}",
        sky.x, sky.y, sky.z
    );
    let (exr, out) = (
        work.join(format!("{tag}.exr")),
        work.join(format!("{tag}.f32")),
    );
    let v = |p: Vec3| format!("[{}, {}, {}]", p.x, p.y, p.z);
    let albedo: Vec<String> = albedo.iter().map(|&a| v(a)).collect();
    let job = format!(
        r#"{{"obj": {:?}, "exr": {:?}, "out": {:?}, "albedo": [{}],
  "light": {{"corner": {}, "edge_u": {}, "edge_v": {}, "radiance": {radiance}}},
  "sky": {},
  "camera": {{"eye": {}, "target": {}, "fov": {}}},
  "width": {width}, "height": {height}, "samples": {samples}, "bounces": {bounces}, "seed": {seed}}}"#,
        obj.to_string_lossy(),
        exr.to_string_lossy(),
        out.to_string_lossy(),
        albedo.join(", "),
        v(light.corner),
        v(light.edge_u),
        v(light.edge_v),
        v(sky),
        v(camera.eye),
        v(camera.target),
        camera.vertical_fov_degrees
    );
    let job_path = work.join(format!("{tag}.json"));
    std::fs::write(&job_path, job)?;
    run_blender("cycles/render_reference.py", &job_path, &out)
}

/// [`render_with_sky`] for an instanced scene, with Cycles instancing:
/// each geometry written once (object space), each instance a point carrying
/// its rotation, scale and albedo, instanced through geometry nodes
/// (`cycles/render_instanced_reference.py`). Positions are relative to
/// `origin`, as are the light and the camera.
#[allow(clippy::too_many_arguments)]
pub fn render_instanced(
    scene: &InstancedScene,
    origin: DVec3,
    light: RectLight,
    radiance: f32,
    sky: Vec3,
    camera: Camera,
    (width, height): (u32, u32),
    samples: u32,
    bounces: u32,
    seed: u32,
    work: &Path,
) -> io::Result<Reference> {
    std::fs::create_dir_all(work)?;
    let mut geometries = Vec::new();
    for (g, surface) in scene.geometries.iter().enumerate() {
        let path = work.join(format!("geometry_{g}.obj"));
        if !path.exists() {
            write_obj(std::slice::from_ref(surface), &path)?;
        }
        geometries.push(format!("{:?}", path.to_string_lossy()));
    }
    let instances = work.join("instances.f32");
    let mut raw: Vec<f32> = Vec::with_capacity(scene.instances.len() * 17);
    for i in &scene.instances {
        raw.push(i.geometry as f32);
        // Row by row.
        let r = i.rotation.transpose();
        raw.extend(r.to_cols_array());
        raw.push(i.scale);
        raw.extend((i.position - origin).as_vec3().to_array());
        raw.extend(i.albedo.to_array());
    }
    let bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&instances, bytes)?;
    let tag = format!(
        "{width}x{height}_{samples}spp_{bounces}b_seed{seed}_r{radiance}_sky{}-{}-{}",
        sky.x, sky.y, sky.z
    );
    let (exr, out) = (
        work.join(format!("{tag}.exr")),
        work.join(format!("{tag}.f32")),
    );
    let v = |p: Vec3| format!("[{}, {}, {}]", p.x, p.y, p.z);
    let job = format!(
        r#"{{"geometries": [{}], "instances": {:?}, "exr": {:?}, "out": {:?},
  "light": {{"corner": {}, "edge_u": {}, "edge_v": {}, "radiance": {radiance}}},
  "sky": {},
  "camera": {{"eye": {}, "target": {}, "fov": {}}},
  "width": {width}, "height": {height}, "samples": {samples}, "bounces": {bounces}, "seed": {seed}}}"#,
        geometries.join(", "),
        instances.to_string_lossy(),
        exr.to_string_lossy(),
        out.to_string_lossy(),
        v(light.corner),
        v(light.edge_u),
        v(light.edge_v),
        v(sky),
        v(camera.eye),
        v(camera.target),
        camera.vertical_fov_degrees
    );
    let job_path = work.join(format!("{tag}.json"));
    std::fs::write(&job_path, job)?;
    run_blender("cycles/render_instanced_reference.py", &job_path, &out)
}

/// Runs a reference script in a background Blender and reads its output.
fn run_blender(script: &str, job_path: &Path, out: &Path) -> io::Result<Reference> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join(script);
    let blender = std::env::var_os("BLENDER")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/Applications/Blender.app/Contents/MacOS/Blender".into());
    let result = Command::new(blender)
        .args(["-b", "--factory-startup", "-noaudio", "--python"])
        .arg(&script)
        .arg("--")
        .arg(job_path)
        .output()?;
    let log = String::from_utf8_lossy(&result.stdout);
    let Some(line) = log.lines().find(|l| l.starts_with("CONTINUUM_REFERENCE")) else {
        return Err(io::Error::other(format!(
            "Blender failed:\n{}\n{}",
            log,
            String::from_utf8_lossy(&result.stderr)
        )));
    };
    let field = |name: &str| {
        line.split_whitespace()
            .find_map(|f| f.strip_prefix(&format!("{name}=")))
            .unwrap_or("")
            .to_string()
    };
    let raw = std::fs::read(out)?;
    let values: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(Reference {
        pixels: values.as_chunks::<3>().0.to_vec(),
        seconds: field("seconds").parse().unwrap_or(f64::NAN),
        device: field("device"),
    })
}

pub fn relative_rmse(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let (mut err, mut sum) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        err += ((x - y) as f64).powi(2);
        sum += *y as f64;
    }
    let n = (3 * b.len()) as f64;
    (err / n).sqrt() / (sum / n)
}

/// Seconds as a `Duration`, for reporting.
pub fn seconds(s: f64) -> Duration {
    Duration::from_secs_f64(s.max(0.0))
}
