//! A small scene built in code, rendered by Continuum through a Bevy
//! camera that circles it: boxes and spheres on a checkered ground, a sun, a
//! warm point light that pulses (re-coloring a light bakes nothing), and a
//! sky.
//!
//! `cargo run --release -p bevy_continuum --example boxes`
//!
//! `-- --screenshot out.png --frames 300` saves the window after that many
//! frames and exits. `-- --spin` spins the first box from a Bevy entity
//! (`ContinuumInstance`), without the sky (movers and a sky are slow
//! together in the cache).

use std::sync::Arc;

use bevy::camera::Hdr;
use bevy::prelude::*;
use bevy::render::view::Tonemapping;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy_continuum::continuum_gpu::{Instance, InstancedScene, Light, RendererOptions, Texture};
use bevy_continuum::{ContinuumCamera, ContinuumInstance, ContinuumPlugin, ContinuumScene};
use continuum::glam::{DVec3, Mat3, Vec2, Vec3 as ZUp};
use continuum::mesh::Surface;

fn checker() -> Texture {
    let rgba = (0..64)
        .flat_map(|k| {
            if (k % 8 + k / 8) % 2 == 0 {
                [225, 215, 195, 255]
            } else {
                [95, 110, 130, 255]
            }
        })
        .collect();
    Texture {
        width: 8,
        height: 8,
        rgba,
    }
}

/// The scene in Continuum's Z-up world.
/// The box `--spin` turns: instance 1, at (−2.1, −2.1, 0.35) in the scene's
/// Z-up world, which is (−2.1, 0.35, 2.1) in Bevy's.
const SPUN: usize = 1;

fn scene() -> ContinuumScene {
    let h = 4.0;
    let c = [(-h, -h), (h, -h), (h, h), (-h, h)].map(|(x, y)| ZUp::new(x, y, 0.0));
    let uv = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)].map(|(u, v)| Vec2::new(u, v));
    let ground = Surface::from_mesh(
        vec![[c[0], c[1], c[2]], [c[0], c[2], c[3]]],
        None,
        Some(vec![[uv[0], uv[1], uv[2]], [uv[0], uv[2], uv[3]]]),
    );
    let cube = Surface::cuboid(ZUp::splat(-0.3), ZUp::splat(0.3));
    let sphere = Surface::parametric(32, 16, |u, v| {
        let (theta, phi) = (u * std::f32::consts::TAU, (v - 0.5) * std::f32::consts::PI);
        ZUp::new(theta.cos() * phi.cos(), theta.sin() * phi.cos(), phi.sin()) * 0.35
    });
    let mut instances = vec![Instance {
        geometry: 0,
        rotation: Mat3::IDENTITY,
        scale: 1.0,
        position: DVec3::ZERO,
        albedo: ZUp::splat(0.8),
    }];
    for i in 0..4 {
        for j in 0..4 {
            let k = i * 4 + j;
            let is_cube = k % 2 == 0;
            instances.push(Instance {
                geometry: if is_cube { 1 } else { 2 },
                rotation: Mat3::from_rotation_z(0.5 * k as f32),
                scale: 1.0,
                position: DVec3::new(-2.1 + 1.4 * i as f64, -2.1 + 1.4 * j as f64, 0.35),
                albedo: ZUp::new(
                    0.35 + 0.15 * (k % 4) as f32,
                    0.55,
                    0.75 - 0.1 * (k % 3) as f32,
                ),
            });
        }
    }
    ContinuumScene {
        scene: Arc::new(InstancedScene {
            geometries: vec![ground, cube, sphere],
            instances,
            textures: vec![checker()],
            geometry_textures: vec![Some(0), None, None],
            geometry_materials: Vec::new(),
            geometry_emission: Vec::new(),
            geometry_emissive_textures: Vec::new(),
            geometry_blend: Vec::new(),
            geometry_normal_maps: Vec::new(),
            geometry_metallic_roughness: Vec::new(),
            geometry_alpha_cutoff: Vec::new(),
            geometry_colors: Vec::new(),
            texture_wrap: Vec::new(),
            geometry_levels: Vec::new(),
            level_source: None,
        }),
        lights: vec![
            Light::sun(
                ZUp::ZERO,
                DVec3::new(0.6, -0.3, 0.6),
                300.0,
                ZUp::new(2.6, 2.4, 2.1),
            ),
            Light::point(ZUp::new(0.0, 0.0, 1.6), 0.1, ZUp::new(1.0, 0.6, 0.3) * 2.0),
        ],
        origin: DVec3::ZERO,
        options: RendererOptions {
            sky: Some(ZUp::new(0.15, 0.2, 0.3)),
            ..Default::default()
        },
    }
}

#[derive(Resource)]
struct Run {
    screenshot: Option<String>,
    frames: u32,
    frame: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spin = args.iter().any(|a| a == "--spin");
    let mut scene = scene();
    if spin {
        scene.options.movable = vec![SPUN];
        scene.options.sky = None;
    }
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    App::new()
        .add_plugins((DefaultPlugins, ContinuumPlugin))
        .insert_resource(scene)
        .insert_resource(Run {
            screenshot: arg("--screenshot"),
            frames: arg("--frames").and_then(|n| n.parse().ok()).unwrap_or(300),
            frame: 0,
        })
        .add_systems(Startup, move |mut commands: Commands| {
            commands.spawn((
                Camera3d::default(),
                Hdr,
                // Bevy's default tone mapper needs its `tonemapping_luts` feature.
                Tonemapping::AcesFitted,
                ContinuumCamera,
                Transform::default(),
            ));
            if spin {
                commands.spawn((
                    ContinuumInstance(SPUN),
                    Transform::from_xyz(-2.1, 0.35, 2.1),
                ));
            }
        })
        .add_systems(Update, (circle, pulse, turn, finish))
        .run();
}

/// The camera circles the scene (Bevy's Y-up world).
fn circle(time: Res<Time>, mut cameras: Query<&mut Transform, With<ContinuumCamera>>) {
    let a = 0.15 * time.elapsed_secs();
    for mut t in &mut cameras {
        *t = Transform::from_xyz(6.0 * a.cos(), 3.0, 6.0 * a.sin()).looking_at(Vec3::ZERO, Vec3::Y);
    }
}

/// The spun box turns about Bevy's up; the cache re-bakes what it changes.
fn turn(time: Res<Time>, mut boxes: Query<&mut Transform, With<ContinuumInstance>>) {
    for mut t in &mut boxes {
        t.rotation = Quat::from_rotation_y(time.elapsed_secs());
    }
}

/// The point light's intensity pulses: re-weighting the cache, no bake.
fn pulse(time: Res<Time>, mut scene: ResMut<ContinuumScene>) {
    let s = 1.0 + 0.8 * (2.0 * time.elapsed_secs()).sin();
    scene.lights[1].radiance =
        Light::point(ZUp::ZERO, 0.1, ZUp::new(1.0, 0.6, 0.3) * 2.0 * s).radiance;
}

fn finish(mut run: ResMut<Run>, mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    let Some(path) = run.screenshot.clone() else {
        return;
    };
    run.frame += 1;
    if run.frame == run.frames {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    if run.frame == run.frames + 10 {
        exit.write(AppExit::Success);
    }
}
