# Using Continuum as a library

Continuum v0.1 is experimental. This guide covers what works today, how to
use it, and what to expect. Measured results live in `docs/results/`, and the
design and its honest status in `docs/DESIGN.md` (§3.6).

## What it does

Continuum renders static scenes made of geometry placed many times (instances
at f64 positions, so worlds can be kilometers wide). It lights them with:

- rectangular area lights and suns;
- point lights with a size;
- spot lights;
- a uniform sky.

Shadows are exact soft shadows from each light's area. They come from *mask
cones*: from a surface point toward a light, the cone finds what covers the
light as exact coverage of 8×8 bins, with distant geometry prefiltered.

A view-dependent lighting cache keeps that work across frames on the
surfaces in view, coarse first. Pixels shade from it with their own normals.
No frame blends in earlier frames, so there is no ghosting and no noise.

Measured on an Apple M4 MacBook Air (fanless) at 1280×720:

| What | Result | Gate |
|---|---|---|
| A million instanced boulders (10^10 triangles), sun, flying | 14.6–15.0 ms per frame | M4 G11 |
| Three colored lights on that field, against Cycles | 1.22% relative RMSE | M5 G6 |
| A textured glTF scene against Cycles | 0.43% | M6 G1 |
| Point and spot lights against Cycles | 1.35% | M5 G8 |
| Re-coloring or dimming a light | free and exact (nothing is re-baked) | M5 G6 |

## Limits, today

- **Moving objects.** Instances listed in `RendererOptions::movable` move
  rigidly (`Renderer::set_transform`; they keep their scale). The cache
  re-bakes, within the frame, exactly what a move can change and never
  shows stale light (M9 gate 1). With the cache, movers and a sky do not
  mix yet.
- **Moving lights, experimental.** Lights listed in
  `RendererOptions::moving_lights` move (`Renderer::set_light`; not
  spots). The cache's ledger never shows stale light (M9 gate 2). But
  while a light moves, pages off screen are set aside and show as missing
  when they come back into view: a third of the pixels in a fast flight
  with a fast sun. Not yet with movers or a sky in the cache. Lights
  change color and intensity freely either way.
- **Diffuse materials only.** Base color and textures are used; metallic,
  roughness, normal maps, alpha and emission are not.
- **One bounce of light, optional** (`CacheOptions::indirect`, M10 gate
  1): 1.79% from Cycles with one bounce on a textured scene, at 1.07× the
  warm frame. At most three lights; not with movers or moving lights.
- **One ray per pixel for visibility,** so distant detail aliases.
  `RendererOptions::pixel_cones` (off by default) traces a cone through each
  pixel instead; it is experimental and far from real time on large scenes
  (M7 gate 1: 105 ms per frame on a small textured scene).
- **The sky converges slowly** in the cache: a view takes hundreds to
  thousands of frames to sharpen (M5 G5). It is opt-in.
- **Sustained speed on fanless laptops.** Flights with several lights hold
  60 Hz while the machine is cool. After minutes of load, the GPU slows and
  frames take 20 ms or more (M5 G8).
- **Measured on Apple Silicon (Metal) only.** CI checks correctness on Linux
  (Vulkan, the lavapipe software adapter), where every test passes, and
  that every shader compiles for Windows (DX12, with DXC). Windows' WARP
  software adapter cannot run the kernel (it runs out of memory building
  it). Real Windows GPUs and NVIDIA are untested.
- **On DX12, wgpu needs DXC** (`dxcompiler.dll` on the path and
  `WGPU_DX12_COMPILER=dxc`, or wgpu's `static-dxc` feature); CI uses it.
  The older FXC compiler is untested.
- **No exposure control.** `Renderer::read_colors` returns linear light as
  it is. Drawn to a window, colors go through a fixed Reinhard curve; a
  float target (such as Bevy's HDR) gets linear light for the engine's own
  tone mapping.

## Setup

Add the crates from this repository (they are not on crates.io):

```toml
[dependencies]
continuum = { path = "../Continuum/crates/continuum" }
continuum-gpu = { path = "../Continuum/crates/continuum-gpu" }
continuum-gltf = { path = "../Continuum/crates/continuum-gltf" }

# Recommended for large instanced scenes: wgpu-core with one fix for
# acceleration structures with many instances (third_party/README.md).
[patch.crates-io]
wgpu-core = { path = "../Continuum/third_party/wgpu-core" }
```

Without the patch everything works, but each frame with hardware rays spends
milliseconds of CPU time on scenes with very many instances (M4 G9).

## Rendering a glTF file

```rust
use continuum::glam::{DVec3, Vec3};
use continuum_gpu::{Camera, Light, MaskConeKernel, Renderer, RendererOptions, wgpu};

let imported = continuum_gltf::load(std::path::Path::new("scene.glb"))?;
// Render relative to a point near the camera (here the scene's center).
let (lo, hi) = imported.bounds;
let origin = (lo + hi) * 0.5;
// The file's lights (point and spot lights get a 5 cm radius), or a sun.
let mut lights = imported.lights(origin, 0.05);
if lights.is_empty() {
    let toward = DVec3::new(0.5, -0.5, 0.7);
    lights.push(Light::sun(Vec3::ZERO, toward, 1000.0, Vec3::splat(3.0)));
}

// A kernel on its own device; or `MaskConeKernel::with_device(device, queue,
// adapter_name)` on yours, created with `continuum_gpu::device_descriptor`.
let kernel = MaskConeKernel::new().expect("a GPU");
let format = wgpu::TextureFormat::Rgba8UnormSrgb;
let mut renderer = Renderer::new(
    kernel,
    format,
    (1280, 720),
    &imported.scene,
    origin,
    &lights,
    RendererOptions::default(),
);

// Each frame, into a texture view of that format and size:
let camera = Camera {
    eye: Vec3::new(0.0, -6.0, 1.7),
    target: Vec3::ZERO,
    vertical_fov_degrees: 50.0,
};
renderer.render(camera, &view);
```

A complete program that writes a PNG is
`crates/continuum-gltf/examples/render.rs`:

```bash
cargo run --release -p continuum-gltf --example render -- scene.glb out.png 1280 720
```

For a window, see `apps/viewer`, which flies through a file with
`--gltf scene.glb`.

## Concepts

- **`InstancedScene`.** It has four parts:
  - `geometries`: triangle meshes with normals and charts for the cache;
    `Surface::from_mesh` makes charts for any mesh.
  - `instances`: a geometry, rotation, uniform scale, f64 position and
    albedo.
  - `textures`.
  - `geometry_textures`: which texture each geometry samples.
- **The render origin.** Positions are uploaded relative to it, and lights
  and cameras are given relative to it. Keep it near the camera in large
  worlds.
- **Lights** have RGB radiance (linear).
  - `Light::rect(rect, radiance)`: a rectangle, one-sided or two-sided.
  - `Light::sun(center, toward, distance, irradiance)`: a square 0.5° across
    giving `irradiance` on surfaces facing it.
  - `Light::point(center, radius, intensity)`: radiant intensity per
    steradian.
  - `Light::spot(center, radius, intensity, Spot { axis, inner, outer })`:
    glTF's falloff between the half-angles.
  - At most `MAX_LIGHTS` (16).
- **Units.** Surfaces reflect `albedo × irradiance / π`. glTF lights come in
  at 683 lumens per watt, as Blender imports them.
- **The sky.** `RendererOptions::sky` is a uniform radiance over the whole
  sphere. The ground (any geometry) blocks it where present.
- **The cache.**
  - `RendererOptions::cache` (on by default) bakes what the view needs,
    within a frame-time budget, coarse first. Shadows are softer while
    flying and sharpen when the view settles. `Renderer::is_settled` tells
    when the view's lighting is complete.
  - Off, every pixel traces a cone per light every frame: exact, and slow at
    scale.
  - Tuning lives in `CacheOptions`.
- **Re-coloring.** `Renderer::set_light_radiance` changes a light's color
  and intensity from the next frame, exactly, with nothing re-baked.
- **Moving objects.** List the instances that may move in
  `RendererOptions::movable`, then call
  `Renderer::set_transform(instance, rotation, position)` before a frame.
  - Each frame something moves, the cache finds the texels whose light
    cones a move can reach (the move's old and new boxes against each
    cone's hull toward each light). It re-bakes those that were on screen
    in the same frame, and drops the rest until they are needed. The
    result equals a fresh cache, bit for bit (M9 gate 1).
  - The cost follows the change: 0.84% of the pages on screen per frame
    for 65 movers among a million boulders, 1.06–1.09× the frame time with
    them still.
  - Rigid motion only (the scale stays). With the cache, not yet with a
    sky.

## Requirements

- Rust 2024 edition, and wgpu 30 (pinned in the workspace).
- A GPU with compute shaders.
- Optional, where available:
  - subgroups (packet traversal);
  - timestamp queries (the bake budget follows GPU time);
  - ray queries (hardware camera rays).

## In Bevy

`crates/bevy_continuum` renders a Bevy 0.20 camera's view with Continuum,
on Bevy's own device (Bevy 0.20 uses wgpu 30, as Continuum does):

```rust
use bevy::prelude::*;
use bevy_continuum::{ContinuumCamera, ContinuumPlugin, ContinuumScene};

App::new()
    .add_plugins((DefaultPlugins, ContinuumPlugin))
    .insert_resource(ContinuumScene { scene, lights, origin, options })
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn((Camera3d::default(), bevy::camera::Hdr, ContinuumCamera));
    })
    .run();
```

- A point `(x, y, z)` in Bevy's Y-up world is `origin + (x, −z, y)` in
  Continuum's Z-up scene, the same change of axes as glTF import.
- The camera's image is Continuum's: Bevy's own meshes and lights do not
  appear in it. Bevy's tone mapping, post-processing and UI apply on top.
- Changing a light's radiance in `ContinuumScene::lights` costs nothing;
  any other change to the resource rebuilds the renderer.
- Run the example (a camera circling boxes and spheres, a pulsing lamp):

  ```bash
  cargo run --release -p bevy_continuum --example boxes
  ```

## API reference

```bash
cargo doc --no-deps -p continuum -p continuum-gpu -p continuum-gltf --open
```

Items used only by the experiments in `experiments/` (the instanced
renderer underneath `Renderer`, the kernel's tuning switches, diagnostics)
are public so the experiments compile, but hidden from the documentation.
They may change without notice.

## Where things are

- `crates/continuum-gpu/src/renderer.rs`: `Renderer`.
- `crates/continuum-gpu/src/mask_cones.wgsl`: the kernel.
- `crates/continuum-gpu/src/vcache.rs`: the view-dependent cache.
- `crates/continuum-gltf`: glTF import and export.
- `apps/viewer`: the real-time viewer.
- `docs/results/`: what was measured, and how to reproduce it.
