//! Continuum viewer: fly through a scene in real time.
//!
//! A panel (H hides it) picks the scene: the showcase (a gallery), a field
//! of boulders (10,000 to a million: about ten billion triangles at the
//! most), or a glTF file (Open…, or drop a `.glb` / `.gltf` on the window).
//! It shows the frame rate, the GPU and its memory, and whether the
//! lighting has settled, and changes how instanced scenes (the field, glTF
//! files) render: bounce light, anti-aliasing, the sky, the render scale
//! (rebuilding the renderer), the tone map, bloom and exposure (at once),
//! and vsync. Problems show in the panel, or in a dialog when they stop the
//! viewer (launched from the Finder there is no terminal).
//!
//! The field and glTF scenes go through the library's front door,
//! `continuum_gpu::Renderer`; the showcase and the room through the
//! lightmap path (`LiveRenderer`), which bakes direct light into a
//! world-space cache at startup and moves objects through it.
//!
//! Command line, all optional: a `.glb`/`.gltf` path (or `--gltf file`) or
//! `--scene showcase|room|field` (`--boulders N`); `--no-bounce`, `--no-aa`,
//! `--sky`, `--scale N` (render at 1/N of the window), `--vsync`,
//! `--exposure K`, `--no-cache` (a mask cone per pixel every frame),
//! `--no-panel`; for glTF files `--no-ground` and `--emitters-as-lights`
//! (emissive rectangles as area lights); `--env sky.hdr` (or a `.png`, or
//! `gradient`; `--env-strength K`) lights the field or a glTF scene with an
//! environment image. A file's first animation plays (P pauses it; not with
//! a sky, as movers and a sky do not mix in the cache yet).
//!
//! Controls: WASD move, Q/E down/up, drag with the left mouse button to look,
//! Shift to move faster, C to play the camera path (showcase), P to pause
//! moving objects, V for vsync, `[` and `]` for exposure a stop at a time,
//! H for the panel, Esc to quit. `--frames N` renders N frames, prints
//! timings and exits (`--screenshot out.png` saves the last one;
//! `VIEWER_SELFTEST=1` scripts a scene load, a rebuild and a failed load).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::wgpu;
use continuum_gpu::{
    Camera, Environment, Instance, InstancedScene, Light, Lightmap, LiveRenderer, MaskConeKernel,
    RectLight, Renderer, RendererOptions,
};
use room3d::field::Field;
use room3d::showcase::{self, PATH_FRAMES, Showcase};
use room3d::{movers, render_scene};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// A scene to view: its surfaces (static first, then moving), light, cache
/// resolution and how its objects move.
struct World {
    name: &'static str,
    surfaces: Vec<Surface>,
    albedo: Vec<Vec3>,
    light: RectLight,
    radiance: f32,
    first_mover: usize,
    texel: f32,
    probes: f32,
    fov: f32,
    /// Walking speed in meters per second.
    speed: f32,
    start: (Vec3, Vec3),
    showcase: Option<Showcase>,
    /// An instanced scene (the boulder field or a glTF file), rendered
    /// through the library's `Renderer`.
    instanced: Option<Instanced>,
    /// Window pixels per rendered pixel (instanced scenes only).
    scale: u32,
    /// Light instanced scenes through the view-dependent lighting cache
    /// (otherwise a mask cone per pixel every frame).
    cache: bool,
    /// Add the scene's sky (instanced scenes only).
    sky: bool,
    /// A sky that varies with direction, in place of the uniform one
    /// (instanced scenes only).
    environment: Option<Environment>,
    /// A glTF file's first animation, played through movable instances.
    animations: Option<continuum_gltf::Animations>,
    /// The display's exposure (instanced scenes): `--exposure`, `[` and `]`.
    exposure: f32,
}

/// An instanced scene with its lights and sky, positioned relative to
/// `origin`.
struct Instanced {
    scene: InstancedScene,
    lights: Vec<Light>,
    sky: Vec3,
    origin: DVec3,
    /// What the scene holds, for the startup line.
    summary: String,
}

impl World {
    /// The instances the first animation moves, if it plays: not with a sky
    /// (movers and a sky do not mix in the cache yet).
    fn animated_instances(&self) -> Vec<usize> {
        match &self.animations {
            Some(a) if !(self.cache && (self.sky || self.environment.is_some())) => a.instances(0),
            _ => Vec::new(),
        }
    }

    /// Skinned and morphing geometry (M12 gate 3), posed from every
    /// animation together (files often animate a rig and a mesh's morph
    /// weights apart): not with a sky, as movers.
    fn deforming_geometries(&self) -> Vec<usize> {
        match &self.animations {
            Some(a) if !(self.cache && (self.sky || self.environment.is_some())) => a.deforming(),
            _ => Vec::new(),
        }
    }

    /// The deforming geometries' poses at `time` (60 Hz frames), looping
    /// over the longest animation.
    fn deforming_poses(&self, time: f32) -> Vec<continuum_gltf::Pose> {
        match &self.animations {
            Some(a) if !self.deforming_geometries().is_empty() => {
                let all: Vec<usize> = (0..a.list.len()).collect();
                let duration = a.list.iter().map(|x| x.duration).fold(1e-3, f32::max);
                a.deform(&all, (time / 60.0) % duration)
            }
            _ => Vec::new(),
        }
    }

    /// The first animation's pose at `time` (60 Hz frames), looping.
    fn animation_pose(&self, time: f32) -> Vec<(usize, Mat3, DVec3)> {
        match &self.animations {
            Some(a) if !self.animated_instances().is_empty() => {
                let duration = a.list[0].duration.max(1e-3);
                a.pose(0, (time / 60.0) % duration)
            }
            _ => Vec::new(),
        }
    }

    fn showcase() -> World {
        let scene = Showcase::new();
        let camera = scene.camera(0.0);
        World {
            name: "showcase",
            surfaces: scene.surfaces.clone(),
            albedo: scene.albedo.clone(),
            light: scene.light,
            radiance: scene.radiance,
            first_mover: scene.first_mover,
            texel: showcase::TEXEL,
            probes: showcase::PROBES,
            fov: scene.vertical_fov_degrees,
            speed: 1.2,
            start: (camera.eye, camera.target),
            // The lightmap path: through `Renderer` (`Showcase::instanced`)
            // its two movers cost 10–35× the still frame (the view cache
            // re-bakes 60,000–90,000 texels a frame for their penumbrae;
            // DESIGN §3.8).
            instanced: None,
            showcase: Some(scene),
            scale: 1,
            cache: true,
            sky: false,
            environment: None,
            animations: None,
            exposure: 1.0,
        }
    }

    /// Whether the scene wants bounce light: not the field, whose bounce rays
    /// reach thousands of boulders, each wanting its own page, so its view
    /// never settles (and outdoors one bounce adds little).
    fn bounce_by_default(&self) -> bool {
        self.name != "field"
    }

    /// Whether the scene looks best with its sky: outdoor scenes (the field,
    /// a glTF file without animations, which do not play with a sky yet).
    fn sky_by_default(&self) -> bool {
        self.instanced.is_some()
            && self.showcase.is_none()
            && self.animations.as_ref().is_none_or(|a| a.list.is_empty())
    }

    fn field(boulders: usize) -> World {
        let field = Field::new(boulders, DVec3::ZERO);
        let origin = field.origin_near_camera();
        let (camera, light) = field.relative(origin);
        let summary = format!(
            "{} boulders, {} triangles",
            field.boulders(),
            field.triangles()
        );
        World {
            name: "field",
            surfaces: Vec::new(),
            albedo: Vec::new(),
            light,
            radiance: field.radiance,
            first_mover: 0,
            texel: 0.0,
            probes: 0.0,
            fov: field.vertical_fov_degrees,
            speed: 5.0,
            start: (camera.eye, camera.target),
            showcase: None,
            instanced: Some(Instanced {
                lights: vec![Light::white(light, field.radiance)],
                // `--sky`: a blue-white sky (M5), slow to warm up in the cache
                // (M5 gate 5).
                sky: field.sky(),
                origin,
                summary,
                scene: field.scene,
            }),
            scale: 1,
            cache: true,
            sky: false,
            environment: None,
            animations: None,
            exposure: 1.0,
        }
    }

    /// The glTF file at `path`, lit by a sun 30° up, over a grey ground
    /// unless `ground` is false. The render origin is the scene's center.
    fn gltf(path: &Path, ground: bool, emitters: bool) -> Result<World, String> {
        let start = Instant::now();
        let imported = continuum_gltf::load(path)
            .map_err(|e| format!("Could not load {}: {e}", path.display()))?;
        if imported.scene.instances.is_empty() {
            return Err(format!("{} has no triangles to show.", path.display()));
        }
        println!(
            "Imported {} in {:.2} s: {} triangles in {} instances of {} geometries, {} charts, {} textures{}.",
            path.display(),
            start.elapsed().as_secs_f64(),
            imported.stats.triangles,
            imported.stats.instances,
            imported.scene.geometries.len(),
            imported.stats.charts,
            imported.stats.textures,
            if imported.stats.skipped_primitives
                + imported.stats.textures_skipped
                + imported.stats.texture_slots_skipped
                > 0
            {
                format!(
                    " ({} primitives, {} textures and {} texture slots skipped)",
                    imported.stats.skipped_primitives,
                    imported.stats.textures_skipped,
                    imported.stats.texture_slots_skipped
                )
            } else {
                String::new()
            }
        );
        let (lo, hi) = imported.bounds;
        let origin = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(1e-3);
        // The file's own lights, if it has any (point and spot lights 5 cm
        // in radius), or else a sun.
        let mut lights = imported.lights(origin, 0.05);
        lights.truncate(continuum_gpu::MAX_LIGHTS);
        let animations = (!imported.animations.is_empty()).then_some(imported.animations);
        let mut scene = imported.scene;
        // `--emitters-as-lights`: emissive rectangles (up to six per
        // geometry) become area lights, if they fit with the file's.
        if emitters {
            let mut converted = scene.clone();
            let made = converted.emitters_to_lights(origin, 6);
            if !made.is_empty() && lights.len() + made.len() <= continuum_gpu::MAX_LIGHTS {
                println!("{} emissive faces made area lights.", made.len());
                lights.extend(made);
                scene = converted;
            } else if !made.is_empty() {
                println!(
                    "{} emissive faces would exceed {} lights: kept as emission.",
                    made.len(),
                    continuum_gpu::MAX_LIGHTS
                );
            }
        }
        if ground {
            // 20 radii across, just below the scene, so it casts shadows.
            let h = (10.0 * radius) as f32;
            let c = [(-h, -h), (h, -h), (h, h), (-h, h)].map(|(x, y)| Vec3::new(x, y, 0.0));
            scene.geometries.push(Surface::from_triangles(vec![
                [c[0], c[1], c[2]],
                [c[0], c[2], c[3]],
            ]));
            scene.geometry_textures.resize(scene.geometries.len(), None);
            scene.instances.push(Instance {
                geometry: scene.geometries.len() - 1,
                rotation: Mat3::IDENTITY,
                scale: 1.0,
                position: DVec3::new(origin.x, origin.y, lo.z - 1e-3 * radius),
                albedo: Vec3::splat(0.5),
            });
        }
        let (eye, target, fov) = match imported.camera {
            Some(c) => (
                (c.eye - origin).as_vec3(),
                (c.target - origin).as_vec3(),
                c.vertical_fov_degrees,
            ),
            None => {
                let (az, el) = (225f64.to_radians(), 25f64.to_radians());
                let dir = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
                ((dir * radius * 2.2).as_vec3(), Vec3::ZERO, 40.0)
            }
        };
        // Irradiance π on a surface facing the sun: white at albedo 1.
        let (az, el) = (200f64.to_radians(), 30f64.to_radians());
        let toward = DVec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
        let sun = Light::sun(
            Vec3::ZERO,
            toward,
            radius * 50.0,
            Vec3::new(1.0, 0.92, 0.8) * std::f32::consts::PI,
        );
        if lights.is_empty() {
            lights.push(sun);
        } else {
            println!("Lit by the file's {} lights.", lights.len());
        }
        let triangles: u64 = scene
            .instances
            .iter()
            .map(|i| scene.geometries[i.geometry].triangles.len() as u64)
            .sum();
        Ok(World {
            name: "glTF",
            surfaces: Vec::new(),
            albedo: Vec::new(),
            light: sun.as_rect().expect("a rectangular sun"),
            radiance: sun.radiance.x,
            first_mover: 0,
            texel: 0.0,
            probes: 0.0,
            fov,
            speed: (0.4 * radius as f32).max(0.2),
            start: (eye, target),
            showcase: None,
            instanced: Some(Instanced {
                scene,
                lights,
                sky: Vec3::new(0.6, 0.75, 1.0) * 0.3,
                origin,
                summary: format!("{triangles} triangles"),
            }),
            scale: 1,
            cache: true,
            sky: false,
            environment: None,
            animations,
            exposure: 1.0,
        })
    }

    fn room() -> World {
        let mut scene = render_scene();
        let first_mover = scene.add_movers(0.0);
        let l = &scene.room.light;
        World {
            name: "room",
            surfaces: scene
                .room
                .meshes
                .iter()
                .map(|m| Surface::from_triangles(m.clone()))
                .collect(),
            albedo: scene.albedo.clone(),
            light: RectLight {
                corner: l.corner,
                edge_u: l.edge_u,
                edge_v: l.edge_v,
                one_sided: false,
            },
            radiance: scene.radiance,
            first_mover,
            texel: 0.005,
            probes: 0.04,
            fov: 60.0,
            speed: 0.5,
            start: (scene.eye, scene.target),
            showcase: None,
            instanced: None,
            scale: 1,
            cache: true,
            sky: false,
            environment: None,
            animations: None,
            exposure: 1.0,
        }
    }

    /// The moving surfaces at `time` (60 Hz frames), with their mesh indices.
    fn movers(&mut self, time: f32) -> Vec<(usize, Surface)> {
        if self.instanced.is_some() {
            return Vec::new();
        }
        let first = self.first_mover;
        let moved: Vec<Surface> = match &mut self.showcase {
            Some(scene) => {
                scene.set_frame(time);
                scene.surfaces[first..].to_vec()
            }
            None => movers(time)
                .into_iter()
                .map(Surface::from_triangles)
                .collect(),
        };
        moved
            .into_iter()
            .enumerate()
            .map(|(k, s)| (first + k, s))
            .collect()
    }

    fn path_camera(&self, frame: f32) -> Option<Camera> {
        self.showcase.as_ref().map(|s| s.camera(frame))
    }
}

struct FlyCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl FlyCamera {
    fn looking_at(eye: Vec3, target: Vec3) -> Self {
        let d = (target - eye).normalize();
        FlyCamera {
            position: eye,
            yaw: d.y.atan2(d.x),
            pitch: d.z.asin(),
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        )
    }

    fn camera(&self, fov: f32) -> Camera {
        Camera {
            eye: self.position,
            target: self.position + self.forward(),
            vertical_fov_degrees: fov,
        }
    }
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    /// None until the first scene is built.
    backend: Option<Backend>,
    ui: Ui,
}

/// The overlay panel: egui's context, its window input and its painter.
struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

/// What the panel changes. Bounce light, anti-aliasing, the sky and the
/// render scale rebuild the renderer; the rest apply at once.
#[derive(Clone, Copy, PartialEq)]
struct Settings {
    bounce: bool,
    anti_aliasing: bool,
    sky: bool,
    scale: u32,
    bloom: bool,
    tone_map: continuum_gpu::ToneMap,
    vsync: bool,
}

impl Settings {
    /// The settings a renderer is built with.
    fn build_key(&self) -> (bool, bool, bool, u32) {
        (self.bounce, self.anti_aliasing, self.sky, self.scale)
    }
}

/// A scene to show.
#[derive(Clone)]
enum SceneChoice {
    Showcase,
    Room,
    Field(usize),
    Gltf(PathBuf),
}

impl SceneChoice {
    fn load(&self, ground: bool, emitters: bool) -> Result<World, String> {
        Ok(match self {
            SceneChoice::Showcase => World::showcase(),
            SceneChoice::Room => World::room(),
            SceneChoice::Field(n) => World::field(*n),
            SceneChoice::Gltf(path) => World::gltf(path, ground, emitters)?,
        })
    }

    fn describe(&self) -> String {
        match self {
            SceneChoice::Showcase => "Showcase (a gallery)".into(),
            SceneChoice::Room => "Room".into(),
            SceneChoice::Field(n) => format!("Boulder field ({} boulders)", thousands(*n)),
            SceneChoice::Gltf(path) => path.file_name().map_or_else(
                || path.display().to_string(),
                |f| f.to_string_lossy().into(),
            ),
        }
    }
}

/// `n` with commas between thousands.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Work the next frames do, after one frame shows that it is coming.
enum Pending {
    Load(SceneChoice),
    Rebuild,
}

/// The lightmap renderer with moving objects (the showcase and the room), or
/// the library's renderer for instanced scenes (the field or a glTF file).
enum Backend {
    Live(Box<LiveRenderer>),
    Instanced(Box<Renderer>),
}

impl Backend {
    fn resize(&mut self, width: u32, height: u32) {
        match self {
            Backend::Live(r) => r.resize(width, height),
            Backend::Instanced(r) => r.resize(width, height),
        }
    }

    /// Poses deforming geometry (instanced scenes).
    fn set_vertices(&mut self, poses: Vec<continuum_gltf::Pose>) {
        if let Backend::Instanced(r) = self {
            for p in poses {
                r.set_vertices(p.geometry, p.triangles, p.normals);
            }
        }
    }

    /// Moves animated instances (instanced scenes).
    fn set_transforms(&mut self, poses: &[(usize, Mat3, DVec3)]) {
        if let Backend::Instanced(r) = self {
            for &(i, rotation, position) in poses {
                r.set_transform(i, rotation, position);
            }
        }
    }

    fn update_surfaces(&mut self, changes: &[(usize, &Surface)]) {
        if let Backend::Live(r) = self
            && !changes.is_empty()
        {
            r.update_surfaces(changes);
        }
    }

    fn frame(&mut self, camera: Camera, view: &wgpu::TextureView) {
        match self {
            Backend::Live(r) => r.frame(camera, camera.vertical_fov_degrees, view),
            Backend::Instanced(r) => r.render(camera, view),
        }
    }
}

struct App {
    gpu: Option<Gpu>,
    world: World,
    scene: SceneChoice,
    /// The glTF flags (`--no-ground`, `--emitters-as-lights`).
    ground: bool,
    emitters: bool,
    settings: Settings,
    /// Animation time in 60 Hz frames.
    time: f32,
    paused: bool,
    /// Playing the camera path: time along it in 60 Hz frames.
    path: Option<f32>,
    camera: FlyCamera,
    keys: HashSet<KeyCode>,
    drag: Option<PhysicalPosition<f64>>,
    last: Instant,
    frame_times: Vec<f32>,
    /// Frames per second and milliseconds a frame, over the last 30.
    rate: (f32, f32),
    frames_left: Option<u32>,
    all_times: Vec<f32>,
    /// Work to do, and whether a frame has shown that it is coming.
    pending: Option<(Pending, bool)>,
    /// The last problem or note, shown in the panel.
    message: Option<String>,
    show_panel: bool,
    /// Boulders for the field's button.
    boulders: usize,
    /// The settings the renderer was built with.
    built: (bool, bool, bool, u32),
    /// Whether `--scale` was given (else Retina displays render at half).
    scale_given: bool,
    /// `--screenshot`: the last frame's image, written as a PNG.
    screenshot: Option<PathBuf>,
}

impl App {
    fn init(event_loop: &ActiveEventLoop, screenshot: bool) -> Gpu {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Continuum Viewer")
                        .with_inner_size(LogicalSize::new(1280, 720)),
                )
                .expect("Could not open a window."),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("Could not draw into the window.");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect(
            "No suitable GPU found. Continuum needs a GPU with compute shaders; it is tested on Apple silicon Macs.",
        );
        let (device, queue) =
            pollster::block_on(adapter.request_device(&continuum_gpu::device_descriptor(&adapter)))
                .expect("The GPU refused the features Continuum needs.");
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: if screenshot {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            },
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        Gpu {
            window,
            surface,
            config,
            device,
            queue,
            adapter_name: adapter.get_info().name,
            backend: None,
            ui: Ui {
                ctx,
                state,
                renderer,
            },
        }
    }

    /// The renderer for `world` with `settings`, into the window's format
    /// and size.
    fn build(gpu: &Gpu, world: &mut World, settings: &Settings) -> Backend {
        let kernel = MaskConeKernel::with_device(
            gpu.device.clone(),
            gpu.queue.clone(),
            gpu.adapter_name.clone(),
        );
        let config = &gpu.config;
        world.sky = settings.sky;
        world.scale = settings.scale;
        if let Some(scene) = &world.instanced {
            let start = Instant::now();
            let options = RendererOptions {
                scale: world.scale,
                cache: world.cache,
                // The sky only with the cache: per pixel, sky cones every
                // frame take seconds.
                sky: (world.sky && world.cache && world.environment.is_none()).then_some(scene.sky),
                environment: world.environment.clone().filter(|_| world.cache),
                movable: world.animated_instances(),
                deforming: world.deforming_geometries(),
                anti_aliasing: settings.anti_aliasing,
                bounce: settings.bounce,
                bloom: settings.bloom.then(continuum_gpu::Bloom::default),
                tone_map: settings.tone_map,
                ..Default::default()
            };
            let mut renderer = Renderer::new(
                kernel,
                config.format,
                (config.width, config.height),
                &scene.scene,
                scene.origin,
                &scene.lights,
                options,
            );
            renderer.set_exposure(world.exposure);
            renderer.wait();
            println!(
                "{} ({} scene): {}, {:.0} MB on the GPU ({}), ready in {:.1} s; window {}×{}, rendering at 1/{} scale; camera rays: {}.",
                renderer.kernel().adapter_name,
                world.name,
                scene.summary,
                renderer.gpu_bytes() as f64 / 1e6,
                if world.cache {
                    "with the lighting cache"
                } else {
                    "a cone per pixel"
                },
                start.elapsed().as_secs_f64(),
                config.width,
                config.height,
                world.scale,
                if renderer.hardware_rays() {
                    "hardware"
                } else {
                    "compute"
                }
            );
            return Backend::Instanced(Box::new(renderer));
        }
        let meshes: Vec<_> = world.surfaces.iter().map(|s| s.triangles.clone()).collect();
        let first_mover = world.first_mover;
        let mut lightmap = Lightmap::new(&world.surfaces, world.texel, |m| m < first_mover)
            .with_probes(world.probes);
        let bake = kernel.bake(&mut lightmap, world.light, &meshes);
        let start = Instant::now();
        let renderer = LiveRenderer::new(
            &kernel,
            config.format,
            config.width,
            config.height,
            world.light,
            &meshes,
            &world.albedo,
            world.radiance,
            &lightmap,
        );
        renderer.wait();
        println!(
            "{} ({} scene): {} texels baked in {:.0} ms, {} probes; caches ready in {:.0} ms; rendering at {}×{}.",
            kernel.adapter_name,
            world.name,
            lightmap.texels(),
            bake.as_secs_f64() * 1000.0,
            renderer.probes(),
            start.elapsed().as_secs_f64() * 1000.0,
            config.width,
            config.height
        );
        Backend::Live(Box::new(renderer))
    }

    /// Does the pending work, if a frame has already shown it coming.
    fn do_pending(&mut self) {
        let Some((work, shown)) = self.pending.take() else {
            return;
        };
        if !shown {
            self.pending = Some((work, true));
            return;
        }
        let Some(gpu) = &mut self.gpu else { return };
        match work {
            Pending::Load(choice) => match choice.load(self.ground, self.emitters) {
                Ok(world) => {
                    // Free the old scene's GPU memory before the new one's.
                    gpu.backend = None;
                    self.settings.sky = world.sky_by_default();
                    self.settings.bounce = world.bounce_by_default();
                    self.world = world;
                    self.world.exposure = 1.0;
                    self.scene = choice;
                    self.camera = FlyCamera::looking_at(self.world.start.0, self.world.start.1);
                    self.time = 0.0;
                    self.path = None;
                    gpu.backend = Some(Self::build(gpu, &mut self.world, &self.settings));
                    self.built = self.settings.build_key();
                    self.message = None;
                    gpu.window
                        .set_title(&format!("Continuum Viewer — {}", self.scene.describe()));
                }
                Err(e) => {
                    eprintln!("{e}");
                    self.message = Some(e);
                }
            },
            Pending::Rebuild => {
                gpu.backend = None;
                gpu.backend = Some(Self::build(gpu, &mut self.world, &self.settings));
                self.built = self.settings.build_key();
            }
        }
        self.frame_times.clear();
    }

    /// The panel; returns what to do (work for the next frames).
    fn panel(&mut self, ctx: &egui::Context) -> Option<Pending> {
        let mut work = None;
        let gpu = self.gpu.as_ref().expect("the window");
        let instanced = self.world.instanced.is_some();
        let settled = match &gpu.backend {
            Some(Backend::Instanced(r)) => Some(r.is_settled(32)),
            _ => None,
        };
        let gpu_mb = match &gpu.backend {
            Some(Backend::Instanced(r)) => Some(r.gpu_bytes() as f64 / 1e6),
            _ => None,
        };
        let loading = self.pending.is_some();
        egui::Window::new("Continuum")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(self.scene.describe()).strong());
                ui.label(format!(
                    "{:.0} fps ({:.1} ms) at {}×{}",
                    self.rate.0, self.rate.1, gpu.config.width, gpu.config.height
                ));
                ui.small(match gpu_mb {
                    Some(mb) => format!("{} · {mb:.0} MB on the GPU", gpu.adapter_name),
                    None => gpu.adapter_name.clone(),
                });
                if let Some(done) = settled {
                    ui.small(if done {
                        "Lighting settled"
                    } else {
                        "Lighting settling…"
                    });
                }
                if loading {
                    ui.colored_label(egui::Color32::YELLOW, "Loading…");
                }
                if let Some(m) = &self.message {
                    ui.colored_label(egui::Color32::from_rgb(255, 120, 100), m);
                }
                ui.separator();
                ui.label("Scene");
                ui.horizontal(|ui| {
                    if ui.button("Showcase").clicked() {
                        work = Some(Pending::Load(SceneChoice::Showcase));
                    }
                    if ui.button("Boulder field").clicked() {
                        work = Some(Pending::Load(SceneChoice::Field(self.boulders)));
                    }
                    egui::ComboBox::from_id_salt("boulders")
                        .selected_text(thousands(self.boulders))
                        .show_ui(ui, |ui| {
                            for n in [10_000, 100_000, 1_000_000] {
                                ui.selectable_value(&mut self.boulders, n, thousands(n));
                            }
                        });
                });
                if ui.button("Open a glTF file…").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("glTF", &["glb", "gltf"])
                        .pick_file()
                {
                    work = Some(Pending::Load(SceneChoice::Gltf(path)));
                }
                ui.small("…or drop a .glb or .gltf file on the window.");
                ui.separator();
                egui::CollapsingHeader::new("Rendering")
                    .default_open(true)
                    .show(ui, |ui| {
                        if !instanced {
                            ui.small("These apply to the boulder field and glTF scenes.");
                        }
                        ui.add_enabled_ui(instanced, |ui| {
                            let s = &mut self.settings;
                            ui.checkbox(&mut s.bounce, "Bounce light");
                            ui.checkbox(&mut s.anti_aliasing, "Anti-aliasing");
                            ui.checkbox(&mut s.sky, "Sky");
                            egui::ComboBox::from_label("Render scale")
                                .selected_text(format!("1/{}", s.scale))
                                .show_ui(ui, |ui| {
                                    for k in 1..=4 {
                                        ui.selectable_value(&mut s.scale, k, format!("1/{k}"));
                                    }
                                });
                            let name = |t: continuum_gpu::ToneMap| match t {
                                continuum_gpu::ToneMap::PbrNeutral => "PBR Neutral",
                                continuum_gpu::ToneMap::Agx => "AgX",
                                continuum_gpu::ToneMap::Reinhard => "Reinhard",
                            };
                            egui::ComboBox::from_label("Tone map")
                                .selected_text(name(s.tone_map))
                                .show_ui(ui, |ui| {
                                    for t in [
                                        continuum_gpu::ToneMap::PbrNeutral,
                                        continuum_gpu::ToneMap::Agx,
                                        continuum_gpu::ToneMap::Reinhard,
                                    ] {
                                        ui.selectable_value(&mut s.tone_map, t, name(t));
                                    }
                                });
                            ui.checkbox(&mut s.bloom, "Bloom");
                            let mut stops = self.world.exposure.log2();
                            if ui
                                .add(
                                    egui::Slider::new(&mut stops, -6.0..=6.0)
                                        .text("Exposure (stops)"),
                                )
                                .changed()
                            {
                                self.world.exposure = stops.exp2();
                            }
                        });
                        ui.checkbox(&mut self.settings.vsync, "Vsync");
                    });
                egui::CollapsingHeader::new("Controls").show(ui, |ui| {
                    ui.small("W A S D: move · Q / E: down, up · Shift: faster");
                    ui.small("Drag: look around");
                    ui.small("P: pause animations · C: camera path (showcase)");
                    ui.small("[ and ]: exposure · V: vsync · H: hide this panel · Esc: quit");
                });
            });
        work
    }

    fn update(&mut self, dt: f32) {
        let speed = if self.keys.contains(&KeyCode::ShiftLeft) {
            3.0
        } else {
            1.0
        } * self.world.speed
            * dt;
        let forward = self.camera.forward();
        let flat = Vec3::new(forward.x, forward.y, 0.0).normalize_or_zero();
        let right = flat.cross(Vec3::Z);
        let mut motion = Vec3::ZERO;
        for (key, dir) in [
            (KeyCode::KeyW, flat),
            (KeyCode::KeyS, -flat),
            (KeyCode::KeyD, right),
            (KeyCode::KeyA, -right),
            (KeyCode::KeyE, Vec3::Z),
            (KeyCode::KeyQ, -Vec3::Z),
        ] {
            if self.keys.contains(&key) {
                motion += dir;
            }
        }
        self.camera.position += motion.normalize_or_zero() * speed;
        if !self.paused {
            self.time += dt * 60.0;
        }
        if let Some(t) = &mut self.path {
            *t = (*t + dt * 60.0) % PATH_FRAMES as f32;
            if let Some(c) = self.world.path_camera(*t) {
                self.camera = FlyCamera::looking_at(c.eye, c.target);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let mut gpu = Self::init(event_loop, self.screenshot.is_some());
            // On a Retina display, render at half the window's pixels (drawn up
            // by the renderer's upscaler) unless `--scale` says otherwise:
            // frames half as long, and lighting fills in sooner.
            if !self.scale_given && gpu.window.scale_factor() >= 2.0 {
                self.settings.scale = 2;
            }
            gpu.backend = Some(Self::build(&gpu, &mut self.world, &self.settings));
            self.built = self.settings.build_key();
            gpu.window
                .set_title(&format!("Continuum Viewer — {}", self.scene.describe()));
            gpu.window.request_redraw();
            self.gpu = Some(gpu);
            self.last = Instant::now();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // The panel first: what it takes, the camera does not.
        let mut consumed = false;
        if let Some(gpu) = &mut self.gpu {
            consumed = gpu.ui.state.on_window_event(&gpu.window, &event).consumed;
        }
        let typing = self
            .gpu
            .as_ref()
            .is_some_and(|g| g.ui.ctx.egui_wants_keyboard_input());
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::DroppedFile(path) => {
                self.pending = Some((Pending::Load(SceneChoice::Gltf(path)), false));
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu
                    && size.width > 0
                    && size.height > 0
                {
                    (gpu.config.width, gpu.config.height) = (size.width, size.height);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    if let Some(b) = &mut gpu.backend {
                        b.resize(size.width, size.height);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        repeat,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed if !typing => {
                    if code == KeyCode::Escape {
                        event_loop.exit();
                    }
                    if code == KeyCode::KeyH && !repeat {
                        self.show_panel = !self.show_panel;
                    }
                    if code == KeyCode::KeyP && !repeat {
                        self.paused = !self.paused;
                    }
                    if code == KeyCode::KeyC && !repeat && self.world.showcase.is_some() {
                        self.path = match self.path {
                            Some(_) => None,
                            None => Some(0.0),
                        };
                    }
                    // Exposure, a stop at a time (instanced scenes).
                    match code {
                        KeyCode::BracketRight => self.world.exposure *= 2.0,
                        KeyCode::BracketLeft => self.world.exposure *= 0.5,
                        _ => {}
                    }
                    if code == KeyCode::KeyV && !repeat {
                        self.settings.vsync = !self.settings.vsync;
                    }
                    self.keys.insert(code);
                }
                ElementState::Pressed => {}
                ElementState::Released => {
                    self.keys.remove(&code);
                }
            },
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                if state == ElementState::Released {
                    self.drag = None;
                } else if !consumed {
                    self.drag = Some(PhysicalPosition::new(f64::NAN, f64::NAN));
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(previous) = self.drag {
                    if !previous.x.is_nan() {
                        self.camera.yaw -= (position.x - previous.x) as f32 * 0.004;
                        self.camera.pitch = (self.camera.pitch
                            - (position.y - previous.y) as f32 * 0.004)
                            .clamp(-1.5, 1.5);
                    }
                    self.drag = Some(position);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }
}

impl App {
    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        // `VIEWER_SELFTEST=1` (development, with `--frames`): what the panel's
        // clicks do, scripted: load the field, turn bounce light off (a
        // rebuild), then a missing file (an error in the panel).
        if std::env::var_os("VIEWER_SELFTEST").is_some() {
            match self.all_times.len() {
                60 => self.pending = Some((Pending::Load(SceneChoice::Field(10_000)), false)),
                150 => self.settings.bounce = false,
                240 => {
                    let missing = PathBuf::from("/nonexistent/scene.glb");
                    self.pending = Some((Pending::Load(SceneChoice::Gltf(missing)), false));
                }
                _ => {}
            }
        }
        self.do_pending();
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        self.update(dt.min(0.1));
        let moved = self.world.movers(self.time);
        let changes: Vec<_> = moved.iter().map(|(m, s)| (*m, s)).collect();
        let (poses, shapes) = if self.paused {
            (Vec::new(), Vec::new())
        } else {
            (
                self.world.animation_pose(self.time),
                self.world.deforming_poses(self.time),
            )
        };
        let fov = self.world.fov;
        let camera = self.camera.camera(fov);
        // The panel's frame, before the scene's: its changes apply now.
        let Some(gpu) = &mut self.gpu else { return };
        let raw = gpu.ui.state.take_egui_input(&gpu.window);
        let ctx = gpu.ui.ctx.clone();
        let mut work = None;
        let show = self.show_panel;
        let output = ctx.run_pass(raw, |ctx| {
            if show {
                work = self.panel(ctx);
            }
        });
        let Some(gpu) = &mut self.gpu else { return };
        // The panel's textures (its font first) now, so a frame skipped
        // below loses none.
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                gpu.ui
                    .renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
        }
        gpu.ui
            .state
            .handle_platform_output(&gpu.window, output.platform_output);
        if let Some(w) = work {
            self.pending = Some((w, false));
        }
        // Settings that need a new renderer (instanced scenes).
        if self.pending.is_none()
            && self.world.instanced.is_some()
            && self.settings.build_key() != self.built
        {
            self.pending = Some((Pending::Rebuild, false));
        }
        // Settings that apply at once.
        let present = if self.settings.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if gpu.config.present_mode != present {
            gpu.config.present_mode = present;
            gpu.surface.configure(&gpu.device, &gpu.config);
        }
        if let Some(Backend::Instanced(r)) = &mut gpu.backend {
            r.set_exposure(self.world.exposure);
            r.set_tone_map(self.settings.tone_map);
            r.set_bloom(self.settings.bloom.then(continuum_gpu::Bloom::default));
        }
        if let Some(b) = &mut gpu.backend {
            b.update_surfaces(&changes);
            b.set_transforms(&poses);
            b.set_vertices(shapes);
        }
        let texture = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                gpu.window.request_redraw();
                return;
            }
        };
        let view = texture.texture.create_view(&Default::default());
        if let Some(b) = &mut gpu.backend {
            b.frame(camera, &view);
        }
        // The panel over the scene.
        let jobs = gpu
            .ui
            .ctx
            .tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut buffers =
            gpu.ui
                .renderer
                .update_buffers(&gpu.device, &gpu.queue, &mut encoder, &jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("panel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            gpu.ui
                .renderer
                .render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        let last_frame = self.frames_left == Some(1);
        let shot = match (&self.screenshot, last_frame) {
            (Some(_), true) => Some(capture(&gpu.device, &mut encoder, &texture.texture)),
            _ => None,
        };
        buffers.push(encoder.finish());
        gpu.queue.submit(buffers);
        for id in &output.textures_delta.free {
            gpu.ui.renderer.free_texture(id);
        }
        if let (Some(path), Some(shot)) = (&self.screenshot, shot) {
            save_capture(&gpu.device, shot, gpu.config.format, path);
        }
        gpu.queue.present(texture);

        self.frame_times.push(dt * 1000.0);
        self.all_times.push(dt * 1000.0);
        if self.frame_times.len() == 30 {
            let avg = self.frame_times.iter().sum::<f32>() / 30.0;
            self.rate = (1000.0 / avg, avg);
            self.frame_times.clear();
        }
        if let Some(left) = &mut self.frames_left {
            *left = left.saturating_sub(1);
            // Exercise a moving camera (the path if there is one), but not
            // for a screenshot.
            if self.world.showcase.is_none() && self.screenshot.is_none() {
                self.camera.yaw += 0.01;
            }
            if *left == 0 {
                let times = &self.all_times[self.all_times.len().min(10)..];
                let mut sorted = times.to_vec();
                sorted.sort_by(f32::total_cmp);
                println!(
                    "{} frames at {}×{}: median {:.2} ms, worst {:.2} ms (first 10 frames skipped).",
                    times.len(),
                    gpu.config.width,
                    gpu.config.height,
                    sorted[sorted.len() / 2],
                    sorted[sorted.len() - 1]
                );
                event_loop.exit();
                return;
            }
        }
        gpu.window.request_redraw();
    }
}

/// A frame's image being copied out (`--screenshot`): the buffer and the
/// texture's size and padded row length.
struct Capture {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    row: u32,
}

fn capture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> Capture {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot"),
        size: (row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    Capture {
        buffer,
        width,
        height,
        row,
    }
}

fn save_capture(device: &wgpu::Device, c: Capture, format: wgpu::TextureFormat, path: &Path) {
    let slice = c.buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU");
    let data = slice
        .get_mapped_range()
        .expect("the screenshot's buffer, mapped");
    let bgra = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let mut image = flatland::img::Image::new(c.width as usize, c.height as usize, [0; 3]);
    for y in 0..c.height as usize {
        for x in 0..c.width as usize {
            let i = y * c.row as usize + 4 * x;
            let p = [data[i], data[i + 1], data[i + 2]];
            image.set(x, y, if bgra { [p[2], p[1], p[0]] } else { p });
        }
    }
    match image.save(path) {
        Ok(()) => println!("Saved {}.", path.display()),
        Err(e) => eprintln!("Could not save {}: {e}", path.display()),
    }
}

/// The `--env` sky: `gradient` (M5 gate 12's, around the scene's uniform
/// sky) or an image file, times `strength`, at most twice the texture size
/// wide (the renderer resamples it to the texture size).
fn load_environment(world: &World, spec: &str, strength: f32) -> Environment {
    let Some(scene) = &world.instanced else {
        eprintln!("--env needs an instanced scene: --scene field or --gltf <file>.");
        std::process::exit(2);
    };
    if !world.cache {
        eprintln!("--env is ignored with --no-cache (sky cones per pixel take seconds).");
    }
    let mut env = if spec == "gradient" {
        let (horizon, zenith) = (
            scene.sky * Vec3::new(1.3, 1.2, 1.05),
            scene.sky * Vec3::new(0.55, 0.72, 1.25),
        );
        Environment::from_fn(512, 256, |d| {
            if d.z <= 0.0 {
                Vec3::ZERO
            } else {
                horizon * (1.0 - d.z) + zenith * d.z
            }
        })
    } else {
        match continuum_gltf::load_environment(Path::new(spec)) {
            Ok(env) => env,
            Err(e) => {
                eprintln!("Could not load the environment {spec}: {e}");
                std::process::exit(1);
            }
        }
    };
    let k = env.width.div_ceil(2 * continuum_gpu::TEXTURE_SIZE).max(1);
    if k > 1 {
        let (w, h) = (env.width / k, (env.height / k).max(1));
        let mut rgb = vec![[0.0f32; 3]; (w * h) as usize];
        for (i, p) in rgb.iter_mut().enumerate() {
            let (x0, y0) = (i as u32 % w * k, i as u32 / w * k);
            for y in y0..(y0 + k).min(env.height) {
                for x in x0..x0 + k {
                    let q = env.rgb[(y * env.width + x) as usize];
                    (0..3).for_each(|c| p[c] += q[c]);
                }
            }
            let n = (k * (k.min(env.height - y0))) as f32;
            p.iter_mut().for_each(|c| *c /= n);
        }
        env = Environment {
            width: w,
            height: h,
            rgb,
        };
    }
    env.rgb
        .iter_mut()
        .for_each(|p| p.iter_mut().for_each(|c| *c *= strength));
    println!(
        "Environment {spec}: {}×{}, times {strength}.",
        env.width, env.height
    );
    env
}

fn main() {
    // Problems in a window as well as on the terminal: launched from the
    // Finder there is no terminal.
    std::panic::set_hook(Box::new(|info| {
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "Continuum Viewer stopped.".into());
        eprintln!(
            "{message} ({})",
            info.location().map_or(String::new(), |l| l.to_string())
        );
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("Continuum Viewer")
            .set_description(&message)
            .set_buttons(rfd::MessageButtons::Ok)
            .show();
    }));
    let arg = |name: &str| std::env::args().skip_while(|a| a != name).nth(1);
    let has = |name: &str| std::env::args().any(|a| a == name);
    let frames_left: Option<u32> = arg("--frames").and_then(|n| n.parse().ok());
    let boulders = arg("--boulders")
        .and_then(|n| n.parse().ok())
        .unwrap_or(1_000_000);
    // A file path alone (as an app opening a document) shows that file.
    let lone = std::env::args()
        .skip(1)
        .find(|a| a.ends_with(".glb") || a.ends_with(".gltf"));
    let scene = match (arg("--gltf").or(lone), arg("--scene").as_deref()) {
        (Some(path), _) => SceneChoice::Gltf(PathBuf::from(path)),
        (None, None | Some("showcase")) => SceneChoice::Showcase,
        (None, Some("room")) => SceneChoice::Room,
        (None, Some("field")) => SceneChoice::Field(boulders),
        (None, Some(other)) => {
            eprintln!("Unknown scene {other:?}: use showcase, room or field, or --gltf <file>.");
            std::process::exit(2);
        }
    };
    let (ground, emitters) = (!has("--no-ground"), has("--emitters-as-lights"));
    let (scene, mut world, message) = match scene.load(ground, emitters) {
        Ok(world) => (scene, world, None),
        Err(e) => {
            eprintln!("{e}");
            (SceneChoice::Showcase, World::showcase(), Some(e))
        }
    };
    world.cache = !has("--no-cache");
    world.exposure = arg("--exposure")
        .and_then(|k| k.parse().ok())
        .unwrap_or(1.0);
    if let Some(spec) = arg("--env") {
        let strength: f32 = arg("--env-strength")
            .and_then(|k| k.parse().ok())
            .unwrap_or(1.0);
        world.environment = Some(load_environment(&world, &spec, strength));
    }
    let settings = Settings {
        bounce: !has("--no-bounce") && (has("--bounce") || world.bounce_by_default()),
        anti_aliasing: !has("--no-aa"),
        sky: has("--sky") || (!has("--no-sky") && world.sky_by_default()),
        scale: arg("--scale").and_then(|n| n.parse().ok()).unwrap_or(1),
        bloom: true,
        tone_map: continuum_gpu::ToneMap::PbrNeutral,
        vsync: has("--vsync"),
    };
    world.sky = settings.sky;
    if let Some(a) = &world.animations {
        let moving = world.animated_instances().len();
        let deforming = world.deforming_geometries().len();
        if moving > 0 {
            println!(
                "Playing animation {} ({:.1} s, looping) on {moving} instances; P pauses.",
                a.list[0].name.as_deref().unwrap_or("0"),
                a.list[0].duration
            );
        }
        if deforming > 0 {
            println!(
                "Posing {deforming} skinned or morphing meshes from {} animations, looping; P pauses.",
                a.list.len()
            );
        }
        if moving == 0
            && deforming == 0
            && (!a.deforming().is_empty() || !a.instances(0).is_empty())
        {
            println!(
                "The file's animation does not play with a sky (movers and a sky do not mix in the cache yet)."
            );
        }
    }
    let path = (frames_left.is_some() && world.showcase.is_some()).then_some(0.0);
    let camera = FlyCamera::looking_at(world.start.0, world.start.1);
    let mut app = App {
        gpu: None,
        world,
        scene,
        ground,
        emitters,
        settings,
        time: 0.0,
        paused: false,
        path,
        camera,
        keys: HashSet::new(),
        drag: None,
        last: Instant::now(),
        frame_times: Vec::new(),
        rate: (0.0, 0.0),
        frames_left,
        all_times: Vec::new(),
        pending: None,
        message,
        show_panel: !has("--no-panel"),
        boulders,
        built: settings.build_key(),
        scale_given: arg("--scale").is_some(),
        screenshot: arg("--screenshot").map(PathBuf::from),
    };
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut app).expect("event loop");
}
