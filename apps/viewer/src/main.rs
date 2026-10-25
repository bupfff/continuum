//! Continuum viewer: fly through a scene in real time.
//!
//! Direct light from the area light is baked into a world-space lighting cache
//! (one mask cone per texel) at startup; each frame is a camera ray per pixel
//! plus a cache lookup, on the GPU. Objects move through the scene: they are
//! shaded per pixel, and each frame re-bakes exactly the cache blocks they can
//! reach, so their soft shadows stay exact. Probes add one bounce of indirect
//! light, gathered each frame through cached G-buffers (probes on moving
//! objects are traced from scratch).
//!
//! Scenes: `--scene showcase` (the default: a gallery with a sculpture,
//! columns and an arch), `--scene room` (the M2 test room), or `--scene field`
//! (a million instanced boulders, about ten billion triangles, lit by a sun
//! through a view-dependent lighting cache baked as you look around;
//! `--boulders N` to change the count, `--scale N` to render at 1/N of the
//! window, `--sky` to add a blue-white sky (experimental: slow to warm up),
//! `--no-cache` for a mask cone per pixel every frame). `--gltf file.glb`
//! (or `.gltf`) flies through your own scene the same way: lit by a sun
//! (`--sky` adds the sky), from the file's camera or one framed around the
//! scene, over a grey ground (`--no-ground` to leave it out).
//!
//! Controls: WASD move, Q/E down/up, drag with the left mouse button to look,
//! Shift to move faster, C to play the camera path, P to pause moving objects,
//! V to toggle vsync, Esc to quit. `--frames N` renders N frames (along the
//! camera path, if the scene has one), prints timings, and exits.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use continuum::glam::{DVec3, Mat3, Vec3};
use continuum::mesh::Surface;
use continuum_gpu::wgpu;
use continuum_gpu::{
    CacheOptions, Camera, Instance, InstancedRenderer, InstancedScene, Light, Lightmap,
    LiveRenderer, MaskConeKernel, RectLight,
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
    /// The instanced boulder field.
    field: Option<Field>,
    /// An imported glTF scene.
    gltf: Option<Gltf>,
    /// Window pixels per rendered pixel (field and glTF scenes only).
    scale: u32,
    /// Light the field through the view-dependent lighting cache (otherwise a
    /// mask cone per pixel every frame).
    cache: bool,
    /// Add a sky to the field (with the cache only).
    sky: bool,
}

/// An imported glTF scene with its lights and sky.
struct Gltf {
    scene: InstancedScene,
    lights: Vec<Light>,
    sky: Vec3,
    origin: DVec3,
    triangles: u64,
}

impl World {
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
            showcase: Some(scene),
            field: None,
            gltf: None,
            scale: 1,
            cache: true,
            sky: false,
        }
    }

    fn field(boulders: usize) -> World {
        let field = Field::new(boulders, DVec3::ZERO);
        let origin = field.origin_near_camera();
        let (camera, light) = field.relative(origin);
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
            field: Some(field),
            gltf: None,
            scale: 1,
            cache: true,
            sky: false,
        }
    }

    /// The glTF file at `path`, lit by a sun 30° up, over a grey ground
    /// unless `ground` is false. The render origin is the scene's center.
    fn gltf(path: &Path, ground: bool) -> World {
        let start = Instant::now();
        let imported = continuum_gltf::load(path).unwrap_or_else(|e| {
            eprintln!("Could not load {}: {e}", path.display());
            std::process::exit(1);
        });
        if imported.scene.instances.is_empty() {
            eprintln!("{} has no triangles to show.", path.display());
            std::process::exit(1);
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
            if imported.stats.skipped_primitives + imported.stats.textures_skipped > 0 {
                format!(
                    " ({} primitives and {} textures skipped)",
                    imported.stats.skipped_primitives, imported.stats.textures_skipped
                )
            } else {
                String::new()
            }
        );
        let mut scene = imported.scene;
        let (lo, hi) = imported.bounds;
        let origin = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(1e-3);
        if ground {
            // 20 radii across, just below the scene, so it casts shadows.
            let h = (10.0 * radius) as f32;
            let c = [(-h, -h), (h, -h), (h, h), (-h, h)].map(|(x, y)| Vec3::new(x, y, 0.0));
            scene.geometries.push(Surface::from_triangles(vec![
                [c[0], c[1], c[2]],
                [c[0], c[2], c[3]],
            ]));
            scene
                .geometry_textures
                .resize(scene.geometries.len(), None);
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
        let triangles = scene
            .instances
            .iter()
            .map(|i| scene.geometries[i.geometry].triangles.len() as u64)
            .sum();
        World {
            name: "glTF",
            surfaces: Vec::new(),
            albedo: Vec::new(),
            light: sun.rect,
            radiance: sun.radiance.x,
            first_mover: 0,
            texel: 0.0,
            probes: 0.0,
            fov,
            speed: (0.4 * radius as f32).max(0.2),
            start: (eye, target),
            showcase: None,
            field: None,
            gltf: Some(Gltf {
                scene,
                lights: vec![sun],
                sky: Vec3::new(0.6, 0.75, 1.0) * 0.3,
                origin,
                triangles,
            }),
            scale: 1,
            cache: true,
            sky: false,
        }
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
            field: None,
            gltf: None,
            scale: 1,
            cache: true,
            sky: false,
        }
    }

    /// The moving surfaces at `time` (60 Hz frames), with their mesh indices.
    fn movers(&mut self, time: f32) -> Vec<(usize, Surface)> {
        if self.field.is_some() || self.gltf.is_some() {
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
    renderer: Renderer,
}

/// The lit scene with its caches, or an instanced scene (the field or a
/// glTF file).
enum Renderer {
    Live(Box<LiveRenderer>),
    Instanced(Box<InstancedRenderer>),
}

impl Renderer {
    fn resize(&mut self, width: u32, height: u32) {
        match self {
            Renderer::Live(r) => r.resize(width, height),
            Renderer::Instanced(r) => r.resize(width, height),
        }
    }

    fn update_surfaces(&mut self, changes: &[(usize, &Surface)]) {
        if let Renderer::Live(r) = self
            && !changes.is_empty()
        {
            r.update_surfaces(changes);
        }
    }

    fn frame(&mut self, camera: Camera, view: &wgpu::TextureView) {
        match self {
            Renderer::Live(r) => r.frame(camera, camera.vertical_fov_degrees, view),
            Renderer::Instanced(r) => r.frame(camera, view),
        }
    }
}

struct App {
    gpu: Option<Gpu>,
    world: World,
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
    frames_left: Option<u32>,
    all_times: Vec<f32>,
}

impl App {
    fn init(event_loop: &ActiveEventLoop, world: &World) -> Gpu {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Continuum")
                        .with_inner_size(LogicalSize::new(1280, 720)),
                )
                .expect("window"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&continuum_gpu::device_descriptor(&adapter)))
                .expect("device");
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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

        let kernel =
            MaskConeKernel::with_device(device.clone(), queue.clone(), adapter.get_info().name);
        if let Some(field) = &world.field {
            kernel.set_prefilter(1.0);
            // `--sky`: a blue-white sky (M5), baked into the view cache, sky
            // cones prefiltered 8× as coarsely. Experimental: sky texels cost
            // about 100× sun texels, and the cache's budget floor makes
            // warm-up slow (M5 gate 4).
            if world.cache && world.sky {
                kernel.set_sky(field.sky());
                kernel.set_sky_prefilter(8.0);
            }
            let start = Instant::now();
            let origin = field.origin_near_camera();
            let renderer = if world.cache {
                InstancedRenderer::with_cache(
                    &kernel,
                    format,
                    config.width,
                    config.height,
                    world.scale,
                    world.light,
                    &field.scene,
                    origin,
                    field.radiance,
                    CacheOptions::default(),
                )
            } else {
                InstancedRenderer::new(
                    &kernel,
                    format,
                    config.width,
                    config.height,
                    world.scale,
                    world.light,
                    &field.scene,
                    origin,
                    field.radiance,
                )
            };
            renderer.wait();
            println!(
                "{} (field scene): {} boulders, {} triangles, {:.0} MB on the GPU ({}), ready in {:.1} s; window {}×{}, rendering at 1/{} scale; camera rays: {}.",
                kernel.adapter_name,
                field.boulders(),
                field.triangles(),
                (renderer.scene_bytes() + renderer.cache_bytes()) as f64 / 1e6,
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
            return Gpu {
                window,
                surface,
                config,
                device,
                queue,
                renderer: Renderer::Instanced(Box::new(renderer)),
            };
        }
        if let Some(g) = &world.gltf {
            kernel.set_prefilter(1.0);
            if world.cache && world.sky {
                kernel.set_sky(g.sky);
                kernel.set_sky_prefilter(8.0);
            }
            let start = Instant::now();
            let renderer = if world.cache {
                InstancedRenderer::with_cache_lights(
                    &kernel,
                    format,
                    config.width,
                    config.height,
                    world.scale,
                    &g.lights,
                    &g.scene,
                    g.origin,
                    CacheOptions::default(),
                )
            } else {
                InstancedRenderer::new_lights(
                    &kernel,
                    format,
                    config.width,
                    config.height,
                    world.scale,
                    &g.lights,
                    &g.scene,
                    g.origin,
                )
            };
            renderer.wait();
            println!(
                "{} (glTF scene): {} triangles, {:.0} MB on the GPU ({}), ready in {:.1} s; window {}×{}, rendering at 1/{} scale; camera rays: {}.",
                kernel.adapter_name,
                g.triangles,
                (renderer.scene_bytes() + renderer.cache_bytes()) as f64 / 1e6,
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
            return Gpu {
                window,
                surface,
                config,
                device,
                queue,
                renderer: Renderer::Instanced(Box::new(renderer)),
            };
        }
        let meshes: Vec<_> = world.surfaces.iter().map(|s| s.triangles.clone()).collect();
        let first_mover = world.first_mover;
        let mut lightmap = Lightmap::new(&world.surfaces, world.texel, |m| m < first_mover)
            .with_probes(world.probes);
        let bake = kernel.bake(&mut lightmap, world.light, &meshes);
        let start = Instant::now();
        let renderer = LiveRenderer::new(
            &kernel,
            format,
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
        Gpu {
            window,
            surface,
            config,
            device,
            queue,
            renderer: Renderer::Live(Box::new(renderer)),
        }
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
            let gpu = Self::init(event_loop, &self.world);
            gpu.window.request_redraw();
            self.gpu = Some(gpu);
            self.last = Instant::now();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu
                    && size.width > 0
                    && size.height > 0
                {
                    (gpu.config.width, gpu.config.height) = (size.width, size.height);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.renderer.resize(size.width, size.height);
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
                ElementState::Pressed => {
                    if code == KeyCode::Escape {
                        event_loop.exit();
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
                    if code == KeyCode::KeyV
                        && !repeat
                        && let Some(gpu) = &mut self.gpu
                    {
                        gpu.config.present_mode = match gpu.config.present_mode {
                            wgpu::PresentMode::AutoNoVsync => wgpu::PresentMode::AutoVsync,
                            _ => wgpu::PresentMode::AutoNoVsync,
                        };
                        gpu.surface.configure(&gpu.device, &gpu.config);
                    }
                    self.keys.insert(code);
                }
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
                } else {
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
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last).as_secs_f32();
                self.last = now;
                self.update(dt.min(0.1));
                let moved = self.world.movers(self.time);
                let changes: Vec<_> = moved.iter().map(|(m, s)| (*m, s)).collect();
                let fov = self.world.fov;
                let Some(gpu) = &mut self.gpu else { return };
                gpu.renderer.update_surfaces(&changes);
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
                gpu.renderer.frame(self.camera.camera(fov), &view);
                gpu.queue.present(texture);

                self.frame_times.push(dt * 1000.0);
                self.all_times.push(dt * 1000.0);
                if self.frame_times.len() == 30 {
                    let avg = self.frame_times.iter().sum::<f32>() / 30.0;
                    let vsync = if gpu.config.present_mode == wgpu::PresentMode::AutoVsync {
                        "on"
                    } else {
                        "off"
                    };
                    gpu.window.set_title(&format!(
                        "Continuum ({}) — {:.0} fps ({avg:.1} ms) at {}×{} — vsync {vsync} — WASD/QE move, drag to look, C camera path, P pause, V vsync",
                        self.world.name,
                        1000.0 / avg,
                        gpu.config.width,
                        gpu.config.height
                    ));
                    self.frame_times.clear();
                }
                if let Some(left) = &mut self.frames_left {
                    *left = left.saturating_sub(1);
                    // Exercise a moving camera: the path if there is one.
                    if self.world.showcase.is_none() {
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
            _ => {}
        }
    }
}

fn main() {
    let arg = |name: &str| std::env::args().skip_while(|a| a != name).nth(1);
    let frames_left: Option<u32> = arg("--frames").and_then(|n| n.parse().ok());
    let world = match (arg("--gltf"), arg("--scene").as_deref()) {
        (Some(path), _) => World::gltf(
            Path::new(&path),
            !std::env::args().any(|a| a == "--no-ground"),
        ),
        (None, None | Some("showcase")) => World::showcase(),
        (None, Some("room")) => World::room(),
        (None, Some("field")) => World::field(
            arg("--boulders")
                .and_then(|n| n.parse().ok())
                .unwrap_or(1_000_000),
        ),
        (None, Some(other)) => {
            eprintln!("Unknown scene {other:?}: use showcase, room or field, or --gltf <file>.");
            std::process::exit(2);
        }
    };
    let mut world = world;
    world.scale = arg("--scale").and_then(|n| n.parse().ok()).unwrap_or(1);
    world.cache = !std::env::args().any(|a| a == "--no-cache");
    world.sky = std::env::args().any(|a| a == "--sky");
    let path = (frames_left.is_some() && world.showcase.is_some()).then_some(0.0);
    let camera = FlyCamera::looking_at(world.start.0, world.start.1);
    let mut app = App {
        gpu: None,
        world,
        time: 0.0,
        paused: false,
        path,
        camera,
        keys: HashSet::new(),
        drag: None,
        last: Instant::now(),
        frame_times: Vec::new(),
        frames_left,
        all_times: Vec::new(),
    };
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut app).expect("event loop");
}
