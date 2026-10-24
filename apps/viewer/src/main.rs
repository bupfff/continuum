//! Continuum viewer: fly through the test room in real time.
//!
//! Direct light from the area light is baked into a world-space lighting cache
//! (one mask cone per texel) at startup; each frame is a camera ray per pixel
//! plus a cache lookup, on the GPU. Three boxes move through the room: they are
//! shaded per pixel, and each frame re-bakes exactly the cache blocks they can
//! reach, so their soft shadows stay exact. Probes on static surfaces add one
//! bounce of indirect light, gathered each frame through cached G-buffers.
//!
//! Controls: WASD move, Q/E down/up, drag with the left mouse button to look,
//! Shift to move faster, P to pause the boxes, V to toggle vsync, Esc to quit.
//! `--frames N` renders N frames, prints timings, and exits.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use continuum::glam::Vec3;
use continuum_gpu::wgpu;
use continuum_gpu::{Camera, Lightmap, LiveRenderer, MaskConeKernel, RectLight};
use room3d::{movers, render_scene};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const TEXEL: f32 = 0.005;
/// Spacing of indirect-light probes.
const PROBES: f32 = 0.04;
const FOV: f32 = 60.0;

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

    fn camera(&self) -> Camera {
        Camera {
            eye: self.position,
            target: self.position + self.forward(),
            vertical_fov_degrees: FOV,
        }
    }
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: LiveRenderer,
}

struct App {
    gpu: Option<Gpu>,
    /// Index of the first moving box.
    first_mover: usize,
    /// Animation time in 60 Hz frames.
    time: f32,
    paused: bool,
    camera: FlyCamera,
    keys: HashSet<KeyCode>,
    drag: Option<PhysicalPosition<f64>>,
    last: Instant,
    frame_times: Vec<f32>,
    frames_left: Option<u32>,
    all_times: Vec<f32>,
}

impl App {
    fn init(event_loop: &ActiveEventLoop) -> (Gpu, usize) {
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("viewer"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
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

        let mut scene = render_scene();
        let first_mover = scene.add_movers(0.0);
        let l = &scene.room.light;
        let light = RectLight {
            corner: l.corner,
            edge_u: l.edge_u,
            edge_v: l.edge_v,
            one_sided: false,
        };
        let kernel =
            MaskConeKernel::with_device(device.clone(), queue.clone(), adapter.get_info().name);
        let mut lightmap = Lightmap::from_meshes(&scene.room.meshes, TEXEL, |m| m < first_mover)
            .with_probes(PROBES);
        let bake = kernel.bake(&mut lightmap, light, &scene.room.meshes);
        println!(
            "{}: baked {} texels in {:.1} ms; rendering at {}×{}.",
            kernel.adapter_name,
            lightmap.texels(),
            bake.as_secs_f64() * 1000.0,
            config.width,
            config.height
        );
        let renderer = LiveRenderer::new(
            &kernel,
            format,
            config.width,
            config.height,
            light,
            &scene.room.meshes,
            &scene.albedo,
            scene.radiance,
            &lightmap,
        );
        (
            Gpu {
                window,
                surface,
                config,
                device,
                queue,
                renderer,
            },
            first_mover,
        )
    }

    fn update(&mut self, dt: f32) {
        let speed = if self.keys.contains(&KeyCode::ShiftLeft) {
            1.5
        } else {
            0.5
        } * dt;
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
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let (gpu, first_mover) = Self::init(event_loop);
            gpu.window.request_redraw();
            self.gpu = Some(gpu);
            self.first_mover = first_mover;
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
                let first_mover = self.first_mover;
                let changes: Vec<_> = movers(self.time)
                    .into_iter()
                    .enumerate()
                    .map(|(k, mesh)| (first_mover + k, mesh))
                    .collect();
                let Some(gpu) = &mut self.gpu else { return };
                gpu.renderer.update_meshes(&changes);
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
                gpu.renderer.frame(self.camera.camera(), FOV, &view);
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
                        "Continuum — {:.0} fps ({avg:.1} ms) at {}×{} — vsync {vsync} — WASD/QE move, drag to look, P pause, V vsync",
                        1000.0 / avg,
                        gpu.config.width,
                        gpu.config.height
                    ));
                    self.frame_times.clear();
                }
                if let Some(left) = &mut self.frames_left {
                    *left = left.saturating_sub(1);
                    // Look around a little so the run exercises a moving camera.
                    self.camera.yaw += 0.01;
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
    let scene = render_scene();
    let frames_left = std::env::args()
        .skip_while(|a| a != "--frames")
        .nth(1)
        .and_then(|n| n.parse().ok());
    let mut app = App {
        gpu: None,
        first_mover: 0,
        time: 0.0,
        paused: false,
        camera: FlyCamera::looking_at(scene.eye, scene.target),
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
