//! Continuum in Bevy: a 3D camera's view rendered with mask cones (exact soft
//! shadows from area lights, a lighting cache kept across frames) instead of
//! Bevy's own passes.
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_continuum::{ContinuumCamera, ContinuumPlugin, ContinuumScene};
//! # fn scene() -> ContinuumScene { unimplemented!() }
//!
//! App::new()
//!     .add_plugins((DefaultPlugins, ContinuumPlugin))
//!     .insert_resource(scene())
//!     .add_systems(Startup, |mut commands: Commands| {
//!         commands.spawn((
//!             Camera3d::default(),
//!             bevy::camera::Hdr,
//!             ContinuumCamera,
//!             Transform::from_xyz(0.0, 2.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
//!         ));
//!     })
//!     .run();
//! ```
//!
//! - **Coordinates.** Continuum is Z-up, and Bevy is Y-up. A point `(x, y, z)`
//!   in Bevy's world is `origin + (x, −z, y)` in the Continuum scene's world
//!   (the same change of axes as glTF import). So a scene loaded with
//!   `continuum_gltf::load` shows where Bevy would place the same file.
//! - **What shows.** The camera's view is Continuum's image: Bevy's own meshes,
//!   lights and materials do not appear in it. Bevy's post-processing (tone
//!   mapping and the rest) and UI apply on top. An HDR camera
//!   (`bevy::camera::Hdr`) gets linear light; others get Continuum's
//!   tone curve.
//! - **What can change.** The camera moves freely. Light colors and
//!   intensities change for free ([`ContinuumScene::lights`]'s radiance, with
//!   the same lights in the same places). Instances listed in the options'
//!   `movable` follow the entities marked [`ContinuumInstance`] (rotation and
//!   translation; the instance keeps its scale), re-baking only what each
//!   move changes. Anything else (geometry, light positions, options)
//!   rebuilds the renderer when the resource changes.
//! - **Limits.** The renderer's limits apply (`docs/GUIDE.md`); cameras do
//!   not roll (Continuum's up is always +Z); one renderer per camera.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::app::PostUpdate;
use bevy::core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy::ecs::prelude::*;
use bevy::math::Vec3 as BevyVec3;
use bevy::prelude::{App, Camera, Plugin};
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::{
    RenderAdapterInfo, RenderContext, RenderDevice, RenderQueue, ViewQuery,
};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::transform::TransformSystems;
use bevy::transform::components::GlobalTransform;
use continuum::glam::{DVec3, Mat3, Vec3};
use continuum_gpu::{
    Camera as ContinuumView, InstancedScene, Light, MaskConeKernel, Renderer, RendererOptions,
};

pub use continuum_gpu;

/// Renders the views of cameras marked [`ContinuumCamera`] from the
/// [`ContinuumScene`] resource.
pub struct ContinuumPlugin;

impl Plugin for ContinuumPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<ContinuumCamera>::default(),
            ExtractResourcePlugin::<ContinuumScene>::default(),
            ExtractResourcePlugin::<Movers>::default(),
        ))
        .init_resource::<Movers>()
        .add_systems(
            PostUpdate,
            collect_movers.after(TransformSystems::Propagate),
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<Views>()
            .add_systems(Render, prepare_views.in_set(RenderSystems::Prepare))
            .add_systems(Core3d, draw_view.in_set(Core3dSystems::EarlyPostProcess));
    }
}

/// Marks a 3D camera whose view Continuum renders.
#[derive(Component, Clone, Copy, Default, Debug, ExtractComponent)]
#[extract_component_filter(With<Camera>)]
#[extract_app(RenderApp)]
pub struct ContinuumCamera;

/// The scene Continuum renders, in its own Z-up world (see the crate's
/// documentation for how it lines up with Bevy's).
#[derive(Resource, Clone)]
pub struct ContinuumScene {
    /// Geometry and its placements, shared rather than copied each time the
    /// resource is extracted.
    pub scene: Arc<InstancedScene>,
    /// The lights, relative to `origin` (at most `continuum_gpu::MAX_LIGHTS`).
    pub lights: Vec<Light>,
    /// Where Bevy's world origin lies in the scene's world (Continuum's
    /// `glam`, which may differ from Bevy's).
    pub origin: DVec3,
    /// How the renderer renders (scale, cache, sky, ...).
    pub options: RendererOptions,
}

/// Moves instance `.0` of the [`ContinuumScene`] (one listed in its
/// `options.movable`) with this entity's `GlobalTransform`: its rotation
/// and translation in Bevy's world (see the crate's coordinates). The
/// instance keeps its own scale. The cache re-bakes what the move changes
/// within the frame (M9 gate 1).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuumInstance(pub usize);

/// Each [`ContinuumInstance`]'s pose this frame, in Continuum's world:
/// instance, rotation (column-major), position.
#[derive(Resource, Clone, Default)]
struct Movers(Vec<(usize, [f32; 9], [f64; 3])>);

impl ExtractResource<RenderApp> for Movers {
    type Source = Movers;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Bevy's Y-up rotation as Continuum's Z-up one (the same change of axes
/// as positions): `C · R · C⁻¹` with `C` taking (x, y, z) to (x, −z, y).
fn rotation_to_z_up(r: [f32; 9]) -> Mat3 {
    let c = Mat3::from_cols(Vec3::X, Vec3::Z, -Vec3::Y);
    c * Mat3::from_cols_array(&r) * c.transpose()
}

/// Gathers the [`ContinuumInstance`]s' poses after transforms propagate.
fn collect_movers(
    instances: Query<(&ContinuumInstance, &GlobalTransform)>,
    scene: Option<Res<ContinuumScene>>,
    mut movers: ResMut<Movers>,
) {
    let origin = scene.map_or(DVec3::ZERO, |s| s.origin);
    movers.0.clear();
    for (instance, transform) in &instances {
        let (_, rotation, translation) = transform.to_scale_rotation_translation();
        let r = bevy::math::Mat3::from_quat(rotation).to_cols_array();
        let p = to_z_up(translation);
        let position = origin + DVec3::new(p.x as f64, p.y as f64, p.z as f64);
        movers.0.push((instance.0, r, position.to_array()));
    }
}

impl ExtractResource<RenderApp> for ContinuumScene {
    type Source = ContinuumScene;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// A camera's renderer and what it was built from.
struct View {
    renderer: Renderer,
    scene: Arc<InstancedScene>,
    lights: usize,
    format: bevy::render::render_resource::TextureFormat,
    size: (u32, u32),
    /// The lights' radiance as last set.
    radiance: Vec<Vec3>,
    /// Movable instances' poses as last set.
    poses: HashMap<usize, ([f32; 9], [f64; 3])>,
}

/// One renderer per camera, by main-world entity. (The kernel's settings
/// are not `Sync`; a mutex makes the resource shareable.)
#[derive(Resource, Default)]
struct Views(Mutex<HashMap<MainEntity, View>>);

/// Bevy's Y-up world to Continuum's Z-up one: (x, y, z) → (x, −z, y).
fn to_z_up(v: BevyVec3) -> Vec3 {
    Vec3::new(v.x, -v.z, v.y)
}

/// The Continuum camera matching a Bevy view (relative to the scene's
/// origin), with its vertical field of view from the projection.
fn view_camera(view: &ExtractedView) -> ContinuumView {
    let world = view.world_from_view;
    let eye = to_z_up(world.translation());
    let forward = to_z_up(world.forward().as_vec3());
    // A perspective projection's [1][1] is 1 / tan(fov / 2).
    let f = view.clip_from_view.y_axis.y;
    let fov = if f > 0.0 {
        2.0 * (1.0 / f).atan()
    } else {
        std::f32::consts::FRAC_PI_4
    };
    ContinuumView {
        eye,
        target: eye + forward,
        vertical_fov_degrees: fov.to_degrees(),
    }
}

/// Builds or updates each Continuum camera's renderer and submits its
/// frame's compute work (lighting, visibility, shading); `draw_view` draws
/// the result into the view later in Bevy's frame.
#[allow(clippy::too_many_arguments)]
fn prepare_views(
    views: Res<Views>,
    scene: Option<Res<ContinuumScene>>,
    movers: Option<Res<Movers>>,
    cameras: Query<(&MainEntity, &ExtractedView, &ExtractedCamera), With<ContinuumCamera>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    adapter: Res<RenderAdapterInfo>,
) {
    let Some(scene) = scene else { return };
    let mut views = views.0.lock().expect("views");
    views.retain(|entity, _| cameras.iter().any(|(e, ..)| e == entity));
    for (entity, view, camera) in &cameras {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        let size = (size.x.max(1), size.y.max(1));
        let format = view.target_format;
        let stale = views.get(entity).is_none_or(|v| {
            !Arc::ptr_eq(&v.scene, &scene.scene)
                || v.lights != scene.lights.len()
                || v.format != format
        });
        if stale {
            let kernel = MaskConeKernel::with_device(
                device.wgpu_device().clone(),
                (**queue).clone(),
                adapter.name.clone(),
            );
            let renderer = Renderer::new(
                kernel,
                format,
                size,
                &scene.scene,
                scene.origin,
                &scene.lights,
                scene.options.clone(),
            );
            views.insert(
                *entity,
                View {
                    renderer,
                    scene: scene.scene.clone(),
                    lights: scene.lights.len(),
                    format,
                    size,
                    radiance: scene.lights.iter().map(|l| l.radiance).collect(),
                    poses: HashMap::new(),
                },
            );
        }
        let v = views.get_mut(entity).expect("just inserted");
        if v.size != size {
            v.renderer.resize(size.0, size.1);
            v.size = size;
        }
        for (k, light) in scene.lights.iter().enumerate() {
            if v.radiance[k] != light.radiance {
                v.renderer.set_light_radiance(k, light.radiance);
                v.radiance[k] = light.radiance;
            }
        }
        for &(i, r, p) in movers.iter().flat_map(|m| m.0.iter()) {
            if scene.options.movable.contains(&i) && v.poses.get(&i) != Some(&(r, p)) {
                v.renderer
                    .set_transform(i, rotation_to_z_up(r), DVec3::from_array(p));
                v.poses.insert(i, (r, p));
            }
        }
        v.renderer.compute(view_camera(view));
    }
}

/// Draws a Continuum camera's frame into its view's main texture, after
/// Bevy's main passes and before its post-processing.
fn draw_view(
    view: ViewQuery<(&MainEntity, &ViewTarget), With<ContinuumCamera>>,
    views: Res<Views>,
    mut ctx: RenderContext,
) {
    let (entity, target) = view.into_inner();
    let views = views.0.lock().expect("views");
    let Some(v) = views.get(entity) else { return };
    v.renderer
        .encode_display(ctx.command_encoder(), target.main_texture_view());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn about Bevy's up (y) is the same turn about Continuum's up
    /// (z), and the change of axes maps a rotated Bevy vector onto the
    /// Continuum rotation of the mapped vector.
    #[test]
    fn rotations_change_axes_like_positions() {
        let angle = 0.7f32;
        let bevy_turn = bevy::math::Mat3::from_rotation_y(angle);
        let ours = rotation_to_z_up(bevy_turn.to_cols_array());
        let want = Mat3::from_rotation_z(angle);
        assert!(ours.abs_diff_eq(want, 1e-6), "{ours} against {want}");
        let r = bevy::math::Mat3::from_euler(bevy::math::EulerRot::XYZ, 0.3, -1.1, 2.0);
        let v = BevyVec3::new(0.2, -0.5, 0.9);
        let mapped = rotation_to_z_up(r.to_cols_array()) * to_z_up(v);
        let direct = to_z_up(r * v);
        assert!(
            (mapped - direct).length() < 1e-6,
            "{mapped} against {direct}"
        );
    }
}
