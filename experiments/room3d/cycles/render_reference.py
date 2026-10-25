"""Render a Continuum scene with Cycles as an independent reference.

Run by `room3d::cycles` in a background Blender:

    Blender -b --factory-startup --python render_reference.py -- job.json

The job (JSON) names an OBJ with one object per mesh (`mesh_<index>`, Z up,
per-vertex normals), the albedo of each mesh, a rectangular light, a pinhole
camera, the resolution and sample count. Materials are Lambertian (Diffuse
BSDF, roughness 0). The light is a one-sided emitting quad facing
`edge_u x edge_v`, visible to the camera. An optional uniform sky (`sky`, linear
RGB) is the world color. Light paths: direct light plus at
most `bounces` diffuse bounces. Each pixel integrates its area
(a one-pixel box filter: Blender 5.2's Cycles applies at least one pixel even
when asked for less, as M5 gate 1 measured), with no denoising or clamping. The linear RGB result is
written top row first as little-endian float32 to `out`.
"""

import json
import math
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

bpy.ops.wm.obj_import(filepath=job["obj"], forward_axis="Y", up_axis="Z")


def diffuse(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.0
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


meshes = 0
for obj in list(scene.objects):
    if obj.type != "MESH":
        continue
    m = int(obj.name.split("_")[1].split(".")[0])
    obj.data.materials.clear()
    obj.data.materials.append(diffuse(f"albedo_{m}", job["albedo"][m]))
    meshes += 1

# The light: a quad facing edge_u x edge_v that emits only from its front.
light = job["light"]
c, u, v = (Vector(light[k]) for k in ("corner", "edge_u", "edge_v"))
mesh = bpy.data.meshes.new("light")
mesh.from_pydata([c, c + u, c + u + v, c + v], [], [(0, 1, 2, 3)])
emitter = bpy.data.objects.new("light", mesh)
scene.collection.objects.link(emitter)
mat = bpy.data.materials.new("emission")
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
out = nodes.new("ShaderNodeOutputMaterial")
front = nodes.new("ShaderNodeEmission")
front.inputs["Strength"].default_value = light["radiance"]
back = nodes.new("ShaderNodeEmission")
back.inputs["Strength"].default_value = 0.0
geometry = nodes.new("ShaderNodeNewGeometry")
mix = nodes.new("ShaderNodeMixShader")
links = mat.node_tree.links
links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
links.new(front.outputs["Emission"], mix.inputs[1])
links.new(back.outputs["Emission"], mix.inputs[2])
links.new(mix.outputs["Shader"], out.inputs["Surface"])
if hasattr(mat.cycles, "emission_sampling"):
    mat.cycles.emission_sampling = "FRONT"
mesh.materials.append(mat)

# Camera: Z up, vertical field of view in degrees.
cam = job["camera"]
eye, target = Vector(cam["eye"]), Vector(cam["target"])
forward = (target - eye).normalized()
right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
up = right.cross(forward)
data = bpy.data.cameras.new("camera")
data.sensor_fit = "VERTICAL"
data.angle_y = math.radians(cam["fov"])
data.clip_start = 0.01
data.clip_end = 1e6
camera = bpy.data.objects.new("camera", data)
scene.collection.objects.link(camera)
camera.matrix_world = Matrix(
    (
        (right.x, up.x, -forward.x, eye.x),
        (right.y, up.y, -forward.y, eye.y),
        (right.z, up.z, -forward.z, eye.z),
        (0.0, 0.0, 0.0, 1.0),
    )
)
scene.camera = camera

# The world: black, or a uniform sky of the given radiance (linear RGB).
sky = job.get("sky", [0.0, 0.0, 0.0])
world = bpy.data.worlds.new("sky")
world.use_nodes = True
background = world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = (*sky, 1.0)
background.inputs["Strength"].default_value = 1.0 if any(sky) else 0.0
scene.world = world

scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
device = "CPU"
for kind in ("METAL", "OPTIX", "CUDA", "HIP"):
    try:
        prefs.compute_device_type = kind
    except TypeError:
        continue
    prefs.get_devices()
    gpus = [d for d in prefs.devices if d.type == kind]
    if gpus:
        for d in prefs.devices:
            d.use = d.type == kind
        device = kind
        break
scene.cycles.device = "GPU" if device != "CPU" else "CPU"
cy = scene.cycles
cy.samples = job["samples"]
cy.use_adaptive_sampling = False
cy.use_denoising = False
cy.max_bounces = job["bounces"]
cy.diffuse_bounces = job["bounces"]
cy.glossy_bounces = 0
cy.transmission_bounces = 0
cy.volume_bounces = 0
cy.transparent_max_bounces = 0
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = 0.0
if hasattr(cy, "light_sampling_threshold"):
    cy.light_sampling_threshold = 0.0
cy.pixel_filter_type = "BOX"
cy.filter_width = 1.0
cy.seed = job["seed"]
scene.render.resolution_x = job["width"]
scene.render.resolution_y = job["height"]
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = job["exr"]

start = time.time()
bpy.ops.render.render(write_still=True)
seconds = time.time() - start
image = bpy.data.images.load(job["exr"])
w, h = job["width"], job["height"]
pixels = np.array(image.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
np.ascontiguousarray(pixels).astype("<f4").tofile(job["out"])
print(f"CONTINUUM_REFERENCE meshes={meshes} device={device} seconds={seconds:.2f}")
