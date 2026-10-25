"""Render an instanced Continuum scene with Cycles as an independent reference.

Run by `room3d::cycles::render_instanced` in a background Blender:

    Blender -b --factory-startup --python render_instanced_reference.py -- job.json

The job (JSON) names one OBJ per geometry (`geometries`, object space, Z up,
per-vertex normals) and a binary file of instances (`instances`: per instance
17 little-endian float32 values: geometry index, the 3x3 rotation row by
row, the uniform scale, the position, and the albedo). Each geometry is
instanced on a point cloud through geometry nodes (Instance on Points), with
per-instance rotation, scale and albedo, so a million instances stay a
million instances (Cycles instancing), not a flattened mesh. Everything else
(lights, sky, camera, sampling, output) is as in `render_reference.py`.
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

raw = np.fromfile(job["instances"], dtype="<f4").reshape(-1, 17)
geometry_of = raw[:, 0].astype(np.int64)

for g, path in enumerate(job["geometries"]):
    mine = raw[geometry_of == g]
    if len(mine) == 0:
        continue
    before = set(scene.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    source = [o for o in scene.objects if o not in before and o.type == "MESH"][0]
    source.name = f"geometry_{g}"
    source.hide_render = True
    source.hide_viewport = True

    # Material: Lambertian, albedo from the instance's attribute.
    mat = bpy.data.materials.new(f"albedo_{g}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Roughness"].default_value = 0.0
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "INSTANCER"
    attr.attribute_name = "albedo"
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Color"])
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    source.data.materials.clear()
    source.data.materials.append(mat)

    # The point cloud: one vertex per instance, with rotation (Euler XYZ),
    # scale and albedo attributes.
    n = len(mine)
    points = bpy.data.meshes.new(f"points_{g}")
    points.vertices.add(n)
    points.vertices.foreach_set("co", mine[:, 11:14].astype(np.float32).ravel())
    eulers = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        r = mine[k, 1:10].reshape(3, 3)
        eulers[k] = Matrix(r.tolist()).to_euler("XYZ")[:]
    a = points.attributes.new("rot", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", eulers.ravel())
    a = points.attributes.new("scl", "FLOAT", "POINT")
    a.data.foreach_set("value", mine[:, 10].astype(np.float32))
    a = points.attributes.new("albedo", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([mine[:, 14:17], np.ones((n, 1))], axis=1).astype(np.float32)
    a.data.foreach_set("color", rgba.ravel())
    holder = bpy.data.objects.new(f"instances_{g}", points)
    scene.collection.objects.link(holder)

    tree = bpy.data.node_groups.new(f"instance_{g}", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = tree.nodes.new("NodeGroupInput")
    gout = tree.nodes.new("NodeGroupOutput")
    inst = tree.nodes.new("GeometryNodeInstanceOnPoints")
    info = tree.nodes.new("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = source
    info.inputs["As Instance"].default_value = True
    info.transform_space = "ORIGINAL"
    rot = tree.nodes.new("GeometryNodeInputNamedAttribute")
    rot.data_type = "FLOAT_VECTOR"
    rot.inputs["Name"].default_value = "rot"
    euler = tree.nodes.new("FunctionNodeEulerToRotation")
    scl = tree.nodes.new("GeometryNodeInputNamedAttribute")
    scl.data_type = "FLOAT"
    scl.inputs["Name"].default_value = "scl"
    links = tree.links
    links.new(gin.outputs["Geometry"], inst.inputs["Points"])
    links.new(info.outputs["Geometry"], inst.inputs["Instance"])
    links.new(rot.outputs["Attribute"], euler.inputs["Euler"])
    links.new(euler.outputs["Rotation"], inst.inputs["Rotation"])
    links.new(scl.outputs["Attribute"], inst.inputs["Scale"])
    links.new(inst.outputs["Instances"], gout.inputs["Geometry"])
    mod = holder.modifiers.new("instances", "NODES")
    mod.node_group = tree

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
print(f"CONTINUUM_REFERENCE instances={len(raw)} device={device} seconds={seconds:.2f}")
