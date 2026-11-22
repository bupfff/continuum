"""A glTF file rendered by Cycles through Blender's own glTF importer (an
independent reading of the file): the reference for Continuum's renders of
real assets.

Run as `blender -b --factory-startup -noaudio --python render_gltf_reference.py
-- job.json`. The job (JSON):

- "gltf": the file; "origin": the point Continuum's coordinates are relative
  to (Blender's importer uses the same Z-up axes: glTF's x, -z, y);
- "camera": {"eye", "target", "fov"} (vertical, degrees), relative to the
  origin, Z up;
- "size": [width, height]; "samples"; "bounces";
- "environment": an equirectangular HDR image or null, "strength" its
  factor; "sky": a uniform sky radiance [r, g, b] when there is none;
- "suns": [{"toward": unit vector to the sun, "irradiance": [r, g, b],
  "angle": degrees}];
- "distribution": "GGX" or "MULTI_GGX" for every Principled BSDF (else
  the importer's, Multiscatter GGX); "thin_glass" (true by default):
  transmission without a volume as thin glass, as glTF defines it;
- "out": where to write the linear image, float32 RGB rows from the top.
"""

import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
scene = bpy.context.scene
for obj in list(scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

bpy.ops.import_scene.gltf(filepath=job["gltf"])
# "distribution": the importer's Multiscatter GGX (Blender's default), or
# "GGX" (single scattering, glTF's own model).
if job.get("distribution"):
    for m in bpy.data.materials:
        for n in (m.node_tree.nodes if m.node_tree else []):
            if n.type == "BSDF_PRINCIPLED":
                n.distribution = job["distribution"]
origin = Vector(job["origin"])


def thin_glass(mat, bsdf):
    """glTF's transmission without a volume as thin glass (as M16's reference,
    `render_instanced_reference.py`): the surface with its transmission off,
    mixed by the transmission factor with its specular layer alone plus a
    Transparent BSDF passing the base colour times one less Schlick's
    Fresnel (F0 0.04, of |N·I|): light goes straight through, and shadow
    rays pass tinted. Blender's importer makes it a solid refracting surface
    with no exit instead."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    t = bsdf.inputs["Transmission Weight"].default_value
    out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    base = bsdf.inputs["Base Color"]
    if base.is_linked:
        base_socket = base.links[0].from_socket
    else:
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = base.default_value
        base_socket = rgb.outputs[0]
    bsdf.inputs["Transmission Weight"].default_value = 0.0
    spec = nodes.new("ShaderNodeBsdfPrincipled")
    spec.distribution = "GGX"
    spec.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    spec.inputs["Metallic"].default_value = 0.0
    spec.inputs["Roughness"].default_value = bsdf.inputs["Roughness"].default_value
    spec.inputs["IOR"].default_value = 1.5
    geometry = nodes.new("ShaderNodeNewGeometry")
    cosine = nodes.new("ShaderNodeVectorMath")
    cosine.operation = "DOT_PRODUCT"
    links.new(geometry.outputs["Normal"], cosine.inputs[0])
    links.new(geometry.outputs["Incoming"], cosine.inputs[1])
    magnitude = nodes.new("ShaderNodeMath")
    magnitude.operation = "ABSOLUTE"
    links.new(cosine.outputs["Value"], magnitude.inputs[0])
    grazing = nodes.new("ShaderNodeMath")
    grazing.operation = "SUBTRACT"
    grazing.inputs[0].default_value = 1.0
    links.new(magnitude.outputs[0], grazing.inputs[1])
    fifth = nodes.new("ShaderNodeMath")
    fifth.operation = "POWER"
    fifth.inputs[1].default_value = 5.0
    links.new(grazing.outputs[0], fifth.inputs[0])
    rest = nodes.new("ShaderNodeMath")
    rest.operation = "SUBTRACT"
    rest.inputs[0].default_value = 1.0
    links.new(fifth.outputs[0], rest.inputs[1])
    unreflected = nodes.new("ShaderNodeMath")
    unreflected.operation = "MULTIPLY"
    unreflected.inputs[1].default_value = 0.96
    links.new(rest.outputs[0], unreflected.inputs[0])
    tint = nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0
    links.new(base_socket, tint.inputs["A"])
    links.new(unreflected.outputs[0], tint.inputs["B"])
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    links.new(tint.outputs["Result"], transparent.inputs["Color"])
    through = nodes.new("ShaderNodeAddShader")
    links.new(spec.outputs["BSDF"], through.inputs[0])
    links.new(transparent.outputs["BSDF"], through.inputs[1])
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = t
    links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    links.new(through.outputs["Shader"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])


# "thin_glass": transmission without a volume as glTF defines it (default).
if job.get("thin_glass", True):
    for m in bpy.data.materials:
        if not m.node_tree:
            continue
        out = next((n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None or out.inputs["Volume"].is_linked:
            continue
        for n in list(m.node_tree.nodes):
            if (
                n.type == "BSDF_PRINCIPLED"
                and n.inputs["Transmission Weight"].default_value > 0.0
                and not n.inputs["Transmission Weight"].is_linked
            ):
                thin_glass(m, n)
                break

cam = job["camera"]
eye = Vector(cam["eye"]) + origin
target = Vector(cam["target"]) + origin
forward = (target - eye).normalized()
right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
up = right.cross(forward)
data = bpy.data.cameras.new("reference camera")
data.sensor_fit = "VERTICAL"
data.angle_y = math.radians(cam["fov"])
data.clip_start = 0.001
data.clip_end = 1e6
camera = bpy.data.objects.new("reference camera", data)
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

for i, sun in enumerate(job.get("suns", [])):
    light = bpy.data.lights.new(f"sun{i}", type="SUN")
    rgb = sun["irradiance"]
    peak = max(rgb) or 1.0
    light.color = [c / peak for c in rgb]
    light.energy = peak
    light.angle = math.radians(sun.get("angle", 0.5))
    obj = bpy.data.objects.new(f"sun{i}", light)
    scene.collection.objects.link(obj)
    # A sun shines along its local -Z.
    obj.rotation_euler = Vector(sun["toward"]).to_track_quat("Z", "Y").to_euler()

world = bpy.data.worlds.new("reference world")
world.use_nodes = True
background = world.node_tree.nodes["Background"]
if job.get("environment"):
    env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(job["environment"])
    env.interpolation = "Linear"
    env.projection = "EQUIRECTANGULAR"
    world.node_tree.links.new(env.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = job.get("strength", 1.0)
else:
    sky = job.get("sky", [0.0, 0.0, 0.0])
    background.inputs["Color"].default_value = (*sky, 1.0)
    background.inputs["Strength"].default_value = 1.0
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
cy.max_bounces = job["bounces"] + 8
cy.diffuse_bounces = job["bounces"]
cy.glossy_bounces = job["bounces"]
cy.transmission_bounces = 8
cy.transparent_max_bounces = 64
cy.volume_bounces = 0
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = 0.0
scene.render.resolution_x, scene.render.resolution_y = job["size"]
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
exr = job["out"] + ".exr"
scene.render.filepath = exr
bpy.ops.render.render(write_still=True)

image = bpy.data.images.load(exr)
w, h = image.size
px = list(image.pixels)
out = bytearray()
import struct

for y in range(h - 1, -1, -1):
    row = px[4 * w * y : 4 * w * (y + 1)]
    for x in range(w):
        out += struct.pack("<3f", row[4 * x], row[4 * x + 1], row[4 * x + 2])
open(job["out"], "wb").write(out)
print("reference written", w, h)
