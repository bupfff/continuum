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
- "time": seconds into the file's animation (its first clip, as
  `compare_skinned_pose.py` poses it), or absent for the rest pose;
- "ground": {"z", "half", "albedo"}: a Lambertian square 2 × half across
  at height z under the origin (the viewer's ground), or absent;
- "cull_back_faces" (true by default): camera rays pass through the back
  faces of single-sided materials, as glTF and Continuum cull them;
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

# "clay" (a diagnostic): every surface Lambertian with albedo 0.5, its alpha
# (cutouts) kept, without normal maps, specular or emission.
if job.get("clay"):
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        outs = [n for n in nodes if n.type == "OUTPUT_MATERIAL"]
        out = next((n for n in outs if n.is_active_output), outs[0] if outs else None)
        if out is None:
            continue
        principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        surface = diffuse.outputs["BSDF"]
        if principled is not None:
            alpha = principled.inputs["Alpha"]
            if alpha.is_linked or alpha.default_value < 1.0:
                clear = nodes.new("ShaderNodeBsdfTransparent")
                mix = nodes.new("ShaderNodeMixShader")
                if alpha.is_linked:
                    links.new(alpha.links[0].from_socket, mix.inputs["Fac"])
                else:
                    mix.inputs["Fac"].default_value = alpha.default_value
                links.new(clear.outputs[0], mix.inputs[1])
                links.new(surface, mix.inputs[2])
                surface = mix.outputs[0]
        links.new(surface, out.inputs["Surface"])

# "nospec" (a diagnostic): every Principled BSDF without its specular
# layer and metals as diffuse (Specular IOR Level 0, Metallic 0, no coat or
# sheen); textures kept.
if job.get("nospec"):
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for n in mat.node_tree.nodes:
            if n.type != "BSDF_PRINCIPLED":
                continue
            for name, value in (("Specular IOR Level", 0.0), ("Metallic", 0.0), ("Coat Weight", 0.0), ("Sheen Weight", 0.0)):
                if name in n.inputs:
                    socket = n.inputs[name]
                    for link in list(socket.links):
                        mat.node_tree.links.remove(link)
                    socket.default_value = value

# glTF culls a single-sided material's back faces (doubleSided false), and
# Continuum's camera rays do; Cycles never culls them. So camera rays pass
# through such back faces here ("cull_back_faces", true by default); other
# rays (shadows, bounces) still meet them. Without it, Sponza's upper
# gallery showed its curtains' backs where Continuum sees the gallery.
if job.get("cull_back_faces", True):
    for mat in bpy.data.materials:
        if not mat.use_nodes or not getattr(mat, "use_backface_culling", False):
            continue
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        outs = [n for n in nodes if n.type == "OUTPUT_MATERIAL"]
        out = next((n for n in outs if n.is_active_output), outs[0] if outs else None)
        if out is None or not out.inputs["Surface"].is_linked:
            continue
        surface = out.inputs["Surface"].links[0].from_socket
        geometry = nodes.new("ShaderNodeNewGeometry")
        path = nodes.new("ShaderNodeLightPath")
        both = nodes.new("ShaderNodeMath")
        both.operation = "MULTIPLY"
        links.new(geometry.outputs["Backfacing"], both.inputs[0])
        links.new(path.outputs["Is Camera Ray"], both.inputs[1])
        clear = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        links.new(both.outputs[0], mix.inputs["Fac"])
        links.new(surface, mix.inputs[1])
        links.new(clear.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], out.inputs["Surface"])

if "time" in job:
    # Every animation plays, as glTF means (see compare_skinned_pose.py):
    # the importer leaves some only as muted NLA tracks.
    for ob in scene.objects:
        ids = [ob]
        if ob.type == "MESH" and ob.data.shape_keys:
            ids.append(ob.data.shape_keys)
        for idb in ids:
            ad = idb.animation_data
            if not ad or not ad.nla_tracks or ad.action is not None:
                continue
            strips = [s for t in ad.nla_tracks for s in t.strips]
            if strips:
                ad.action = strips[0].action
                if hasattr(ad, "action_slot") and getattr(strips[0], "action_slot", None):
                    ad.action_slot = strips[0].action_slot
    fps = scene.render.fps / scene.render.fps_base
    # The importer puts time t at frame t × fps.
    frame = job["time"] * fps
    scene.frame_set(int(frame), subframe=frame - int(frame))

if job.get("ground"):
    g = job["ground"]
    h, z = g["half"], g["z"]
    mesh = bpy.data.meshes.new("ground")
    mesh.from_pydata(
        [(origin.x + x, origin.y + y, origin.z + z) for x, y in ((-h, -h), (h, -h), (h, h), (-h, h))],
        [],
        [(0, 1, 2, 3)],
    )
    mat = bpy.data.materials.new("ground")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for n in list(nodes):
        if n.type != "OUTPUT_MATERIAL":
            nodes.remove(n)
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (*([g["albedo"]] * 3), 1.0)
    mat.node_tree.links.new(
        diffuse.outputs["BSDF"],
        next(n for n in nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"],
    )
    mesh.materials.append(mat)
    ground = bpy.data.objects.new("ground", mesh)
    scene.collection.objects.link(ground)

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
