# Makes a small skinned, morphing glTF test asset for M12 gate 2, with
# Blender's own armature and shape keys (no download): a vertical tube of
# 32 segments by 24 rings, 2 m tall and 0.3 m in radius, skinned to a chain
# of three bones with automatic weights, with one shape key ("bulge") that
# swells its middle; a 48-frame animation (24 fps) bends the middle bone 60°
# about x and back, twists the top bone, and drives the shape key 0 → 1 → 0.
#
# Run headless:
#   Blender -b --factory-startup --python make_skinned_test.py -- out.glb
import math
import sys

import bpy
import bmesh

out = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 48

# The tube: rings from z = 0 to 2, capped.
mesh = bpy.data.meshes.new("tube")
bm = bmesh.new()
segments, rings, radius, height = 32, 24, 0.3, 2.0
grid = []
for r in range(rings + 1):
    z = height * r / rings
    row = []
    for s in range(segments):
        a = 2.0 * math.pi * s / segments
        row.append(bm.verts.new((radius * math.cos(a), radius * math.sin(a), z)))
    grid.append(row)
for r in range(rings):
    for s in range(segments):
        a, b = grid[r][s], grid[r][(s + 1) % segments]
        c, d = grid[r + 1][(s + 1) % segments], grid[r + 1][s]
        bm.faces.new((a, b, c, d))
bm.faces.new(list(reversed(grid[0])))
bm.faces.new(grid[rings])
bmesh.ops.triangulate(bm, faces=bm.faces[:])
bm.normal_update()
bm.to_mesh(mesh)
bm.free()
tube = bpy.data.objects.new("tube", mesh)
scene.collection.objects.link(tube)
for p in mesh.polygons:
    p.use_smooth = True

# Three bones along z.
arm_data = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", arm_data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
names = []
prev = None
for k, (z0, z1) in enumerate([(0.0, 0.7), (0.7, 1.4), (1.4, 2.0)]):
    b = arm_data.edit_bones.new(f"bone{k}")
    b.head = (0.0, 0.0, z0)
    b.tail = (0.0, 0.0, z1)
    if prev is not None:
        b.parent = prev
        b.use_connect = True
    prev = b
    names.append(b.name)
bpy.ops.object.mode_set(mode="OBJECT")

# Skin with automatic weights.
bpy.ops.object.select_all(action="DESELECT")
tube.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

# The shape key: the middle swells.
tube.shape_key_add(name="Basis")
bulge = tube.shape_key_add(name="bulge")
for i, v in enumerate(mesh.vertices):
    co = v.co
    swell = math.exp(-((co.z - 1.0) / 0.35) ** 2)
    bulge.data[i].co = (co.x * (1.0 + 0.6 * swell), co.y * (1.0 + 0.6 * swell), co.z)

# The animation.
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
mid, top = rig.pose.bones[names[1]], rig.pose.bones[names[2]]
for frame, bend, twist, key in [(1, 0.0, 0.0, 0.0), (25, 60.0, 45.0, 1.0), (48, 0.0, 0.0, 0.0)]:
    mid.rotation_mode = "XYZ"
    top.rotation_mode = "XYZ"
    mid.rotation_euler = (math.radians(bend), 0.0, 0.0)
    top.rotation_euler = (0.0, 0.0, math.radians(twist))
    mid.keyframe_insert("rotation_euler", frame=frame)
    top.keyframe_insert("rotation_euler", frame=frame)
    bulge.value = key
    bulge.keyframe_insert("value", frame=frame)
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.export_scene.gltf(
    filepath=out,
    export_format="GLB",
    export_animations=True,
    export_skins=True,
    export_morph=True,
    export_morph_normal=True,
    export_apply=False,
)
print("wrote", out)
