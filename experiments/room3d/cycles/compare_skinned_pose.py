# Compares a pose of a skinned glTF file posed by Continuum with Blender's
# own glTF import, armature and shape keys at the same time (M12 gate 2).
#
# Input: the .glb, the time in seconds, and a file of Continuum's posed
# triangle corners (little-endian f32 x, y, z in Blender's Z-up world axes,
# which are Continuum's), and the animation's name (or "" for the first).
# Output (stdout, one line): "max <m> mean <m> corners <n> verts <n>": the
# largest and mean distance from each of Continuum's corners to Blender's
# evaluated surface (its nearest evaluated vertex), and the counts.
#
# Run headless:
#   Blender -b --factory-startup --python compare_skinned_pose.py -- in.glb 0.5 corners.f32 ""
import struct
import sys

import bpy
from mathutils import kdtree

args = sys.argv[sys.argv.index("--") + 1 :]
glb, seconds, corners_path = args[0], float(args[1]), args[2]
name = args[3] if len(args) > 3 else ""

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
scene = bpy.context.scene
# Every animation plays, as glTF means: Blender's importer leaves the first
# one active on each datablock it animates and some only as muted NLA
# tracks (a mesh's shape key weights): those become active.
for ob in scene.objects:
    ids = [ob]
    if ob.type == "MESH" and ob.data.shape_keys:
        ids.append(ob.data.shape_keys)
    for idb in ids:
        ad = idb.animation_data
        if not ad or not ad.nla_tracks:
            continue
        if name:
            for t in ad.nla_tracks:
                t.mute = t.name != name
        elif ad.action is None:
            strips = [s for t in ad.nla_tracks for s in t.strips]
            if strips:
                ad.action = strips[0].action
                # (Blender 4.4 on: an action animates through its slot.)
                if hasattr(ad, "action_slot") and getattr(strips[0], "action_slot", None):
                    ad.action_slot = strips[0].action_slot
fps = scene.render.fps / scene.render.fps_base
# Blender's glTF importer puts time t at frame t × fps (its exporter wrote
# frame f at f / fps: frame 1 at 1/24 s).
frame = seconds * fps
scene.frame_set(int(frame), subframe=frame - int(frame))
depsgraph = bpy.context.evaluated_depsgraph_get()

points = []
for ob in scene.objects:
    if ob.type != "MESH":
        continue
    ev = ob.evaluated_get(depsgraph)
    m = ev.to_mesh()
    mw = ob.matrix_world
    for v in m.vertices:
        points.append(mw @ v.co)
    ev.to_mesh_clear()

tree = kdtree.KDTree(len(points))
for i, p in enumerate(points):
    tree.insert(p, i)
tree.balance()

data = open(corners_path, "rb").read()
n = len(data) // 12
worst = 0.0
total = 0.0
for k in range(n):
    x, y, z = struct.unpack_from("<3f", data, 12 * k)
    _, _, d = tree.find((x, y, z))
    worst = max(worst, d)
    total += d
print(f"max {worst:.6g} mean {total / max(n, 1):.6g} corners {n} verts {len(points)}")
