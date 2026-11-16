# Applies Blender's own view transforms to a linear image (M17): the
# display values Blender shows for it with each view, for comparing
# Continuum's tone maps.
#
# Input: a raw file of little-endian f32 RGB, rows from the top, its width
# and height, and an output stem. Output: for each view (Khronos PBR
# Neutral, AgX, Standard), `<stem>_<view>.f32` with the display values
# (sRGB-encoded, 0–1), rows from the top.
#
# Run headless:
#   Blender -b --factory-startup --python view_transform.py -- in.f32 W H stem
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1 :]
path, w, h, stem = args[0], int(args[1]), int(args[2]), args[3]
rgb = np.fromfile(path, dtype="<f4").reshape(h, w, 3)
rgba = np.ones((h, w, 4), np.float32)
# Blender's images run from the bottom row.
rgba[..., :3] = rgb[::-1]
img = bpy.data.images.new("linear", w, h, float_buffer=True, alpha=True)
img.colorspace_settings.name = "Linear Rec.709"
img.pixels.foreach_set(rgba.ravel())
scene = bpy.context.scene
for view in ["Khronos PBR Neutral", "AgX", "Standard"]:
    scene.view_settings.view_transform = view
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    png = f"{stem}_{view.replace(' ', '_')}.png"
    img.save_render(png, scene=scene)
    out = bpy.data.images.load(png)
    out.colorspace_settings.name = "Non-Color"
    a = np.array(out.pixels[:], np.float32).reshape(h, w, 4)[::-1, :, :3]
    a.astype("<f4").tofile(f"{stem}_{view.replace(' ', '_')}.f32")
print("views done")
