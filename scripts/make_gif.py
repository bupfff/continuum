"""Turns a folder of PNG frames into a looping GIF for the README.

    python3 scripts/make_gif.py FRAMES_DIR OUT.gif [--width 480] [--fps 20] [--every 1]

Frames are downscaled (which also smooths their edges), quantized to one
shared palette so colors do not flicker between frames, and written with
GIF frame optimization.
"""

import argparse
import glob
import os

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("frames")
p.add_argument("out")
p.add_argument("--width", type=int, default=480)
p.add_argument("--fps", type=float, default=20)
p.add_argument("--every", type=int, default=1, help="keep every n-th frame")
args = p.parse_args()

paths = sorted(glob.glob(os.path.join(args.frames, "*.png")))[:: args.every]
assert paths, "no frames"
frames = []
for path in paths:
    im = Image.open(path).convert("RGB")
    h = round(im.height * args.width / im.width)
    frames.append(im.resize((args.width, h), Image.LANCZOS))
# One palette from a strip of sampled frames.
step = max(1, len(frames) // 12)
strip = Image.new("RGB", (frames[0].width, frames[0].height * len(frames[::step])))
for k, f in enumerate(frames[::step]):
    strip.paste(f, (0, k * f.height))
palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
quantized = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
quantized[0].save(
    args.out,
    save_all=True,
    append_images=quantized[1:],
    duration=round(1000 / args.fps),
    loop=0,
    optimize=True,
    disposal=1,
)
print(args.out, len(frames), "frames", os.path.getsize(args.out) // 1024, "KB")
