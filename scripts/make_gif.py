"""Turns a folder of PNG frames into a looping GIF (or animated WebP) for the
README.

    python3 scripts/make_gif.py FRAMES_DIR OUT.gif [--width 480] [--fps 20] [--every 1] [--no-dither]
    python3 scripts/make_gif.py FRAMES_DIR OUT.webp [--width 720] [--quality 75]

Frames are downscaled (which also smooths their edges). For a GIF they are
quantized to one shared palette so colors do not flicker between frames, and
written with GIF frame optimization. A moving camera changes every pixel,
which a GIF cannot compress (Sponza walking: 36 MB at 720 wide); an
animated WebP (lossy) holds it in a few MB, and browsers show it as an image.
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
p.add_argument("--no-dither", action="store_true", help="no dithering (much smaller for detailed scenes)")
p.add_argument("--quality", type=int, default=75, help="WebP quality (0-100)")
args = p.parse_args()

paths = sorted(glob.glob(os.path.join(args.frames, "*.png")))[:: args.every]
assert paths, "no frames"
frames = []
for path in paths:
    im = Image.open(path).convert("RGB")
    h = round(im.height * args.width / im.width)
    frames.append(im.resize((args.width, h), Image.LANCZOS))
if args.out.endswith(".webp"):
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        quality=args.quality,
        method=6,
    )
    print(args.out, len(frames), "frames", os.path.getsize(args.out) // 1024, "KB")
    raise SystemExit
# One palette from a strip of sampled frames.
step = max(1, len(frames) // 12)
strip = Image.new("RGB", (frames[0].width, frames[0].height * len(frames[::step])))
for k, f in enumerate(frames[::step]):
    strip.paste(f, (0, k * f.height))
palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
dither = Image.Dither.NONE if args.no_dither else Image.Dither.FLOYDSTEINBERG
quantized = [f.quantize(palette=palette, dither=dither) for f in frames]
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
