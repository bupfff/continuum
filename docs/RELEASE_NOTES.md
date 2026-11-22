Continuum 0.3.1 makes real-world files look right and run well. Every
number below is measured against Blender's Cycles (through Blender's own
glTF importer) or the engine's own reference; the details are in
`docs/results/assets-vs-cycles.md`.

**Try it on a Mac with Apple silicon:** paste this into Terminal, which
installs Continuum Viewer into `~/Applications` and opens it:

    curl -fsSL https://raw.githubusercontent.com/bupfff/continuum/main/scripts/install_macos.sh | sh

Or download `continuum-viewer-0.3.1-macos-arm64.zip` below, unzip it and
open the app; it is not notarized by Apple, so the first time macOS refuses
a browser download: open System Settings → Privacy & Security, scroll down
and click Open Anyway. Then drop a `.glb` or `.gltf` file on the window.

**The viewer**
- A Lighting menu: the sun, the sun and sky, a studio or an outdoor HDRI
  (both CC0, from Poly Haven, built in), or your own `.hdr`. Objects open
  in the studio, larger scenes outdoors and from inside at eye height.

**Real-world files**
- Alpha-cutout foliage (Sponza's ivy and flowers): 3.0 million triangles
  instead of 24.9 million, 1.3 GB of GPU memory instead of 6.7, ready in
  under 9 s instead of about 70.
- Glossy reflections through hardware ray queries: Sponza's frames 116 →
  36 ms at 1280×720 with the same image.
- HDR environments: the image keeps its range (RGBM), rough surfaces
  reflect it blurred, and an outdoor HDRI's sun becomes a sun light with
  exact soft shadows. A helmet under two HDRIs matches Cycles' mean within
  0.4%.
- Metals 15–20% brighter where they were too dark (the split sum's
  directional albedo from a table), metal seen in metal no longer black,
  rough surfaces reflect what the cache knows they see.
- Sky light through narrow openings no longer shows sharp strips
  (Sponza's floor).

**Known limits:** measured on Apple silicon only; animations do not play
under a sky or HDRI yet; see the README's Limits.

Free for noncommercial use (PolyForm Noncommercial 1.0.0). Commercial use
needs a license from Penumbral Labs: open an issue.
