Continuum 0.3.2 brings fabrics, studio lighting and animated characters
to real-world files. Every number below is measured against Blender's
Cycles (through Blender's own glTF importer) or the engine's own reference;
the details are in `docs/results/assets-vs-cycles.md`.

**Try it on a Mac with Apple silicon:** paste this into Terminal, which
installs Continuum Viewer into `~/Applications` and opens it:

    curl -fsSL https://raw.githubusercontent.com/bupfff/continuum/main/scripts/install_macos.sh | sh

Or download `continuum-viewer-0.3.2-macos-arm64.zip` below, unzip it and
open the app; it is not notarized by Apple, so the first time macOS refuses
a browser download: open System Settings → Privacy & Security, scroll down
and click Open Anyway. Then drop a `.glb` or `.gltf` file on the window.

**Animated characters**
- Animations now play under the sky and the studio or outdoor HDRIs: sky
  light through the moving objects' own ray-tracing structure, and the sky
  around them catching up within a few frames (`CacheOptions::sky_lag`):
  an animated fox under a sky in 10.9 ms a frame instead of 78.
- Fixed: moving or deforming objects could vanish from the image in the
  viewer after a few frames (in 0.3.1 too).
- A file's alternative clips of the same rig (Walk, Run) no longer play
  at once.

**Materials and lighting**
- Sheen for fabrics (`KHR_materials_sheen`): a velvet sofa against Cycles
  from 4.35% to 2.11% (relative RMSE).
- Studio HDRIs: their softboxes and lamps become lights with exact soft
  shadows (`RendererOptions::lights_from_environment`, which replaces
  0.3.1's `sun_from_environment`; an outdoor HDRI's sun as before).
- glTF import keeps primitives with different transmission, specular,
  clear coat or sheen apart where they share a mesh.
- Fixed: sunlight through walls onto what stands close in front of them
  (Sponza's lion head and arches lit from behind the wall), from two
  causes: large triangles behind a receiver lost to f32 clipping, and
  bumps beside texels that face away from the light left unshadowed.
  Sponza against Cycles under the sun and sky 65.4% → 46.4% (clamped
  relative RMSE), about 6% longer to warm up.

**The viewer**
- Objects open in the studio with their own lights too; the view from
  inside at eye height only where walls stand around.

**Known limits:** measured on Apple silicon only; see the README's Limits.

Free for noncommercial use (PolyForm Noncommercial 1.0.0). Commercial use
needs a license from Penumbral Labs: open an issue.
