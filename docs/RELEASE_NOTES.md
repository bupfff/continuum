Continuum 0.3.0 makes it easy to try and adds most of what a finished
image needs. Every number below is measured against Blender's Cycles or
the engine's own exact reference; the details are in `docs/results/`.

**Try it on a Mac with Apple silicon:** download
`continuum-viewer-0.3.0-macos-arm64.zip` below, unzip it and open
Continuum Viewer. The app is not notarized by Apple, so the first time
macOS refuses it: open System Settings → Privacy & Security, scroll down
and click Open Anyway. Then pick a scene in the panel, or drop a `.glb` or
`.gltf` file on the window. Only Apple silicon Macs are tested so far;
other platforms build from source (see the README).

**The viewer**
- A panel: the showcase, a field of up to a million boulders, or your own
  glTF file (Open…, or drag and drop); bounce light, anti-aliasing, the
  sky, render scale, tone map, bloom, exposure and vsync; frame rate, GPU
  memory, and whether the lighting has settled.
- Problems show in the panel or a dialog instead of a terminal.

**Lighting**
- Bounce light is on by default in `Renderer` (1.06–1.12× the frame), and a
  settled view's bounce light now matches a fresh bake (0.1–0.8% of pages
  off by more than 1%, from 2.4–7.1%).
- Bounce light stays right while objects move (a ledger re-bakes what a
  move can change), and several bounces on request.
- Settling under the default time budget is faster where the GPU's
  timestamps are unreliable (Apple GPUs).

**Scale and detail**
- Unique geometry streamed beyond memory: 4,096 unique rocks carrying 33
  GB of detail produced on demand, and fields of 36,864 flown with no
  frame over 16.7 ms on the M4, with lighting streamed too and still
  exact.
- Coarser versions of imported meshes (`InstancedScene::simplify_geometries`).
- An experimental sub-pixel path for fields of tiny distant objects
  (`RendererOptions::sub_pixel`).

**The image**
- Anti-aliasing on by default, with nothing taken from past frames.
- Glass, thin and solid (refraction, absorption), clear coats, index of
  refraction and specular color (`KHR_materials_transmission`, `_volume`,
  `_clearcoat`, `_ior`, `_specular`).
- PBR Neutral and AgX tone maps, auto-exposure, bloom, upscaling, fog with
  light shafts, depth of field and motion blur.

**Animation**
- glTF skins and morph targets, posed within a micrometre of Blender's own
  evaluation, and any mesh the application deforms each frame.

**Known limits:** measured on Apple silicon only; bounce light is not yet
available with moving lights; see the README's Limits.

Free for noncommercial use (PolyForm Noncommercial 1.0.0). Commercial use
needs a license from Penumbral Labs: open an issue.
